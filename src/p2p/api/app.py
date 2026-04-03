import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp

from p2p.api.routes import router
from p2p.api.session_enrichment_service import shutdown_pool
from p2p.scheduler.routes import scheduler_router

logger = logging.getLogger(__name__)

_ORPHAN_CHECK_INTERVAL = 30  # seconds
_NODE_MONITOR_INTERVAL = 10  # seconds


def _cleanup_orphan_jobs() -> int:
    """Find jobs whose scheduler is dead but runs are still 'running'.

    Kills remote processes and marks them as error.  Returns the number
    of orphaned runs cleaned up.
    """
    from p2p.scheduler.manifest_io import list_job_ids, read_job_manifest, write_job_manifest
    from p2p.scheduler.ssh_utils import find_node, kill_ssh_process
    from p2p.scheduler.types import now_iso
    from p2p.utils.process_safety import (
        force_kill_pids,
        get_descendant_pids,
        safe_killpg,
        verify_pid_ownership,
    )

    cleaned = 0
    for job_id in list_job_ids():
        manifest = read_job_manifest(job_id)
        if manifest is None or manifest.get("status") != "running":
            continue

        pid = manifest.get("scheduler_pid")
        if pid and verify_pid_ownership(pid, expected_cmdline="p2p.scheduler.job_scheduler"):
            continue  # scheduler alive and verified, nothing to do

        # Scheduler is dead — clean up orphaned runs
        has_orphans = False
        for run in manifest.get("runs", []):
            if run.get("state") != "running":
                continue
            has_orphans = True
            cleaned += 1

            # Kill remote process
            rpid = run.get("pid")
            node_id = run.get("node_id", "")
            if rpid and node_id and node_id != "local":
                node = find_node(node_id)
                if node:
                    kill_ssh_process(pid=rpid, node=node)
            elif rpid and node_id == "local":
                session_id = run.get("spec", {}).get("parameters", {}).get("session_id", "")
                children = get_descendant_pids(rpid)
                if safe_killpg(rpid, expected_cmdline=session_id or None):
                    force_kill_pids(children)

            run["state"] = "error"
            run["error"] = "Orphaned: scheduler process died"
            run["completed_at"] = now_iso()

        if has_orphans:
            # Check if all runs are terminal now
            all_terminal = all(
                r.get("state") in ("completed", "error", "cancelled")
                for r in manifest.get("runs", [])
            )
            if all_terminal:
                manifest["status"] = "error"
                manifest["error"] = "Scheduler process died with running jobs"
                manifest["completed_at"] = now_iso()
            manifest.pop("scheduler_pid", None)
            write_job_manifest(manifest)
            logger.info(
                "Cleaned up orphaned job %s: killed %d run(s)",
                job_id,
                sum(1 for r in manifest["runs"] if r.get("error", "").startswith("Orphaned")),
            )

    return cleaned


async def _orphan_cleanup_loop() -> None:
    """Background task that periodically cleans up orphaned jobs."""
    while True:
        await asyncio.sleep(_ORPHAN_CHECK_INTERVAL)
        try:
            cleaned = await asyncio.get_event_loop().run_in_executor(None, _cleanup_orphan_jobs)
            if cleaned:
                logger.info("Orphan cleanup: %d run(s) cleaned", cleaned)
        except Exception:
            logger.exception("Orphan cleanup failed")


async def _node_monitor_loop() -> None:
    """Background task that polls remote node resources."""
    while True:
        await asyncio.sleep(_NODE_MONITOR_INTERVAL)
        try:
            from p2p.api.node_monitor import poll_all_nodes

            await asyncio.get_event_loop().run_in_executor(None, poll_all_nodes)
        except Exception:
            logger.exception("Node monitor poll failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Startup: recover sessions that were running when backend last stopped
    from p2p.api.process_manager import recover_stale_sessions

    actions = recover_stale_sessions()
    if actions:
        for sid, action in actions.items():
            logger.info("Session recovery: %s → %s", sid, action)
        logger.info("Recovered %d stale session(s)", len(actions))

    # Recover stale job scheduler subprocesses (benchmarks + sessions)
    from p2p.scheduler.controllers import _spawn_job_scheduler
    from p2p.scheduler.manifest_io import list_job_ids, read_job_manifest

    for job_id in list_job_ids():
        manifest = read_job_manifest(job_id)
        if manifest and manifest.get("status") == "running":
            pid = manifest.get("scheduler_pid")
            if pid:
                from p2p.utils.process_safety import verify_pid_ownership

                if verify_pid_ownership(pid, expected_cmdline="p2p.scheduler.job_scheduler"):
                    continue  # subprocess still alive and verified
            logger.info("Re-spawning job scheduler for stale job: %s", job_id)
            _spawn_job_scheduler(job_id)

    # Seed psutil baseline so the first /resources/cpu-usage call
    # returns meaningful values instead of all zeros.
    import psutil

    psutil.cpu_percent(interval=None, percpu=True)

    # Start background orphan cleanup loop
    cleanup_task = asyncio.create_task(_orphan_cleanup_loop())
    monitor_task = asyncio.create_task(_node_monitor_loop())

    yield

    # Shutdown
    cleanup_task.cancel()
    monitor_task.cancel()
    shutdown_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="Prompt2Policy API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure browsers revalidate cached static files (etag handles 304).
    # Uses raw ASGI middleware instead of BaseHTTPMiddleware to avoid
    # buffering FileResponse bodies (which breaks browser video streaming).
    class _NoCacheStaticMiddleware:
        def __init__(self, asgi_app: ASGIApp) -> None:
            self.app = asgi_app

        async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
            if scope["type"] != "http" or not scope["path"].startswith("/static/"):
                await self.app(scope, receive, send)
                return

            async def _send(message):  # type: ignore[no-untyped-def]
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"cache-control", b"no-cache"))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, _send)

    app.add_middleware(_NoCacheStaticMiddleware)

    # Static files for iteration artifacts (videos, frames)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    app.mount("/static/runs", StaticFiles(directory=str(runs_dir)), name="run_files")

    app.include_router(router, prefix="/api")
    # Scheduler runs as a separate track from process_manager (see scheduler/__init__.py)
    app.include_router(scheduler_router, prefix="/api")
    return app


app = create_app()
