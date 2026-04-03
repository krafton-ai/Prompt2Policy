"""Job query helpers: manifest-based CRUD and sync operations.

Provides functions to query, list, cancel, and sync scheduler jobs
by reading/writing job manifests on disk.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from p2p.api.entity_lifecycle import is_entity_deleted
from p2p.scheduler.manifest_io import (
    _jobs_dir,
    list_job_ids,
    read_job_manifest,
    write_job_manifest,
)
from p2p.scheduler.types import TERMINAL_STATES, Job, JobManifest, now_iso
from p2p.settings import RUNS_DIR
from p2p.utils.process_safety import verify_pid_ownership

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_allocation(manifest: JobManifest) -> dict[str, Any]:
    """Compute resource allocation metadata from manifest runs."""
    runs = manifest["runs"]

    # state_counts: {"pending": 2, "running": 3, ...}
    state_counts: dict[str, int] = defaultdict(int)
    for r in runs:
        state_counts[r["state"]] += 1

    # node_allocation: {"gpu-1": {"total": 3, "running": 2}, "unassigned": {"total": 1}}
    node_alloc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in runs:
        node = r.get("node_id") or "unassigned"
        if not node:
            node = "unassigned"
        node_alloc[node]["total"] += 1
        node_alloc[node][r["state"]] += 1

    # session_affinity: all runs share the same session_group
    groups = {r.get("session_group", "") for r in runs}
    groups.discard("")
    session_affinity = len(groups) == 1 and len(runs) > 0

    # affinity_node: the node assigned when affinity applies
    affinity_node: str | None = None
    if session_affinity:
        assigned_nodes = {r.get("node_id") for r in runs if r.get("node_id")}
        if len(assigned_nodes) == 1:
            affinity_node = assigned_nodes.pop()

    return {
        "state_counts": dict(state_counts),
        "node_allocation": {k: dict(v) for k, v in node_alloc.items()},
        "session_affinity": session_affinity,
        "affinity_node": affinity_node,
        "sessions": _compute_sessions(manifest),
    }


def _compute_sessions(manifest: JobManifest) -> list[dict[str, Any]]:
    """Group runs by session_group and compute per-session summaries."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in manifest["runs"]:
        sg = r.get("session_group") or r["run_id"]
        grouped[sg].append(r)

    sessions: list[dict[str, Any]] = []
    for sg, sg_runs in grouped.items():
        sc: dict[str, int] = defaultdict(int)
        for r in sg_runs:
            sc[r["state"]] += 1

        # Node from first assigned run
        node_id = ""
        for r in sg_runs:
            if r.get("node_id"):
                node_id = r["node_id"]
                break

        # Extract test-case metadata from spec tags / parameters
        first_spec = sg_runs[0].get("spec", {})
        tags = first_spec.get("tags", {})
        params = first_spec.get("parameters", {})

        sessions.append(
            {
                "session_group": sg,
                "case_index": tags.get("case_index"),
                "node_id": node_id,
                "total_runs": len(sg_runs),
                "state_counts": dict(sc),
                "run_ids": [r["run_id"] for r in sg_runs],
                "env_id": params.get("env_id", ""),
                "instruction": params.get("prompt", ""),
            }
        )

    return sessions


def _manifest_to_job(manifest: JobManifest) -> Job:
    """Convert a manifest to the Job TypedDict for API responses."""
    metadata: dict[str, Any] = dict(manifest.get("metadata", {}))
    metadata.update(_compute_allocation(manifest))

    config = dict(manifest.get("config", {}))

    # Backfill cores_per_run / num_envs for manifests that lack these fields.
    if "cores_per_run" not in config and manifest["runs"]:
        first_spec = manifest["runs"][0].get("spec", {})
        config["cores_per_run"] = first_spec.get("cpu_cores", 0)
    if "num_envs" not in config and manifest["runs"]:
        first_spec = manifest["runs"][0].get("spec", {})
        lc_json = first_spec.get("parameters", {}).get("loop_config", "")
        if lc_json:
            try:
                lc = json.loads(lc_json)
                config["num_envs"] = lc.get("train", {}).get("num_envs", 0)
            except (json.JSONDecodeError, TypeError):
                pass

    # Derive effective job status from run states: if manifest says
    # "cancelled" but some runs are still running, report as "running".
    status = manifest["status"]
    if status == "cancelled":
        runs = manifest.get("runs", [])
        has_active = any(r.get("state") not in TERMINAL_STATES for r in runs)
        if has_active:
            status = "running"

    job: Job = {
        "job_id": manifest["job_id"],
        "job_type": manifest["job_type"],
        "run_ids": [r["run_id"] for r in manifest["runs"]],
        "status": status,
        "created_at": manifest["created_at"],
        "metadata": metadata,
        "backend": manifest.get("backend", "local"),
        "config": config,
    }
    if "completed_at" in manifest:
        job["completed_at"] = manifest["completed_at"]
    if "error" in manifest:
        job["error"] = manifest["error"]
    return job


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_job(job_id: str) -> Job | None:
    manifest = read_job_manifest(job_id)
    if manifest is None:
        return None
    return _manifest_to_job(manifest)


def list_jobs() -> list[Job]:
    jobs: list[Job] = []
    for job_id in list_job_ids():
        if is_entity_deleted(_jobs_dir() / job_id):
            continue
        manifest = read_job_manifest(job_id)
        if manifest is not None:
            jobs.append(_manifest_to_job(manifest))
    return jobs


def find_job_for_session(session_id: str) -> str | None:
    """Find the job_id managing a given session.

    Scans running/pending job manifests for a run whose
    ``spec.parameters.session_id`` matches *session_id*.
    Returns the job_id if found, ``None`` otherwise.
    """
    for job_id in list_job_ids():
        manifest = read_job_manifest(job_id)
        if manifest is None:
            continue
        if manifest.get("status") not in ("running", "pending"):
            continue
        for run in manifest.get("runs", []):
            sid = run.get("spec", {}).get("parameters", {}).get("session_id", "")
            if sid == session_id and run.get("state") in ("running", "pending"):
                return job_id
    return None


def cancel_job(job_id: str) -> None:
    """Cancel a job by killing all running/pending runs.

    If the scheduler subprocess is alive it will detect the cancelled
    status on its next poll loop.  If it is dead we kill the run
    processes directly and update run states ourselves.

    The job-level status transitions to ``"cancelled"`` only after all
    runs have reached a terminal state.
    """
    manifest = read_job_manifest(job_id)
    if manifest is None:
        return

    # Verify the PID belongs to the scheduler — not a recycled PID (issue #380).
    pid = manifest.get("scheduler_pid")
    scheduler_alive = pid is not None and verify_pid_ownership(
        pid, expected_cmdline="p2p.scheduler.job_scheduler"
    )

    if scheduler_alive:
        # Scheduler is alive — just set the flag; it will handle the rest.
        manifest["status"] = "cancelled"
        write_job_manifest(manifest)
        return

    # Scheduler is dead — kill runs directly and update states.
    from p2p.scheduler.job_scheduler import kill_run_process_standalone

    runs = manifest.get("runs", [])
    for run in runs:
        if run.get("state", "") not in ("running", "pending"):
            continue
        if run.get("state") == "running":
            try:
                kill_run_process_standalone(run)
            except Exception as exc:
                logger.warning("Failed to kill run %s: %s", run.get("run_id"), exc)
        run["state"] = "cancelled"
        run["completed_at"] = now_iso()

    # Only mark job as cancelled when all runs are terminal.
    if all(r.get("state", "") in TERMINAL_STATES for r in runs):
        manifest["status"] = "cancelled"
        manifest["completed_at"] = now_iso()
        manifest.pop("scheduler_pid", None)

    write_job_manifest(manifest)


def is_scheduler_alive(job_id: str) -> bool:
    """Check if the job scheduler subprocess is still running.

    Verifies PID ownership to guard against recycled PIDs (issue #380).
    """
    manifest = read_job_manifest(job_id)
    if manifest is None:
        return False
    pid = manifest.get("scheduler_pid")
    if pid is None:
        return False
    return verify_pid_ownership(pid, expected_cmdline="p2p.scheduler.job_scheduler")


def sync_job_run(job_id: str, run_id: str, *, mode: str = "full") -> dict:
    """Manually sync a single run's results from the remote node.

    Args:
        mode: "lite" excludes videos/trajectories, "full" syncs everything.

    Returns a dict with ``synced`` (bool), ``mode`` (str), and ``error``.
    """
    from p2p.scheduler.ssh_utils import (
        find_node,
        is_localhost,
        sync_full_results,
        sync_lite_results,
    )

    manifest = read_job_manifest(job_id)
    if manifest is None:
        return {"synced": False, "mode": mode, "error": "Job not found"}

    target: dict | None = None
    for run in manifest["runs"]:
        if run["run_id"] == run_id:
            target = run
            break
    if target is None:
        return {"synced": False, "mode": mode, "error": "Run not found in job"}

    # For full sync, skip if already fully synced
    if mode == "full" and target.get("synced"):
        return {"synced": True, "mode": mode, "error": None}

    node_id = target.get("node_id", "")
    if not node_id or node_id == "local":
        return {"synced": True, "mode": mode, "error": None}

    node = find_node(node_id)
    # SSH localhost node — results are already local, no sync needed.
    if node is not None and is_localhost(node):
        if mode == "full":
            target["synced"] = True
            write_job_manifest(manifest)
        return {"synced": True, "mode": mode, "error": None}
    if node is None:
        return {"synced": False, "mode": mode, "error": f"Node '{node_id}' not found"}

    remote_dir = target.get("remote_dir", "")
    session_id = target["spec"]["parameters"].get("session_id", run_id)

    sync_fn = sync_lite_results if mode == "lite" else sync_full_results
    ok = sync_fn(
        session_id=session_id,
        node=node,
        remote_dir=remote_dir,
        runs_dir=RUNS_DIR,
    )

    if ok:
        if mode == "full":
            target["synced"] = True
            write_job_manifest(manifest)
        return {"synced": True, "mode": mode, "error": None}
    return {"synced": False, "mode": mode, "error": "rsync failed"}


def sync_job_all(job_id: str) -> dict:
    """Sync all unsynced runs for a job.

    Returns ``{"synced": <count>, "failed": <count>, "skipped": <count>}``.
    """
    from p2p.scheduler.ssh_utils import find_node, is_localhost, sync_full_results

    manifest = read_job_manifest(job_id)
    if manifest is None:
        return {"synced": 0, "failed": 0, "skipped": 0}

    synced = 0
    failed = 0
    skipped = 0

    for run in manifest["runs"]:
        if run.get("synced"):
            skipped += 1
            continue

        node_id = run.get("node_id", "")
        if not node_id or node_id == "local":
            run["synced"] = True
            skipped += 1
            continue

        node = find_node(node_id)
        # SSH localhost node — already local, skip sync.
        if node is not None and is_localhost(node):
            run["synced"] = True
            skipped += 1
            continue
        if node is None:
            failed += 1
            continue

        remote_dir = run.get("remote_dir", "")
        session_id = run["spec"]["parameters"].get("session_id", run["run_id"])

        ok = sync_full_results(
            session_id=session_id,
            node=node,
            remote_dir=remote_dir,
            runs_dir=RUNS_DIR,
        )
        if ok:
            run["synced"] = True
            synced += 1
        else:
            failed += 1

    write_job_manifest(manifest)
    return {"synced": synced, "failed": failed, "skipped": skipped}
