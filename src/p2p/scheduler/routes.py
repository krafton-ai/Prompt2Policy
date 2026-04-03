"""FastAPI routes for the scheduler module."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from p2p.api.services import generate_hp_configs
from p2p.config import loop_config_from_params
from p2p.scheduler import node_store
from p2p.scheduler.backend import check_node, setup_node
from p2p.scheduler.benchmark_aggregation import (
    get_benchmark_case_metrics,
    get_job_benchmark,
)
from p2p.scheduler.controllers import (
    BenchmarkController,
)
from p2p.scheduler.job_queries import (
    cancel_job,
    get_job,
    list_jobs,
    sync_job_all,
    sync_job_run,
)
from p2p.scheduler.manifest_io import read_job_manifest
from p2p.scheduler.schemas import (
    JobListResponse,
    JobResponse,
    NodeCheckResponse,
    NodeCreateRequest,
    NodeResponse,
    NodeUpdateRequest,
    RunStatusResponse,
    SubmitBenchmarkJobRequest,
)
from p2p.scheduler.types import NodeConfig
from p2p.settings import RUNS_DIR, resolve_session_dir

scheduler_router = APIRouter(prefix="/scheduler", tags=["scheduler"])


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------


@scheduler_router.get("/nodes", response_model=list[NodeResponse])
def api_list_nodes() -> list[NodeResponse]:
    nodes = node_store.list_nodes()
    return [NodeResponse(**n, online=False, active_runs=0) for n in nodes]


@scheduler_router.post("/nodes", response_model=NodeResponse, status_code=201)
def api_add_node(req: NodeCreateRequest) -> NodeResponse:
    config: NodeConfig = {
        "node_id": req.node_id,
        "host": req.host,
        "user": req.user,
        "port": req.port,
        "base_dir": req.base_dir,
        "max_cores": req.max_cores,
        "enabled": req.enabled,
    }
    try:
        node_store.add_node(config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return NodeResponse(**config)


@scheduler_router.put("/nodes/{node_id}", response_model=NodeResponse)
def api_update_node(node_id: str, req: NodeUpdateRequest) -> NodeResponse:
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        updated = node_store.update_node(node_id, updates)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return NodeResponse(**updated)


@scheduler_router.delete("/nodes/{node_id}")
def api_remove_node(node_id: str) -> dict:
    removed = node_store.remove_node(node_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return {"detail": f"Node '{node_id}' removed"}


@scheduler_router.post("/nodes/{node_id}/check", response_model=NodeCheckResponse)
def api_check_node(node_id: str) -> NodeCheckResponse:
    node = node_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    result = check_node(node_id)
    return NodeCheckResponse(node_id=node_id, **result)


@scheduler_router.post("/nodes/{node_id}/setup")
def api_setup_node(node_id: str) -> dict:
    """Install uv, sync code, and run uv sync on a node."""
    node = node_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return setup_node(node_id)


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


@scheduler_router.post("/jobs/benchmark", response_model=JobResponse, status_code=201)
def api_submit_benchmark_job(req: SubmitBenchmarkJobRequest) -> JobResponse:
    loop_config = loop_config_from_params(
        total_timesteps=req.total_timesteps,
        seed=req.seed,
        num_envs=req.num_envs,
        side_info=req.side_info,
        trajectory_stride=req.trajectory_stride,
        device=req.device,
        max_iterations=req.max_iterations,
        pass_threshold=req.pass_threshold,
        model=req.model,
        vlm_model=req.vlm_model,
        thinking_effort=req.thinking_effort,
        refined_initial_frame=req.refined_initial_frame,
        criteria_diagnosis=req.criteria_diagnosis,
        motion_trail_dual=req.motion_trail_dual,
        cores_per_run=req.cores_per_run,
        hp_tuning=req.hp_tuning,
        use_code_judge=req.use_code_judge,
        review_reward=req.review_reward,
        review_judge=req.review_judge,
        judgment_select=req.judgment_select,
        use_zoo_preset=req.use_zoo_preset,
    )
    # Convert num_configs -> configs list with proper config_id/label/params
    configs = generate_hp_configs(req.num_configs) if req.num_configs > 1 else None
    ctrl = BenchmarkController()
    job = ctrl.run(
        loop_config=loop_config,
        backend=req.backend,
        csv_file=req.csv_file,
        mode=req.mode,
        num_stages=req.num_stages,
        gate_threshold=req.gate_threshold,
        start_from_stage=req.start_from_stage,
        max_parallel=req.max_parallel,
        configs=configs,
        seeds=req.seeds,
        filter_envs=req.filter_envs or None,
        filter_categories=req.filter_categories or None,
        filter_difficulties=req.filter_difficulties or None,
        allowed_nodes=req.allowed_nodes or None,
    )
    return JobResponse(**job)


@scheduler_router.get("/jobs", response_model=JobListResponse)
def api_list_jobs() -> JobListResponse:
    jobs = list_jobs()
    return JobListResponse(jobs=[JobResponse(**j) for j in jobs])


@scheduler_router.get("/jobs/{job_id}", response_model=JobResponse)
def api_get_job(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(**job)


@scheduler_router.delete("/jobs/{job_id}")
def api_delete_job(job_id: str) -> dict:
    from p2p.api.entity_lifecycle import delete_job

    try:
        delete_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"detail": "Job moved to trash"}


@scheduler_router.post("/jobs/{job_id}/restore")
def api_restore_job(job_id: str) -> dict:
    from p2p.api.entity_lifecycle import restore_job

    try:
        restore_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"detail": "Job restored"}


@scheduler_router.post("/jobs/bulk-trash")
def api_bulk_trash_jobs(req: dict) -> dict:
    from p2p.api.entity_lifecycle import bulk_trash_jobs

    job_ids: list[str] = req.get("job_ids", [])
    if not job_ids:
        raise HTTPException(status_code=400, detail="job_ids is required")
    trashed, failed = bulk_trash_jobs(job_ids)
    detail = f"Moved {trashed} job(s) to trash"
    if failed:
        detail += f", {len(failed)} failed"
    return {"trashed": trashed, "failed": failed, "detail": detail}


@scheduler_router.post("/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    cancel_job(job_id)
    return {"detail": "Job cancellation requested"}


@scheduler_router.get("/jobs/{job_id}/disk-usage")
def api_get_job_disk_usage(job_id: str) -> dict:
    """Return disk usage breakdown for a job's run directory."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Find the benchmark root directory from run_ids.
    # Benchmark run_ids look like bm_XXXXXXXX_caseN; resolve_session_dir gives
    # RUNS_DIR/bm_XXX/caseN, so .parent is the bm_ root.
    run_ids: list[str] = job.get("run_ids", [])
    target_dir: Path | None = None
    for run_id in run_ids:
        try:
            session_dir = resolve_session_dir(run_id)
        except ValueError:
            continue
        candidate = session_dir.parent
        # For plain sessions parent == RUNS_DIR; skip those
        if candidate != RUNS_DIR and candidate.is_dir():
            target_dir = candidate
            break
        # Plain session: use session_dir itself
        if session_dir.is_dir():
            target_dir = session_dir
            break

    if target_dir is None:
        return {"total_bytes": 0, "trajectory_bytes": 0, "video_bytes": 0, "checkpoint_bytes": 0}

    total_bytes = 0
    trajectory_bytes = 0
    video_bytes = 0
    checkpoint_bytes = 0

    for f in target_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        total_bytes += size
        suffix = f.suffix.lower()
        if suffix == ".jsonl":
            trajectory_bytes += size
        elif suffix == ".mp4":
            video_bytes += size
        elif suffix == ".zip":
            checkpoint_bytes += size

    return {
        "total_bytes": total_bytes,
        "trajectory_bytes": trajectory_bytes,
        "video_bytes": video_bytes,
        "checkpoint_bytes": checkpoint_bytes,
    }


@scheduler_router.get("/jobs/{job_id}/benchmark")
def api_get_job_benchmark(job_id: str) -> dict:
    """Get aggregated benchmark data for a scheduler benchmark job."""
    detail = get_job_benchmark(job_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark data not found for job '{job_id}'",
        )
    return detail


@scheduler_router.get("/jobs/{job_id}/cases/{case_index}/metrics")
def api_get_benchmark_case_metrics(job_id: str, case_index: int, iteration: int = 0) -> dict:
    """Get cross-config aggregated training metrics for a benchmark case."""
    result = get_benchmark_case_metrics(job_id, case_index, iteration)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for job '{job_id}' case {case_index}",
        )
    return result


@scheduler_router.post("/jobs/{job_id}/sync")
def api_sync_job(job_id: str) -> dict:
    """Sync all unsynced runs for a job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return sync_job_all(job_id)


@scheduler_router.post("/runs/{run_id}/sync")
def api_sync_run(
    run_id: str,
    job_id: str | None = None,
    mode: str = "full",
) -> dict:
    """Sync a single run's results from the remote node.

    Args:
        mode: "lite" excludes videos/trajectories, "full" syncs everything.
    """
    if job_id:
        return sync_job_run(job_id, run_id, mode=mode)

    # Find the job containing this run
    from p2p.scheduler.manifest_io import list_job_ids

    for jid in list_job_ids():
        manifest = read_job_manifest(jid)
        if manifest is None:
            continue
        for run in manifest["runs"]:
            if run["run_id"] == run_id:
                return sync_job_run(jid, run_id, mode=mode)
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


# ---------------------------------------------------------------------------
# Run status (reads from manifest)
# ---------------------------------------------------------------------------


@scheduler_router.get("/runs", response_model=list[RunStatusResponse])
def api_list_runs() -> list[RunStatusResponse]:
    """List all runs across all jobs."""
    from p2p.scheduler.manifest_io import list_job_ids

    results: list[RunStatusResponse] = []
    for jid in list_job_ids():
        manifest = read_job_manifest(jid)
        if manifest is None:
            continue
        for run in manifest["runs"]:
            results.append(
                RunStatusResponse(
                    run_id=run["run_id"],
                    state=run["state"],
                    node_id=run.get("node_id", ""),
                    pid=run.get("pid"),
                    started_at=run.get("started_at"),
                    completed_at=run.get("completed_at"),
                    error=run.get("error"),
                )
            )
    return results


@scheduler_router.get("/runs/{run_id}", response_model=RunStatusResponse)
def api_get_run(run_id: str) -> RunStatusResponse:
    from p2p.scheduler.manifest_io import list_job_ids

    for jid in list_job_ids():
        manifest = read_job_manifest(jid)
        if manifest is None:
            continue
        for run in manifest["runs"]:
            if run["run_id"] == run_id:
                return RunStatusResponse(
                    run_id=run["run_id"],
                    state=run["state"],
                    node_id=run.get("node_id", ""),
                    pid=run.get("pid"),
                    started_at=run.get("started_at"),
                    completed_at=run.get("completed_at"),
                    error=run.get("error"),
                )
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@scheduler_router.get("/runs/{run_id}/log")
def api_get_run_log(run_id: str, tail: int = 100) -> dict:
    """Read the last N lines of a run's subprocess.log."""
    try:
        log_path = (resolve_session_dir(run_id) / "subprocess.log").resolve()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    if not str(log_path).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid run_id")

    if not log_path.exists():
        return {"run_id": run_id, "log": "", "available": False}
    try:
        lines = log_path.read_text().splitlines()
        return {"run_id": run_id, "log": "\n".join(lines[-tail:]), "available": True}
    except OSError:
        return {"run_id": run_id, "log": "", "available": False}
