from __future__ import annotations

import dataclasses
import logging
import queue
import subprocess as _subprocess
import threading
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from p2p.analysis.trajectory_metrics import resolve_trajectory_path
from p2p.api import (
    benchmark_service,
    entity_lifecycle,
    process_manager,
    services,
    vlm_service,
)
from p2p.api.schemas import (
    AggregatedMetricsResponse,
    BenchmarkOptionsResponse,
    BenchmarkRunDetail,
    BenchmarkRunSummary,
    CpuUsageResponse,
    ElaborateIntentRequest,
    ElaborateIntentResponse,
    EnvInfo,
    EventDetail,
    EventSummary,
    GpuUsageResponse,
    HumanLabelRequest,
    HumanLabelResponse,
    IntentCriterionSchema,
    IterationDetail,
    IterationRunEntry,
    IterationSummary,
    LabelingStatusResponse,
    LoopIterationSummary,
    MemoryInfo,
    MetricsResponse,
    NodeResourcesResponse,
    ResourceAutoResponse,
    ResourceStatusResponse,
    RunMetricsResponse,
    SessionAnalysisResponse,
    SessionDetail,
    StartSessionRequest,
    StartSessionResponse,
    StopBenchmarkResponse,
    StopResponse,
    TrashItem,
    UpdateMetadataRequest,
    UpdateMetadataResponse,
    VlmChatRequest,
    VlmChatResponse,
    VlmStatusResponse,
)
from p2p.api.sse import sse_event
from p2p.config import LoopConfig, loop_config_from_params
from p2p.contracts import SessionConfig
from p2p.scheduler.controllers import SessionController
from p2p.scheduler.job_queries import cancel_job, find_job_for_session
from p2p.session.iteration_record import SessionRecord
from p2p.session.session_id import generate_session_id
from p2p.settings import resolve_session_dir

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/providers")
def list_providers() -> dict[str, bool]:
    """Report which LLM/VLM provider API keys are configured."""
    from p2p import settings

    return {
        "anthropic": bool(settings.ANTHROPIC_API_KEY),
        "gemini": bool(settings.GEMINI_API_KEY),
        "openai": bool(settings.OPENAI_API_KEY),
    }


@router.get("/envs", response_model=list[EnvInfo])
def list_envs() -> list[EnvInfo]:
    return services.list_envs()


@router.get("/iterations", response_model=list[IterationSummary])
def list_iterations() -> list[IterationSummary]:
    return services.list_iterations()


@router.get("/sessions/{session_id}/iterations/{iteration_id}", response_model=IterationDetail)
def get_session_iteration(session_id: str, iteration_id: str) -> IterationDetail:
    iteration = services.get_iteration(iteration_id, session_id=session_id)
    if iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return iteration


@router.get(
    "/sessions/{session_id}/iterations/{iteration_id}/metrics",
    response_model=MetricsResponse,
)
def get_session_iteration_metrics(session_id: str, iteration_id: str) -> MetricsResponse:
    metrics = services.get_metrics(iteration_id, session_id=session_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return metrics


@router.get("/iterations/{iteration_id}", response_model=IterationDetail)
def get_iteration(iteration_id: str) -> IterationDetail:
    """Legacy endpoint — prefer /sessions/{session_id}/iterations/{iteration_id}."""
    iteration = services.get_iteration(iteration_id)
    if iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return iteration


@router.get("/iterations/{iteration_id}/metrics", response_model=MetricsResponse)
def get_metrics(iteration_id: str) -> MetricsResponse:
    """Legacy endpoint — prefer /sessions/{session_id}/iterations/{iteration_id}/metrics."""
    metrics = services.get_metrics(iteration_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return metrics


@router.post("/elaborate-intent", response_model=ElaborateIntentResponse)
def elaborate_intent_endpoint(req: ElaborateIntentRequest) -> ElaborateIntentResponse:
    from p2p.agents.intent_elicitor import elaborate_intent
    from p2p.inference.llm_client import get_client

    try:
        client = get_client()
        criteria = elaborate_intent(
            req.prompt,
            req.env_id,
            client=client,
            **({"model": req.model} if req.model is not None else {}),
        )
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown env_id: {req.env_id}")
    return ElaborateIntentResponse(criteria=[IntentCriterionSchema(**c) for c in criteria])


# ---------------------------------------------------------------------------
# Session helpers (CPU tracking for scheduler-routed sessions)
# ---------------------------------------------------------------------------


def _allocate_cpu_for_session(session_id: str, loop_config: LoopConfig) -> list[int] | None:
    """Reserve CPU cores in the API server's CPUManager for ResourceBar display."""
    cores_per_run = loop_config.cores_per_run
    if cores_per_run <= 0:
        return None

    from p2p.training.cpu_manager import get_cpu_manager

    cpu_mgr = get_cpu_manager()
    configs = loop_config.configs
    seeds = loop_config.seeds
    num_configs = len(configs) if configs else 1
    num_seeds = len(seeds) if seeds else 1
    total_runs = num_configs * num_seeds
    max_par = loop_config.max_parallel
    if max_par > 0:
        concurrent = min(max_par, total_runs)
    else:
        concurrent = min(
            max(1, cpu_mgr.available_count() // cores_per_run),
            total_runs,
        )
    total_needed = concurrent * cores_per_run
    alloc_id = f"session_{session_id}"
    allocated = cpu_mgr.allocate(alloc_id, total_needed)
    if allocated is not None:
        with process_manager._session_cpu_allocs_lock:
            process_manager._session_cpu_allocs[session_id] = alloc_id
        # Persist for recovery after backend restart
        session_dir = resolve_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "cpu_alloc_count").write_text(str(total_needed))
    return allocated


def _track_scheduler_process(session_id: str, proc: _subprocess.Popen) -> None:
    """Register a scheduler subprocess in _active_procs for cpu-usage tracking."""
    with process_manager._active_procs_lock:
        process_manager._active_procs[session_id] = proc

    def _watchdog() -> None:
        try:
            proc.wait()
        except Exception:
            logger.exception("Watchdog: proc.wait() failed for %s", session_id)
        # Always clean up, even if proc.wait() raised
        with process_manager._active_procs_lock:
            process_manager._active_procs.pop(session_id, None)
        try:
            process_manager._release_session_cores(session_id)
        except Exception:
            logger.exception("Watchdog: failed to release cores for %s", session_id)

    threading.Thread(target=_watchdog, daemon=True, name=f"watch-{session_id}").start()


@router.post("/sessions", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    has_vlm = bool(req.vlm_model)
    if not req.use_code_judge and not has_vlm:
        raise HTTPException(
            status_code=422,
            detail="Please choose at least one judge: code-based or VLM-based",
        )
    configs = [c.model_dump() for c in req.configs] if req.configs else None
    num_configs = req.num_configs if req.hp_tuning else 0
    if not configs and num_configs > 0:
        configs = services.generate_hp_configs(
            max(1, num_configs),
            env_id=req.env_id,
            num_envs=req.num_envs,
        )

    loop_config = loop_config_from_params(
        total_timesteps=req.total_timesteps,
        seed=req.seed,
        env_id=req.env_id,
        num_envs=req.num_envs,
        side_info=req.side_info,
        trajectory_stride=req.trajectory_stride,
        num_evals=req.num_evals,
        configs=configs,
        seeds=req.seeds if req.seeds else None,
        max_iterations=req.max_iterations,
        pass_threshold=req.pass_threshold,
        model=req.model,
        vlm_model=req.vlm_model,
        thinking_effort=req.thinking_effort,
        cores_per_run=req.cores_per_run,
        max_parallel=req.max_parallel,
        hp_tuning=req.hp_tuning,
        use_code_judge=req.use_code_judge,
        review_reward=req.review_reward,
        review_judge=req.review_judge,
        judgment_select=req.judgment_select,
        use_zoo_preset=req.use_zoo_preset,
        elaborated_intent=req.elaborated_intent,
        refined_initial_frame=req.refined_initial_frame,
        criteria_diagnosis=req.criteria_diagnosis,
        motion_trail_dual=req.motion_trail_dual,
        terminate_when_unhealthy=req.terminate_when_unhealthy,
    )
    session_id = generate_session_id()

    # Allocate CPU cores for ResourceBar tracking (API-server-side only).
    # The scheduler subprocess handles actual pinning independently.
    allocated_cores = _allocate_cpu_for_session(session_id, loop_config)
    if allocated_cores:
        loop_config = dataclasses.replace(loop_config, cores_pool=allocated_cores)

    # Route through unified scheduler
    ctrl = SessionController()
    try:
        ctrl.run(
            prompt=req.prompt,
            loop_config=loop_config,
            backend="local",
            session_id=session_id,
        )
    except Exception:
        # Release allocated cores to prevent pool exhaustion
        process_manager._release_session_cores(session_id)
        raise

    # Register scheduler process for cpu-usage endpoint tracking
    if ctrl.scheduler_proc is not None:
        _track_scheduler_process(session_id, ctrl.scheduler_proc)

    return StartSessionResponse(session_id=session_id, status="running")


@router.get("/sessions", response_model=list[SessionDetail])
def list_sessions() -> list[SessionDetail]:
    return services.list_sessions()


@router.get("/sessions/{session_id}/config")
def get_session_config(session_id: str) -> SessionConfig:
    config = services.get_session_config(session_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Session config not found")
    return config


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> SessionDetail:
    session = services.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions/{session_id}/stop", response_model=StopResponse)
def stop_session(session_id: str) -> StopResponse:
    # Try scheduler path first (manifest-based cancel)
    job_id = find_job_for_session(session_id)
    if job_id:
        cancel_job(job_id)
        try:
            sr = SessionRecord(resolve_session_dir(session_id))
            sr.set_status_if("cancelled", only_if=("running", "pending"))
        except OSError:
            logger.warning("Failed to update status.json for session %s", session_id)
        process_manager._release_session_cores(session_id)
        return StopResponse(stopped=True, detail="Session stopped")

    # Fallback: stop via process_manager directly
    stopped = process_manager.stop_session(session_id)
    if not stopped:
        raise HTTPException(status_code=409, detail="Session is not running")
    return StopResponse(stopped=True, detail="Session stopped")


@router.patch("/sessions/{session_id}", response_model=UpdateMetadataResponse)
def update_session(session_id: str, req: UpdateMetadataRequest) -> UpdateMetadataResponse:
    try:
        meta = entity_lifecycle.update_session_metadata(
            session_id, alias=req.alias, starred=req.starred, tags=req.tags
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return UpdateMetadataResponse(
        alias=meta.get("alias", ""),
        starred=meta.get("starred", False),
        tags=meta.get("tags", []),
    )


@router.delete("/sessions/{session_id}", response_model=StopResponse)
def delete_session(session_id: str) -> StopResponse:
    try:
        entity_lifecycle.delete_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StopResponse(stopped=True, detail="Session moved to trash")


@router.post("/sessions/{session_id}/restore", response_model=StopResponse)
def restore_session(session_id: str) -> StopResponse:
    try:
        entity_lifecycle.restore_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return StopResponse(stopped=True, detail="Session restored")


@router.get("/sessions/{session_id}/loop-iterations", response_model=list[LoopIterationSummary])
def get_session_loop_iterations(session_id: str) -> list[LoopIterationSummary]:
    iterations = services.get_session_iterations(session_id)
    if iterations is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return iterations


@router.get(
    "/sessions/{session_id}/iterations/{iter_num}/runs",
    response_model=list[IterationRunEntry],
)
def get_iteration_runs(session_id: str, iter_num: int) -> list[IterationRunEntry]:
    runs = services.get_iteration_runs(session_id, iter_num)
    if runs is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return runs


@router.get(
    "/sessions/{session_id}/iterations/{iter_num}/runs/{run_id}/metrics",
    response_model=RunMetricsResponse,
)
def get_run_metrics(
    session_id: str,
    iter_num: int,
    run_id: str,
) -> RunMetricsResponse:
    result = services.get_run_metrics(session_id, iter_num, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


@router.get(
    "/sessions/{session_id}/iterations/{iter_num}/configs/{config_id}/aggregated-metrics",
    response_model=AggregatedMetricsResponse,
)
def get_aggregated_metrics(
    session_id: str,
    iter_num: int,
    config_id: str,
) -> AggregatedMetricsResponse:
    result = services.get_aggregated_metrics(session_id, iter_num, config_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Config or iteration not found")
    return result


@router.get("/resources/status", response_model=ResourceStatusResponse)
def resource_status() -> ResourceStatusResponse:
    return ResourceStatusResponse(**services.get_resource_status())


@router.get("/resources/cpu-usage", response_model=CpuUsageResponse)
def cpu_usage() -> CpuUsageResponse:
    """Real-time per-core CPU utilization + process→core mapping."""
    import psutil

    from p2p.api.process_manager import _active_procs, _active_procs_lock
    from p2p.api.schemas import CoreProcessInfo, RunProcessInfo

    per_core = psutil.cpu_percent(interval=None, percpu=True)
    avg = sum(per_core) / len(per_core) if per_core else 0.0

    # Build session → pinned cores from active subprocess PIDs
    processes: list[CoreProcessInfo] = []
    with _active_procs_lock:
        snapshot = dict(_active_procs)
    for session_id, proc_or_pid in snapshot.items():
        pid = proc_or_pid if isinstance(proc_or_pid, int) else proc_or_pid.pid
        try:
            p = psutil.Process(pid)
            affinity = p.cpu_affinity() or []
            if not affinity:
                continue

            # Discover individual run subprocesses via process tree.
            # taskset wraps the actual python executor, so both the
            # taskset process and its python child carry --iteration-id.
            # Deduplicate by run_id, keeping the later (actual executor).
            runs_map: dict[str, RunProcessInfo] = {}
            try:
                for child in p.children(recursive=True):
                    try:
                        cmdline = child.cmdline()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    if "--iteration-id" not in cmdline:
                        continue
                    idx = cmdline.index("--iteration-id")
                    run_id = cmdline[idx + 1] if idx + 1 < len(cmdline) else "unknown"
                    try:
                        child_affinity = child.cpu_affinity() or []
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        child_affinity = []
                    runs_map[run_id] = RunProcessInfo(
                        run_id=run_id,
                        pid=child.pid,
                        cores=sorted(child_affinity),
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            runs = sorted(runs_map.values(), key=lambda r: r.run_id)

            processes.append(
                CoreProcessInfo(
                    session_id=session_id,
                    pid=pid,
                    cores=sorted(affinity),
                    runs=runs,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    vm = psutil.virtual_memory()
    used = vm.total - vm.available
    memory = MemoryInfo(
        total_mb=vm.total // (1024 * 1024),
        used_mb=used // (1024 * 1024),
        available_mb=vm.available // (1024 * 1024),
        percent=round(used / vm.total * 100, 1) if vm.total else 0.0,
    )

    return CpuUsageResponse(per_core=per_core, avg=avg, processes=processes, memory=memory)


@router.get("/resources/gpu-usage", response_model=GpuUsageResponse)
def gpu_usage() -> GpuUsageResponse:
    """GPU utilization via nvidia-smi, including per-process memory."""
    import subprocess as sp

    import psutil

    from p2p.api.process_manager import _active_procs, _active_procs_lock
    from p2p.api.schemas import GpuInfo, GpuProcessInfo

    try:
        result = sp.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,temperature.gpu,"
                "utilization.gpu,utilization.memory,"
                "memory.used,memory.total,"
                "power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return GpuUsageResponse(gpus=[])
    except (FileNotFoundError, sp.TimeoutExpired):
        return GpuUsageResponse(gpus=[])

    uuid_to_idx: dict[str, int] = {}
    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue
        idx = int(parts[0])
        uuid_to_idx[parts[2]] = idx
        gpus.append(
            GpuInfo(
                index=idx,
                name=parts[1],
                temperature=float(parts[3]),
                utilization=float(parts[4]),
                memory_utilization=float(parts[5]),
                memory_used_mb=float(parts[6]),
                memory_total_mb=float(parts[7]),
                power_draw_w=float(parts[8]),
                power_limit_w=float(parts[9]),
            )
        )

    if not gpus:
        return GpuUsageResponse(gpus=gpus)

    # Query per-process GPU memory
    try:
        proc_result = sp.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, sp.TimeoutExpired):
        return GpuUsageResponse(gpus=gpus)

    if proc_result.returncode != 0 or not proc_result.stdout.strip():
        return GpuUsageResponse(gpus=gpus)

    # Build PID → (session_id, run_id) map for cross-referencing
    pid_session: dict[int, tuple[str, str]] = {}
    with _active_procs_lock:
        snapshot = dict(_active_procs)
    for sid, proc_or_pid in snapshot.items():
        root_pid = proc_or_pid if isinstance(proc_or_pid, int) else proc_or_pid.pid
        try:
            p = psutil.Process(root_pid)
            pid_session[root_pid] = (sid, "")
            for child in p.children(recursive=True):
                rid = ""
                try:
                    cmd = child.cmdline()
                    if "--iteration-id" in cmd:
                        i = cmd.index("--iteration-id")
                        if i + 1 < len(cmd):
                            rid = cmd[i + 1]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                pid_session[child.pid] = (sid, rid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Parse process entries and group by GPU
    gpu_procs: dict[int, list[GpuProcessInfo]] = {g.index: [] for g in gpus}
    for line in proc_result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_idx = uuid_to_idx.get(parts[0])
        if gpu_idx is None:
            continue
        try:
            pid = int(parts[1])
            mem = float(parts[2])
        except ValueError:
            continue
        name = parts[3].rsplit("/", 1)[-1]
        sid, rid = pid_session.get(pid, ("", ""))
        gpu_procs[gpu_idx].append(
            GpuProcessInfo(
                pid=pid,
                gpu_memory_mb=mem,
                process_name=name,
                session_id=sid,
                run_id=rid,
            )
        )

    for gpu in gpus:
        gpu.processes = gpu_procs.get(gpu.index, [])

    return GpuUsageResponse(gpus=gpus)


@router.get("/resources/auto", response_model=ResourceAutoResponse)
def resource_auto(
    num_configs: int = 3,
    num_seeds: int = 3,
    env_id: str | None = None,
) -> ResourceAutoResponse:
    """Compute optimal resource allocation for given experiment size."""
    from p2p.training.resource_auto import find_best_allocation

    alloc = find_best_allocation(
        num_configs=num_configs,
        num_seeds=num_seeds,
        env_id=env_id,
    )
    return ResourceAutoResponse(**alloc)


@router.get("/resources/nodes", response_model=NodeResourcesResponse)
def node_resources() -> NodeResourcesResponse:
    """Cached CPU/memory/GPU data for all remote SSH nodes."""
    from p2p.api.node_monitor import POLL_INTERVAL, get_cached_snapshots

    return NodeResourcesResponse(nodes=get_cached_snapshots(), poll_interval_s=POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Session analysis endpoints
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/analysis", response_model=SessionAnalysisResponse)
def get_session_analysis(session_id: str) -> SessionAnalysisResponse:
    cached = services.get_cached_analysis(session_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="No analysis found")
    return SessionAnalysisResponse(**cached)


@router.post("/sessions/{session_id}/analyze")
def analyze_session(session_id: str) -> StreamingResponse:
    session = services.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def generate():  # type: ignore[return]
        q: queue.Queue[tuple[str, str | dict]] = queue.Queue()

        def run() -> None:
            try:
                result = services.run_analysis(
                    session_id,
                    on_status=lambda msg: q.put(("status", msg)),
                )
                q.put(("done", result))
            except Exception as e:
                q.put(("error", str(e)))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while True:
            try:
                event_type, data = q.get(timeout=120)
            except queue.Empty:
                yield sse_event("error", {"error": "Timeout"})
                break

            if event_type == "status":
                yield sse_event("status", {"message": data})
            elif event_type == "done":
                yield sse_event("analysis", data)
                break
            elif event_type == "error":
                yield sse_event("error", {"error": data})
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Event log endpoints
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/events", response_model=list[EventSummary])
def list_events(session_id: str) -> list[EventSummary]:
    return services.list_events(session_id)


@router.get("/sessions/{session_id}/events/{seq}", response_model=EventDetail)
def get_event_detail(session_id: str, seq: int) -> EventDetail:
    event = services.get_event_detail(session_id, seq)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------


@router.get("/benchmarks/options", response_model=BenchmarkOptionsResponse)
def benchmark_options(csv_file: str | None = None) -> BenchmarkOptionsResponse:
    opts = benchmark_service.get_benchmark_options(csv_file=csv_file)
    return BenchmarkOptionsResponse(**opts)


@router.get("/benchmarks", response_model=list[BenchmarkRunSummary])
def list_benchmarks() -> list[BenchmarkRunSummary]:
    return benchmark_service.list_benchmarks()


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkRunDetail)
def get_benchmark(benchmark_id: str) -> BenchmarkRunDetail:
    detail = benchmark_service.get_benchmark(benchmark_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return detail


@router.get("/benchmarks/{benchmark_id}/config")
def get_benchmark_config(benchmark_id: str) -> dict:
    config = benchmark_service.get_benchmark_config(benchmark_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Benchmark config not found")
    return config


@router.post("/benchmarks/{benchmark_id}/stop", response_model=StopBenchmarkResponse)
def stop_benchmark(benchmark_id: str) -> StopBenchmarkResponse:
    stopped, count = benchmark_service.stop_benchmark(benchmark_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return StopBenchmarkResponse(
        stopped=True,
        stopped_sessions=count,
        detail=f"Stopped {count} sessions",
    )


@router.patch("/benchmarks/{benchmark_id}", response_model=UpdateMetadataResponse)
def update_benchmark(benchmark_id: str, req: UpdateMetadataRequest) -> UpdateMetadataResponse:
    try:
        meta = entity_lifecycle.update_benchmark_metadata(
            benchmark_id, alias=req.alias, starred=req.starred, tags=req.tags
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return UpdateMetadataResponse(
        alias=meta.get("alias", ""),
        starred=meta.get("starred", False),
        tags=meta.get("tags", []),
    )


@router.delete("/benchmarks/{benchmark_id}", response_model=StopResponse)
def delete_benchmark_route(benchmark_id: str) -> StopResponse:
    try:
        entity_lifecycle.delete_benchmark(benchmark_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StopResponse(stopped=True, detail="Benchmark moved to trash")


@router.post("/benchmarks/{benchmark_id}/restore", response_model=StopResponse)
def restore_benchmark(benchmark_id: str) -> StopResponse:
    try:
        entity_lifecycle.restore_benchmark(benchmark_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return StopResponse(stopped=True, detail="Benchmark restored")


# ---------------------------------------------------------------------------
# Experiment lineage
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/lineage")
def get_session_lineage(session_id: str) -> dict[str, Any]:
    """Return the experiment lineage tree for a session."""
    from p2p.session.lineage import load_lineage

    session_dir = resolve_session_dir(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    lineage = load_lineage(session_dir)
    return lineage


# ---------------------------------------------------------------------------
# Trash (soft-deleted entities)
# ---------------------------------------------------------------------------


@router.get("/trash", response_model=list[TrashItem])
def list_trash() -> list[TrashItem]:
    return [TrashItem(**item) for item in entity_lifecycle.list_trash()]


@router.delete("/trash", response_model=StopResponse)
def hard_delete_all() -> StopResponse:
    deleted = entity_lifecycle.hard_delete_all_trash()
    return StopResponse(stopped=True, detail=f"Permanently deleted {deleted} item(s)")


@router.delete("/trash/{entity_id}", response_model=StopResponse)
def hard_delete(entity_id: str) -> StopResponse:
    try:
        entity_lifecycle.hard_delete_entity(entity_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Entity not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StopResponse(stopped=True, detail="Permanently deleted")


# ---------------------------------------------------------------------------
# Human labeling proxy
# ---------------------------------------------------------------------------


@router.get("/human-label/status", response_model=LabelingStatusResponse)
def labeling_status() -> LabelingStatusResponse:
    """Check if the human labeling server is configured."""
    from p2p.settings import LABELING_ANNOTATOR, LABELING_SERVER_URL

    return LabelingStatusResponse(
        enabled=bool(LABELING_SERVER_URL),
        annotator=LABELING_ANNOTATOR,
    )


def _send_to_labeling_server(
    session_id: str,
    iteration: int,
    annotator: str,
    intent_score: float,
    video_url: str = "",
) -> None:
    """Background task: collect iteration videos and POST to the labeling server.

    Uses ``urllib.request`` (stdlib) to avoid adding new runtime dependencies.
    Persists status transitions to ``human_label.json`` via read-then-update.
    """
    import hashlib
    import json
    import urllib.request
    from datetime import datetime, timezone
    from pathlib import Path

    from p2p.session.iteration_record import IterationRecord, SessionRecord, read_json_safe
    from p2p.settings import LABELING_SERVER_URL, resolve_session_dir

    if not LABELING_SERVER_URL:
        logger.warning("LABELING_SERVER_URL not set, skipping label submission")
        return

    session_dir = resolve_session_dir(session_id)
    if not session_dir.is_dir():
        logger.error("Session directory not found: %s", session_dir)
        return

    # Find iteration directory and create record for human_label persistence
    iter_dir = session_dir / f"iter_{iteration}"
    rec = IterationRecord(iter_dir)

    video_key = video_url.split("/")[-1] if video_url else ""

    def _update_label_scored(result: dict) -> None:
        if not video_key:
            return
        all_labels = rec.read_human_labels() or {}
        entry = all_labels.get(video_key, {})
        entry["status"] = "scored"
        entry["scored_at"] = datetime.now(timezone.utc).isoformat()
        entry["labeling_server_result"] = result
        all_labels[video_key] = entry
        rec.save_human_labels(all_labels)  # type: ignore[arg-type]

    def _update_label_error(exc: BaseException) -> None:
        if not video_key:
            return
        all_labels = rec.read_human_labels() or {}
        entry = all_labels.get(
            video_key,
            {
                "status": "error",
                "annotator": annotator,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        entry["status"] = "error"
        entry["error"] = str(exc)
        all_labels[video_key] = entry
        rec.save_human_labels(all_labels)  # type: ignore[arg-type]

    try:
        # Read session config for env_id and prompt
        sr = SessionRecord(session_dir)
        config = sr.read_session_config()
        env_id = "unknown"
        prompt = ""
        if config:
            train = config.get("train", {})
            env_id = train.get("env_id", config.get("env_id", "unknown"))
            prompt = config.get("prompt", "")

        # Fallback: read from loop_history.json if session_config.json is missing
        if env_id == "unknown" or not prompt:
            history_path = session_dir / "loop_history.json"
            if history_path.exists():
                try:
                    history = json.loads(history_path.read_text())
                    if not prompt:
                        prompt = history.get("prompt", "")
                except (json.JSONDecodeError, OSError):
                    pass

        # Fallback: read from parent benchmark.json (for benchmark sessions)
        if env_id == "unknown" or not prompt:
            bm_json = session_dir.parent / "benchmark.json"
            if bm_json.exists():
                try:
                    bm = json.loads(bm_json.read_text())
                    # Match by session_id or by directory name (caseN)
                    case_name = session_dir.name  # e.g. "case58"
                    for tc in bm.get("test_cases", []):
                        tc_sid = tc.get("session_id", "")
                        tc_idx = f"case{tc.get('index', '')}"
                        if tc_sid == session_id or tc_idx == case_name:
                            if env_id == "unknown":
                                env_id = tc.get("env_id", "unknown")
                            if not prompt:
                                prompt = tc.get("instruction", "")
                            break
                except (json.JSONDecodeError, OSError):
                    pass

        # Determine source host from session config
        source_host = ""
        if config:
            source_host = config.get("source_host", "")

        if not iter_dir.is_dir():
            logger.error("Iteration directory not found: %s", iter_dir)
            return

        # Collect video files: check iteration-level first, then best sub-run
        video_files: list[Path] = []
        videos_dir = rec.videos_dir
        if videos_dir.exists():
            video_files = sorted(
                f for f in videos_dir.glob("*.mp4") if not f.stem.endswith("_vlm")
            )

        # Fall back to sub-run directories
        if not video_files:
            # Try best_run.json
            best_run_data = read_json_safe(iter_dir / "best_run.json")
            best_run_id = best_run_data.get("best_run_id", "") if best_run_data else ""
            target_sub: Path | None = None
            if best_run_id:
                candidate = iter_dir / best_run_id
                if candidate.is_dir() and (candidate / "videos").exists():
                    target_sub = candidate
            if target_sub is None:
                for sub in sorted(iter_dir.iterdir()):
                    if sub.is_dir() and (sub / "videos").exists():
                        target_sub = sub
                        break
            if target_sub:
                video_files = sorted((target_sub / "videos").glob("*.mp4"))

        if not video_files:
            logger.warning("No video files found for session %s iter %d", session_id, iteration)
            return

        # Collect matching trajectory files (.npz preferred, .jsonl fallback)
        traj_dir = video_files[0].parent.parent  # videos/ -> parent run dir
        trajectory_files: list[Path] = []
        for vf in video_files:
            stem = vf.stem.replace("eval_", "trajectory_")
            found = resolve_trajectory_path(traj_dir, stem)
            if found is not None:
                trajectory_files.append(found)

        # Compute SHA256 (first 64KB) for each video
        video_hashes: dict[str, str] = {}
        for vf in video_files:
            with open(vf, "rb") as f:
                head = f.read(65536)
            video_hashes[vf.name] = hashlib.sha256(head).hexdigest()

        # Determine which video was scored
        scored_filename = video_url.split("/")[-1] if video_url else ""
        scored_hash = video_hashes.get(scored_filename, "")

        # Check if labeling server already has these videos
        base_url = LABELING_SERVER_URL.rstrip("/")
        all_hashes = list(video_hashes.values())
        existing_hashes: set[str] = set()
        try:
            check_body = json.dumps({"hashes": all_hashes}).encode()
            check_req = urllib.request.Request(
                f"{base_url}/api/check-hashes",
                data=check_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(check_req, timeout=10) as resp:
                check_result = json.loads(resp.read().decode())
                existing_hashes = set(check_result.get("existing", []))
        except Exception:
            logger.debug("check-hashes failed, will send all videos")

        # If the scored video already exists, just send score-only
        if scored_hash and scored_hash in existing_hashes:
            score_body = json.dumps(
                {
                    "video_hash": scored_hash,
                    "annotator": annotator,
                    "intent_score": intent_score,
                    "video_name": scored_filename,
                }
            ).encode()
            score_req = urllib.request.Request(
                f"{base_url}/api/score-label",
                data=score_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(score_req, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
                    logger.info(
                        "Score-only label for %s iter %d: %s",
                        session_id,
                        iteration,
                        result,
                    )
                    _update_label_scored(result)
                # Upload only NEW videos (no score attached)
                new_files = [
                    vf for vf in video_files if video_hashes[vf.name] not in existing_hashes
                ]
                if new_files:
                    # Trajectories matching new videos only
                    new_trajs = [
                        tf
                        for tf in trajectory_files
                        if any(
                            tf.stem.replace("trajectory_", "") == vf.stem.replace("eval_", "")
                            for vf in new_files
                        )
                    ]
                    _upload_videos_only(
                        base_url,
                        new_files,
                        video_hashes,
                        env_id,
                        prompt,
                        session_id,
                        iteration,
                        source_host,
                        annotator,
                        new_trajs,
                    )
                return
            except Exception:
                logger.debug("score-label failed, falling back to full upload")

        # Full upload: filter out already-existing videos
        files_to_send = [vf for vf in video_files if video_hashes[vf.name] not in existing_hashes]
        # If all exist but scored video wasn't handled above, still send metadata
        if not files_to_send and not scored_hash:
            _update_label_scored({})
            return

        # If no new files but we need to score, include at least the scored video
        if not files_to_send and scored_filename:
            scored_file = next((vf for vf in video_files if vf.name == scored_filename), None)
            if scored_file:
                files_to_send = [scored_file]

        # Trajectories matching files to send
        send_trajs = [
            tf
            for tf in trajectory_files
            if any(
                tf.stem.replace("trajectory_", "") == vf.stem.replace("eval_", "")
                for vf in files_to_send
            )
        ]
        _upload_with_score(
            base_url,
            files_to_send,
            video_hashes,
            env_id,
            prompt,
            annotator,
            intent_score,
            session_id,
            iteration,
            source_host,
            scored_filename,
            send_trajs,
        )
        _update_label_scored({})

    except Exception as exc:
        logger.exception("Failed to send to labeling server")
        _update_label_error(exc)


def _upload_videos_only(
    base_url: str,
    video_files: list,
    video_hashes: dict[str, str],
    env_id: str,
    prompt: str,
    session_id: str,
    iteration: int,
    source_host: str,
    submitted_by: str,
    trajectory_files: list | None = None,
) -> None:
    """Upload new videos without attaching a score."""
    import urllib.request
    from uuid import uuid4

    boundary = uuid4().hex
    meta = {
        "env_id": env_id,
        "intent": prompt,
        "annotator": submitted_by,
        "intent_score": 0.0,
        "session_id": session_id,
        "iteration": iteration,
        "source_host": source_host,
        "video_hashes": {vf.name: video_hashes[vf.name] for vf in video_files},
        "scored_video_url": "",
    }
    body = _build_multipart(boundary, meta, video_files, trajectory_files)
    req = urllib.request.Request(
        f"{base_url}/api/ingest-label",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            logger.info("Uploaded %d new videos: %s", len(video_files), resp.read().decode())
    except Exception:
        logger.exception("Failed to upload new videos")


def _upload_with_score(
    base_url: str,
    video_files: list,
    video_hashes: dict[str, str],
    env_id: str,
    prompt: str,
    annotator: str,
    intent_score: float,
    session_id: str,
    iteration: int,
    source_host: str,
    scored_filename: str,
    trajectory_files: list | None = None,
) -> None:
    """Upload videos with a score for a specific video."""
    import json
    import socket
    import urllib.request
    from uuid import uuid4

    boundary = uuid4().hex
    meta = {
        "env_id": env_id,
        "intent": prompt,
        "annotator": annotator,
        "intent_score": intent_score,
        "session_id": session_id,
        "iteration": iteration,
        "source_host": source_host,
        "video_hashes": {vf.name: video_hashes[vf.name] for vf in video_files},
        "scored_video_url": scored_filename,
    }
    body = _build_multipart(boundary, meta, video_files, trajectory_files)
    req = urllib.request.Request(
        f"{base_url}/api/ingest-label",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            logger.info(
                "Label submission for session %s iter %d: %s",
                session_id,
                iteration,
                result,
            )
    except (urllib.error.URLError, socket.timeout) as exc:
        logger.error("Failed to send label to %s: %s", base_url, exc)
    except Exception:
        logger.exception("Unexpected error sending label to labeling server")


def _build_multipart(
    boundary: str,
    meta: dict,
    video_files: list,
    trajectory_files: list | None = None,
) -> bytes:
    """Build a multipart/form-data body with metadata JSON + video + trajectory files."""
    import json

    parts: list[bytes] = []
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="metadata"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{json.dumps(meta)}\r\n".encode()
    )
    for vf in video_files:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="videos"; filename="{vf.name}"\r\n'
            f"Content-Type: video/mp4\r\n\r\n".encode()
        )
        parts.append(vf.read_bytes())
        parts.append(b"\r\n")
    for tf in trajectory_files or []:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="trajectories"; filename="{tf.name}"\r\n'
            f"Content-Type: application/jsonl\r\n\r\n".encode()
        )
        parts.append(tf.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


@router.post("/human-label", response_model=HumanLabelResponse)
def submit_human_label(
    req: HumanLabelRequest,
    background_tasks: BackgroundTasks,
) -> HumanLabelResponse:
    """Accept a human score and forward iteration videos to the labeling server."""
    from p2p.session.iteration_record import IterationRecord, read_json_safe
    from p2p.settings import resolve_session_dir

    session_dir = resolve_session_dir(req.session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")

    iter_dir = session_dir / f"iter_{req.iteration}"
    if not iter_dir.is_dir():
        raise HTTPException(status_code=404, detail="Iteration not found")

    # Count videos to report back
    rec = IterationRecord(iter_dir)
    video_count = len(rec.video_filenames())

    # Fall back to sub-run videos
    if video_count == 0:
        best_run_data = read_json_safe(iter_dir / "best_run.json")
        best_run_id = best_run_data.get("best_run_id", "") if best_run_data else ""
        target_sub = None
        if best_run_id:
            candidate = iter_dir / best_run_id
            if candidate.is_dir() and (candidate / "videos").exists():
                target_sub = candidate
        if target_sub is None:
            for sub in sorted(iter_dir.iterdir()):
                if sub.is_dir() and (sub / "videos").exists():
                    target_sub = sub
                    break
        if target_sub:
            video_count = len(list((target_sub / "videos").glob("*.mp4")))

    # Persist initial "sent" status per-video so the frontend can show it immediately
    from datetime import datetime, timezone

    from p2p.contracts import HumanLabelEntry

    video_key = req.video_url.split("/")[-1] if req.video_url else ""
    if video_key:
        all_labels = rec.read_human_labels() or {}
        all_labels[video_key] = HumanLabelEntry(
            status="sent",
            annotator=req.annotator,
            sent_at=datetime.now(timezone.utc).isoformat(),
            intent_score=req.intent_score,
        )
        rec.save_human_labels(all_labels)

    background_tasks.add_task(
        _send_to_labeling_server,
        req.session_id,
        req.iteration,
        req.annotator,
        req.intent_score,
        req.video_url,
    )

    return HumanLabelResponse(status="accepted", video_count=video_count)


# ---------------------------------------------------------------------------
# VLM proxy endpoints
# ---------------------------------------------------------------------------


@router.post("/vlm/api/chat", response_model=VlmChatResponse)
def vlm_chat(req: VlmChatRequest) -> VlmChatResponse:
    try:
        return vlm_service.vlm_chat(req)
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/vlm/api/tags")
def vlm_tags() -> dict[str, Any]:
    try:
        return vlm_service.vlm_tags()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/vlm/status", response_model=VlmStatusResponse)
def vlm_status() -> VlmStatusResponse:
    return vlm_service.vlm_status()
