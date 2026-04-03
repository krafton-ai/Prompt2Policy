"""Standalone job scheduler — runs as a subprocess, independent of the web server.

Usage::

    python -m p2p.scheduler.job_scheduler --job-id <job_id>

The web server writes the initial manifest
(``runs/scheduler/jobs/<job_id>/manifest.json``) with all parameters, run specs,
and config.  This process reads the manifest, launches runs, polls for
completion, syncs results, and updates the manifest as it progresses.

Communication with the web server is entirely file-based:

- **Monitoring**: the server reads ``manifest.json`` for status display.
- **Stop/cancel**: the server writes ``"status": "cancelled"`` to the manifest;
  this process detects it on the next loop iteration and exits.
- **Liveness**: the server verifies ``scheduler_pid`` ownership via
  ``verify_pid_ownership()`` (cmdline match against ``/proc``).

Handles both session jobs and benchmark jobs (flat and staged modes).
"""

# ruff: noqa: I001 — p2p.settings must be imported before gymnasium/mujoco
from __future__ import annotations

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from p2p.scheduler.backend import Backend, LocalBackend, SSHBackend
from p2p.scheduler.manifest_io import read_job_manifest, write_job_manifest
from p2p.scheduler.ssh_utils import (
    is_localhost,
    find_node,
    kill_ssh_process,
    load_nodes,
    resolve_node,
)
from p2p.scheduler.types import TERMINAL_STATES, JobManifest, RunRecord, now_iso
from p2p.settings import resolve_session_dir
from p2p.utils.process_safety import force_kill_pids, get_descendant_pids, safe_killpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10  # seconds


def _detect_local_gpu_count() -> int:
    """Detect the number of GPUs on the local machine."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def _is_local_run(run: RunRecord) -> bool:
    """Check if a run executes on the local machine (local backend or SSH localhost)."""
    node_id = run.get("node_id", "")
    if node_id == "local":
        return True
    node = find_node(node_id)
    return node is not None and is_localhost(node)


# ---------------------------------------------------------------------------
# Per-node CPU core allocator
# ---------------------------------------------------------------------------


class _NodeCoreAllocator:
    """Per-node CPU core pool for affinity pinning."""

    def __init__(self) -> None:
        self._pools: dict[str, list[int]] = {}

    def ensure_node(self, node_id: str, max_cores: int) -> None:
        if node_id not in self._pools:
            self._pools[node_id] = list(range(max_cores))

    def allocate(self, node_id: str, max_cores: int, num_cores: int) -> list[int] | None:
        self.ensure_node(node_id, max_cores)
        pool = self._pools[node_id]
        if len(pool) < num_cores:
            return None
        cores = pool[:num_cores]
        self._pools[node_id] = pool[num_cores:]
        return cores

    def release(self, node_id: str, cores: list[int]) -> None:
        self._pools.setdefault(node_id, []).extend(cores)
        self._pools[node_id].sort()

    def reserve(self, node_id: str, cores: list[int]) -> None:
        """Remove specific cores from the pool (for recovery of in-progress runs)."""
        self.ensure_node(node_id, 0)
        pool = self._pools.get(node_id, [])
        for c in cores:
            if c in pool:
                pool.remove(c)
        self._pools[node_id] = pool

    def available(self, node_id: str) -> int:
        return len(self._pools.get(node_id, []))


# ---------------------------------------------------------------------------
# Per-node GPU allocator
# ---------------------------------------------------------------------------


class _NodeGPUAllocator:
    """Per-node GPU allocator for CUDA_VISIBLE_DEVICES assignment.

    Two modes based on ``gpu_count`` in the RunSpec:

    - **Exclusive** (MuJoCo, ``gpu_count=0``): not used — MuJoCo sessions
      share GPUs implicitly via PyTorch.
    - **Round-robin** (IsaacLab, ``gpu_count=1``): each session is pinned to
      one GPU via ``CUDA_VISIBLE_DEVICES``.  Multiple sessions may share the
      same physical GPU — the allocator cycles through GPUs to spread load
      evenly instead of blocking when the pool is exhausted.
    """

    def __init__(self) -> None:
        self._num_gpus: dict[str, int] = {}
        self._counters: dict[str, int] = {}

    def ensure_node(self, node_id: str, num_gpus: int) -> None:
        if node_id not in self._num_gpus:
            self._num_gpus[node_id] = num_gpus
            self._counters[node_id] = 0

    def allocate(self, node_id: str, num_gpus: int, count: int) -> list[int] | None:
        self.ensure_node(node_id, num_gpus)
        n = self._num_gpus[node_id]
        if n == 0:
            return None
        gpus = []
        for _ in range(count):
            gpu_id = self._counters[node_id] % n
            self._counters[node_id] += 1
            gpus.append(gpu_id)
        return gpus

    def release(self, node_id: str, gpus: list[int]) -> None:
        pass  # round-robin — nothing to return

    def reserve(self, node_id: str, gpus: list[int]) -> None:
        """Advance counter past reserved GPUs (recovery of in-progress runs)."""
        self.ensure_node(node_id, 0)

    def available(self, node_id: str) -> int:
        return self._num_gpus.get(node_id, 0)


# ---------------------------------------------------------------------------
# Session initialization (pre-submit)
# ---------------------------------------------------------------------------


def _init_session(run: RunRecord) -> None:
    """Initialize status.json + session_config.json before launching.

    This ensures the dashboard can poll immediately.
    """
    spec = run["spec"]
    params = spec.get("parameters", {})
    session_id = params.get("session_id", run["run_id"])
    log_dir = resolve_session_dir(session_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    from p2p.session.iteration_record import SessionRecord

    session_rec = SessionRecord(log_dir)
    session_rec.set_status("running")

    if "loop_config" in params:
        try:
            start_config = json.loads(params["loop_config"])
            start_config["prompt"] = params.get("prompt", "")
            session_rec.config_path.write_text(json.dumps(start_config, indent=2))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to save session_config.json for %s: %s", session_id, exc)


# ---------------------------------------------------------------------------
# Backend-delegated operations
# ---------------------------------------------------------------------------


def _submit_run(
    run: RunRecord,
    backend: Backend,
    *,
    allocated_cores: list[int] | None = None,
    allocated_gpus: list[int] | None = None,
) -> None:
    """Submit a run via the Backend Protocol and update RunRecord."""
    _init_session(run)
    status = backend.submit(
        run["spec"],
        allocated_cores=allocated_cores,
        allocated_gpus=allocated_gpus,
    )
    run["state"] = status["state"]
    if "pid" in status:
        run["pid"] = status["pid"]
    if "node_id" in status:
        run["node_id"] = status["node_id"]
    if "remote_dir" in status:
        run["remote_dir"] = status["remote_dir"]
    if "error" in status:
        run["error"] = status["error"]
    if "started_at" in status:
        run["started_at"] = status["started_at"]
    if status["state"] in TERMINAL_STATES:
        run["completed_at"] = status.get("completed_at", now_iso())
    if allocated_cores is not None and status["state"] == "running":
        run["allocated_cores"] = allocated_cores
    if allocated_gpus is not None and status["state"] == "running":
        run["allocated_gpus"] = allocated_gpus


def _check_liveness(run: RunRecord, backend: Backend) -> None:
    """Check run liveness via Backend.status() and update RunRecord."""
    status = backend.status(run["run_id"])
    if status["state"] in TERMINAL_STATES:
        run["state"] = status["state"]
        run["completed_at"] = status.get("completed_at", now_iso())
        if "error" in status:
            run["error"] = status["error"]


def _kill_run(run: RunRecord, backend: Backend) -> None:
    """Cancel a running process via the Backend Protocol."""
    backend.cancel(run["run_id"])


def kill_run_process_standalone(run: RunRecord) -> None:
    """Kill a running process without a Backend instance.

    Used in SIGTERM and crash handlers where Backend instances may not be
    available.  Falls back to direct OS/SSH calls.

    Verifies PID ownership before killing local processes to prevent
    collateral damage from recycled PIDs (see issue #380).
    """
    pid = run.get("pid")
    if not pid:
        return
    if _is_local_run(run):
        session_id = run.get("spec", {}).get("parameters", {}).get("session_id", "") or None
        children = get_descendant_pids(pid)
        killed = safe_killpg(pid, expected_cmdline=session_id)
        if killed:
            force_kill_pids(children)
    else:
        node_id = run.get("node_id", "")
        node = find_node(node_id)
        if node:
            kill_ssh_process(pid=pid, node=node)


# ---------------------------------------------------------------------------
# Main scheduling loop
# ---------------------------------------------------------------------------


class _SchedulerState:
    """Mutable state shared across poll iterations."""

    def __init__(
        self,
        manifest: JobManifest,
        allocator: _NodeCoreAllocator,
        gpu_allocator: _NodeGPUAllocator,
        local_backend: LocalBackend,
        ssh_backends: dict[str, SSHBackend],
    ) -> None:
        self.manifest = manifest
        self.allocator = allocator
        self.gpu_allocator = gpu_allocator
        self.local_backend = local_backend
        self.ssh_backends = ssh_backends
        self.status_sync_counter = 0
        self.STATUS_SYNC_EVERY = 3

    @property
    def backend_type(self) -> str:
        return self.manifest["backend"]

    @property
    def runs(self) -> list[RunRecord]:
        return self.manifest["runs"]

    def get_backend(self, node_id: str) -> Backend:
        """Return the Backend instance for a given node_id."""
        if node_id == "local" or not node_id:
            return self.local_backend
        if node_id in self.ssh_backends:
            return self.ssh_backends[node_id]
        # Create on-demand for nodes not yet seen
        node = find_node(node_id)
        if node is None:
            msg = f"Node {node_id!r} not found in node store"
            raise ValueError(msg)
        be = SSHBackend(node)
        self.ssh_backends[node_id] = be
        return be


def _check_cancelled(state: _SchedulerState, job_id: str) -> bool:
    """Re-read manifest; cancel all runs if server wrote 'cancelled'.

    Returns True if cancelled.
    """
    current = read_job_manifest(job_id)
    if not current or current.get("status") != "cancelled":
        return False
    logger.info("Job %s cancelled by server", job_id)
    for run in state.runs:
        if run["state"] == "running":
            backend = state.get_backend(run.get("node_id", "local"))
            _kill_run(run, backend)
            run["state"] = "cancelled"
            run["completed_at"] = now_iso()
        elif run["state"] == "pending":
            run["state"] = "cancelled"
            run["completed_at"] = now_iso()
    state.manifest["status"] = "cancelled"
    state.manifest.pop("scheduler_pid", None)
    write_job_manifest(state.manifest)
    return True


def _poll_liveness_and_sync(state: _SchedulerState) -> None:
    """Check liveness, sync terminal runs, periodic running status sync."""
    runs = state.runs

    # Check liveness of running runs via Backend.status()
    for run in runs:
        if run["state"] != "running":
            continue
        backend = state.get_backend(run.get("node_id", "local"))
        _check_liveness(run, backend)
        if run["state"] in TERMINAL_STATES:
            nid = run.get("node_id", "")
            ac = run.get("allocated_cores")
            if ac and nid:
                state.allocator.release(nid, ac)
            ag = run.get("allocated_gpus")
            if ag and nid:
                state.gpu_allocator.release(nid, ag)

    # Sync terminal runs via Backend.sync_results() + cleanup()
    for run in runs:
        if run["state"] in ("completed", "error") and not run.get("synced"):
            node_id = run.get("node_id", "")
            backend = state.get_backend(node_id)
            if backend.sync_results(run["run_id"]):
                run["synced"] = True
                backend.cleanup(run["run_id"])
            else:
                logger.warning("Failed to sync %s, will retry", run["run_id"])

    # Periodic lightweight sync for running runs via Backend.sync_running()
    state.status_sync_counter += 1
    if state.status_sync_counter >= state.STATUS_SYNC_EVERY:
        state.status_sync_counter = 0
        for run in runs:
            if run["state"] != "running":
                continue
            backend = state.get_backend(run.get("node_id", "local"))
            backend.sync_running(run["run_id"])


def _submit_pending_runs(
    state: _SchedulerState,
    eligible_runs: list[RunRecord],
    max_parallel: int,
) -> None:
    """Submit pending runs from *eligible_runs*, respecting max_parallel by session_group."""
    backend_type = state.backend_type
    runs = state.runs

    # Compute used cores per node
    used_cores_per_node: dict[str, int] = {}
    for run in runs:
        if run["state"] == "running" and run.get("node_id"):
            nid = run["node_id"]
            cores = run["spec"].get("cpu_cores", 1)
            used_cores_per_node[nid] = used_cores_per_node.get(nid, 0) + cores

    # Build session_group → node_id mapping (for node affinity)
    group_node: dict[str, str] = {}
    for run in runs:
        sg = run.get("session_group", "")
        if sg and run.get("node_id"):
            group_node.setdefault(sg, run["node_id"])

    # Count running groups (across ALL runs, not just eligible)
    running_groups: set[str] = set()
    for r in runs:
        if r["state"] == "running":
            running_groups.add(r.get("session_group", r["run_id"]))
    running_group_count = len(running_groups)

    allowed_nodes_list = state.manifest.get("config", {}).get("allowed_nodes")
    allowed_nodes_set = set(allowed_nodes_list) if allowed_nodes_list else None

    failed_nodes: set[str] = set()

    # IsaacLab benchmark sessions serialize AppLauncher init through a
    # machine-level filelock.  Launching too many at once creates a long
    # queue (~30s per init).  Cap new launches per poll cycle so each wave
    # finishes init before the next starts.  MuJoCo and single E2E sessions
    # are unaffected (gpu_count=0 or job_type != "benchmark").
    _ISAACLAB_LAUNCH_CAP = 8
    isaaclab_launched_this_cycle = 0

    for run in eligible_runs:
        if run["state"] != "pending":
            continue
        sg = run.get("session_group", run["run_id"])
        is_new_group = sg not in running_groups
        if is_new_group and max_parallel > 0 and running_group_count >= max_parallel:
            break

        tags = run["spec"].get("tags", {})
        is_isaaclab_bm = (
            run["spec"].get("gpu_count", 0) > 0 and tags.get("job_type") == "benchmark"
        )
        if is_isaaclab_bm and isaaclab_launched_this_cycle >= _ISAACLAB_LAUNCH_CAP:
            break

        if sg and sg in group_node:
            run["spec"].setdefault("tags", {})["node_id"] = group_node[sg]

        try:
            spec = run["spec"]
            skip_affinity = spec.get("tags", {}).get("no_cpu_affinity") == "true"

            if backend_type == "local":
                if skip_affinity:
                    local_cores = None
                else:
                    required_cores = spec.get("cpu_cores", 1)
                    local_cores = state.allocator.allocate("local", 0, required_cores)
                gpu_count = spec.get("gpu_count", 0)
                local_gpus = None
                if gpu_count > 0:
                    local_gpus = state.gpu_allocator.allocate("local", 0, gpu_count)
                _submit_run(
                    run,
                    state.local_backend,
                    allocated_cores=local_cores,
                    allocated_gpus=local_gpus,
                )
                if run["state"] != "running":
                    if local_cores is not None:
                        state.allocator.release("local", local_cores)
                    if local_gpus is not None:
                        state.gpu_allocator.release("local", local_gpus)
            else:
                # Resolve SSH node and get/create Backend instance
                required_cores = 0 if skip_affinity else spec.get("cpu_cores", 1)
                node_id_hint = spec.get("tags", {}).get("node_id")
                node = resolve_node(
                    node_id_hint,
                    used_cores_per_node,
                    required_cores,
                    skip_nodes=failed_nodes,
                    allowed_nodes=allowed_nodes_set,
                )
                if node is None:
                    run["state"] = "error"
                    run["error"] = "No available SSH node"
                    run["completed_at"] = now_iso()
                elif skip_affinity:
                    ssh_be = state.get_backend(node["node_id"])
                    gpu_count = spec.get("gpu_count", 0)
                    ssh_gpus = None
                    if gpu_count > 0:
                        ssh_gpus = state.gpu_allocator.allocate(
                            node["node_id"],
                            node.get("num_gpus", 0),
                            gpu_count,
                        )
                    _submit_run(
                        run,
                        ssh_be,
                        allocated_cores=None,
                        allocated_gpus=ssh_gpus,
                    )
                    if run["state"] != "running" and ssh_gpus is not None:
                        state.gpu_allocator.release(node["node_id"], ssh_gpus)
                else:
                    allocated_cores = state.allocator.allocate(
                        node["node_id"],
                        node["max_cores"],
                        required_cores,
                    )
                    if allocated_cores is None:
                        run["state"] = "error"
                        run["error"] = f"No free cores on node {node['node_id']}"
                        run["completed_at"] = now_iso()
                    else:
                        # GPU allocation
                        gpu_count = spec.get("gpu_count", 0)
                        allocated_gpus = None
                        if gpu_count > 0:
                            allocated_gpus = state.gpu_allocator.allocate(
                                node["node_id"],
                                node.get("num_gpus", 0),
                                gpu_count,
                            )
                        ssh_be = state.get_backend(node["node_id"])
                        _submit_run(
                            run,
                            ssh_be,
                            allocated_cores=allocated_cores,
                            allocated_gpus=allocated_gpus,
                        )
                        if run["state"] != "running":
                            state.allocator.release(node["node_id"], allocated_cores)
                            if allocated_gpus is not None:
                                state.gpu_allocator.release(node["node_id"], allocated_gpus)

                if run["state"] == "running" and run.get("node_id"):
                    nid = run["node_id"]
                    cores = spec.get("cpu_cores", 1)
                    used_cores_per_node[nid] = used_cores_per_node.get(nid, 0) + cores
                elif run["state"] == "error" and run.get("node_id"):
                    failed_nodes.add(run["node_id"])
                elif run["state"] == "error" and not run.get("node_id"):
                    run["state"] = "pending"
                    run.pop("error", None)
                    run.pop("completed_at", None)
                    break
                if sg and run.get("node_id") and sg not in group_node:
                    group_node[sg] = run["node_id"]
            if run["state"] == "running":
                running_groups.add(sg)
                if is_new_group:
                    running_group_count += 1
                if is_isaaclab_bm:
                    isaaclab_launched_this_cycle += 1
        except Exception:
            logger.exception("Failed to submit run %s", run["run_id"])
            run["state"] = "error"
            run["error"] = traceback.format_exc()
            run["completed_at"] = now_iso()


def _run_flat(
    state: _SchedulerState,
    job_id: str,
    runs: list[RunRecord],
    max_parallel: int,
) -> bool:
    """Run all *runs* in a flat loop. Returns False if cancelled."""
    while True:
        if _check_cancelled(state, job_id):
            return False

        _poll_liveness_and_sync(state)
        _submit_pending_runs(state, runs, max_parallel)
        write_job_manifest(state.manifest)

        all_terminal = all(r["state"] in TERMINAL_STATES for r in runs)
        all_synced = all(r.get("synced", True) for r in runs if r["state"] == "completed")
        if all_terminal and all_synced:
            return True

        time.sleep(_POLL_INTERVAL)


def _run_staged(
    state: _SchedulerState,
    job_id: str,
) -> bool:
    """Run benchmark in staged mode. Returns False if cancelled."""
    from p2p.benchmark.benchmark_helpers import evaluate_gate

    metadata = state.manifest.get("metadata", {})
    stage_defs: list[dict] = metadata.get("stages", [])
    config = state.manifest.get("config", {})
    start_from_stage = config.get("start_from_stage", 1)
    pass_threshold = config.get("pass_threshold", 0.7)
    default_max_parallel = config.get("max_parallel", 0)
    runs = state.runs

    for stage_def in stage_defs:
        stage_num = stage_def["stage"]

        if stage_num < start_from_stage:
            stage_def["status"] = "skipped"
            write_job_manifest(state.manifest)
            continue

        case_indices = stage_def.get("case_indices", [])
        if not case_indices:
            stage_def["status"] = "skipped"
            write_job_manifest(state.manifest)
            continue

        stage_def["status"] = "running"
        metadata["current_stage"] = stage_num
        write_job_manifest(state.manifest)

        # Filter runs belonging to this stage
        stage_runs = [r for r in runs if r["spec"].get("tags", {}).get("stage") == str(stage_num)]
        stage_max_parallel = stage_def.get("max_parallel", default_max_parallel)

        ok = _run_flat(state, job_id, stage_runs, stage_max_parallel)
        if not ok:
            return False  # cancelled

        # Evaluate gate
        gt = stage_def.get("gate_threshold", 0.0)
        if gt > 0:
            # Build entries list for evaluate_gate: case_index → session_id
            entries: list[dict] = []
            test_cases_meta = metadata.get("test_cases", [])
            for tc in test_cases_meta:
                # Find the run for this case
                idx = tc["index"]
                session_id = ""
                for r in runs:
                    if r["spec"].get("tags", {}).get("case_index") == str(idx):
                        session_id = r["spec"]["parameters"].get("session_id", r["run_id"])
                        break
                entries.append({"session_id": session_id})

            gate_result = evaluate_gate(entries, case_indices, pass_threshold, gt)
            stage_def["gate_result"] = gate_result
            if gate_result["passed"]:
                stage_def["status"] = "gate_passed"
            else:
                stage_def["status"] = "gate_failed"
                # Skip remaining stages and cancel their pending runs
                for remaining in stage_defs:
                    if remaining["stage"] > stage_num:
                        remaining["status"] = "skipped"
                for r in runs:
                    r_stage = r["spec"].get("tags", {}).get("stage", "")
                    try:
                        r_stage_num = int(r_stage) if r_stage else 0
                    except (ValueError, TypeError):
                        continue
                    if r_stage_num > stage_num and r["state"] == "pending":
                        r["state"] = "cancelled"
                        r["completed_at"] = now_iso()
                write_job_manifest(state.manifest)
                return True  # gate failed, but not cancelled
        else:
            stage_def["status"] = "completed"

        write_job_manifest(state.manifest)

    return True


def _run_scheduler(job_id: str) -> None:
    """Main scheduler logic — reads manifest and drives the job to completion."""
    manifest = read_job_manifest(job_id)
    if manifest is None:
        logger.error("Manifest not found for %s", job_id)
        sys.exit(1)

    # Write our PID so the server can check liveness / kill us
    manifest["scheduler_pid"] = os.getpid()
    write_job_manifest(manifest)

    backend = manifest["backend"]
    runs = manifest["runs"]
    config = manifest.get("config", {})
    max_parallel = config.get("max_parallel", 0)  # 0 = unlimited

    allocator = _NodeCoreAllocator()
    gpu_allocator = _NodeGPUAllocator()
    ssh_backends: dict[str, SSHBackend] = {}
    if backend == "ssh":
        for n in load_nodes():
            if n.get("enabled", True):
                allocator.ensure_node(n["node_id"], n.get("max_cores", 0))
                gpu_allocator.ensure_node(n["node_id"], n.get("num_gpus", 0))
                ssh_backends[n["node_id"]] = SSHBackend(n)
    elif backend == "local":
        import multiprocessing

        allocator.ensure_node("local", multiprocessing.cpu_count())
        gpu_allocator.ensure_node("local", _detect_local_gpu_count())
    # Reserve cores/GPUs for already-running runs (scheduler recovery)
    for run in runs:
        if run["state"] == "running":
            nid = run.get("node_id", "")
            prev_cores = run.get("allocated_cores")
            if nid and prev_cores:
                allocator.reserve(nid, prev_cores)
            prev_gpus = run.get("allocated_gpus")
            if nid and prev_gpus:
                gpu_allocator.reserve(nid, prev_gpus)

    local_backend = LocalBackend()
    state = _SchedulerState(manifest, allocator, gpu_allocator, local_backend, ssh_backends)

    # Dispatch to flat or staged mode
    metadata = manifest.get("metadata", {})
    mode = metadata.get("mode", config.get("mode", "flat"))
    if mode == "staged" and metadata.get("stages"):
        _run_staged(state, job_id)
    else:
        _run_flat(state, job_id, runs, max_parallel)

    # Final status
    errors = [r for r in runs if r["state"] == "error"]
    cancelled = manifest.get("status") == "cancelled"
    # Check for gate failure in staged mode
    gate_failed = False
    if mode == "staged":
        for sd in metadata.get("stages", []):
            if sd.get("status") == "gate_failed":
                gate_failed = True
                break
    if cancelled:
        pass  # already handled
    elif gate_failed:
        manifest["status"] = "completed"  # job completed (gate stopped it early)
    elif len(errors) == len(runs):
        manifest["status"] = "error"
        manifest["error"] = "All runs failed"
    else:
        manifest["status"] = "completed"

    if not cancelled:
        manifest["completed_at"] = now_iso()
        manifest.pop("scheduler_pid", None)
        write_job_manifest(manifest)
        logger.info(
            "Job %s completed: %d/%d successful",
            job_id,
            len(runs) - len(errors),
            len(runs),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Job scheduler subprocess")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    job_id = args.job_id
    logger.info("Job scheduler starting: %s (pid=%d)", job_id, os.getpid())

    # Graceful shutdown on SIGTERM
    def _handle_sigterm(signum: int, frame: object) -> None:
        logger.info("Received SIGTERM, killing running processes and marking cancelled")
        try:
            m = read_job_manifest(job_id)
            if m and m.get("status") == "running":
                for run in m.get("runs", []):
                    if run.get("state") == "running":
                        try:
                            kill_run_process_standalone(run)
                        except Exception as exc:
                            logger.warning("Failed to kill run %s: %s", run.get("run_id"), exc)
                m["status"] = "cancelled"
                m.pop("scheduler_pid", None)
                write_job_manifest(m)
        except Exception:
            logger.exception("Failed to write cancellation during SIGTERM")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        _run_scheduler(job_id)
    except Exception:
        logger.exception("Job scheduler crashed: %s", job_id)
        try:
            m = read_job_manifest(job_id)
            if m:
                # Kill all running processes before marking as error
                for run in m.get("runs", []):
                    if run.get("state") == "running":
                        try:
                            kill_run_process_standalone(run)
                        except Exception as exc:
                            logger.warning("Failed to kill run %s: %s", run.get("run_id"), exc)
                m["status"] = "error"
                m["error"] = traceback.format_exc()
                m.pop("scheduler_pid", None)
                write_job_manifest(m)
        except Exception:
            logger.exception("Failed to write error to manifest")
        sys.exit(1)


if __name__ == "__main__":
    main()
