"""Handler functions for the scheduler CLI.

Each handler takes an ``argparse.Namespace`` and returns an ``int`` exit code
(0 = success, 1 = error).  Output goes to stdout/stderr.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict], columns: list[str]) -> None:
    """Print a simple aligned table from a list of dicts."""
    if not rows:
        print("(none)")
        return
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    header = "  ".join(c.upper().ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


# ---------------------------------------------------------------------------
# Node handlers
# ---------------------------------------------------------------------------


def handle_node_list(args: object) -> int:
    from p2p.scheduler import node_store

    nodes = node_store.list_nodes()
    _print_table(
        [
            {
                "node_id": n["node_id"],
                "host": n["host"],
                "user": n["user"],
                "port": n["port"],
                "max_cores": n["max_cores"],
                "enabled": n.get("enabled", True),
            }
            for n in nodes
        ],
        ["node_id", "host", "user", "port", "max_cores", "enabled"],
    )
    return 0


def handle_node_add(args: object) -> int:
    from p2p.scheduler import node_store
    from p2p.scheduler.types import NodeConfig

    ns = args  # type: ignore[assignment]
    config: NodeConfig = {
        "node_id": ns.node_id,
        "host": ns.host,
        "user": ns.user,
        "port": ns.port,
        "base_dir": ns.base_dir,
        "max_cores": ns.max_cores,
    }
    try:
        node_store.add_node(config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Node '{ns.node_id}' added.")
    return 0


def handle_node_check(args: object) -> int:
    from p2p.scheduler.backend import check_node

    ns = args  # type: ignore[assignment]
    result = check_node(ns.node_id)
    if result.get("online"):
        print(f"Node '{ns.node_id}': online")
        if result.get("uv_available"):
            print("  uv: available")
        if result.get("gpu"):
            print(f"  GPU: {result['gpu']}")
    else:
        print(f"Node '{ns.node_id}': offline")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        return 1
    return 0


def handle_node_setup(args: object) -> int:
    from p2p.scheduler.backend import setup_node

    ns = args  # type: ignore[assignment]
    result = setup_node(ns.node_id)
    if result.get("success"):
        print(f"Node '{ns.node_id}' setup complete.")
    else:
        print(f"Node '{ns.node_id}' setup failed: {result.get('error', 'unknown')}")
        return 1
    return 0


def handle_node_update(args: object) -> int:
    from p2p.scheduler import node_store

    ns = args  # type: ignore[assignment]
    updates: dict = {}
    if ns.max_cores is not None:
        updates["max_cores"] = ns.max_cores
    if ns.enabled is not None:
        updates["enabled"] = ns.enabled
    if not updates:
        print("Error: no fields to update", file=sys.stderr)
        return 1
    try:
        node_store.update_node(ns.node_id, updates)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Node '{ns.node_id}' updated.")
    return 0


def handle_node_remove(args: object) -> int:
    from p2p.scheduler import node_store

    ns = args  # type: ignore[assignment]
    if node_store.remove_node(ns.node_id):
        print(f"Node '{ns.node_id}' removed.")
        return 0
    print(f"Error: node '{ns.node_id}' not found", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------


def handle_job_list(args: object) -> int:
    from p2p.scheduler.job_queries import list_jobs

    jobs = list_jobs()
    _print_table(
        [
            {
                "job_id": j["job_id"],
                "type": j["job_type"],
                "status": j["status"],
                "runs": len(j.get("run_ids", [])),
                "created_at": j["created_at"][:19],
            }
            for j in jobs
        ],
        ["job_id", "type", "status", "runs", "created_at"],
    )
    return 0


def handle_job_view(args: object) -> int:
    from p2p.scheduler.job_queries import get_job

    ns = args  # type: ignore[assignment]
    job = get_job(ns.job_id)
    if job is None:
        print(f"Error: job '{ns.job_id}' not found", file=sys.stderr)
        return 1
    print(json.dumps(job, indent=2, default=str))
    return 0


def handle_job_cancel(args: object) -> int:
    from p2p.scheduler.job_queries import cancel_job, get_job

    ns = args  # type: ignore[assignment]
    job = get_job(ns.job_id)
    if job is None:
        print(f"Error: job '{ns.job_id}' not found", file=sys.stderr)
        return 1
    cancel_job(ns.job_id)
    print(f"Job '{ns.job_id}' cancellation requested.")
    return 0


def handle_job_sync(args: object) -> int:
    from p2p.scheduler.job_queries import sync_job_all

    ns = args  # type: ignore[assignment]
    result = sync_job_all(ns.job_id)
    print(f"Synced: {result['synced']}, failed: {result['failed']}, skipped: {result['skipped']}")
    return 0


def handle_job_submit_session(args: object) -> int:
    from p2p.api.services import generate_hp_configs
    from p2p.config import loop_config_from_params
    from p2p.scheduler.controllers import SessionController

    ns = args  # type: ignore[assignment]

    loop_config = loop_config_from_params(
        total_timesteps=ns.total_timesteps,
        seed=ns.seed,
        env_id=ns.env_id,
        num_envs=ns.num_envs,
        side_info=ns.side_info,
        num_evals=ns.num_evals,
        trajectory_stride=ns.trajectory_stride,
        device=ns.device,
        max_iterations=ns.max_iterations,
        pass_threshold=ns.pass_threshold,
        model=ns.model,
        vlm_model=ns.vlm_model,
        thinking_effort=ns.thinking_effort,
        refined_initial_frame=getattr(ns, "refined_initial_frame", True),
        criteria_diagnosis=getattr(ns, "criteria_diagnosis", False),
        motion_trail_dual=getattr(ns, "motion_trail_dual", False),
        hp_tuning=ns.hp_tuning,
        use_code_judge=ns.use_code_judge,
        review_reward=ns.review_reward,
        review_judge=ns.review_judge,
        judgment_select=ns.judgment_select,
        use_zoo_preset=not ns.no_zoo_preset,
    )

    seeds = [ns.seed]
    if ns.seeds:
        seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip()]

    configs = None
    if ns.configs:
        configs = json.loads(ns.configs)
    elif ns.num_configs and ns.num_configs > 1:
        configs = generate_hp_configs(
            ns.num_configs,
            env_id=ns.env_id,
            num_envs=ns.num_envs,
        )

    allowed_nodes = None
    if ns.nodes:
        allowed_nodes = [n.strip() for n in ns.nodes.split(",") if n.strip()]

    ctrl = SessionController()
    job = ctrl.run(
        prompt=ns.prompt,
        loop_config=loop_config,
        backend=ns.backend,
        node_id=ns.node_id,
        configs=configs,
        seeds=seeds if len(seeds) > 1 else None,
        cores_per_run=ns.cores_per_run,
        max_parallel=ns.max_parallel,
        allowed_nodes=allowed_nodes,
        spawn=not ns.foreground,
    )

    job_id = job["job_id"]
    print(f"Session job submitted: {job_id}")
    print(f"  Runs: {len(job.get('run_ids', []))}")

    if ns.foreground:
        return _run_foreground(job_id)
    return 0


def handle_job_submit_benchmark(args: object) -> int:
    import csv as csv_mod

    from p2p.api.services import generate_hp_configs
    from p2p.config import loop_config_from_params
    from p2p.scheduler.controllers import BenchmarkController

    ns = args  # type: ignore[assignment]

    # Load and filter test cases from CSV
    csv_path = Path(ns.csv_file) if ns.csv_file else Path("benchmark/test_cases.csv")
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open() as f:
        test_cases = list(csv_mod.DictReader(f))

    if ns.filter_envs:
        envs = {e.strip() for e in ns.filter_envs.split(",")}
        test_cases = [tc for tc in test_cases if tc["env_id"] in envs]
    if ns.filter_categories:
        cats = {c.strip() for c in ns.filter_categories.split(",")}
        test_cases = [tc for tc in test_cases if tc.get("category") in cats]
    if ns.filter_difficulties:
        diffs = {d.strip() for d in ns.filter_difficulties.split(",")}
        test_cases = [tc for tc in test_cases if tc.get("difficulty") in diffs]

    if not test_cases:
        print("Error: no test cases match the filters.", file=sys.stderr)
        return 1

    seeds = [ns.seed]
    if ns.seeds:
        seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip()]

    loop_config = loop_config_from_params(
        total_timesteps=ns.total_timesteps,
        seed=ns.seed,
        num_envs=ns.num_envs,
        side_info=ns.side_info,
        trajectory_stride=ns.trajectory_stride,
        device=ns.device,
        max_iterations=ns.max_iterations,
        pass_threshold=ns.pass_threshold,
        model=ns.model,
        vlm_model=ns.vlm_model,
        thinking_effort=ns.thinking_effort,
        refined_initial_frame=getattr(ns, "refined_initial_frame", True),
        criteria_diagnosis=getattr(ns, "criteria_diagnosis", False),
        motion_trail_dual=getattr(ns, "motion_trail_dual", False),
        cores_per_run=ns.cores_per_run,
        hp_tuning=ns.hp_tuning,
        use_code_judge=ns.use_code_judge,
        review_reward=ns.review_reward,
        review_judge=ns.review_judge,
        judgment_select=ns.judgment_select,
        use_zoo_preset=not ns.no_zoo_preset,
    )

    configs = None
    if ns.num_configs and ns.num_configs > 1:
        configs = generate_hp_configs(ns.num_configs)

    allowed_nodes = None
    if ns.nodes:
        allowed_nodes = [n.strip() for n in ns.nodes.split(",") if n.strip()]

    ctrl = BenchmarkController()
    job = ctrl.run(
        loop_config=loop_config,
        backend=ns.backend,
        test_cases=test_cases,
        mode=ns.mode,
        num_stages=ns.num_stages,
        gate_threshold=ns.gate_threshold,
        start_from_stage=ns.start_from_stage,
        max_parallel=ns.max_parallel,
        configs=configs,
        seeds=seeds,
        allowed_nodes=allowed_nodes,
        spawn=not ns.foreground,
    )

    job_id = job["job_id"]
    metadata = job.get("metadata", {})
    total_stages = metadata.get("total_stages", 0)

    benchmark_id = metadata.get("benchmark_id", job_id)
    print(f"Benchmark {benchmark_id} (job {job_id})")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Runs: {len(job.get('run_ids', []))}")
    if total_stages:
        print(f"  Stages: {total_stages} (gate={ns.gate_threshold})")

    if ns.foreground:
        return _run_foreground(job_id)
    return 0


# ---------------------------------------------------------------------------
# Run handlers
# ---------------------------------------------------------------------------


def handle_run_list(args: object) -> int:
    from p2p.scheduler.manifest_io import list_job_ids, read_job_manifest

    ns = args  # type: ignore[assignment]
    rows: list[dict] = []
    for jid in list_job_ids():
        if ns.job_id and jid != ns.job_id:
            continue
        manifest = read_job_manifest(jid)
        if manifest is None:
            continue
        for run in manifest["runs"]:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "job_id": jid,
                    "state": run["state"],
                    "node_id": run.get("node_id", ""),
                }
            )
    _print_table(rows, ["run_id", "job_id", "state", "node_id"])
    return 0


def handle_run_view(args: object) -> int:
    from p2p.scheduler.manifest_io import list_job_ids, read_job_manifest

    ns = args  # type: ignore[assignment]
    for jid in list_job_ids():
        manifest = read_job_manifest(jid)
        if manifest is None:
            continue
        for run in manifest["runs"]:
            if run["run_id"] == ns.run_id:
                print(json.dumps(run, indent=2, default=str))
                return 0
    print(f"Error: run '{ns.run_id}' not found", file=sys.stderr)
    return 1


def handle_run_log(args: object) -> int:
    from p2p.settings import RUNS_DIR, resolve_session_dir

    ns = args  # type: ignore[assignment]
    try:
        log_path = resolve_session_dir(ns.run_id) / "subprocess.log"
    except ValueError:
        print("Error: invalid run_id", file=sys.stderr)
        return 1

    if not log_path.exists():
        # Check if the path is within RUNS_DIR for security
        try:
            log_path.resolve().relative_to(RUNS_DIR.resolve())
        except ValueError:
            print("Error: invalid run_id", file=sys.stderr)
            return 1
        print(f"No log file found at {log_path}", file=sys.stderr)
        return 1

    lines = log_path.read_text().splitlines()
    tail = ns.tail
    for line in lines[-tail:]:
        print(line)
    return 0


def handle_run_sync(args: object) -> int:
    from p2p.scheduler.job_queries import sync_job_run
    from p2p.scheduler.manifest_io import list_job_ids, read_job_manifest

    ns = args  # type: ignore[assignment]
    job_id = ns.job_id
    if not job_id:
        # Find the job containing this run
        for jid in list_job_ids():
            manifest = read_job_manifest(jid)
            if manifest is None:
                continue
            for run in manifest["runs"]:
                if run["run_id"] == ns.run_id:
                    job_id = jid
                    break
            if job_id:
                break
    if not job_id:
        print(f"Error: run '{ns.run_id}' not found in any job", file=sys.stderr)
        return 1

    result = sync_job_run(job_id, ns.run_id, mode=ns.mode)
    if result.get("synced"):
        print(f"Run '{ns.run_id}' synced ({ns.mode}).")
    else:
        print(f"Sync failed: {result.get('error', 'unknown')}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Foreground scheduler runner
# ---------------------------------------------------------------------------


def _run_foreground(job_id: str) -> int:
    """Run the job scheduler in-process with signal handling."""
    from p2p.scheduler.job_scheduler import _run_scheduler
    from p2p.scheduler.manifest_io import read_job_manifest, write_job_manifest

    def _handle_sigterm(signum: int, frame: object) -> None:
        logger.info("Received signal %d, marking job as cancelled", signum)
        try:
            m = read_job_manifest(job_id)
            if m and m.get("status") == "running":
                m["status"] = "cancelled"
                m.pop("scheduler_pid", None)
                write_job_manifest(m)
        except Exception:
            logger.exception("Failed to write cancellation during signal handler")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print("\nRunning scheduler in foreground (Ctrl+C to cancel)...\n")

    try:
        _run_scheduler(job_id)
    except Exception:
        logger.exception("Job scheduler crashed: %s", job_id)
        try:
            m = read_job_manifest(job_id)
            if m:
                m["status"] = "error"
                m["error"] = traceback.format_exc()
                m.pop("scheduler_pid", None)
                write_job_manifest(m)
        except Exception:
            logger.exception("Failed to write error to manifest")
        return 1
    return 0
