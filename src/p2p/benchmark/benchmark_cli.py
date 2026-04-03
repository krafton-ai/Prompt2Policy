"""CLI for running benchmarks directly from the command line.

Usage::

    uv run python -m p2p.benchmark.benchmark_cli --csv benchmark/test_cases.csv
    uv run python -m p2p.benchmark.benchmark_cli \\
        --csv benchmark/test_cases_humanoid_skills.csv \\
        --total-timesteps 500000 --max-iterations 3 --max-parallel 10

Delegates to ``BenchmarkController`` to create a job manifest and pointer,
then runs the job scheduler in-process (foreground).  Ctrl+C or SIGTERM
will mark the job as cancelled and exit gracefully.
"""

# ruff: noqa: I001 — p2p.settings must be imported before gymnasium/mujoco
from __future__ import annotations

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports

import argparse
import csv
import logging
import signal
import sys
import traceback
from pathlib import Path

from p2p.settings import VLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_csv(path: str) -> list[dict]:
    """Load test cases from a CSV file (env_id, instruction, category, difficulty)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a benchmark from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m p2p.benchmark.benchmark_cli --csv benchmark/test_cases.csv\n"
            "  uv run python -m p2p.benchmark.benchmark_cli \\\n"
            "      --csv benchmark/test_cases_humanoid_skills.csv \\\n"
            "      --total-timesteps 500000 --max-iterations 3 --max-parallel 10\n"
        ),
    )

    # Required
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to test cases CSV (columns: env_id, instruction, category, difficulty)",
    )

    # Training params (defaults match dashboard: BenchmarkFormFields.tsx)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pass-threshold", type=float, default=0.9)
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--vlm-model", default=VLM_MODEL)

    # Parallelism
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--cores-per-run", type=int, default=0)

    parser.add_argument(
        "--no-cpu-affinity",
        action="store_true",
        default=False,
        help="Disable CPU affinity pinning (no taskset, rely on OS scheduler)",
    )

    # Multi-config / multi-seed
    parser.add_argument("--num-configs", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list, e.g. '1,2,3'",
    )

    # Staging
    parser.add_argument(
        "--mode",
        default="staged",
        choices=["flat", "staged"],
        help="'flat' runs all cases at once; 'staged' groups into gated batches (default: staged)",
    )
    parser.add_argument("--num-stages", type=int, default=25)
    parser.add_argument("--gate-threshold", type=float, default=0.7)
    parser.add_argument("--start-from-stage", type=int, default=1)

    # Filters
    parser.add_argument(
        "--filter-envs",
        default=None,
        help="Comma-separated env IDs to include, e.g. 'Humanoid-v5,Ant-v5'",
    )
    parser.add_argument(
        "--filter-categories",
        default=None,
        help="Comma-separated categories, e.g. 'locomotion,balance'",
    )
    parser.add_argument(
        "--filter-difficulties",
        default=None,
        help="Comma-separated difficulties, e.g. 'easy,medium'",
    )

    # Feature flags (defaults match dashboard: BenchmarkFormFields.tsx)
    parser.add_argument(
        "--side-info",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-zoo-preset",
        action="store_true",
        default=False,
        help="Disable RL Baselines3 Zoo tuned HPs",
    )
    parser.add_argument("--hp-tuning", action="store_true", default=False)
    parser.add_argument(
        "--use-code-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load and filter test cases
    # ------------------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    test_cases = _load_csv(str(csv_path))
    if not test_cases:
        print(f"Error: CSV file is empty: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.filter_envs:
        envs = {e.strip() for e in args.filter_envs.split(",")}
        test_cases = [tc for tc in test_cases if tc["env_id"] in envs]
    if args.filter_categories:
        cats = {c.strip() for c in args.filter_categories.split(",")}
        test_cases = [tc for tc in test_cases if tc["category"] in cats]
    if args.filter_difficulties:
        diffs = {d.strip() for d in args.filter_difficulties.split(",")}
        test_cases = [tc for tc in test_cases if tc["difficulty"] in diffs]

    if not test_cases:
        print("Error: no test cases match the filters.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse seeds
    # ------------------------------------------------------------------
    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # When HP tuning is off, force single config
    num_configs = args.num_configs
    if not args.hp_tuning:
        num_configs = 1

    # ------------------------------------------------------------------
    # Build LoopConfig and create job via BenchmarkController
    # ------------------------------------------------------------------
    from p2p.config import loop_config_from_params
    from p2p.scheduler.controllers import BenchmarkController
    from p2p.scheduler.job_scheduler import kill_run_process_standalone, _run_scheduler
    from p2p.scheduler.manifest_io import read_job_manifest, write_job_manifest

    loop_config = loop_config_from_params(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        num_envs=args.num_envs,
        side_info=args.side_info,
        seeds=seeds,
        max_iterations=args.max_iterations,
        pass_threshold=args.pass_threshold,
        vlm_model=args.vlm_model,
        max_parallel=args.max_parallel,
        cores_per_run=args.cores_per_run,
        no_cpu_affinity=args.no_cpu_affinity,
        hp_tuning=args.hp_tuning,
        use_code_judge=args.use_code_judge,
        use_zoo_preset=not args.no_zoo_preset,
    )

    configs = None
    if num_configs > 1:
        from p2p.api.services import generate_hp_configs

        configs = generate_hp_configs(max(1, num_configs))

    # Create manifest + pointer without spawning subprocess (we run in-process below)
    ctrl = BenchmarkController()
    job = ctrl.run(
        loop_config=loop_config,
        backend="local",
        test_cases=test_cases,
        mode=args.mode,
        num_stages=args.num_stages,
        gate_threshold=args.gate_threshold,
        start_from_stage=args.start_from_stage,
        max_parallel=args.max_parallel,
        configs=configs,
        seeds=seeds,
        spawn=False,
    )

    job_id = job["job_id"]
    metadata = job.get("metadata", {})
    benchmark_id = metadata.get("benchmark_id", job_id)
    total_stages = metadata.get("total_stages", 0)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"Benchmark {benchmark_id} (job {job_id})")
    print(f"  CSV:             {csv_path} ({len(test_cases)} test cases)")
    print(f"  Mode:            {args.mode}", end="")
    if total_stages:
        print(f" ({total_stages} stages, gate={args.gate_threshold})")
    else:
        print()
    print(f"  Timesteps:       {args.total_timesteps:,}")
    print(f"  Max iterations:  {args.max_iterations}")
    print(f"  Max parallel:    {args.max_parallel}")
    print()

    # ------------------------------------------------------------------
    # Run job scheduler in-process
    # ------------------------------------------------------------------
    def _handle_sigterm(signum: int, frame: object) -> None:
        logger.info("Received signal %d, killing running processes and marking cancelled", signum)
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
            logger.exception("Failed to write cancellation during signal handler")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        _run_scheduler(job_id)
    except Exception:
        logger.exception("Job scheduler crashed: %s", job_id)
        try:
            m = read_job_manifest(job_id)
            if m:
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
