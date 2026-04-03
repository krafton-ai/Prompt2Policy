"""Scheduler CLI entry point.

Usage::

    python -m p2p.scheduler node list
    python -m p2p.scheduler node add --node-id gpu1 --host 10.0.0.1 --user rlws
    python -m p2p.scheduler job list
    python -m p2p.scheduler job submit-session --prompt "walk forward" --env-id HalfCheetah-v5
    python -m p2p.scheduler job submit-benchmark --csv-file test_cases.csv
    python -m p2p.scheduler run list --job-id <job_id>
"""

# ruff: noqa: I001 — p2p.settings must be imported before gymnasium/mujoco
from __future__ import annotations

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports

import argparse
import logging
import sys

from p2p.config import DEFAULT_JUDGMENT_SELECT
from p2p.settings import LLM_MODEL, VLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------------
# Shared training args
# ---------------------------------------------------------------------------


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training arguments shared by submit-session and submit-benchmark."""
    parser.add_argument("--backend", default="local", choices=["local", "ssh"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. '1,2,3'")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--pass-threshold", type=float, default=0.7)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--model", default=LLM_MODEL, help="LLM model for reward/judge agents")
    parser.add_argument("--vlm-model", default=VLM_MODEL)
    parser.add_argument("--side-info", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hp-tuning", action="store_true", default=False)
    parser.add_argument("--use-code-judge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--review-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cores-per-run", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thinking-effort", default="")
    parser.add_argument(
        "--refined-initial-frame",
        action="store_true",
        default=True,
        help="Pad video + send first frame JPEG in Turn 2 (Gemini only)",
    )
    parser.add_argument(
        "--criteria-diagnosis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Criteria-guided holistic scoring (Method D style)",
    )
    parser.add_argument(
        "--motion-trail-dual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send normal + motion trail dual videos to VLM (Method F style)",
    )
    parser.add_argument("--judgment-select", default=DEFAULT_JUDGMENT_SELECT)
    parser.add_argument(
        "--no-zoo-preset",
        action="store_true",
        default=False,
        help="Disable RL Baselines3 Zoo tuned HPs",
    )
    parser.add_argument("--trajectory-stride", type=int, default=1)
    parser.add_argument(
        "--foreground",
        action="store_true",
        default=False,
        help="Run scheduler in-process instead of spawning a subprocess",
    )
    parser.add_argument(
        "--nodes",
        default=None,
        help="Comma-separated node IDs to restrict job placement, e.g. 'gpu1,gpu2'",
    )


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m p2p.scheduler",
        description="Scheduler CLI — manage nodes, jobs, and runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Top-level command")

    # ---- node ----
    node_parser = subparsers.add_parser("node", help="Manage SSH nodes")
    node_sub = node_parser.add_subparsers(dest="action", help="Node action")

    node_sub.add_parser("list", help="List all nodes")

    add_p = node_sub.add_parser("add", help="Add a new node")
    add_p.add_argument("--node-id", required=True)
    add_p.add_argument("--host", required=True)
    add_p.add_argument("--user", required=True)
    add_p.add_argument("--port", type=int, default=22)
    add_p.add_argument("--base-dir", default="")
    add_p.add_argument("--max-cores", type=int, default=1)

    check_p = node_sub.add_parser("check", help="Check node connectivity")
    check_p.add_argument("node_id", help="Node ID to check")

    setup_p = node_sub.add_parser("setup", help="Setup node (install uv, sync code)")
    setup_p.add_argument("node_id", help="Node ID to setup")

    update_p = node_sub.add_parser("update", help="Update node settings")
    update_p.add_argument("node_id", help="Node ID to update")
    update_p.add_argument("--max-cores", type=int, default=None)
    enabled_group = update_p.add_mutually_exclusive_group()
    enabled_group.add_argument("--enabled", action="store_true", dest="enabled", default=None)
    enabled_group.add_argument("--disabled", action="store_false", dest="enabled")

    remove_p = node_sub.add_parser("remove", help="Remove a node")
    remove_p.add_argument("node_id", help="Node ID to remove")

    # ---- job ----
    job_parser = subparsers.add_parser("job", help="Manage jobs")
    job_sub = job_parser.add_subparsers(dest="action", help="Job action")

    job_sub.add_parser("list", help="List all jobs")

    view_p = job_sub.add_parser("view", help="View job details")
    view_p.add_argument("job_id", help="Job ID")

    cancel_p = job_sub.add_parser("cancel", help="Cancel a running job")
    cancel_p.add_argument("job_id", help="Job ID")

    sync_p = job_sub.add_parser("sync", help="Sync all unsynced runs for a job")
    sync_p.add_argument("job_id", help="Job ID")

    # submit-session
    ss_p = job_sub.add_parser("submit-session", help="Submit a session job")
    ss_p.add_argument("--prompt", required=True, help="Training prompt")
    ss_p.add_argument("--env-id", default="HalfCheetah-v5")
    ss_p.add_argument("--node-id", default=None, help="Target node (None=auto)")
    ss_p.add_argument("--num-evals", type=int, default=4)
    ss_p.add_argument("--num-configs", type=int, default=1)
    ss_p.add_argument("--configs", default=None, help="JSON array of config overrides")
    _add_training_args(ss_p)

    # submit-benchmark
    sb_p = job_sub.add_parser("submit-benchmark", help="Submit a benchmark job")
    sb_p.add_argument(
        "--csv-file",
        default=None,
        help="Path to test cases CSV (default: benchmark/test_cases.csv)",
    )
    sb_p.add_argument(
        "--mode",
        default="staged",
        choices=["flat", "staged"],
        help="'flat' or 'staged' (default: staged)",
    )
    sb_p.add_argument("--num-stages", type=int, default=25)
    sb_p.add_argument("--gate-threshold", type=float, default=0.7)
    sb_p.add_argument("--start-from-stage", type=int, default=1)
    sb_p.add_argument("--num-configs", type=int, default=1)
    sb_p.add_argument("--filter-envs", default=None, help="Comma-separated env IDs")
    sb_p.add_argument("--filter-categories", default=None, help="Comma-separated categories")
    sb_p.add_argument("--filter-difficulties", default=None, help="Comma-separated difficulties")
    _add_training_args(sb_p)

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="Manage individual runs")
    run_sub = run_parser.add_subparsers(dest="action", help="Run action")

    list_r = run_sub.add_parser("list", help="List runs")
    list_r.add_argument("--job-id", default=None, help="Filter by job ID")

    view_r = run_sub.add_parser("view", help="View run details")
    view_r.add_argument("run_id", help="Run ID")

    log_r = run_sub.add_parser("log", help="View run subprocess log")
    log_r.add_argument("run_id", help="Run ID")
    log_r.add_argument("--tail", type=int, default=100, help="Number of lines")

    sync_r = run_sub.add_parser("sync", help="Sync run results from remote")
    sync_r.add_argument("run_id", help="Run ID")
    sync_r.add_argument("--job-id", default=None, help="Job ID (optional)")
    sync_r.add_argument(
        "--mode",
        default="full",
        choices=["full", "lite"],
        help="Sync mode: full or lite (default: full)",
    )

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    ("node", "list"): "handle_node_list",
    ("node", "add"): "handle_node_add",
    ("node", "check"): "handle_node_check",
    ("node", "setup"): "handle_node_setup",
    ("node", "update"): "handle_node_update",
    ("node", "remove"): "handle_node_remove",
    ("job", "list"): "handle_job_list",
    ("job", "view"): "handle_job_view",
    ("job", "cancel"): "handle_job_cancel",
    ("job", "sync"): "handle_job_sync",
    ("job", "submit-session"): "handle_job_submit_session",
    ("job", "submit-benchmark"): "handle_job_submit_benchmark",
    ("run", "list"): "handle_run_list",
    ("run", "view"): "handle_run_view",
    ("run", "log"): "handle_run_log",
    ("run", "sync"): "handle_run_sync",
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1
    if not args.action:
        # Print help for the specific subcommand
        parser.parse_args([args.command, "--help"])
        return 1

    key = (args.command, args.action)
    handler_name = _HANDLERS.get(key)
    if handler_name is None:
        print(f"Unknown command: {args.command} {args.action}", file=sys.stderr)
        return 1

    from p2p.scheduler import cli

    handler = getattr(cli, handler_name)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
