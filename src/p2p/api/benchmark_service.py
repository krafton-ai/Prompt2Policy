"""Benchmark service — read/list/stop operations for benchmark runs.

New benchmarks are created via ``BenchmarkController`` (scheduler path).
Benchmarks with ``benchmark_`` prefix use the standalone manifest format;
``bm_`` prefix benchmarks delegate to the scheduler job manifest.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from p2p.api.entity_lifecycle import (
    _is_benchmark_dir,
    inject_metadata,
    is_entity_deleted,
)
from p2p.api.process_manager import stop_session
from p2p.api.schemas import (
    BenchmarkGroupStats,
    BenchmarkRunDetail,
    BenchmarkRunSummary,
    BenchmarkTestCaseResult,
    StageDetail,
    StageGateResult,
)
from p2p.benchmark.benchmark_helpers import (
    lightweight_session_info,
    read_manifest,
    write_manifest,
)
from p2p.config import LoopConfig
from p2p.contracts import BenchmarkOptions
from p2p.settings import RUNS_DIR
from p2p.utils.process_safety import safe_killpg, verify_pid_ownership

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_job_id(manifest: dict) -> str | None:
    """Return the job_id if this is a pointer manifest from BenchmarkController."""
    if manifest.get("type") == "pointer":
        return manifest.get("job_id")
    return None


def _loop_config_from_manifest(manifest: dict) -> LoopConfig | None:
    """Extract LoopConfig from manifest, or None if unavailable.

    New-format manifests have a ``loop_config`` key.  Legacy manifests
    that predate the LoopConfig format return None.
    """
    if "loop_config" in manifest:
        return LoopConfig.from_json(manifest["loop_config"])
    return None


_test_cases_cache: dict[str, list[dict]] = {}
BENCHMARK_DIR = Path("benchmark")
BENCHMARK_CSV = BENCHMARK_DIR / "test_cases.csv"


def _list_csv_files() -> list[str]:
    """Return available CSV filenames in the benchmark directory."""
    if not BENCHMARK_DIR.exists():
        return []
    return sorted(f.name for f in BENCHMARK_DIR.glob("*.csv"))


def _load_test_cases(csv_file: str | None = None) -> list[dict]:
    """Load benchmark test cases from CSV (cached per filename)."""
    filename = csv_file or BENCHMARK_CSV.name
    # Fallback: if default CSV doesn't exist, use the first available CSV
    if not (BENCHMARK_DIR / filename).exists() and not csv_file:
        available = _list_csv_files()
        if available:
            filename = available[0]
    if "/" in filename or ".." in filename:
        raise ValueError(f"Invalid CSV filename: {filename}")
    if filename in _test_cases_cache:
        return _test_cases_cache[filename]
    csv_path = (BENCHMARK_DIR / filename).resolve()
    if not csv_path.is_relative_to(BENCHMARK_DIR.resolve()):
        raise ValueError(f"Invalid CSV filename: {filename}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        cases = list(csv.DictReader(f))
    _test_cases_cache[filename] = cases
    return cases


# ---------------------------------------------------------------------------
# Scheduler subprocess management
# ---------------------------------------------------------------------------


def _is_scheduler_alive(manifest: dict) -> bool:
    """Check if the scheduler subprocess is still running.

    Handles both legacy manifests (scheduler_pid in benchmark.json)
    and job pointer manifests (delegate to job_queries).
    Verifies PID ownership to guard against recycled PIDs (issue #380).
    """
    job_id = _get_job_id(manifest)
    if job_id:
        from p2p.scheduler.job_queries import is_scheduler_alive as _job_alive

        return _job_alive(job_id)
    pid = manifest.get("scheduler_pid")
    if pid is None:
        return False
    return verify_pid_ownership(pid, expected_cmdline="p2p.scheduler.job_scheduler")


def _kill_scheduler(manifest: dict) -> bool:
    """Send SIGTERM to the scheduler subprocess. Returns True if signal was sent."""
    pid = manifest.get("scheduler_pid")
    if pid is None:
        return False
    return safe_killpg(pid, expected_cmdline="p2p.scheduler.job_scheduler")


# ---------------------------------------------------------------------------
# Session info helpers
# ---------------------------------------------------------------------------


_lightweight_session_info = lightweight_session_info
_read_manifest = read_manifest


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _build_group_stats(results: list[BenchmarkTestCaseResult]) -> BenchmarkGroupStats:
    total = len(results)
    completed = sum(1 for r in results if r.session_status not in ("running", "pending", "queued"))
    passed = sum(1 for r in results if r.passed)
    cumulative = sum(r.best_score for r in results)
    avg = cumulative / completed if completed > 0 else 0.0
    rate = passed / completed if completed > 0 else 0.0
    return BenchmarkGroupStats(
        total=total,
        completed=completed,
        passed=passed,
        success_rate=rate,
        average_score=avg,
        cumulative_score=cumulative,
    )


def _benchmark_test_case_result(
    entry: dict,
    session_info: dict | None,
    node_id: str = "",
) -> BenchmarkTestCaseResult:
    stage = entry.get("stage", 0)
    if session_info is None:
        # Empty session_id means queued (not yet launched by scheduler)
        status = "queued" if not entry.get("session_id") else "pending"
        return BenchmarkTestCaseResult(
            index=entry["index"],
            env_id=entry["env_id"],
            instruction=entry["instruction"],
            category=entry["category"],
            difficulty=entry["difficulty"],
            session_id=entry.get("session_id", ""),
            session_status=status,
            best_score=0.0,
            passed=False,
            iterations_completed=0,
            video_urls=[],
            stage=stage,
            node_id=node_id,
        )

    # Treat stale sessions as failed (process died)
    status = "stale" if session_info["is_stale"] else session_info["status"]

    return BenchmarkTestCaseResult(
        index=entry["index"],
        env_id=entry["env_id"],
        instruction=entry["instruction"],
        category=entry["category"],
        difficulty=entry["difficulty"],
        session_id=entry.get("session_id", ""),
        session_status=status,
        best_score=session_info["best_score"],
        passed=session_info["status"] == "passed",
        iterations_completed=session_info["iterations_completed"],
        video_urls=[],
        stage=stage,
        iteration_scores=session_info.get("iteration_scores", []),
        node_id=node_id,
    )


def _derive_benchmark_status(
    manifest_status: str,
    results: list[BenchmarkTestCaseResult],
    benchmark_id: str = "",
    manifest: dict | None = None,
) -> str:
    """Derive benchmark status from manifest + session states.

    Uses the scheduler subprocess PID to determine whether the scheduler
    is still alive.  If the scheduler is dead, queued entries will never
    be launched, so they should not count as active.
    """
    if manifest_status == "cancelled":
        return "cancelled"

    if manifest_status == "error":
        return "error"

    # Check if any stage gate_failed (staged mode early termination)
    if manifest:
        for stage in manifest.get("stages", []):
            if stage.get("status") == "gate_failed":
                if not _is_scheduler_alive(manifest):
                    return "gate_failed"

    # Check if the scheduler subprocess is still alive
    scheduler_alive = bool(manifest and _is_scheduler_alive(manifest))

    statuses = [r.session_status for r in results]
    # "queued" only counts as active when the scheduler can still launch them
    active = {"running", "pending", "queued"} if scheduler_alive else {"running", "pending"}
    has_active = any(s in active for s in statuses)
    has_stale = any(s == "stale" for s in statuses)

    if not has_active:
        if has_stale:
            return "stale"
        return "completed"

    return "running"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_benchmark_options(csv_file: str | None = None) -> BenchmarkOptions:
    """Return distinct env_ids, categories, and difficulties from the test case CSV."""
    cases = _load_test_cases(csv_file)
    envs = sorted({tc["env_id"] for tc in cases})
    categories = sorted({tc["category"] for tc in cases})
    difficulties_order = {"easy": 0, "medium": 1, "hard": 2}
    difficulties = sorted(
        {tc["difficulty"] for tc in cases},
        key=lambda d: difficulties_order.get(d, 99),
    )
    case_list = [
        {"env_id": tc["env_id"], "category": tc["category"], "difficulty": tc["difficulty"]}
        for tc in cases
    ]
    return BenchmarkOptions(
        envs=envs,
        categories=categories,
        difficulties=difficulties,
        cases=case_list,
        csv_files=_list_csv_files(),
    )


def list_benchmarks(include_deleted: bool = False) -> list[BenchmarkRunSummary]:
    """List all benchmark runs with aggregated stats."""
    if not RUNS_DIR.exists():
        return []
    summaries: list[BenchmarkRunSummary] = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or not _is_benchmark_dir(d):
            continue
        if not include_deleted and is_entity_deleted(d):
            continue
        manifest = _read_manifest(d.name)
        if manifest is None:
            continue

        # Delegate to job-based aggregation for pointer manifests
        job_id = _get_job_id(manifest)
        if job_id:
            from p2p.scheduler.benchmark_aggregation import get_job_benchmark

            data = get_job_benchmark(job_id)
            if data is None:
                logger.debug("Job manifest unavailable for %s (job_id=%s)", d.name, job_id)
                continue
            summary = BenchmarkRunSummary(
                benchmark_id=data["benchmark_id"],
                created_at=data["created_at"],
                completed_at=data.get("completed_at"),
                status=data["status"],
                total_cases=data["total_cases"],
                completed_cases=data["completed_cases"],
                passed_cases=data["passed_cases"],
                success_rate=data["success_rate"],
                average_score=data["average_score"],
                cumulative_score=data["cumulative_score"],
                mode=data.get("mode", "flat"),
                current_stage=data.get("current_stage", 0),
                total_stages=data.get("total_stages", 0),
            )
            inject_metadata(summary, d)
            summaries.append(summary)
            continue

        entries = manifest.get("test_cases", [])
        mode = manifest.get("mode", "flat")
        manifest_stages = manifest.get("stages", [])

        # For stats, only count cases from launched stages
        if mode == "staged" and manifest_stages:
            launched_statuses = {"running", "completed", "gate_passed", "gate_failed"}
            launched_indices: set[int] = set()
            for s in manifest_stages:
                if s.get("status") in launched_statuses:
                    launched_indices.update(s.get("case_indices", []))
            active_entries = [e for e in entries if e["index"] in launched_indices]
        else:
            active_entries = entries

        node_alloc = manifest.get("node_allocation", {})
        results = [
            _benchmark_test_case_result(
                e,
                _lightweight_session_info(e["session_id"]) if e.get("session_id") else None,
                node_id=node_alloc.get(str(e["index"]), ""),
            )
            for e in active_entries
        ]
        stats = _build_group_stats(results)
        status = _derive_benchmark_status(
            manifest.get("status", "running"),
            results,
            benchmark_id=manifest["benchmark_id"],
            manifest=manifest,
        )

        current_stage = manifest.get("current_stage", 0)

        summary = BenchmarkRunSummary(
            benchmark_id=manifest["benchmark_id"],
            created_at=manifest["created_at"],
            completed_at=manifest.get("completed_at"),
            status=status,
            total_cases=stats.total,
            completed_cases=stats.completed,
            passed_cases=stats.passed,
            success_rate=stats.success_rate,
            average_score=stats.average_score,
            cumulative_score=stats.cumulative_score,
            mode=mode,
            current_stage=current_stage,
            total_stages=len(manifest_stages),
        )
        inject_metadata(summary, d)
        summaries.append(summary)
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_benchmark_config(benchmark_id: str) -> dict | None:
    """Return the benchmark config (for preset saving).

    Handles both the new LoopConfig-based manifest and the legacy flat format.
    """
    manifest = _read_manifest(benchmark_id)
    if manifest is None:
        return None

    # Delegate to job manifest config if this is a pointer
    job_id = _get_job_id(manifest)
    if job_id:
        from p2p.scheduler.manifest_io import read_job_manifest

        job_manifest = read_job_manifest(job_id)
        if job_manifest is None:
            return None
        return dict(job_manifest.get("config", {}))

    lc = _loop_config_from_manifest(manifest)
    if lc is not None:
        result: dict = {
            "total_timesteps": lc.train.total_timesteps,
            "max_iterations": lc.max_iterations,
            "pass_threshold": lc.pass_threshold,
            "seed": lc.train.seed,
            "num_envs": lc.train.num_envs,
            "vlm_model": lc.vlm_model,
            "max_parallel": lc.max_parallel,
            "cores_per_run": lc.cores_per_run,
            "side_info": lc.train.side_info,
            "use_zoo_preset": lc.use_zoo_preset,
            "hp_tuning": lc.hp_tuning,
            "use_code_judge": lc.use_code_judge,
            "review_reward": lc.review_reward,
            "review_judge": lc.review_judge,
            "device": lc.train.device,
            "thinking_effort": lc.thinking_effort,
            "seeds": lc.seeds or [lc.train.seed],
        }
    else:
        result = {}

    # Add benchmark-specific manifest fields
    for k in ("num_configs", "mode", "start_from_stage", "csv_file"):
        if k in manifest:
            result[k] = manifest[k]

    return result


def get_benchmark(benchmark_id: str) -> BenchmarkRunDetail | None:
    """Get detailed benchmark run with breakdowns."""
    manifest = _read_manifest(benchmark_id)
    if manifest is None:
        return None

    # Delegate to job-based aggregation if this is a pointer manifest
    job_id = _get_job_id(manifest)
    if job_id:
        from p2p.scheduler.benchmark_aggregation import get_job_benchmark

        data = get_job_benchmark(job_id)
        if data is None:
            return None
        detail = BenchmarkRunDetail(**data)
        inject_metadata(detail, RUNS_DIR / benchmark_id)
        return detail

    entries = manifest.get("test_cases", [])
    mode = manifest.get("mode", "flat")
    manifest_stages = manifest.get("stages", [])
    start_from = manifest.get("start_from_stage", 1)

    node_alloc = manifest.get("node_allocation", {})
    results: list[BenchmarkTestCaseResult] = []
    for e in entries:
        info = _lightweight_session_info(e["session_id"]) if e.get("session_id") else None
        results.append(
            _benchmark_test_case_result(
                e,
                info,
                node_id=node_alloc.get(str(e["index"]), ""),
            )
        )

    # For stats + test_cases, only include cases from launched stages
    if mode == "staged" and manifest_stages:
        launched_statuses = {"running", "completed", "gate_passed", "gate_failed"}
        launched_indices: set[int] = set()
        for s in manifest_stages:
            if s.get("status") in launched_statuses:
                launched_indices.update(s.get("case_indices", []))
        active_results = [r for r in results if r.index in launched_indices]
    else:
        active_results = results

    stats = _build_group_stats(active_results)
    status = _derive_benchmark_status(
        manifest.get("status", "running"),
        active_results,
        benchmark_id=manifest["benchmark_id"],
        manifest=manifest,
    )

    # Group by category, difficulty, env (stats use active only)
    by_category: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    by_difficulty: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    by_env: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    for r in active_results:
        by_category[r.category].append(r)
        by_difficulty[r.difficulty].append(r)
        by_env[r.env_id].append(r)

    # Return ALL cases (including pending stages) so the frontend can display them
    all_results = results

    # Build stage details
    stage_details: list[StageDetail] = []
    for s in manifest_stages:
        gate_result = None
        if s.get("gate_result"):
            gate_result = StageGateResult(**s["gate_result"])
        stage_details.append(
            StageDetail(
                stage=s["stage"],
                name=s.get("name", f"Stage {s['stage']}"),
                status=s.get("status", "pending"),
                gate_threshold=s.get("gate_threshold", 0.0),
                max_parallel=s.get("max_parallel", 0),
                case_count=len(s.get("case_indices", [])),
                case_indices=s.get("case_indices", []),
                gate_result=gate_result,
            )
        )

    lc = _loop_config_from_manifest(manifest)
    detail = BenchmarkRunDetail(
        benchmark_id=manifest["benchmark_id"],
        created_at=manifest["created_at"],
        completed_at=manifest.get("completed_at"),
        status=status,
        total_cases=stats.total,
        completed_cases=stats.completed,
        passed_cases=stats.passed,
        success_rate=stats.success_rate,
        average_score=stats.average_score,
        cumulative_score=stats.cumulative_score,
        pass_threshold=lc.pass_threshold if lc else manifest.get("pass_threshold", 0.7),
        by_category={k: _build_group_stats(v) for k, v in by_category.items()},
        by_difficulty={k: _build_group_stats(v) for k, v in by_difficulty.items()},
        by_env={k: _build_group_stats(v) for k, v in by_env.items()},
        test_cases=all_results,
        mode=mode,
        current_stage=manifest.get("current_stage", 0),
        total_stages=len(manifest_stages),
        stages=stage_details,
        start_from_stage=start_from,
        max_iterations=lc.max_iterations if lc else manifest.get("max_iterations", 5),
    )
    inject_metadata(detail, RUNS_DIR / benchmark_id)
    return detail


def stop_benchmark(benchmark_id: str) -> tuple[bool, int]:
    """Stop a running benchmark.

    1. Writes ``"cancelled"`` to the manifest (scheduler detects on next loop).
    2. Sends SIGTERM to the scheduler subprocess.
    3. Stops all running training sessions.

    Returns (stopped, count_of_stopped_sessions).
    """
    manifest = _read_manifest(benchmark_id)
    if manifest is None:
        return False, 0

    # Delegate to job-based cancel if this is a pointer
    job_id = _get_job_id(manifest)
    if job_id:
        from p2p.scheduler.job_queries import cancel_job

        cancel_job(job_id)
        # Update pointer status too
        manifest["status"] = "cancelled"
        write_manifest(manifest, benchmark_id)
        return True, 0

    # Legacy path: kill the scheduler subprocess first
    _kill_scheduler(manifest)

    # Stop individual training sessions
    stopped_count = 0
    for entry in manifest.get("test_cases", []):
        sid = entry.get("session_id")
        if sid and stop_session(sid):
            stopped_count += 1

    # Update manifest status
    manifest["status"] = "cancelled"
    manifest.pop("scheduler_pid", None)
    write_manifest(manifest, benchmark_id)

    return True, stopped_count
