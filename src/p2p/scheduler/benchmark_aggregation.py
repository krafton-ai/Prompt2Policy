"""Benchmark aggregation: result collection and training curve metrics.

Provides ``get_job_benchmark`` (overall benchmark summary) and
``get_benchmark_case_metrics`` (per-case cross-config training curves).
"""

from __future__ import annotations

import re as _re
from collections import defaultdict
from pathlib import Path

from p2p.api.services import aggregate_scalars
from p2p.benchmark.benchmark_helpers import lightweight_session_info
from p2p.contracts import (
    BenchmarkCostSummary,
    BenchmarkGroupStats,
    BenchmarkTestCaseResult,
    TokenUsageByModel,
)
from p2p.scheduler.manifest_io import read_job_manifest
from p2p.session.iteration_record import IterationRecord
from p2p.settings import RUNS_DIR, resolve_session_dir

# Regex fallback to extract config_id and seed from run_id when tags are absent
_RUN_ID_PATTERN = _re.compile(r"_case\d+_(.+)_s(\d+)$")


def _static_prefix_for(iter_path: Path) -> str:
    """Build /static/runs/... prefix from an iteration directory path."""
    try:
        rel = iter_path.resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        return f"/static/runs/{iter_path.parent.name}/{iter_path.name}"
    return f"/static/runs/{rel.as_posix()}"


def _read_best_run_and_checkpoint(
    iter_dir: Path,
) -> tuple[str | None, str | None]:
    """Read best_run_id and best_checkpoint from iteration metadata files."""
    import json as _json

    best_run_id: str | None = None
    best_run_path = iter_dir / "best_run.json"
    if best_run_path.exists():
        try:
            best_run_id = _json.loads(best_run_path.read_text()).get("best_run_id")
        except (ValueError, KeyError):
            pass

    best_checkpoint: str | None = None
    judgment_path = iter_dir / "judgment.json"
    if judgment_path.exists():
        try:
            best_checkpoint = str(
                _json.loads(judgment_path.read_text()).get("best_checkpoint", "")
            )
        except (ValueError, KeyError):
            pass

    return best_run_id, best_checkpoint


def _best_video_urls(session_id: str, best_iteration: int) -> list[str]:
    """Return the median video URL from the best checkpoint of the best run.

    For multi-config iterations, reads ``best_run.json`` to identify the
    winning config/seed sub-directory and ``judgment.json`` to find the
    best checkpoint step.
    """
    if not session_id or best_iteration < 1:
        return []
    iter_dir = resolve_session_dir(session_id) / f"iter_{best_iteration}"
    if not iter_dir.is_dir():
        return []
    prefix = _static_prefix_for(iter_dir)
    best_run_id, best_checkpoint = _read_best_run_and_checkpoint(iter_dir)

    # 1) Single-config iteration (no sub-runs)
    rec = IterationRecord(iter_dir)
    filenames = rec.video_filenames()
    if filenames and not best_run_id:
        video = _pick_median_video(filenames, best_checkpoint)
        return [f"{prefix}/videos/{video}"]

    # 2) Multi-config: try best_run_id sub-directory first
    if best_run_id:
        sub = iter_dir / best_run_id
        if sub.is_dir() and (sub / "videos").exists():
            sub_fns = IterationRecord(sub).video_filenames()
            if sub_fns:
                video = _pick_median_video(sub_fns, best_checkpoint)
                return [f"{prefix}/{best_run_id}/videos/{video}"]

    # 3) Fallback: first sub-run with videos
    for sub in sorted(iter_dir.iterdir()):
        if sub.is_dir() and (sub / "videos").exists():
            sub_fns = IterationRecord(sub).video_filenames()
            if sub_fns:
                video = _pick_median_video(sub_fns, best_checkpoint)
                return [f"{prefix}/{sub.name}/videos/{video}"]
    return []


def _find_median_rollout(session_id: str, best_iteration: int) -> dict | None:
    """Locate the median rollout judgment dict for the best checkpoint."""
    import json as _json

    if not session_id or best_iteration < 1:
        return None
    iter_dir = resolve_session_dir(session_id) / f"iter_{best_iteration}"
    if not iter_dir.is_dir():
        return None
    best_run_id, _ = _read_best_run_and_checkpoint(iter_dir)

    # Read judgment from best run sub-dir or iteration root
    if best_run_id:
        judgment_path = iter_dir / best_run_id / "judgment.json"
    else:
        judgment_path = iter_dir / "judgment.json"
    if not judgment_path.exists():
        judgment_path = iter_dir / "judgment.json"
    if not judgment_path.exists():
        return None

    try:
        judgment = _json.loads(judgment_path.read_text())
    except (ValueError, KeyError):
        return None

    best_checkpoint = str(judgment.get("best_checkpoint", ""))
    if not best_checkpoint:
        return None

    cp_data = judgment.get("checkpoint_judgments", {}).get(best_checkpoint, {})
    rollout_judgments = cp_data.get("rollout_judgments", [])

    for rj in rollout_judgments:
        if rj.get("rollout_label") == "median":
            return rj
    if rollout_judgments:
        return rollout_judgments[0]
    # Legacy: return checkpoint-level data directly
    return cp_data if "intent_score" in cp_data else None


def _best_judge_scores(session_id: str, best_iteration: int) -> dict[str, float | None]:
    """Return code/vlm/synthesizer scores for the median rollout."""
    rj = _find_median_rollout(session_id, best_iteration)
    if not rj:
        return {}
    return {
        "code": rj.get("code_score"),
        "vlm": rj.get("vlm_score"),
        "synthesizer": rj.get("intent_score"),
    }


def _best_judge_diagnoses(session_id: str, best_iteration: int) -> dict[str, str]:
    """Return code/vlm/synthesizer diagnoses for the median rollout."""
    rj = _find_median_rollout(session_id, best_iteration)
    if not rj:
        return {}
    result: dict[str, str] = {}
    if rj.get("code_diagnosis"):
        result["code"] = str(rj["code_diagnosis"])
    if rj.get("vlm_diagnosis"):
        result["vlm"] = str(rj["vlm_diagnosis"])
    if rj.get("diagnosis"):
        result["synthesizer"] = str(rj["diagnosis"])
    return result


def _pick_median_video(filenames: list[str], best_checkpoint: str | None) -> str:
    """Select the median rollout video for *best_checkpoint*, else first file."""
    if best_checkpoint:
        median = f"eval_{best_checkpoint}_median.mp4"
        if median in filenames:
            return median
        for fn in filenames:
            if fn.startswith(f"eval_{best_checkpoint}"):
                return fn
    return filenames[0]


# ---------------------------------------------------------------------------
# Cost aggregation
# ---------------------------------------------------------------------------


def _aggregate_token_usage(manifest: dict) -> BenchmarkCostSummary:
    """Aggregate LLM/VLM token usage across all sessions in a benchmark job."""
    import json as _json

    seen_sessions: set[str] = set()
    sessions_counted = 0
    # model_name -> [input_tokens, output_tokens, call_count]
    usage: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for run in manifest["runs"]:
        session_id = run["run_id"]
        if session_id in seen_sessions:
            continue
        seen_sessions.add(session_id)

        events_path = resolve_session_dir(session_id) / "events.jsonl"
        if not events_path.exists():
            continue
        sessions_counted += 1

        for line in events_path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except (ValueError, _json.JSONDecodeError):
                continue
            event = entry.get("event", "")
            if event not in ("llm.call", "vlm.call"):
                continue
            data = entry.get("data", {})
            model = str(data.get("model", "unknown"))
            in_tok = int(data.get("input_tokens", 0) or 0)
            out_tok = int(data.get("output_tokens", 0) or 0)
            acc = usage[model]
            acc[0] += in_tok
            acc[1] += out_tok
            acc[2] += 1

    models: list[TokenUsageByModel] = [
        TokenUsageByModel(
            model=model,
            input_tokens=vals[0],
            output_tokens=vals[1],
            call_count=vals[2],
        )
        for model, vals in sorted(usage.items())
    ]

    return BenchmarkCostSummary(
        models=models,
        total_input_tokens=sum(v[0] for v in usage.values()),
        total_output_tokens=sum(v[1] for v in usage.values()),
        total_calls=sum(v[2] for v in usage.values()),
        sessions_counted=sessions_counted,
        sessions_total=len(seen_sessions),
    )


# ---------------------------------------------------------------------------
# Benchmark job detail
# ---------------------------------------------------------------------------


def get_job_benchmark(job_id: str) -> dict | None:
    """Aggregate benchmark data from a scheduler job manifest.

    Reads the job manifest + per-session status to build a response
    matching the ``BenchmarkRunDetail`` schema that the frontend expects.
    """
    manifest = read_job_manifest(job_id)
    if manifest is None or manifest["job_type"] != "benchmark":
        return None

    metadata = manifest.get("metadata", {})
    config = manifest.get("config", {})
    test_cases_meta: list[dict] = metadata.get("test_cases", [])
    mode = config.get("mode", "flat")
    benchmark_id = metadata.get("benchmark_id", job_id)
    pass_threshold = config.get("pass_threshold", 0.7)
    max_iterations = config.get("max_iterations", 5)

    # Stage definitions from metadata
    stage_defs: list[dict] = metadata.get("stages", [])
    total_stages = metadata.get("total_stages", len(stage_defs))
    current_stage = metadata.get("current_stage", 0)

    # Build case_index → stage mapping from stage defs
    case_stage_map: dict[int, int] = {}
    for sd in stage_defs:
        for idx in sd.get("case_indices", []):
            case_stage_map[idx] = sd["stage"]

    # Launched stage indices (for stats filtering in staged mode)
    launched_statuses = {"running", "completed", "gate_passed", "gate_failed"}
    launched_indices: set[int] = set()
    if mode == "staged" and stage_defs:
        for sd in stage_defs:
            if sd.get("status") in launched_statuses:
                launched_indices.update(sd.get("case_indices", []))

    # Group runs by case_index
    runs_by_case: dict[int, list[dict]] = defaultdict(list)
    for run in manifest["runs"]:
        tags = run.get("spec", {}).get("tags", {})
        idx_str = tags.get("case_index")
        if idx_str is not None:
            runs_by_case[int(idx_str)].append(run)

    # Build test case results
    results: list[BenchmarkTestCaseResult] = []
    for tc in test_cases_meta:
        i = tc["index"]
        runs = runs_by_case.get(i, [])

        # Find the best result across all runs for this test case
        best_info: dict | None = None
        best_score = 0.0
        best_node_id = ""
        best_session_id = ""
        best_pid: int | None = None
        for run in runs:
            session_id = run["run_id"]
            info = lightweight_session_info(session_id)
            if info is not None:
                if best_info is None or info["best_score"] > best_score:
                    best_score = info["best_score"]
                    best_info = info
                    best_node_id = run.get("node_id", "")
                    best_session_id = session_id
                    best_pid = int(run["pid"]) if run.get("pid") else None
            elif best_info is None:
                best_node_id = run.get("node_id", "")
                best_pid = int(run["pid"]) if run.get("pid") else None

        case_stage = case_stage_map.get(i, 0)

        # Determine status from run state if no session info
        if best_info is None:
            run_states = [r["state"] for r in runs]
            if "running" in run_states:
                status = "running"
            elif "pending" in run_states:
                status = "pending"
            elif not runs:
                status = "queued"
            else:
                status = "pending"
            results.append(
                BenchmarkTestCaseResult(
                    index=i,
                    env_id=tc.get("env_id", ""),
                    instruction=tc.get("instruction", ""),
                    category=tc.get("category", ""),
                    difficulty=tc.get("difficulty", ""),
                    session_id=runs[0]["run_id"] if runs else "",
                    session_status=status,
                    best_score=0.0,
                    passed=False,
                    iterations_completed=0,
                    video_urls=[],
                    stage=case_stage,
                    node_id=best_node_id,
                    pids=[best_pid] if best_pid else [],
                    max_iterations=max_iterations,
                ),
            )
        else:
            status = "stale" if best_info["is_stale"] else best_info["status"]
            results.append(
                BenchmarkTestCaseResult(
                    index=i,
                    env_id=tc.get("env_id", ""),
                    instruction=tc.get("instruction", ""),
                    category=tc.get("category", ""),
                    difficulty=tc.get("difficulty", ""),
                    session_id=best_session_id,
                    session_status=status,
                    best_score=best_info["best_score"],
                    passed=best_info["status"] == "passed",
                    iterations_completed=best_info["iterations_completed"],
                    video_urls=_best_video_urls(
                        best_session_id, best_info.get("best_iteration", 0)
                    ),
                    judge_scores=_best_judge_scores(
                        best_session_id, best_info.get("best_iteration", 0)
                    ),
                    judge_diagnoses=_best_judge_diagnoses(
                        best_session_id, best_info.get("best_iteration", 0)
                    ),
                    stage=case_stage,
                    iteration_scores=best_info.get("iteration_scores", []),
                    node_id=best_node_id,
                    pids=[best_pid] if best_pid else [],
                    max_iterations=max_iterations,
                ),
            )

    # In staged mode, filter to launched stages for stats
    if mode == "staged" and launched_indices:
        active_results = [r for r in results if r["index"] in launched_indices]
    else:
        active_results = results

    # Aggregate stats
    def _group_stats(items: list[BenchmarkTestCaseResult]) -> BenchmarkGroupStats:
        total = len(items)
        completed = sum(
            1 for r in items if r["session_status"] not in ("running", "pending", "queued")
        )
        passed = sum(1 for r in items if r["passed"])
        cumulative = sum(r["best_score"] for r in items)
        return BenchmarkGroupStats(
            total=total,
            completed=completed,
            passed=passed,
            success_rate=passed / total if total > 0 else 0.0,
            average_score=cumulative / total if total > 0 else 0.0,
            cumulative_score=cumulative,
        )

    by_category: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    by_difficulty: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    by_env: dict[str, list[BenchmarkTestCaseResult]] = defaultdict(list)
    for r in active_results:
        by_category[r["category"]].append(r)
        by_difficulty[r["difficulty"]].append(r)
        by_env[r["env_id"]].append(r)

    overall = _group_stats(active_results)

    # Build per-case sync status: True if any run for that case is synced or local
    sync_status: dict[int, bool] = {}
    for tc in test_cases_meta:
        i = tc["index"]
        runs = runs_by_case.get(i, [])
        synced = any(r.get("synced") or r.get("node_id", "") in ("", "local") for r in runs)
        sync_status[i] = synced

    # Determine job-level status
    job_status = manifest["status"]
    if job_status == "running":
        statuses = [r["session_status"] for r in results]
        has_active = any(s in ("running", "pending", "queued") for s in statuses)
        if not has_active:
            job_status = "completed"

    # Build stage details for response
    stage_details: list[dict] = []
    for sd in stage_defs:
        stage_details.append(
            {
                "stage": sd["stage"],
                "name": sd.get("name", f"Batch {sd['stage']}"),
                "status": sd.get("status", "pending"),
                "gate_threshold": sd.get("gate_threshold", 0.0),
                "max_parallel": sd.get("max_parallel", 0),
                "case_count": len(sd.get("case_indices", [])),
                "case_indices": sd.get("case_indices", []),
                "gate_result": sd.get("gate_result"),
            }
        )

    return {
        "benchmark_id": benchmark_id,
        "created_at": manifest["created_at"],
        "completed_at": manifest.get("completed_at"),
        "status": job_status,
        "total_cases": overall["total"],
        "completed_cases": overall["completed"],
        "passed_cases": overall["passed"],
        "success_rate": overall["success_rate"],
        "average_score": overall["average_score"],
        "cumulative_score": overall["cumulative_score"],
        "pass_threshold": pass_threshold,
        "by_category": {k: _group_stats(v) for k, v in by_category.items()},
        "by_difficulty": {k: _group_stats(v) for k, v in by_difficulty.items()},
        "by_env": {k: _group_stats(v) for k, v in by_env.items()},
        "test_cases": results,
        "mode": mode,
        "current_stage": current_stage,
        "total_stages": total_stages,
        "stages": stage_details,
        "start_from_stage": config.get("start_from_stage", 1),
        "max_iterations": max_iterations,
        "alias": "",
        "starred": False,
        "tags": [],
        "sync_status": {str(k): v for k, v in sync_status.items()},
        "cost_summary": _aggregate_token_usage(manifest),
    }


# ---------------------------------------------------------------------------
# Benchmark case metrics (cross-config training curve comparison)
# ---------------------------------------------------------------------------


def get_benchmark_case_metrics(
    job_id: str,
    case_index: int,
    iteration: int = 0,
) -> dict | None:
    """Aggregate cross-config training metrics for a single benchmark case.

    Args:
        job_id: The scheduler job ID.
        case_index: Zero-based test case index.
        iteration: Iteration number (1-based). 0 means latest.

    Returns:
        Dict with case metadata, per-config aggregated metrics, and run info.
        None if the job is not found or is not a benchmark.
    """
    manifest = read_job_manifest(job_id)
    if manifest is None or manifest["job_type"] != "benchmark":
        return None

    metadata = manifest.get("metadata", {})
    test_cases_meta: list[dict] = metadata.get("test_cases", [])

    # Validate case_index
    tc_meta: dict | None = None
    for tc in test_cases_meta:
        if tc["index"] == case_index:
            tc_meta = tc
            break
    if tc_meta is None:
        return None

    # Filter runs for this case
    case_runs: list[dict] = []
    for run in manifest["runs"]:
        tags = run.get("spec", {}).get("tags", {})
        idx_str = tags.get("case_index")
        if idx_str is not None and int(idx_str) == case_index:
            case_runs.append(run)

    if not case_runs:
        return None

    if len(case_runs) != 1:
        return None
    return _case_metrics_bundled(case_runs[0], tc_meta, iteration)


# -- New-format helper (1 run per case, multi-config dirs inside iter_N) ------


def _case_metrics_bundled(
    run: dict,
    tc_meta: dict,
    iteration: int,
) -> dict:
    session_id = run["run_id"]
    session_dir = resolve_session_dir(session_id)

    # Discover iterations
    available_iterations: set[int] = set()
    if session_dir.is_dir():
        for d in session_dir.iterdir():
            if d.is_dir() and d.name.startswith("iter_"):
                try:
                    available_iterations.add(int(d.name.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    sorted_iters = sorted(available_iterations)
    target_iter = _resolve_iteration(iteration, available_iterations, sorted_iters)

    iter_dir = session_dir / f"iter_{target_iter}"

    # Discover {config_id}_seed_{seed} sub-dirs
    config_seed_re = _re.compile(r"^(.+)_seed_(\d+)$")
    runs_by_config: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    if iter_dir.is_dir():
        for sub in iter_dir.iterdir():
            m = config_seed_re.match(sub.name)
            if m and sub.is_dir():
                runs_by_config[m.group(1)].append((int(m.group(2)), sub))

    configs_result: list[dict] = []
    for config_id, seed_dirs in sorted(runs_by_config.items()):
        seed_scalars: list[tuple[int, list[dict]]] = []
        for seed, run_dir in seed_dirs:
            rec = IterationRecord(run_dir)
            training, evaluation = rec.parse_scalars()
            scalars = training or _flatten_eval(evaluation)
            if scalars:
                seed_scalars.append((seed, scalars))

        agg = aggregate_scalars(seed_scalars, config_id)
        if agg is not None:
            configs_result.append(dict(agg))

    # Build runs info from discovered sub-dirs
    runs_info: list[dict] = []
    for config_id, seed_dirs in sorted(runs_by_config.items()):
        for seed, _ in seed_dirs:
            runs_info.append(
                {
                    "run_id": f"{session_id}/{config_id}_seed_{seed}",
                    "config_id": config_id,
                    "seed": seed,
                    "state": run["state"],
                    "node_id": run.get("node_id", ""),
                }
            )

    return {
        "case_index": tc_meta["index"],
        "test_case": {
            "env_id": tc_meta.get("env_id", ""),
            "instruction": tc_meta.get("instruction", ""),
            "category": tc_meta.get("category", ""),
            "difficulty": tc_meta.get("difficulty", ""),
        },
        "configs": configs_result,
        "iterations": sorted_iters,
        "selected_iteration": target_iter,
        "runs": runs_info,
    }


# -- Shared helpers -----------------------------------------------------------


def _resolve_iteration(
    iteration: int,
    available: set[int],
    sorted_iters: list[int],
) -> int:
    """Return the target iteration number (0 = latest)."""
    if iteration == 0 and sorted_iters:
        return sorted_iters[-1]
    if iteration in available:
        return iteration
    if sorted_iters:
        return sorted_iters[-1]
    return 1


def _flatten_eval(evaluation: list[dict]) -> list[dict]:
    """Convert evaluation records to flat scalar rows."""
    flat: list[dict] = []
    for e in evaluation:
        row: dict = {
            "global_step": e.get("global_step", 0),
            "episodic_return": e.get("total_reward", 0),
            "episode_length": e.get("episode_length", 0),
        }
        for tk, tv in (e.get("reward_terms") or {}).items():
            if isinstance(tv, int | float):
                row[tk] = tv
        flat.append(row)
    return flat
