"""Shared helpers for benchmark service and job scheduler.

Used by both the web server (benchmark_service.py) and the job scheduler
subprocess (job_scheduler.py).  Keeping them here avoids copy-paste drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from p2p.api.process_manager import is_stale
from p2p.session.iteration_record import SessionRecord
from p2p.settings import RUNS_DIR, resolve_session_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def read_manifest(benchmark_id: str) -> dict | None:
    """Read benchmark.json for *benchmark_id*, or ``None`` if missing/corrupt."""
    path = RUNS_DIR / benchmark_id / "benchmark.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_manifest(manifest: dict, benchmark_id: str) -> None:
    path = RUNS_DIR / benchmark_id / "benchmark.json"
    path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Session info (lightweight, file-based)
# ---------------------------------------------------------------------------


def best_streaming_score(session_dir: Path) -> float:
    """Scan ``streaming_judgments/`` across all iterations for the best live score."""
    best = 0.0
    for sj_dir in session_dir.glob("iter_*/streaming_judgments"):
        for f in sj_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                raw = d.get("intent_score")
                score = float(raw) if raw is not None else 0.0
                if score > best:
                    best = score
            except (json.JSONDecodeError, ValueError, OSError):
                logger.warning("Failed to read streaming judgment: %s", f)
                continue
    return best


def lightweight_session_info(session_id: str) -> dict | None:
    """Read minimal session info (status, score, iters) without heavy processing."""
    session_dir = resolve_session_dir(session_id)
    if not session_dir.exists():
        return None

    sr = SessionRecord(session_dir)
    status_data = sr.read_status()
    stale = is_stale(status_data, session_id)

    history = sr.read_history()
    if history is None:
        status = "running"
        if status_data:
            status = status_data.get("status", "running")
        live_score = best_streaming_score(session_dir)
        return {
            "status": status,
            "best_score": live_score,
            "iterations_completed": 0,
            "is_stale": stale,
            "best_iteration": 0,
            "iteration_scores": [live_score] if live_score > 0 else [],
        }

    best_score = history.get("best_score", 0.0)
    if best_score == 0.0:
        best_score = best_streaming_score(session_dir)

    iteration_scores: list[float] = []
    for it in history.get("iterations", []):
        j = it.get("judgment", {})
        score = j.get("intent_score")
        if score is not None:
            iteration_scores.append(float(score))
        else:
            iteration_scores.append(0.0)

    return {
        "status": history.get("status", "unknown"),
        "best_score": best_score,
        "iterations_completed": len(history.get("iterations", [])),
        "is_stale": stale,
        "best_iteration": history.get("best_iteration", 0),
        "iteration_scores": iteration_scores,
    }


# ---------------------------------------------------------------------------
# Stage building
# ---------------------------------------------------------------------------


def build_default_stages(
    test_cases: list[dict],
    num_stages: int = 25,
    gate_threshold: float = 0.7,
    max_parallel: int = 10,
) -> list[dict]:
    """Assign test cases to *num_stages* stages with uniform mix of difficulty AND environment.

    Within each difficulty group, cases are interleaved by environment (round-robin)
    so that no single env dominates.  Then the three difficulty streams are interleaved.
    Finally, consecutive chunks are assigned to stages.  Fully deterministic — depends
    only on CSV ordering.

    **Env diversity tip**: the env order per difficulty is set by first appearance
    in the CSV.  Staggering env first-appearances across difficulties (e.g. Ant
    appears first among easy cases, Walker2d first among medium, HalfCheetah
    first among hard) maximises the number of envs covered in stage 1.
    """
    from collections import OrderedDict

    by_diff: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    for i, tc in enumerate(test_cases):
        d = tc.get("difficulty", "hard").lower()
        by_diff.setdefault(d, []).append(i)

    def _env_interleave(indices: list[int]) -> list[int]:
        by_env: OrderedDict[str, list[int]] = OrderedDict()
        for idx in indices:
            env = test_cases[idx].get("env_id", "")
            by_env.setdefault(env, []).append(idx)
        result: list[int] = []
        env_lists = list(by_env.values())
        max_len = max((len(v) for v in env_lists), default=0)
        for pos in range(max_len):
            for el in env_lists:
                if pos < len(el):
                    result.append(el[pos])
        return result

    for d in by_diff:
        by_diff[d] = _env_interleave(by_diff[d])

    interleaved: list[int] = []
    max_len = max((len(v) for v in by_diff.values()), default=0)
    for pos in range(max_len):
        for d in ("easy", "medium", "hard"):
            if pos < len(by_diff[d]):
                interleaved.append(by_diff[d][pos])

    total = len(interleaved)
    chunk_size = max(1, (total + num_stages - 1) // num_stages)
    buckets: list[list[int]] = []
    for s in range(num_stages):
        buckets.append(interleaved[s * chunk_size : (s + 1) * chunk_size])

    stages: list[dict] = []
    for stage_num, bucket in enumerate(buckets, start=1):
        if not bucket:
            continue
        is_last = stage_num == num_stages
        stages.append(
            {
                "stage": stage_num,
                "name": f"Batch {stage_num}",
                "gate_threshold": 0.0 if is_last else gate_threshold,
                "max_parallel": max_parallel,
                "case_indices": bucket,
                "status": "pending",
                "gate_result": None,
            }
        )

    return stages


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gate(
    entries: list[dict],
    case_indices: list[int],
    pass_threshold: float,  # noqa: ARG001
    gate_threshold: float,
) -> dict:
    """Evaluate the gate for a completed stage.

    Returns a StageGateResult-compatible dict.
    """
    scores: list[float] = []
    passed_count = 0
    for idx in case_indices:
        sid = entries[idx].get("session_id", "")
        if not sid:
            continue
        info = lightweight_session_info(sid)
        if info is None:
            continue
        scores.append(info["best_score"])
        if info["status"] == "passed":
            passed_count += 1

    total = len(case_indices)
    completed = len(scores)
    avg = sum(scores) / completed if completed > 0 else 0.0
    rate = passed_count / completed if completed > 0 else 0.0
    return {
        "passed": avg >= gate_threshold,
        "avg_score": round(avg, 4),
        "success_rate": round(rate, 4),
        "completed": completed,
        "total": total,
        "threshold": gate_threshold,
    }
