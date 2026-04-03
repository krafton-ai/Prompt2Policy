"""Behavior judge — scores alignment between policy behavior and intent.

Scoring pipeline (per rollout):
  1. Reward term analysis — per-term breakdown (mean, std, trend, share).
  2. Judge evaluation — code-based judge, VLM video critique, or both.
  3. LLM synthesis — merges judge outputs into final pass/fail decision.
     Falls back to direct score passthrough when no LLM client is available.

Streaming mode:
  StreamingJudge watches for eval videos during training and judges them
  as they appear, overlapping VLM inference with PPO training.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from p2p.analysis.trajectory_metrics import (
    analyze_trajectory,
    load_trajectory,
    resolve_trajectory_path,
)
from p2p.config import DEFAULT_JUDGMENT_SELECT
from p2p.contracts import (
    CheckpointAggregateJudgment,
    FailureTagEntry,
    JudgmentResult,
    RolloutJudgment,
    StageJudgment,
    SynthesisToolCall,
)
from p2p.inference.llm_client import create_message, extract_response_text
from p2p.inference.vlm import (
    MAX_VLM_TOKENS,
    VLMError,
    call_vlm_auto,
    call_vlm_two_turn,
    create_composite,
    extract_json,
    save_vlm_preview,
)
from p2p.prompts.judge_agent import (
    AGENTIC_SYNTHESIS_SYSTEM,
    AGENTIC_SYNTHESIS_USER,
    VLM_SCORING_RUBRIC,
    build_vlm_expectations_prompt,
    build_vlm_scoring_prompt,
)
from p2p.settings import LLM_MODEL, VLLM_HOST, VLLM_PORT, VLM_BASE_URL, VLM_MODEL
from p2p.training.env_spec import engine_display_name

logger = logging.getLogger(__name__)


def _with_event_context(fn):
    """Wrap *fn* so it restores the caller's event logger context in a worker thread.

    ``ThreadPoolExecutor`` does not propagate ``contextvars``, so VLM cost
    events emitted inside worker threads are silently lost.  This wrapper
    captures the current event logger and iteration at construction time
    and restores them when the callable executes in the pool.
    """
    from p2p.event_log import (
        _current_iteration,
        get_event_logger,
        reset_current_iteration,
        reset_event_logger,
        set_current_iteration,
        set_event_logger,
    )

    ev_logger = get_event_logger()
    ev_iteration = _current_iteration.get(None)

    def _wrapper(*args, **kwargs):
        logger_token = set_event_logger(ev_logger)
        iter_token = set_current_iteration(ev_iteration)
        try:
            return fn(*args, **kwargs)
        finally:
            reset_current_iteration(iter_token)
            reset_event_logger(logger_token)

    return _wrapper


# Module-level compiled regex for eval video suffixes:
#   _ep{N}  (sequential eval)
#   _p10, _median, _p90  (parallel eval percentile selection)
_EP_SUFFIX_RE = re.compile(r"_(?:ep\d+|p\d+|median)$")


# ---------------------------------------------------------------------------
# Failure tag history (cross-iteration tracking)
# ---------------------------------------------------------------------------


def compute_tag_history(
    iterations: list[dict],
) -> list[FailureTagEntry]:
    """Compute failure tag history with cross-iteration tracking.

    Walks through all past iterations' failure_tags, tracking persistence.
    Tags not seen in the most recent iteration are marked "resolved".

    Args:
        iterations: Past iteration dicts (each with "judgment.failure_tags").

    Returns:
        List of FailureTagEntry dicts sorted by count (descending).
    """
    if not iterations:
        return []

    tag_tracker: dict[str, FailureTagEntry] = {}

    for it in iterations:
        it_num = it.get("iteration", 0)
        judgment = it.get("judgment")
        if not isinstance(judgment, dict):
            continue
        tags = judgment.get("failure_tags", [])

        seen_this_iter: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str) or tag in seen_this_iter:
                continue
            seen_this_iter.add(tag)
            if tag in tag_tracker:
                entry = tag_tracker[tag]
                entry["count"] += 1
                entry["last_seen"] = it_num
                # Reactivate resolved tags. first_seen intentionally
                # retains the original iteration — this shows the full
                # history span even across resolved/reactivated gaps.
                entry["status"] = "active"
            else:
                tag_tracker[tag] = {
                    "tag": tag,
                    "count": 1,
                    "first_seen": it_num,
                    "last_seen": it_num,
                    "status": "active",
                }

        # Mark tags not seen this iteration as resolved
        for tag, entry in tag_tracker.items():
            if tag not in seen_this_iter and entry["status"] == "active":
                entry["status"] = "resolved"

    return sorted(tag_tracker.values(), key=lambda x: -x["count"])


def format_tag_history(tag_history: list[FailureTagEntry]) -> str:
    """Format tag history for LLM prompt injection."""
    if not tag_history:
        return "No previous failure tags (first iteration)."

    active = [t for t in tag_history if t.get("status") != "resolved"]
    resolved = [t for t in tag_history if t.get("status") == "resolved"]

    lines: list[str] = []
    if active:
        lines.append("Active failure tags (persisting from previous iterations):")
        for t in active:
            lines.append(
                f'- "{t["tag"]}" [{t["count"]}x, iterations {t["first_seen"]}-{t["last_seen"]}]'
            )
    if resolved:
        lines.append("Recently resolved tags:")
        recent_resolved = sorted(resolved, key=lambda x: x["last_seen"], reverse=True)[:3]
        for t in recent_resolved:
            lines.append(
                f'- "{t["tag"]}" [RESOLVED after {t["count"]}x, '
                f"iterations {t['first_seen']}-{t['last_seen']}]"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider capability helpers
# ---------------------------------------------------------------------------


def _provider_supports_video(vlm_model: str) -> bool:
    """Check if the VLM provider supports native video input."""
    model_lower = vlm_model.lower()
    return model_lower.startswith(("vllm-", "gemini"))


def _vllm_available(host: str = VLLM_HOST, port: int = VLLM_PORT) -> bool:
    """Quick check if vLLM server is reachable."""
    from p2p.inference.vllm_server import vllm_health_check

    return vllm_health_check(host, port)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_all_checkpoints(
    prompt: str,
    iteration_dir: str | Path,
    summary: dict[str, Any],
    *,
    pass_threshold: float = 0.7,
    model: str = LLM_MODEL,
    client: Any = None,
    reward_code: str = "",
    reward_terms: dict[str, float] | None = None,
    env_name: str,
    vlm_model: str = VLM_MODEL,
    thinking_effort: str = "",
    streaming_results: dict[str, JudgmentResult] | None = None,
    judge_code: str = "",
    judgment_select: str = DEFAULT_JUDGMENT_SELECT,
    tag_history: list[FailureTagEntry] | None = None,
    engine: str = "mujoco",
) -> JudgmentResult:
    """Judge each eval checkpoint independently, return the selected one.

    Groups frames by step-label prefix, judges each checkpoint separately,
    and returns the selected judgment with per-checkpoint details.

    When *streaming_results* is provided (from StreamingJudge), those
    checkpoints are reused and only missing ones are judged.

    Args:
        prompt: User's intent description.
        iteration_dir: Root directory for the training run.
        summary: Training summary dict (from summary.json).
        pass_threshold: Minimum intent_score to pass.
        model: Claude model ID for LLM analysis/synthesis.
        client: Anthropic client instance (None → direct score passthrough).
        reward_code: The reward function source code.
        reward_terms: Reward term values from the last eval step.
        env_name: Environment display name.
        vlm_model: VLM model ID routed to the correct provider.
        thinking_effort: Thinking effort level for LLM calls (e.g. "high").
        streaming_results: Pre-computed judgments from StreamingJudge.
        judge_code: Source code for code-based judge (empty string disables).
        judgment_select: "last" (final checkpoint) or "best" (highest
            intent_score).
        tag_history: Cross-iteration failure tag history for synthesis.

    Returns:
        JudgmentResult dict with best_checkpoint and checkpoint_judgments.
    """
    iteration_dir = Path(iteration_dir)
    videos_dir = iteration_dir / "videos"

    # Discover checkpoints from eval video files: eval_983040.mp4 → step "983040"
    checkpoint_videos = _discover_checkpoint_videos(videos_dir)

    if not checkpoint_videos:
        logger.warning("No eval videos found in %s", videos_dir)
        return {
            "intent_score": 0.0,
            "passed": False,
            "diagnosis": "No eval videos found in iteration directory.",
            "failure_tags": ["no_eval_videos"],
            "evidence": [],
            "reward_term_analysis": {},
            "vlm_score": None,
            "scoring_method": "no_judge",
            "best_checkpoint": "",
            "checkpoint_judgments": {},
        }

    if len(checkpoint_videos) == 1 and not streaming_results:
        # Only one checkpoint — no need for per-checkpoint overhead
        step = next(iter(checkpoint_videos))
        result = _judge_single_checkpoint(
            prompt=prompt,
            iteration_dir=iteration_dir,
            video_path=checkpoint_videos[step],
            step_label=step,
            pass_threshold=pass_threshold,
            env_name=env_name,
            vlm_model=vlm_model,
            thinking_effort=thinking_effort,
            client=client,
            model=model,
            judge_code=judge_code,
            tag_history=tag_history,
        )
        checkpoint_copy = dict(result)
        result["best_checkpoint"] = step
        result["checkpoint_judgments"] = {step: checkpoint_copy}
        return result

    # Start from streaming results (already judged during training)
    checkpoint_judgments: dict[str, JudgmentResult] = dict(streaming_results or {})

    # Find steps that still need judging
    sorted_steps = sorted(checkpoint_videos.keys(), key=int)
    remaining = [s for s in sorted_steps if s not in checkpoint_judgments]

    if streaming_results:
        logger.info(
            "Streaming judge provided %d/%d checkpoints, judging %d remaining",
            len(streaming_results),
            len(sorted_steps),
            len(remaining),
        )

    # Judge remaining checkpoints in parallel
    if remaining:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from concurrent.futures import as_completed

        def _judge_step(step: str) -> tuple[str, JudgmentResult]:
            video = checkpoint_videos[step]
            logger.info("Judging checkpoint step=%s video=%s", step, video.name)
            result = _judge_single_checkpoint(
                prompt=prompt,
                iteration_dir=iteration_dir,
                video_path=video,
                step_label=step,
                pass_threshold=pass_threshold,
                env_name=env_name,
                vlm_model=vlm_model,
                thinking_effort=thinking_effort,
                client=client,
                model=model,
                judge_code=judge_code,
                tag_history=tag_history,
            )
            return step, result

        max_workers = min(len(remaining), 4)
        _judge_step_ctx = _with_event_context(_judge_step)
        with _TPE(max_workers=max_workers) as pool:
            futures = {pool.submit(_judge_step_ctx, step): step for step in remaining}
            for future in as_completed(futures):
                step, result = future.result()
                checkpoint_judgments[step] = result

    # Pick the selected checkpoint
    if judgment_select == "last":
        selected_step = max(checkpoint_judgments, key=lambda s: int(s))
    else:
        selected_step = max(
            checkpoint_judgments,
            key=lambda s: checkpoint_judgments[s].get("intent_score", 0),
        )
    best = checkpoint_judgments[selected_step]

    # Build the final result: selected checkpoint's fields + per-checkpoint details
    result = {**best}
    result["best_checkpoint"] = selected_step
    result["checkpoint_judgments"] = checkpoint_judgments

    logger.info(
        "Selected checkpoint (mode=%s): step=%s score=%.3f (of %d checkpoints)",
        judgment_select,
        selected_step,
        best.get("intent_score", 0),
        len(checkpoint_judgments),
    )
    return result


def _discover_checkpoint_videos(videos_dir: Path) -> dict[str, Path]:
    """Discover eval videos by step label.

    Videos are named ``eval_{step_label}.mp4`` (legacy) or
    ``eval_{step_label}_ep{N}.mp4`` (multi-rollout).
    Returns {step_label: first_video_path} for backward compat with callers
    that expect a single representative video per checkpoint.
    """
    if not videos_dir.exists():
        return {}

    result: dict[str, Path] = {}
    for f in sorted(videos_dir.glob("eval_*.mp4")):
        if f.stem.endswith(("_vlm", "_flow", "_motion")):
            continue  # skip VLM / flow / motion trail preview videos
        name = f.stem.replace("eval_", "")
        step = _EP_SUFFIX_RE.sub("", name)
        if step not in result:
            result[step] = f  # first (or legacy) video as representative
    return result


def _discover_rollout_videos(videos_dir: Path, step_label: str) -> list[Path]:
    """Find all per-episode eval videos for a given checkpoint step.

    Returns sorted list of rollout videos.  Checks three naming conventions:
      1. ``eval_{step}_{p10|median|p90}.mp4``  (parallel eval)
      2. ``eval_{step}_ep{N}.mp4``             (sequential eval)
      3. ``eval_{step}.mp4``                   (legacy single video)
    """
    # Parallel eval: percentile-selected rollouts
    percentile = sorted(videos_dir.glob(f"eval_{step_label}_p*.mp4")) + sorted(
        videos_dir.glob(f"eval_{step_label}_median.mp4")
    )
    percentile = [p for p in percentile if not p.stem.endswith(("_vlm", "_flow", "_motion"))]
    if percentile:
        return sorted(set(percentile), key=lambda p: p.name)
    # Sequential eval: per-episode videos
    per_ep = sorted(videos_dir.glob(f"eval_{step_label}_ep*.mp4"))
    per_ep = [p for p in per_ep if not p.stem.endswith(("_vlm", "_flow", "_motion"))]
    if per_ep:
        return per_ep
    # Legacy fallback: single video per checkpoint
    legacy = videos_dir / f"eval_{step_label}.mp4"
    return [legacy] if legacy.exists() else []


def _vlm_judge_max_workers(vlm_model: str) -> int:
    """Determine max concurrent VLM calls based on provider.

    Remote APIs (Gemini, remote vLLM) → high concurrency.
    Local vLLM → conservative to avoid GPU overload.
    """
    model_lower = vlm_model.lower()
    if model_lower.startswith("gemini"):
        return 20
    if model_lower.startswith("vllm-"):
        from p2p.inference.vlm import _is_remote_host

        if _is_remote_host(VLLM_HOST):
            return 20
        return 2  # local GPU
    return 4  # Ollama / other local


def _judge_single_rollout(
    *,
    prompt: str,
    iteration_dir: Path,
    video_path: Path,
    step_label: str,
    episode_idx: int,
    pass_threshold: float,
    env_name: str,
    vlm_model: str = VLM_MODEL,
    thinking_effort: str = "",
    client: Any = None,
    model: str = LLM_MODEL,
    judge_code: str = "",
    eval_return: float | None = None,
    tag_history: list[FailureTagEntry] | None = None,
) -> RolloutJudgment:
    """Run the full 3-stage judge pipeline on a single rollout video.

    Returns a RolloutJudgment dict.
    """
    # Find per-episode trajectory file matching the video suffix (.npz or .jsonl)
    video_suffix = video_path.stem.replace(f"eval_{step_label}", "").lstrip("_")
    candidates = []
    if video_suffix:
        candidates.append(f"trajectory_{step_label}_{video_suffix}")
    candidates += [
        f"trajectory_{step_label}_ep{episode_idx}",
        f"trajectory_{step_label}",
        "trajectory",
    ]
    traj_path: Path = iteration_dir / "trajectory.jsonl"
    for stem in candidates:
        found = resolve_trajectory_path(iteration_dir, stem)
        if found is not None:
            traj_path = found
            break

    traj_analysis = _run_trajectory_analysis_from_path(traj_path)

    # Stage 2: Run judge(s) — code-based, VLM, or both
    has_code_judge = bool(judge_code)
    has_vlm = bool(vlm_model)

    code_result = None
    vlm_result = None

    if has_code_judge:
        code_result = _run_code_judge(traj_path, judge_code, summary={})

    if has_vlm:
        vlm_result = _run_vlm_judgment_on_video(
            video_path,
            prompt,
            env_name=env_name,
            vlm_model=vlm_model,
        )

    # Stage 3: Synthesis
    if has_code_judge and has_vlm and code_result and vlm_result:
        judgment = _synthesize_dual_judges_agentic(
            traj_analysis,
            code_result,
            vlm_result,
            prompt,
            pass_threshold,
            client,
            model,
            thinking_effort=thinking_effort,
            video_path=video_path,
            traj_path=traj_path,
            vlm_model=vlm_model,
            tag_history=tag_history,
            env_name=env_name,
        )
    elif has_code_judge and code_result:
        judgment = _synthesize(traj_analysis, code_result, pass_threshold, is_code_judge=True)
    elif has_vlm and vlm_result:
        judgment = _synthesize(traj_analysis, vlm_result, pass_threshold, is_code_judge=False)
    else:
        judgment = _synthesize(
            traj_analysis, _empty_vlm_result("No judge configured"), pass_threshold
        )

    rj: RolloutJudgment = {
        "episode_idx": episode_idx,
        "intent_score": judgment.get("intent_score", 0.0),
        "diagnosis": judgment.get("diagnosis", ""),
        "failure_tags": judgment.get("failure_tags", []),
    }
    # Set rollout_label from video suffix so the dashboard can match
    # rollout judgments to video URLs (e.g. "p10", "median", "p90", "ep0").
    video_suffix = video_path.stem.replace(f"eval_{step_label}", "").lstrip("_")
    if video_suffix:
        rj["rollout_label"] = video_suffix
    if eval_return is not None:
        rj["eval_return"] = eval_return
    if "scoring_method" in judgment:
        rj["scoring_method"] = judgment["scoring_method"]
    # Preserve per-judge raw outputs so frontend can display them
    if judgment.get("code_diagnosis"):
        rj["code_diagnosis"] = judgment["code_diagnosis"]
    if judgment.get("code_score") is not None:
        rj["code_score"] = judgment["code_score"]
    if judgment.get("vlm_diagnosis"):
        rj["vlm_diagnosis"] = judgment["vlm_diagnosis"]
    if judgment.get("vlm_score") is not None:
        rj["vlm_score"] = judgment["vlm_score"]
    if judgment.get("vlm_criteria"):
        rj["vlm_criteria"] = judgment["vlm_criteria"]
    # criteria_scores comes from VLM result, not synthesis
    if vlm_result and vlm_result.get("criteria_scores"):
        rj["criteria_scores"] = vlm_result["criteria_scores"]
    elif judgment.get("criteria_scores"):
        rj["criteria_scores"] = judgment["criteria_scores"]
    # Propagate agentic synthesis tool call traces
    if judgment.get("synthesis_tool_calls"):
        rj["synthesis_tool_calls"] = judgment["synthesis_tool_calls"]
    # Generate VLM preview video (center-of-interval sampled at VLM fps)
    if has_vlm and vlm_result:
        preview_path = save_vlm_preview(video_path)
        if preview_path is not None:
            rj["vlm_preview_filename"] = preview_path.name
    return rj


def _aggregate_rollout_judgments(
    rollout_judgments: list[RolloutJudgment],
    step_label: str,
    pass_threshold: float,
) -> CheckpointAggregateJudgment:
    """Aggregate per-rollout judgments into checkpoint-level statistics."""
    import statistics

    scores = [rj["intent_score"] for rj in rollout_judgments]
    mean_score = statistics.mean(scores) if scores else 0.0
    score_std = statistics.stdev(scores) if len(scores) >= 2 else 0.0
    success_rate = sum(1 for s in scores if s >= pass_threshold) / max(len(scores), 1)

    # Use failure tags from the median rollout (closest to mean score).
    # Free-form tags are naturally diverse across rollouts — frequency-based
    # aggregation drops most of them. The median rollout's tags are the most
    # representative of the checkpoint's behavior.
    median_rj = min(rollout_judgments, key=lambda rj: abs(rj["intent_score"] - mean_score))
    common_tags = list(dict.fromkeys(median_rj.get("failure_tags", [])))

    # Build aggregate diagnosis
    diag_parts = [
        f"{len(rollout_judgments)} rollouts evaluated",
        f"mean_score={mean_score:.3f}",
        f"success_rate={success_rate:.0%}",
        f"std={score_std:.3f}",
    ]
    if common_tags:
        diag_parts.append(f"common_issues=[{', '.join(common_tags)}]")

    return {
        "step": step_label,
        "rollout_judgments": list(rollout_judgments),
        "mean_intent_score": round(mean_score, 3),
        "success_rate": round(success_rate, 3),
        "score_std": round(score_std, 3),
        "common_failure_tags": common_tags,
        "aggregate_diagnosis": "; ".join(diag_parts),
    }


def _judge_single_checkpoint(
    *,
    prompt: str,
    iteration_dir: Path,
    video_path: Path,
    step_label: str,
    pass_threshold: float,
    env_name: str,
    vlm_model: str = VLM_MODEL,
    thinking_effort: str = "",
    client: Any = None,
    model: str = LLM_MODEL,
    judge_code: str = "",
    tag_history: list[FailureTagEntry] | None = None,
) -> JudgmentResult:
    """Run the full judge pipeline on all rollouts at a single checkpoint.

    If multiple per-episode videos exist (eval_{step}_ep{N}.mp4), judges
    each rollout in parallel and aggregates results. Falls back to single-
    video judging for legacy runs.
    """
    videos_dir = video_path.parent
    rollout_videos = _discover_rollout_videos(videos_dir, step_label)

    if len(rollout_videos) <= 1:
        # Single video — derive trajectory filename from video suffix
        v_suffix = ""
        if rollout_videos:
            v_suffix = rollout_videos[0].stem.replace(f"eval_{step_label}", "").lstrip("_")
        candidates = []
        if v_suffix:
            candidates.append(f"trajectory_{step_label}_{v_suffix}")
        candidates += [
            f"trajectory_{step_label}_ep0",
            f"trajectory_{step_label}",
            "trajectory",
        ]
        traj_path: Path = iteration_dir / "trajectory.jsonl"
        for stem in candidates:
            found = resolve_trajectory_path(iteration_dir, stem)
            if found is not None:
                traj_path = found
                break
        traj_analysis = _run_trajectory_analysis_from_path(traj_path)

        # Stage 2: Run judge(s) — code-based, VLM, or both
        has_code_judge = bool(judge_code)
        has_vlm = bool(vlm_model)

        code_result = None
        vlm_result = None

        if has_code_judge:
            code_result = _run_code_judge(traj_path, judge_code, summary={})

        if has_vlm:
            actual_video = rollout_videos[0] if rollout_videos else video_path
            vlm_result = _run_vlm_judgment_on_video(
                actual_video,
                prompt,
                env_name=env_name,
                vlm_model=vlm_model,
            )

        # Generate VLM preview for the single-video checkpoint path
        _single_preview_filename: str | None = None
        if has_vlm and vlm_result:
            actual_video = rollout_videos[0] if rollout_videos else video_path
            preview = save_vlm_preview(actual_video)
            if preview is not None:
                _single_preview_filename = preview.name

        def _attach_preview(result: dict) -> dict:
            if _single_preview_filename:
                result["vlm_preview_filename"] = _single_preview_filename
            return result

        # Stage 3: Synthesis
        if has_code_judge and has_vlm and code_result and vlm_result:
            actual_video_for_synth = rollout_videos[0] if rollout_videos else video_path
            return _attach_preview(
                _synthesize_dual_judges_agentic(
                    traj_analysis,
                    code_result,
                    vlm_result,
                    prompt,
                    pass_threshold,
                    client,
                    model,
                    thinking_effort=thinking_effort,
                    video_path=actual_video_for_synth,
                    traj_path=traj_path,
                    vlm_model=vlm_model,
                    tag_history=tag_history,
                    env_name=env_name,
                )
            )
        elif has_code_judge and code_result:
            return _synthesize(traj_analysis, code_result, pass_threshold, is_code_judge=True)
        elif has_vlm and vlm_result:
            return _attach_preview(
                _synthesize(traj_analysis, vlm_result, pass_threshold, is_code_judge=False)
            )
        return _synthesize(traj_analysis, _empty_vlm_result("No judge configured"), pass_threshold)

    # Multi-rollout: judge each episode in parallel
    max_workers = _vlm_judge_max_workers(vlm_model)
    logger.info(
        "Judging %d rollouts for step=%s (max_workers=%d)",
        len(rollout_videos),
        step_label,
        max_workers,
    )

    # Read per-episode returns from scalars.jsonl if available
    ep_returns = _read_per_episode_returns(iteration_dir, step_label)

    def _judge_ep(idx_video: tuple[int, Path]) -> RolloutJudgment:
        idx, vid = idx_video
        return _judge_single_rollout(
            prompt=prompt,
            iteration_dir=iteration_dir,
            video_path=vid,
            step_label=step_label,
            episode_idx=idx,
            pass_threshold=pass_threshold,
            env_name=env_name,
            vlm_model=vlm_model,
            thinking_effort=thinking_effort,
            client=client,
            model=model,
            judge_code=judge_code,
            eval_return=ep_returns[idx] if idx < len(ep_returns) else None,
            tag_history=tag_history,
        )

    indexed_videos = list(enumerate(rollout_videos))
    rollout_judgments: list[RolloutJudgment] = []
    _judge_ep_ctx = _with_event_context(_judge_ep)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        from concurrent.futures import as_completed

        futures = {pool.submit(_judge_ep_ctx, iv): iv[0] for iv in indexed_videos}
        for future in as_completed(futures):
            ep_idx = futures[future]
            try:
                rj = future.result()
                rollout_judgments.append(rj)
            except Exception:
                logger.exception("Failed to judge rollout ep=%d step=%s", ep_idx, step_label)

    rollout_judgments.sort(key=lambda rj: rj["episode_idx"])

    if not rollout_judgments:
        logger.error(
            "All %d rollout judgments failed for step=%s",
            len(indexed_videos),
            step_label,
        )
        return _synthesize(
            traj_analysis,
            _empty_vlm_result("All rollout judgments failed"),
            pass_threshold,
        )

    # Aggregate
    aggregate = _aggregate_rollout_judgments(rollout_judgments, step_label, pass_threshold)

    # Compute per-judge score means from individual rollouts
    import statistics

    code_scores = [
        rj["code_score"] for rj in rollout_judgments if rj.get("code_score") is not None
    ]
    vlm_scores = [rj["vlm_score"] for rj in rollout_judgments if rj.get("vlm_score") is not None]

    result: JudgmentResult = {
        "intent_score": aggregate["mean_intent_score"],
        "passed": aggregate["mean_intent_score"] >= pass_threshold,
        "diagnosis": aggregate["aggregate_diagnosis"],
        "failure_tags": aggregate["common_failure_tags"],
        "evidence": [],
        "reward_term_analysis": {},
        "vlm_score": None,
        "scoring_method": "multi_rollout",
        "rollout_judgments": rollout_judgments,
        "checkpoint_aggregate": aggregate,
    }
    if code_scores:
        result["code_score"] = round(statistics.mean(code_scores), 3)
    if vlm_scores:
        result["vlm_score"] = round(statistics.mean(vlm_scores), 3)
    # Propagate vlm_criteria from first rollout (all rollouts share the same Turn 1)
    for rj in rollout_judgments:
        if rj.get("vlm_criteria"):
            result["vlm_criteria"] = rj["vlm_criteria"]
            break
    return result


def _read_per_episode_returns(iteration_dir: Path, step_label: str) -> list[float]:
    """Read per-episode returns from scalars.jsonl for a given eval step."""
    scalars_path = iteration_dir / "metrics" / "scalars.jsonl"
    if not scalars_path.exists():
        return []
    try:
        for line in scalars_path.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            if (
                entry.get("type") == "eval"
                and str(entry.get("global_step")) == step_label
                and "per_episode_returns" in entry
            ):
                return entry["per_episode_returns"]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to read per-episode returns for step=%s: %s", step_label, exc)
    return []


# ---------------------------------------------------------------------------
# Code-based judge (custom analysis code against trajectory data)
# ---------------------------------------------------------------------------


def _run_code_judge(traj_path: Path, judge_code: str, summary: dict[str, Any]) -> StageJudgment:
    """Execute code-based judge against trajectory data.

    ``execute_judge_code`` handles all exceptions internally and always
    returns a well-formed dict, so no additional try/except is needed here.
    """
    from p2p.agents.judge_author import execute_judge_code

    trajectory = load_trajectory(traj_path) if traj_path.exists() else []
    data = execute_judge_code(judge_code, trajectory, summary)
    result = StageJudgment(
        intent_score=data.get("intent_score"),
        diagnosis=data.get("diagnosis", ""),
        failure_tags=data.get("failure_tags", []),
    )
    if "evidence" in data:
        result["evidence"] = data["evidence"]
    return result


# ---------------------------------------------------------------------------
# Stage 1: Reward term analysis
# ---------------------------------------------------------------------------


def _run_trajectory_analysis_from_path(traj_path: Path) -> dict[str, Any]:
    """Load a specific trajectory file and compute reward term statistics."""
    if not traj_path.exists():
        return analyze_trajectory([])

    trajectory = load_trajectory(traj_path)
    return analyze_trajectory(trajectory)


def _synthesize_dual_judges(
    traj_analysis: dict[str, Any],
    code_result: StageJudgment,
    vlm_result: StageJudgment,
    intent: str,
    pass_threshold: float,
    client: Any,
    model: str,
    thinking_effort: str = "",
    tag_history: list[FailureTagEntry] | None = None,
) -> JudgmentResult:
    """Synthesize code-based judge + VLM judge outputs via LLM.

    Used when both judges are active. Falls back to code judge result on failure.
    """
    from p2p.prompts.judge_agent import DUAL_JUDGE_SYNTHESIS_PROMPT

    prompt_text = DUAL_JUDGE_SYNTHESIS_PROMPT.format(
        intent=intent,
        code_score=code_result.get("intent_score", "N/A"),
        code_diagnosis=code_result.get("diagnosis", "N/A"),
        code_tags=code_result.get("failure_tags", []),
        vlm_score=vlm_result.get("intent_score", "N/A"),
        vlm_diagnosis=vlm_result.get("diagnosis", "N/A"),
        vlm_tags=vlm_result.get("failure_tags", []),
        tag_history=format_tag_history(tag_history or []),
    )

    try:
        response = create_message(
            client,
            model=model,
            thinking_effort=thinking_effort,
            messages=[{"role": "user", "content": prompt_text}],
        )
        raw = extract_response_text(response)
        data = extract_json(raw)

        score = float(data.get("intent_score", 0))
        score = max(0.0, min(1.0, score))
        intent_score = round(score, 3)

        return {
            "intent_score": intent_score,
            "passed": intent_score >= pass_threshold,
            "diagnosis": data.get("diagnosis", ""),
            "failure_tags": data.get("failure_tags", []),
            "evidence": vlm_result.get("evidence", []),
            "reward_term_analysis": traj_analysis,
            "vlm_score": vlm_result.get("intent_score"),
            "vlm_diagnosis": vlm_result.get("diagnosis", ""),
            "vlm_criteria": vlm_result.get("vlm_criteria", ""),
            "code_score": code_result.get("intent_score"),
            "code_diagnosis": code_result.get("diagnosis", ""),
            "scoring_method": "dual_judge+llm_synthesis",
        }
    except Exception:
        logger.exception("Dual judge synthesis failed, falling back to code judge")
        return _synthesize(traj_analysis, code_result, pass_threshold, is_code_judge=True)


# ---------------------------------------------------------------------------
# Stage 3 (Agentic): Synthesis with tool use
# ---------------------------------------------------------------------------


def _build_reask_vlm_tool(
    video_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Build the reask_vlm tool schema with current VLM_FPS default."""
    from p2p.inference.vlm import REASK_MAX_FPS, REASK_MAX_FRAMES, VLM_FPS

    duration_hint = ""
    if video_duration_sec is not None:
        duration_hint = f"The evaluation video is {video_duration_sec:.1f}s long. "

    return {
        "name": "reask_vlm",
        "description": (
            "Ask a targeted follow-up question about the evaluation video. "
            "Use this when the VLM judge's diagnosis is ambiguous or conflicts "
            "with the code judge's findings. "
            f"{duration_hint}"
            "You can specify a time range and FPS to focus on a specific segment "
            "at higher temporal resolution. "
            f"Default: full video at {VLM_FPS} FPS. "
            f"Max FPS: {REASK_MAX_FPS}, max frames: {REASK_MAX_FRAMES}. "
            "Higher FPS on short segments is cheap (1s @ 15 FPS = 15 frames). "
            "Avoid high FPS on long segments (10s @ 15 FPS = 150 frames, will be clamped)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A specific question about the video to clarify the disagreement."
                    ),
                },
                "start_time": {
                    "type": "number",
                    "description": (
                        "Start time in seconds (default: 0.0 = video start). "
                        "Use to focus on a specific segment."
                    ),
                },
                "end_time": {
                    "type": "number",
                    "description": (
                        "End time in seconds (default: video end). "
                        "Use with start_time to isolate a time window."
                    ),
                },
                "fps": {
                    "type": "integer",
                    "description": (
                        f"Frames per second for video sampling (default: {VLM_FPS}). "
                        "Increase for finer temporal detail on short segments "
                        "(e.g., 10-15 FPS for a 1-2s window)."
                    ),
                },
            },
            "required": ["question"],
        },
    }


_TRAJECTORY_KEYS_HINTS: dict[str, str] = {
    "mujoco": (
        "'qpos', 'qvel', 'obs', 'action', 'reward', 'xpos', 'xquat', 'cvel', 'cfrc_ext', etc."
    ),
    "isaaclab": (
        "'joint_pos' (list[float]), 'joint_vel' (list[float]), "
        "'body_pos_w' (list[list[float]], shape num_bodies×3), "
        "'body_quat_w' (list[list[float]], shape num_bodies×4, wxyz), "
        "'root_pos_w' (list[float], [x,y,z]), "
        "'root_quat_w' (list[float], wxyz), "
        "'obs', 'action', 'reward', etc."
    ),
}
_TRAJECTORY_KEYS_DEFAULT = "'obs', 'action', 'reward', etc."


def _build_synthesis_tools(
    engine: str = "mujoco",
    video_duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Build synthesis tools with engine-appropriate hints and current VLM_FPS."""
    keys_hint = _TRAJECTORY_KEYS_HINTS.get(engine, _TRAJECTORY_KEYS_DEFAULT)

    return [
        _build_reask_vlm_tool(video_duration_sec=video_duration_sec),
        {
            "name": "run_trajectory_check",
            "description": (
                "Run a short Python snippet against the trajectory data to verify "
                "a specific claim. The code must define a function "
                "`check_fn(trajectory, summary) -> dict` that returns a dict with "
                f"a 'result' key. `trajectory` is a list of dicts with keys like "
                f"{keys_hint} "
                "numpy is available as `np`."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": (
                            "Python code defining `check_fn(trajectory, summary) -> dict`. "
                            "Must return a dict with a 'result' key."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "What this check verifies.",
                    },
                },
                "required": ["python_code", "description"],
            },
        },
    ]


def _execute_trajectory_check(
    code: str,
    trajectory: list[dict],
    summary: dict,
) -> dict[str, Any]:
    """Execute a trajectory check function in a sandboxed namespace.

    Uses ``run_sandboxed_fn`` from ``judge_author`` to execute
    ``check_fn(trajectory, summary)`` in a restricted environment.
    Returns the raw dict result (no score clamping).
    """
    from p2p.agents.judge_author import run_sandboxed_fn

    return run_sandboxed_fn(code, "check_fn", (trajectory, summary))


def _synthesize_dual_judges_agentic(
    traj_analysis: dict[str, Any],
    code_result: StageJudgment,
    vlm_result: StageJudgment,
    intent: str,
    pass_threshold: float,
    client: Any,
    model: str,
    env_name: str,
    thinking_effort: str = "",
    video_path: Path | None = None,
    traj_path: Path | None = None,
    vlm_model: str = "",
    tag_history: list[FailureTagEntry] | None = None,
) -> JudgmentResult:
    """Synthesize code + VLM judges via agentic tool-use loop.

    Uses ``run_tool_loop()`` with ``reask_vlm`` and ``run_trajectory_check``
    tools.  Falls back to the single-shot ``_synthesize_dual_judges()`` on
    any error.
    """
    from p2p.inference.agent_tools import emit_conversation, run_tool_loop
    from p2p.training.env_spec import get_spec_by_name
    from p2p.training.simulator import get_simulator

    # Build coordinate convention text for trajectory check code generation.
    env_conventions_section = ""
    spec = get_spec_by_name(env_name) if env_name else None
    engine_label = engine_display_name(spec.engine if spec else "mujoco")
    if spec:
        backend = get_simulator(spec.engine)
        parts: list[str] = []
        joint_sem = backend.extract_joint_semantics(spec.env_id)
        if joint_sem:
            parts.append(joint_sem)
        body_info = backend.extract_body_info(spec.env_id)
        if body_info:
            parts.append(body_info)
        if parts:
            env_conventions_section = (
                f"\n## {engine_label} Coordinate Conventions\n"
                "Use these conventions when writing trajectory check code:\n"
                + "\n".join(parts)
                + "\n"
            )

    # Duration is surfaced in the prompt and reask_vlm tool schema so the
    # LLM can request meaningful time-windowed re-evaluations.
    from p2p.inference.vlm import VLM_FPS, get_video_duration

    video_duration_sec = get_video_duration(video_path) if video_path else None
    video_duration_info = f" over {video_duration_sec:.1f}s video" if video_duration_sec else ""

    user_text = AGENTIC_SYNTHESIS_USER.format(
        intent=intent,
        code_score=code_result.get("intent_score", "N/A"),
        code_diagnosis=code_result.get("diagnosis", "N/A"),
        code_tags=code_result.get("failure_tags", []),
        vlm_score=vlm_result.get("intent_score", "N/A"),
        vlm_diagnosis=vlm_result.get("diagnosis", "N/A"),
        vlm_tags=vlm_result.get("failure_tags", []),
        vlm_rubric=VLM_SCORING_RUBRIC,
        tag_history=format_tag_history(tag_history or []),
        env_conventions_section=env_conventions_section,
        vlm_fps=VLM_FPS,
        video_duration_info=video_duration_info,
    )

    # Track tool call traces as a side effect of handlers
    tool_call_traces: list[SynthesisToolCall] = []

    # Lazy trajectory cache — loaded on first run_trajectory_check call only,
    # avoiding redundant I/O when judges agree and no tools are used.
    _cached_trajectory: list[dict] | None = None

    def _get_trajectory() -> list[dict]:
        nonlocal _cached_trajectory
        if _cached_trajectory is None:
            _cached_trajectory = (
                load_trajectory(traj_path) if traj_path and traj_path.exists() else []
            )
        return _cached_trajectory

    # --- Tool handlers (closures capturing video_path, traj_path, vlm_model) ---

    def _handle_reask_vlm(inp: dict) -> dict[str, Any]:
        question = inp.get("question", "")
        start_time = inp.get("start_time")
        end_time = inp.get("end_time")
        fps = inp.get("fps")

        if not video_path or not video_path.exists():
            result_text = "No video available for re-query."
        elif not vlm_model:
            result_text = "No VLM model configured for re-query."
        else:
            segment: Path | None = None
            try:
                from p2p.inference.vlm import extract_video_segment

                segment = extract_video_segment(
                    video_path,
                    start_time=start_time,
                    end_time=end_time,
                    target_fps=int(fps) if fps is not None else None,
                )
                effective_video = segment if segment is not None else video_path
                result_text = call_vlm_auto(
                    question, [], vlm_model=vlm_model, video_path=effective_video
                )
            except Exception as exc:
                result_text = f"VLM re-query failed: {exc}"
            finally:
                if segment is not None and segment != video_path:
                    segment.unlink(missing_ok=True)

        tool_input: dict[str, Any] = {"question": question}
        if start_time is not None:
            tool_input["start_time"] = start_time
        if end_time is not None:
            tool_input["end_time"] = end_time
        if fps is not None:
            tool_input["fps"] = fps
        tool_call_traces.append(
            SynthesisToolCall(
                tool_name="reask_vlm",
                input=tool_input,
                output=result_text[:2000],
            )
        )
        return {"vlm_response": result_text}

    def _handle_run_trajectory_check(inp: dict) -> dict[str, Any]:
        python_code = inp.get("python_code", "")
        description = inp.get("description", "")
        try:
            result = _execute_trajectory_check(python_code, _get_trajectory(), summary={})
        except Exception as exc:
            result = {"error": str(exc)}
        result_text = json.dumps(result, default=str)[:2000]
        tool_call_traces.append(
            SynthesisToolCall(
                tool_name="run_trajectory_check",
                input={"python_code": python_code, "description": description},
                output=result_text,
            )
        )
        return result

    # Wrap handlers to append remaining-budget note to each tool result.
    # Sequential dispatch in run_tool_loop — no lock needed.
    max_tool_calls = 3
    _calls_used = 0

    def _counting_wrapper(handler):
        def _wrapper(inp: dict) -> dict[str, Any]:
            nonlocal _calls_used
            result = handler(inp)
            _calls_used += 1
            remaining = max_tool_calls - _calls_used
            budget_msg = (
                f"{remaining} tool call{'s' if remaining != 1 else ''} remaining"
                if remaining > 0
                else "No tool calls remaining — produce the final JSON verdict now."
            )
            return {**result, "_tool_budget": budget_msg}

        return _wrapper

    tool_dispatch = {
        "reask_vlm": _counting_wrapper(_handle_reask_vlm),
        "run_trajectory_check": _counting_wrapper(_handle_run_trajectory_check),
    }

    try:
        system_prompt = AGENTIC_SYNTHESIS_SYSTEM.format(
            max_tool_calls=max_tool_calls,
            engine_label=engine_label,
        )
        loop_result = run_tool_loop(
            client=client,
            model=model,
            system=system_prompt,
            tools=_build_synthesis_tools(
                spec.engine if spec else "mujoco",
                video_duration_sec=video_duration_sec,
            ),
            messages=[{"role": "user", "content": user_text}],
            tool_dispatch=tool_dispatch,
            max_rounds=max_tool_calls,
            force_final_turn=True,
            agent_name="synthesis",
        )

        # Emit conversation so the frontend dedup hides individual llm.call events
        if loop_result.tool_calls_used > 0:
            emit_conversation("synthesis", model, loop_result.conversation_log)

        raw = extract_response_text(loop_result.response)
        data = extract_json(raw)

        score = float(data.get("intent_score", 0))
        score = max(0.0, min(1.0, score))
        intent_score = round(score, 3)

        result: JudgmentResult = {
            "intent_score": intent_score,
            "passed": intent_score >= pass_threshold,
            "diagnosis": data.get("diagnosis", ""),
            "failure_tags": data.get("failure_tags", []),
            "evidence": vlm_result.get("evidence", []),
            "reward_term_analysis": traj_analysis,
            "vlm_score": vlm_result.get("intent_score"),
            "vlm_diagnosis": vlm_result.get("diagnosis", ""),
            "vlm_criteria": vlm_result.get("vlm_criteria", ""),
            "code_score": code_result.get("intent_score"),
            "code_diagnosis": code_result.get("diagnosis", ""),
            "scoring_method": "dual_judge+llm_synthesis",
        }
        if tool_call_traces:
            result["synthesis_tool_calls"] = tool_call_traces
        return result

    except Exception:
        logger.exception("Agentic synthesis failed, falling back to single-shot synthesis")
        return _synthesize_dual_judges(
            traj_analysis,
            code_result,
            vlm_result,
            intent,
            pass_threshold,
            client,
            model,
            thinking_effort=thinking_effort,
            tag_history=tag_history,
        )


# ---------------------------------------------------------------------------
# Stage 2: VLM judgment (Qwen3.5-27B primary, Claude fallback)
# ---------------------------------------------------------------------------


def _run_vlm_judgment_on_video(
    video_path: Path,
    intent: str,
    *,
    env_name: str,
    vlm_model: str = VLM_MODEL,
) -> StageJudgment:
    """Get VLM judgment from video using 2-turn protocol.

    Turn 1 (text-only): VLM pre-commits visual success criteria.
    Turn 2 (with media): VLM scores against its own criteria.

    Routes to the correct provider via call_vlm_two_turn().
    """
    if not video_path.exists():
        return _empty_vlm_result("Video file not found")

    # Look up engine from registry by display name
    from p2p.training.env_spec import get_spec_by_name as _get_spec

    _spec = _get_spec(env_name)
    if _spec:
        _engine = _spec.engine
    else:
        logger.warning("No spec found for '%s'; falling back to engine='mujoco'", env_name)
        _engine = "mujoco"

    use_video = _provider_supports_video(vlm_model)

    from p2p.settings import (
        VLM_CRITERIA_DIAGNOSIS,
        VLM_MOTION_TRAIL_DUAL,
        VLM_REFINED_INITIAL_FRAME,
    )

    turn1 = build_vlm_expectations_prompt(
        intent,
        env_name,
        engine=_engine,
        initial_frame=VLM_REFINED_INITIAL_FRAME,
    )

    turn2 = build_vlm_scoring_prompt(
        mode="video" if use_video else "image",
        intent=intent,
        env_name=env_name,
        engine=_engine,
        criteria_diagnosis=VLM_CRITERIA_DIAGNOSIS,
        motion_trail_dual=VLM_MOTION_TRAIL_DUAL,
    )

    # Defer composite creation — only needed for non-video providers or fallback.
    # This avoids decoding the entire video into frames when the provider uses
    # native video input (vLLM, Gemini).
    def _get_images() -> list[str]:
        composite_b64 = create_composite(video_path)
        return [composite_b64] if composite_b64 else []

    images_b64: list[str] | None = None  # lazily populated

    def _do_fallback() -> StageJudgment:
        nonlocal images_b64
        if images_b64 is None:
            images_b64 = _get_images()
        return _fallback_claude_image(turn1, turn2, images_b64)

    # Provider-specific availability checks + fallback to Claude image
    model_lower = vlm_model.lower()
    if model_lower.startswith("vllm-"):
        if not _vllm_available():
            logger.warning("vLLM unavailable, falling back to Claude image judge")
            return _do_fallback()
    elif not model_lower.startswith(("claude", "gemini")):
        if not _ollama_available():
            logger.warning("Ollama unavailable, falling back to Claude image judge")
            return _do_fallback()

    # Build images_b64 now if the provider needs it (non-video providers)
    if not use_video:
        images_b64 = _get_images()

    try:
        criteria, response = call_vlm_two_turn(
            turn1,
            turn2,
            images_b64 or [],
            vlm_model=vlm_model,
            max_tokens=MAX_VLM_TOKENS,
            video_path=video_path if use_video else None,
            refined_initial_frame=VLM_REFINED_INITIAL_FRAME,
        )
    except VLMError:
        logger.warning("VLM call failed (%s), falling back to Claude", vlm_model)
        return _do_fallback()
    result = _parse_vlm_response(response)
    result["vlm_criteria"] = criteria
    return result


def _ollama_available() -> bool:
    """Quick check if Ollama server is reachable."""
    try:
        import requests

        resp = requests.get(f"{VLM_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _parse_vlm_response(text: str) -> StageJudgment:
    """Parse VLM JSON response into judgment fields."""
    try:
        data = extract_json(text)
    except ValueError:
        return _empty_vlm_result(f"JSON parse failed: {text[:200]}")

    score = data.get("intent_score", data.get("score"))
    if score is not None:
        score = float(score)
        if score > 1.0:
            score = score / 10.0
        score = max(0.0, min(1.0, score))

    result = StageJudgment(
        intent_score=score,
        diagnosis=data.get("diagnosis", ""),
        failure_tags=data.get("failure_tags", []),
    )
    if "evidence" in data:
        result["evidence"] = data["evidence"]
    if "criteria" in data:
        result["criteria_scores"] = [
            {
                "criterion": c.get("criterion", ""),
                "assessment": c.get("assessment", ""),
                "status": c.get("status", ""),
            }
            for c in data["criteria"]
        ]
    return result


def _empty_vlm_result(reason: str) -> StageJudgment:
    return {
        "intent_score": None,
        "diagnosis": reason,
        "failure_tags": [],
    }


def _fallback_claude_image(
    turn1_prompt: str,
    turn2_prompt: str,
    images_b64: list[str],
) -> StageJudgment:
    """Fall back to Claude Vision with composite image when VLM video fails.

    Accepts pre-built prompts from the caller to avoid duplicating prompt
    construction logic.
    """
    if not images_b64:
        return _empty_vlm_result("No composite image available for Claude fallback")

    try:
        criteria, response = call_vlm_two_turn(
            turn1_prompt,
            turn2_prompt,
            images_b64,
            vlm_model=LLM_MODEL,
            max_tokens=MAX_VLM_TOKENS,
            video_path=None,
        )
    except VLMError as exc:
        return _empty_vlm_result(f"Claude image fallback also failed: {exc}")
    result = _parse_vlm_response(response)
    result["vlm_criteria"] = criteria
    return result


# ---------------------------------------------------------------------------
# Stage 3: Synthesis
# ---------------------------------------------------------------------------


def _synthesize(
    traj_analysis: dict[str, Any],
    vlm_result: StageJudgment,
    pass_threshold: float,
    *,
    is_code_judge: bool = False,
) -> JudgmentResult:
    """Combine VLM/code-judge score + reward term analysis into final judgment."""
    vlm_score = vlm_result.get("intent_score")

    # Use judge score directly
    if vlm_score is not None:
        intent_score = vlm_score
    else:
        intent_score = 0.0

    intent_score = round(intent_score, 3)

    # Use judge tags directly (dedupe, preserve order)
    all_tags = list(dict.fromkeys(vlm_result.get("failure_tags", [])))

    # Diagnosis from judge
    diagnosis = vlm_result.get("diagnosis", "")

    result: dict[str, Any] = {
        # --- Core output fields ---
        "intent_score": intent_score,
        "passed": intent_score >= pass_threshold,
        "diagnosis": diagnosis,
        "failure_tags": all_tags,
        "evidence": vlm_result.get("evidence", []),
        # --- Metadata fields ---
        "reward_term_analysis": traj_analysis,
        "vlm_score": None if is_code_judge else vlm_score,
        # When is_code_judge=True, code_result is passed as vlm_result.
        # The method is "code_judge" regardless of score value — vlm_score
        # check only distinguishes VLM-present vs no-judge for non-code paths.
        "scoring_method": (
            "code_judge" if is_code_judge else "vlm" if vlm_score is not None else "no_judge"
        ),
    }
    if is_code_judge:
        result["code_diagnosis"] = vlm_result.get("diagnosis", "")
        result["code_score"] = vlm_score
    if vlm_result.get("vlm_criteria"):
        result["vlm_criteria"] = vlm_result["vlm_criteria"]
    return result


# ---------------------------------------------------------------------------
# Streaming judge — judges checkpoints during training
# ---------------------------------------------------------------------------


class StreamingJudge:
    """Watches a run directory for new eval videos and judges them as they appear.

    Usage::

        sj = StreamingJudge(iteration_dir, prompt, env_name=..., vlm_model=...)
        sj.start()
        # ... training runs concurrently ...
        sj.stop()
        results = sj.results  # {step: judgment_dict}
    """

    def __init__(
        self,
        iteration_dir: Path,
        prompt: str,
        *,
        pass_threshold: float = 0.7,
        env_name: str,
        vlm_model: str = VLM_MODEL,
        thinking_effort: str = "",
        poll_interval: float = 10.0,
        max_workers: int = 4,
        client: Any = None,
        model: str = LLM_MODEL,
        judge_code: str = "",
        judge_code_future: Future[str] | None = None,
    ) -> None:
        self.iteration_dir = Path(iteration_dir)
        self.prompt = prompt
        self.pass_threshold = pass_threshold
        self.env_name = env_name
        self.vlm_model = vlm_model
        self.thinking_effort = thinking_effort
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self.client = client
        self.model = model
        self.judge_code = judge_code
        self._judge_code_future = judge_code_future

        self._results: dict[str, JudgmentResult] = {}
        self._lock = threading.Lock()
        self._judge_code_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._judged_steps: set[str] = set()
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._event_logger: Any = None
        self._event_iteration: int | None = None

    @property
    def results(self) -> dict[str, JudgmentResult]:
        with self._lock:
            return dict(self._results)

    def start(self) -> None:
        """Start the background watcher thread."""
        # Capture event logger context — Python 3.11 ThreadPoolExecutor
        # does not propagate contextvars to worker threads, so VLM cost
        # events would be silently dropped without this.
        from p2p.event_log import _current_iteration, get_event_logger

        self._event_logger = get_event_logger()
        self._event_iteration = _current_iteration.get(None)

        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("StreamingJudge started for %s", self.iteration_dir)

    def stop(self) -> None:
        """Stop watching and wait for in-flight judgments to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        if self._pool:
            self._pool.shutdown(wait=True)
        logger.info("StreamingJudge stopped, %d checkpoints judged", len(self._results))

    def _watch_loop(self) -> None:
        """Poll for new eval checkpoints and submit them for judging.

        Uses the eval entry in scalars.jsonl as the completion signal — it is
        written only after ALL episodes have been saved.  This avoids the race
        condition of triggering while episode videos are still being written.
        Falls back to video-based detection for legacy runs without scalars.
        """
        videos_dir = self.iteration_dir / "videos"
        scalars_path = self.iteration_dir / "metrics" / "scalars.jsonl"

        while not self._stop_event.is_set():
            # Primary: detect completed evals from scalars.jsonl
            completed_steps = self._completed_eval_steps(scalars_path)
            for step in completed_steps:
                if step in self._judged_steps:
                    continue
                # Wait for all percentile videos before judging.
                # MuJoCo Phase 2 is synchronous (all ready at once).
                # IsaacLab Phase 2 is async (worker renders one by one).
                p10 = videos_dir / f"eval_{step}_p10.mp4"
                median = videos_dir / f"eval_{step}_median.mp4"
                p90 = videos_dir / f"eval_{step}_p90.mp4"
                if p10.exists() or median.exists() or p90.exists():
                    # Percentile videos detected — wait for all 3
                    if not (p10.exists() and median.exists() and p90.exists()):
                        continue
                    ref_video = median
                else:
                    # Fallback: sequential eval (ep0) or legacy single video
                    ref_video = videos_dir / f"eval_{step}_ep0.mp4"
                    if not ref_video.exists():
                        ref_video = videos_dir / f"eval_{step}.mp4"
                    if not ref_video.exists():
                        continue
                self._judged_steps.add(step)
                self._pool.submit(self._judge_checkpoint, step, ref_video)

            # Fallback: runs without scalars — detect from video files
            if not scalars_path.exists() and videos_dir.exists():
                for mp4 in sorted(videos_dir.glob("eval_*.mp4")):
                    if mp4.stem.endswith("_vlm") or mp4.stem.endswith("_flow"):
                        continue
                    name = mp4.stem.replace("eval_", "")
                    step = _EP_SUFFIX_RE.sub("", name)
                    if step in self._judged_steps:
                        continue
                    self._judged_steps.add(step)
                    self._pool.submit(self._judge_checkpoint, step, mp4)

            self._stop_event.wait(timeout=self.poll_interval)

    @staticmethod
    def _completed_eval_steps(scalars_path: Path) -> list[str]:
        """Extract step labels of completed eval checkpoints from scalars."""
        if not scalars_path.exists():
            return []
        steps: list[str] = []
        try:
            for line in scalars_path.read_text().strip().split("\n"):
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("type") == "eval":
                    steps.append(str(entry["global_step"]))
        except Exception:
            logger.debug("Failed to parse scalars line", exc_info=True)
        return steps

    def _resolve_judge_code(self) -> None:
        """Block until the judge code future completes (if any).

        Uses double-checked locking: the outer check avoids lock acquisition
        on the fast path (after code is already resolved), and the inner check
        prevents a race when multiple worker threads hit this simultaneously.
        The outer lock-free read of ``self.judge_code`` is safe under CPython's
        GIL (str assignment is atomic); the lock guards the state transition.
        """
        if not self.judge_code and self._judge_code_future is not None:
            with self._judge_code_lock:
                if self._judge_code_future is not None and not self.judge_code:
                    logger.info("StreamingJudge: waiting for code judge generation to finish...")
                    try:
                        self.judge_code = self._judge_code_future.result()
                        n_lines = self.judge_code.count("\n") + 1
                        logger.info("StreamingJudge: code judge ready (%d lines)", n_lines)
                    except Exception:
                        logger.warning(
                            "StreamingJudge: code judge generation failed, "
                            "falling back to VLM-only judging",
                            exc_info=True,
                        )
                    finally:
                        self._judge_code_future = None

    def _judge_checkpoint(self, step: str, video_path: Path) -> None:
        """Judge a single checkpoint (runs in thread pool)."""
        # Resolve judge code from background generation if still pending.
        # This blocks only on the very first checkpoint if the LLM call
        # hasn't finished yet — an unlikely but handled edge case.
        self._resolve_judge_code()

        # Restore event logger context so VLM cost events are emitted
        from p2p.event_log import (
            reset_current_iteration,
            reset_event_logger,
            set_current_iteration,
            set_event_logger,
        )

        logger_token = None
        iter_token = None
        try:
            logger_token = set_event_logger(self._event_logger)
            iter_token = set_current_iteration(self._event_iteration)
            logger.info("StreamingJudge: judging step=%s", step)
            result = _judge_single_checkpoint(
                prompt=self.prompt,
                iteration_dir=self.iteration_dir,
                video_path=video_path,
                step_label=step,
                pass_threshold=self.pass_threshold,
                env_name=self.env_name,
                vlm_model=self.vlm_model,
                thinking_effort=self.thinking_effort,
                client=self.client,
                model=self.model,
                judge_code=self.judge_code,
            )
            with self._lock:
                self._results[step] = result
            # Persist to disk so the API can serve live scores
            self._persist_result(step, result)
            logger.info(
                "StreamingJudge: step=%s score=%.3f",
                step,
                result.get("intent_score", 0),
            )
        except Exception:
            logger.exception("StreamingJudge: failed to judge step=%s", step)
        finally:
            if iter_token is not None:
                reset_current_iteration(iter_token)
            if logger_token is not None:
                reset_event_logger(logger_token)

    def _persist_result(self, step: str, result: JudgmentResult) -> None:
        """Write streaming judgment to disk for live API access."""
        try:
            out_dir = self.iteration_dir / "streaming_judgments"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{step}.json"
            out_file.write_text(json.dumps(result, default=str))
        except Exception:
            logger.exception("StreamingJudge: failed to persist step=%s", step)
