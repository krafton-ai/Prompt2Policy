"""Orchestrator: LLM reward generation -> training -> VLM judgment -> revision."""

from __future__ import annotations

import json
import logging
import resource
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic

from p2p.agents.judge_agent import compute_tag_history
from p2p.agents.revise_agent import generate, revise_multi
from p2p.agents.reward_author import (
    MAX_CODE_RETRIES,
    fix_code,
    review_reward_code,
    validate_reward_code,
)
from p2p.config import LoopConfig
from p2p.contracts import round_metric
from p2p.event_log import (
    EventLogger,
    emit,
    reset_current_iteration,
    reset_event_logger,
    set_current_iteration,
    set_event_logger,
    span,
)
from p2p.inference.llm_client import LLMRateLimitError
from p2p.session.iteration_record import IterationData, SessionRecord
from p2p.session.iteration_runner import run_iteration
from p2p.session.lineage import (
    record_iteration as _record_lineage,
)
from p2p.session.lineage import (
    record_multi_config_iteration as _record_lineage_multi,
)
from p2p.session.session_id import generate_session_id
from p2p.settings import resolve_session_subpath
from p2p.training.env_spec import get_env_spec

if TYPE_CHECKING:
    from p2p.contracts import LoopResult, ReviseResult, RunConfigEntry
    from p2p.training.env_spec import EnvSpec
    from p2p.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _resolve_scalars_path(record: IterationData) -> Path:
    """Return the scalars.jsonl path for the best run."""
    iter_dir = Path(record.iteration_dir)
    best_run_path = iter_dir / "best_run.json"
    if best_run_path.exists():
        best_info = json.loads(best_run_path.read_text())
        return iter_dir / best_info["best_run_id"] / "metrics" / "scalars.jsonl"
    return iter_dir / "metrics" / "scalars.jsonl"


def _apply_revise_to_record(record: IterationData, result: ReviseResult) -> None:
    """Copy ReviseResult fields onto an IterationData record."""
    record.reward_reasoning = result["reward_reasoning"]
    record.hp_reasoning = result["hp_reasoning"]
    record.hp_changes = result["hp_changes"]
    record.training_dynamics = result.get("training_dynamics", "")
    record.revise_diagnosis = result["diagnosis"]
    record.lesson = result.get("lesson", "")
    record.based_on = result.get("based_on", 0)


def _collect_judge_code(
    future: Future[str],
    session_path: Path,
    pool: ThreadPoolExecutor | None,
) -> str:
    """Resolve the judge code future, persist to disk, emit event, and shut down the pool.

    Returns the generated code, or ``""`` if generation failed (code judge
    is an optional enhancement — a failure should not abort the session).
    """
    try:
        code = future.result()
    except Exception:
        logger.warning("Code judge generation failed, skipping code judge", exc_info=True)
        return ""
    finally:
        if pool:
            pool.shutdown(wait=False)
    (session_path / "judge_fn.py").write_text(code)
    emit("judge_code.ready", data={"code_lines": len(code.splitlines())})
    return code


def _try_review_reward(
    code: str,
    prompt: str,
    *,
    phase: str,
    client: anthropic.Anthropic,
    env_spec: EnvSpec,
    model: str,
    config: LoopConfig,
    thinking_effort: str,
) -> tuple[str, bool]:
    """Run reward code review, returning original code on failure.

    Returns:
        (reviewed_code, succeeded) — succeeded is False when review failed.
    """
    try:
        reviewed = review_reward_code(
            code,
            prompt,
            client=client,
            env=env_spec,
            model=model,
            side_info=config.side_info,
            env_id=config.env_id,
            thinking_effort=thinking_effort,
        )
        return reviewed, True
    except Exception:
        logger.warning(
            "Reward review failed (%s), using unreviewed code",
            phase,
            exc_info=True,
        )
        return code, False


def _build_multiconfig_context(judgment: dict) -> str:
    """Build multi-config summary for lesson generation."""
    config_judgments = judgment.get("config_judgments")
    if not config_judgments:
        return ""
    lines = ["Multi-config results:"]
    for cid, cj in sorted(config_judgments.items()):
        mean_s = cj.get("mean_intent_score", 0)
        std_s = cj.get("score_std", 0)
        tags = ", ".join(cj.get("common_failure_tags", [])) or "none"
        lines.append(f"  {cid}: {mean_s:.3f} +/- {std_s:.3f} (failures: {tags})")
    return "\n".join(lines)


def _record_lineage_safe(
    session_dir: Path,
    session_id: str,
    iteration: int,
    record: IterationData,
    judgment: dict,
    result: dict,
    client: anthropic.Anthropic,
    model: str,
    *,
    configs: list[RunConfigEntry] | None = None,
    lineage_based_on: int = 0,
    thinking_effort: str = "",
) -> None:
    """Record iteration into session lineage tree (non-fatal on error).

    Called AFTER revise fields are applied to the record, so
    reward_reasoning, revise_diagnosis, and lesson are available.
    The lesson is produced by the revise agent (full context), not a
    separate LLM call.

    When *configs* is provided and the judgment contains per-config data,
    each config is recorded as a separate lineage node.

    *lineage_based_on* overrides record.based_on for the lineage parent
    pointer when provided (> 0).  This avoids mutating the record after
    asdict() has already serialized it into loop_history.
    """
    based_on = lineage_based_on if lineage_based_on > 0 else record.based_on
    config_judgments = judgment.get("config_judgments")

    # Build rich diagnosis (shared by both paths)
    diagnosis = judgment.get("diagnosis", "")
    mc_context = _build_multiconfig_context(judgment)
    if mc_context:
        diagnosis = f"{diagnosis}\n\n{mc_context}"

    try:
        if configs and config_judgments and len(config_judgments) > 1:
            # Multi-config: record each config as a first-class node
            _record_lineage_multi(
                session_dir,
                session_id=session_id,
                iteration=iteration,
                based_on=based_on,
                lesson=record.lesson,
                config_judgments=config_judgments,
                configs=configs,
                diagnosis=diagnosis,
                client=client,
                model=model,
                thinking_effort=thinking_effort,
            )
        else:
            # Single-config (or trivial multi-config): iteration-level node
            score = judgment.get("intent_score", 0)

            summary = record.summary if isinstance(record.summary, dict) else {}
            final_return = summary.get("final_episodic_return")
            best_checkpoint = judgment.get("best_checkpoint", "")

            _record_lineage(
                session_dir,
                session_id=session_id,
                iteration=iteration,
                based_on=based_on,
                lesson=record.lesson,
                score=score,
                diagnosis=diagnosis,
                failure_tags=judgment.get("failure_tags", []),
                final_return=final_return,
                best_checkpoint=best_checkpoint,
                client=client,
                model=model,
                thinking_effort=thinking_effort,
            )
    except Exception:
        logger.warning("Failed to update lineage (non-fatal)", exc_info=True)


def run_loop(
    prompt: str,
    loop_config: LoopConfig | None = None,
    *,
    client: anthropic.Anthropic | None = None,
    session_id: str | None = None,
    trainer: Trainer | None = None,
) -> LoopResult:
    """Run the LLM reward loop: generate -> train -> judge -> revise.

    Args:
        prompt: Natural language description of desired behavior.
        loop_config: Complete loop configuration (defaults to LoopConfig()).
        client: Anthropic client instance. None = auto-create via get_client().
        session_id: Optional pre-created session ID (used by API service).
        trainer: Trainer instance for training execution. None = auto-create
            LocalTrainer from loop_config execution settings.

    Returns:
        LoopResult with all iteration records and best score.
    """
    if loop_config is None:
        loop_config = LoopConfig()

    if client is None:
        from p2p.inference.llm_client import get_client

        client = get_client()

    # Propagate motion overlay flags to environment + settings module
    import os

    import p2p.settings as _settings

    os.environ["VLM_CRITERIA_DIAGNOSIS"] = "true" if loop_config.criteria_diagnosis else "false"
    os.environ["VLM_MOTION_TRAIL_DUAL"] = "true" if loop_config.motion_trail_dual else "false"
    _settings.VLM_CRITERIA_DIAGNOSIS = loop_config.criteria_diagnosis
    _settings.VLM_MOTION_TRAIL_DUAL = loop_config.motion_trail_dual

    # Unpack LoopConfig for local use
    config = loop_config.train
    configs: list[RunConfigEntry] | None = loop_config.configs
    seeds = loop_config.seeds
    max_iterations = loop_config.max_iterations
    pass_threshold = loop_config.pass_threshold
    model = loop_config.model
    vlm_model = loop_config.vlm_model
    thinking_effort = loop_config.thinking_effort
    hp_tuning = loop_config.hp_tuning
    use_code_judge = loop_config.use_code_judge
    review_reward = loop_config.review_reward
    judgment_select = loop_config.judgment_select

    # Build default trainer if not provided
    if trainer is None:
        from p2p.training.trainer import LocalTrainer

        trainer = LocalTrainer(
            cores_per_run=loop_config.cores_per_run,
            max_parallel=loop_config.max_parallel,
            cores_pool=loop_config.cores_pool,
            no_cpu_affinity=loop_config.no_cpu_affinity,
            gpu_pool=loop_config.gpu_pool,
        )

    # Intent elicitation: use elaborated intent for all downstream consumers
    effective_intent = loop_config.elaborated_intent or prompt

    # Raise soft FD limit — SubprocVecEnv with num_envs=64 uses ~256 FDs
    # per iteration, and the default soft limit (often 1024) is exhausted
    # after a few iterations.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))

    session_id = session_id or generate_session_id()

    session = SessionRecord(loop_config.runs_dir / resolve_session_subpath(session_id))

    # Set up event logger for this session
    events = EventLogger(session.path)
    logger_token = set_event_logger(events)
    iter_token = set_current_iteration(None)

    # Normalize configs and seeds — a single run is just 1×1 multi-config.
    configs = configs or [{"config_id": "default", "label": "default", "params": {}}]
    seeds = seeds or [config.seed]

    # Look up environment spec (falls back to HalfCheetah-v5 via defaults)
    try:
        env_spec = get_env_spec(config.env_id)
    except KeyError:
        logger.warning("Unknown env_id %r, proceeding without env_spec", config.env_id)
        env_spec = None

    result: LoopResult = {
        "session_id": session_id,
        "prompt": prompt,
        "status": "running",
        "iterations": [],
        "best_iteration": 0,
        "best_score": 0.0,
        "pass_threshold": pass_threshold,
    }
    if config.env_id is not None:
        result["env_id"] = config.env_id
    session.set_status("running")
    session.save_history(result)  # persist prompt immediately for dashboard

    emit(
        "session.started",
        data={
            "prompt": prompt,
            "max_iterations": max_iterations,
            "model": model,
            "vlm_model": vlm_model,
            "num_configs": len(configs),
            "num_seeds": len(seeds),
            "pass_threshold": pass_threshold,
        },
    )

    try:
        # -- Parallel generation of code judge + reward code ----------------
        # Both are LLM calls; running them concurrently saves wall-clock time.
        # The code judge is only consumed at the first eval checkpoint, which
        # comes well after reward gen + validation + training start.  In the
        # unlikely case the first checkpoint arrives before the judge code is
        # ready, StreamingJudge will block until the future resolves.
        judge_code = ""
        judge_code_future: Future[str] | None = None
        _judge_pool: ThreadPoolExecutor | None = None

        if use_code_judge:
            # Override stride immediately — training must record full-res data.
            if config.trajectory_stride != 1:
                logger.info(
                    "Code judge active: overriding trajectory_stride %d -> 1",
                    config.trajectory_stride,
                )
                config.trajectory_stride = 1

            from p2p.agents.judge_author import generate_judge_code
            from p2p.event_log import get_event_logger

            # Capture event logger so the background thread can emit span
            # events (contextvars don't auto-propagate to threads).
            _captured_logger = get_event_logger()

            def _generate_judge_code_bg() -> str:
                tok = set_event_logger(_captured_logger)
                try:
                    with span("judge_code.generate"):
                        code = generate_judge_code(
                            effective_intent,
                            client=client,
                            env=env_spec,
                            model=model,
                            thinking_effort=thinking_effort,
                            max_episode_steps=config.max_episode_steps,
                        )
                    return code
                finally:
                    reset_event_logger(tok)

            _judge_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-codegen")
            judge_code_future = _judge_pool.submit(_generate_judge_code_bg)
            logger.info("Code judge generation started in background thread")

        # Generate reward code on the main thread (runs concurrently with
        # judge code generation above).
        with span("reward.generate") as gen_result:
            reward_code = generate(
                effective_intent,
                config=config,
                client=client,
                env=env_spec,
                model=model,
                thinking_effort=thinking_effort,
            )
            gen_result["code_lines"] = len(reward_code.splitlines())
        if review_reward:
            with span("reward.review.generate") as review_result:
                reward_code, ok = _try_review_reward(
                    reward_code,
                    prompt,
                    phase="generate",
                    client=client,
                    env_spec=env_spec,
                    model=model,
                    config=config,
                    thinking_effort=thinking_effort,
                )
                review_result["review_failed"] = not ok

        # Validate generated code; retry with fix_code if invalid
        for retry in range(MAX_CODE_RETRIES):
            try:
                validate_reward_code(reward_code, engine=config.engine)
                break
            except (SyntaxError, SyntaxWarning, ValueError) as exc:
                if retry == MAX_CODE_RETRIES - 1:
                    raise
                logger.warning(
                    "Generated code failed validation (retry %d/%d): %s",
                    retry + 1,
                    MAX_CODE_RETRIES,
                    exc,
                )
                reward_code = fix_code(
                    reward_code,
                    exc,
                    model=model,
                    client=client,
                    env=env_spec,
                    thinking_effort=thinking_effort,
                )

        # Try to collect judge code now (may already be done after reward gen).
        if judge_code_future is not None and judge_code_future.done():
            judge_code = _collect_judge_code(judge_code_future, session.path, _judge_pool)
            judge_code_future = None
            _judge_pool = None

        prev_based_on = 0

        for iteration in range(1, max_iterations + 1):
            current_based_on = prev_based_on  # What previous revise declared for THIS iter
            prev_based_on = 0  # Reset

            set_current_iteration(iteration)
            session.set_status_if("running", only_if=("running",))  # refresh heartbeat
            iter_id = f"iter_{iteration}"
            iteration_dir = session.path / iter_id
            iteration_dir.mkdir(parents=True, exist_ok=True)

            emit("iteration.started", data={"iter_id": iter_id})

            env_name = env_spec.name if env_spec else config.env_id

            tag_history = compute_tag_history(result["iterations"])

            record, judgment = run_iteration(
                iteration=iteration,
                iteration_dir=iteration_dir,
                reward_code=reward_code,
                config=config,
                configs=configs,
                seeds=seeds,
                env_id=config.env_id,
                env_name=env_name,
                prompt=effective_intent,
                pass_threshold=pass_threshold,
                vlm_model=vlm_model,
                thinking_effort=thinking_effort,
                client=client,
                model=model,
                trainer=trainer,
                judge_code=judge_code,
                judge_code_future=judge_code_future,
                session=session,
                judgment_select=judgment_select,
                tag_history=tag_history,
            )

            # Resolve judge code future after first iteration (if still pending).
            # By this point training has completed, so the future is certainly done.
            if judge_code_future is not None:
                if not judge_code:
                    judge_code = _collect_judge_code(judge_code_future, session.path, _judge_pool)
                judge_code_future = None
                _judge_pool = None

            result["iterations"].append(asdict(record))

            # Track best
            score = judgment.get("intent_score", 0)
            if score > result["best_score"]:
                result["best_score"] = score
                result["best_iteration"] = iteration

            emit(
                "iteration.completed",
                data={
                    "intent_score": round_metric(score),
                    "passed": judgment.get("passed", False),
                    "diagnosis": judgment.get("diagnosis", "")[:300],
                    "failure_tags": judgment.get("failure_tags", []),
                },
            )

            # Save incrementally + heartbeat
            session.save_history(result)
            session.touch_heartbeat()

            if judgment.get("passed"):
                _record_lineage_safe(
                    session.path,
                    session_id,
                    iteration,
                    record,
                    judgment,
                    result,
                    client,
                    model,
                    configs=configs,
                    thinking_effort=thinking_effort,
                )
                result["status"] = "passed"
                emit(
                    "session.completed",
                    data={
                        "status": "passed",
                        "best_score": result["best_score"],
                    },
                )
                break

            # Revise for next iteration (skip if this was the last)
            if iteration < max_iterations:
                scalars_path = _resolve_scalars_path(record)

                with span("revise") as revise_result:
                    revise_results = revise_multi(
                        n_variants=len(configs),
                        prompt=effective_intent,
                        reward_code=reward_code,
                        judgment=judgment,
                        summary=record.summary,
                        config=config,
                        iterations=result["iterations"],
                        scalars_path=scalars_path,
                        client=client,
                        env=env_spec,
                        model=model,
                        best_iteration=result["best_iteration"],
                        best_score=result["best_score"],
                        hp_tuning=hp_tuning,
                        session_dir=session.path,
                        thinking_effort=thinking_effort,
                    )
                    primary = revise_results[0]
                    revise_result["hp_changes"] = primary["hp_changes"]
                    revise_result["has_diagnosis"] = bool(primary["diagnosis"])
                    revise_result["n_variants"] = len(revise_results)

                # Review revised reward code (guarded by the same review_reward flag)
                revised_code = primary["reward_code"]
                if review_reward:
                    with span("reward.review.revise") as review_result:
                        revised_code, ok = _try_review_reward(
                            revised_code,
                            prompt,
                            phase="revise",
                            client=client,
                            env_spec=env_spec,
                            model=model,
                            config=config,
                            thinking_effort=thinking_effort,
                        )
                        review_result["review_failed"] = not ok

                # Validate shared reward code; retry with fix_code if invalid
                for retry in range(MAX_CODE_RETRIES):
                    try:
                        validate_reward_code(revised_code, engine=config.engine)
                        break
                    except (SyntaxError, SyntaxWarning, ValueError) as exc:
                        if retry == MAX_CODE_RETRIES - 1:
                            raise
                        logger.warning(
                            "Revised code failed validation (retry %d/%d): %s",
                            retry + 1,
                            MAX_CODE_RETRIES,
                            exc,
                        )
                        revised_code = fix_code(
                            revised_code,
                            exc,
                            model=model,
                            client=client,
                            env=env_spec,
                            thinking_effort=thinking_effort,
                        )
                reward_code = revised_code

                # Apply per-config HP changes
                for cfg, rev in zip(configs, revise_results):
                    cfg["params"] = rev["hp_changes"]

                _apply_revise_to_record(record, primary)
                if primary["hp_changes"]:
                    config = config.apply_updates(primary["hp_changes"])

                prev_based_on = record.based_on

            # Re-serialize the last iteration to capture revise fields.
            # record.based_on still holds the revise's raw value (next iter's
            # parent), which is the correct value for loop_history.json.
            result["iterations"][-1] = asdict(record)

            # Record into lineage AFTER revise fields are applied.
            # Pass current_based_on separately so we don't mutate record
            # after asdict() — that would cause loop_history and lineage to
            # disagree on based_on.
            _record_lineage_safe(
                session.path,
                session_id,
                iteration,
                record,
                judgment,
                result,
                client,
                model,
                configs=configs,
                lineage_based_on=current_based_on,
                thinking_effort=thinking_effort,
            )

            # Save incrementally (includes reasoning fields if revised)
            session.save_history(result)
        else:
            # for loop completed without break → max iterations reached
            result["status"] = "max_iterations"
            emit(
                "session.completed",
                data={
                    "status": "max_iterations",
                    "best_score": result["best_score"],
                },
            )
    except (anthropic.RateLimitError, LLMRateLimitError) as exc:
        result["status"] = "rate_limited"
        result["error"] = f"API rate limit exceeded: {exc}"
        logger.warning("Rate limit hit: %s", exc)
    except anthropic.AuthenticationError as exc:
        result["status"] = "auth_error"
        result["error"] = f"API authentication failed: {exc}"
        logger.error("Authentication error: %s", exc)
    except (SyntaxError, SyntaxWarning) as exc:
        result["status"] = "invalid_code"
        result["error"] = f"Generated code error: {exc}"
        logger.error("Invalid generated code: %s", exc)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.exception("Unexpected error in loop")
        emit("session.error", data={"error": result["error"][:500]})
    finally:
        # Shut down the judge-codegen thread pool if it was never consumed
        if _judge_pool is not None:
            _judge_pool.shutdown(wait=False)
        reset_current_iteration(iter_token)
        reset_event_logger(logger_token)

    # Final save
    session.save_history(result)
    session.set_status(result["status"], error=result.get("error"))
    return result
