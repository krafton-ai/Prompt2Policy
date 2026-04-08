"""Iteration execution: training, judgment, and aggregation for one loop iteration."""

from __future__ import annotations

import copy as _copy
import json
import logging
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from p2p.agents.judge_agent import (
    SharedCriteriaHolder,
    StreamingJudge,
    judge_all_checkpoints,
)
from p2p.analysis.guardrails import check_training_plateau, detect_reward_hacking
from p2p.config import DEFAULT_JUDGMENT_SELECT
from p2p.contracts import round_metric
from p2p.event_log import emit
from p2p.session.iteration_record import IterationData, IterationRecord, SessionRecord

if TYPE_CHECKING:
    from concurrent.futures import Future

    import anthropic

    from p2p.config import TrainConfig
    from p2p.contracts import FailureTagEntry, RunConfigEntry
    from p2p.training.trainer import Trainer

logger = logging.getLogger(__name__)


def run_iteration(
    *,
    iteration: int,
    iteration_dir: Path,
    reward_code: str,
    config: TrainConfig,
    configs: list[RunConfigEntry],
    seeds: list[int],
    env_id: str,
    env_name: str,
    prompt: str,
    pass_threshold: float,
    vlm_model: str,
    thinking_effort: str = "",
    client: anthropic.Anthropic,
    model: str,
    trainer: Trainer,
    judge_code: str = "",
    judge_code_future: Future[str] | None = None,
    session: SessionRecord | None = None,
    judgment_select: str = DEFAULT_JUDGMENT_SELECT,
    tag_history: list[FailureTagEntry] | None = None,
) -> tuple[IterationData, dict]:
    """Run a training iteration: parallel training, judge all runs, aggregate.

    Training execution is delegated to the *trainer* instance, which
    encapsulates backend details (local subprocess, SSH, etc.).
    """

    # Save prompt for traceability
    (iteration_dir / "prompt.txt").write_text(prompt)

    # Write shared reward file
    reward_fn_path = iteration_dir / "reward_fn.py"
    reward_fn_path.write_text(reward_code)

    # Load reviewed VLM criteria from session cache if available.
    # If not cached yet, StreamingJudge will generate lazily on first eval video
    # (so the first frame can be included for grounding).
    cached_criteria: str | None = None
    session_dir = iteration_dir.parent
    criteria_path = session_dir / "vlm_criteria.json"
    if criteria_path.exists():
        try:
            cached_criteria = json.loads(criteria_path.read_text())["criteria"]
            logger.info("Loaded cached VLM criteria from %s", criteria_path.name)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to load cached criteria, will regenerate on first eval")

    # Single shared holder for all StreamingJudge instances in this iteration.
    # On first iteration (no cache), the first instance to receive an eval video
    # generates criteria; all others block and reuse the same value.
    criteria_holder = SharedCriteriaHolder(initial=cached_criteria)

    # Pre-create run directories and start streaming judges
    base_config_dict = json.loads(config.to_json())
    all_run_ids = [f"{cfg['config_id']}_seed_{seed}" for cfg in configs for seed in seeds]
    streaming_judges: list[StreamingJudge] = []
    for run_id in all_run_ids:
        run_dir = iteration_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        sj = StreamingJudge(
            run_dir,
            prompt,
            pass_threshold=pass_threshold,
            env_name=env_name,
            vlm_model=vlm_model,
            thinking_effort=thinking_effort,
            client=client,
            model=model,
            judge_code=judge_code,
            judge_code_future=judge_code_future,
            criteria_holder=criteria_holder,
        )
        sj.start()
        streaming_judges.append(sj)

    # Launch persistent Phase 2 worker for IsaacLab (one bootstrap, one env).
    # The worker polls a queue dir for render requests written by training
    # eval callbacks.  Videos/trajectories appear during training for the
    # streaming judge and dashboard.
    worker_proc = None
    if config.engine == "isaaclab":
        worker_proc = _launch_phase2_worker(
            iteration_dir=iteration_dir,
            config=config,
            reward_code_path=str(reward_fn_path),
        )

    # Run all configs x seeds in parallel.
    # Wrap in try/finally to ensure streaming judges are always stopped,
    # even if parallel training raises (e.g. all runs fail).
    train_completed = False
    try:
        if session:
            session.touch_heartbeat()

        emit(
            "train.start",
            iteration=iteration,
            data={
                "total_timesteps": config.total_timesteps,
                "env_id": env_id,
                "num_runs": len(all_run_ids),
            },
        )
        train_start = _time.monotonic()

        aggregation = trainer.train(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_fn_path,
            base_config_dict=base_config_dict,
            iteration_dir=iteration_dir,
            env_id=env_id,
        )

        train_duration = int((_time.monotonic() - train_start) * 1000)

        if session:
            session.touch_heartbeat()
        train_completed = True
    finally:
        # Stop streaming judges and collect results (or just stop on error)
        for sj in streaming_judges:
            sj.stop()
        # Force-kill Phase 2 worker on abnormal exit (normal graceful stop
        # happens below after post-training processing).
        if not train_completed and worker_proc is not None and worker_proc.poll() is None:
            logger.warning("Force-killing Phase 2 worker on early exit (pid=%s)", worker_proc.pid)
            _kill_process_group(worker_proc)

    streaming_results_per_run: dict[str, dict[str, dict]] = {}
    for run_id, sj in zip(all_run_ids, streaming_judges):
        streaming_results_per_run[run_id] = sj.results

    best_run_id = aggregation["best_run_id"]

    emit(
        "train.end",
        iteration=iteration,
        data={
            "num_runs": len(all_run_ids),
            "best_run_id": best_run_id,
        },
        duration_ms=train_duration,
    )

    # Resolve judge code future after training (guaranteed done by now).
    if judge_code_future is not None and not judge_code:
        try:
            judge_code = judge_code_future.result()
            n_lines = judge_code.count("\n") + 1
            logger.info("Judge code resolved after training (%d lines)", n_lines)
        except Exception:
            logger.warning("Code judge generation failed, skipping code judge", exc_info=True)

    # Stop the Phase 2 worker and wait for pending renders to finish.
    # If unprocessed requests remain (worker crashed or rendered invisible
    # robots), retry with a fresh worker before proceeding to judgment.
    if worker_proc is not None:
        _stop_phase2_worker(iteration_dir, worker_proc)
        _retry_failed_renders(
            iteration_dir=iteration_dir,
            config=config,
            reward_code_path=str(reward_fn_path),
        )

    # Judge ALL config x seed runs (not just the best)
    emit(
        "judge.start",
        iteration=iteration,
        data={"num_runs": len(all_run_ids)},
    )
    judge_start = _time.monotonic()

    run_judgments: dict[str, dict] = {}

    for run_id in all_run_ids:
        run_dir = iteration_dir / run_id
        if not run_dir.exists():
            logger.warning("Run dir not found: %s", run_dir)
            continue
        logger.info("Judging run %s", run_id)
        _, judgment = judge_run(
            run_id=run_id,
            run_dir=run_dir,
            prompt=prompt,
            reward_code=reward_code,
            pass_threshold=pass_threshold,
            env_name=env_name,
            vlm_model=vlm_model,
            client=client,
            model=model,
            judge_code=judge_code,
            streaming_results=streaming_results_per_run.get(run_id),
            judgment_select=judgment_select,
            thinking_effort=thinking_effort,
            tag_history=tag_history,
            engine=config.engine,
            cached_criteria=criteria_holder.value,
        )
        run_judgments[run_id] = judgment
        if session:
            session.touch_heartbeat()

    if not run_judgments:
        raise RuntimeError(
            f"All {len(all_run_ids)} runs failed to produce judgments. "
            "Check run directories and VLM availability."
        )

    # Build per-config judgment aggregates
    config_judgments = aggregate_judgments(
        configs,
        seeds,
        run_judgments,
        aggregation,
    )

    # Enrich aggregation with judgment data
    aggregation["config_judgments"] = config_judgments

    # Re-select best config by intent_score (may differ from return-based best)
    best_by_score = max(
        config_judgments.items(),
        key=lambda kv: kv[1].get("mean_intent_score", 0),
        default=(aggregation["best_config_id"], {}),
    )
    aggregation["best_config_id"] = best_by_score[0]

    # Re-select best run: highest intent_score in best config
    best_config_id = aggregation["best_config_id"]
    best_score = -1.0
    for seed in seeds:
        rid = f"{best_config_id}_seed_{seed}"
        j = run_judgments.get(rid, {})
        s = j.get("intent_score", 0)
        if s > best_score:
            best_score = s
            best_run_id = rid
    aggregation["best_run_id"] = best_run_id

    # Overwrite aggregation.json with enriched version
    (iteration_dir / "aggregation.json").write_text(json.dumps(aggregation, indent=2))

    # Use best run's judgment as the iteration-level judgment (deep copy to
    # avoid circular reference: best_judgment IS run_judgments[best_run_id],
    # so attaching all_run_judgments would make it reference itself).
    best_judgment = _copy.deepcopy(run_judgments.get(best_run_id, {}))
    # Attach multi-config aggregation for the revise agent
    best_judgment["config_judgments"] = config_judgments
    best_judgment["all_run_judgments"] = run_judgments

    # Save iteration-level judgment
    (iteration_dir / "judgment.json").write_text(json.dumps(best_judgment, indent=2, default=str))

    emit(
        "judge.end",
        iteration=iteration,
        data={
            "intent_score": best_judgment.get("intent_score"),
            "best_config": best_config_id,
            "best_run": best_run_id,
            "config_scores": {
                cid: cj.get("mean_intent_score", 0) for cid, cj in config_judgments.items()
            },
        },
        duration_ms=int((_time.monotonic() - judge_start) * 1000),
    )

    # Save best_run info
    best_cfg_stats = aggregation.get("configs", {}).get(best_config_id, {})
    (iteration_dir / "best_run.json").write_text(
        json.dumps(
            {
                "best_config_id": best_config_id,
                "best_run_id": best_run_id,
                "mean_final_return": best_cfg_stats.get("mean_final_return", 0),
                "mean_intent_score": config_judgments.get(best_config_id, {}).get(
                    "mean_intent_score", 0
                ),
            },
            indent=2,
        )
    )

    # Read best run summary and apply guardrails
    best_rec = IterationRecord(iteration_dir / best_run_id)
    summary = best_rec.read_summary() or {}

    _, evals = best_rec.parse_scalars()
    if evals:
        last_eval = evals[-1]
        terms = last_eval.get("reward_terms", {})
        warning = detect_reward_hacking(terms)
        if warning:
            summary["guardrail_warning"] = warning
            emit(
                "guardrail.warning",
                iteration=iteration,
                data={"type": "reward_hacking", "warning": warning},
            )

    scalars_path = iteration_dir / best_run_id / "metrics" / "scalars.jsonl"
    if check_training_plateau(scalars_path):
        summary["guardrail_warning"] = (
            summary.get("guardrail_warning", "") + " Training plateau detected."
        ).strip()
        emit(
            "guardrail.warning",
            iteration=iteration,
            data={"type": "plateau", "warning": "Training plateau detected."},
        )

    record = IterationData(
        iteration=iteration,
        iteration_dir=str(iteration_dir),
        reward_code=reward_code,
        summary=summary,
        judgment=best_judgment,
        is_multi_config=True,
        aggregation=aggregation.get("configs"),
    )
    return record, best_judgment


def judge_run(
    run_id: str,
    run_dir: Path,
    prompt: str,
    reward_code: str,
    pass_threshold: float,
    env_name: str,
    vlm_model: str,
    client: Any,
    model: str,
    judge_code: str,
    streaming_results: dict[str, dict] | None = None,
    judgment_select: str = DEFAULT_JUDGMENT_SELECT,
    thinking_effort: str = "",
    tag_history: list[FailureTagEntry] | None = None,
    engine: str = "mujoco",
    cached_criteria: str | None = None,
) -> tuple[str, dict]:
    """Judge a single config x seed run. Returns (run_id, judgment)."""
    rec = IterationRecord(run_dir)
    summary = rec.read_summary() or {}
    _, evals = rec.parse_scalars()
    eval_reward_terms: dict[str, float] | None = None
    if evals:
        eval_reward_terms = evals[-1].get("reward_terms")

    judgment = judge_all_checkpoints(
        prompt,
        run_dir,
        summary,
        pass_threshold=pass_threshold,
        reward_code=reward_code,
        reward_terms=eval_reward_terms,
        env_name=env_name,
        vlm_model=vlm_model,
        thinking_effort=thinking_effort,
        client=client,
        model=model,
        judge_code=judge_code,
        streaming_results=streaming_results,
        judgment_select=judgment_select,
        tag_history=tag_history,
        engine=engine,
        cached_criteria=cached_criteria,
    )
    rec.save_judgment(judgment)
    return run_id, judgment


def aggregate_judgments(
    configs: list,
    seeds: list[int],
    run_judgments: dict[str, dict],
    aggregation: dict,
) -> dict[str, dict]:
    """Build per-config judgment aggregates from all run judgments.

    Returns {config_id: judgment_aggregate} dict.
    """
    config_judgments: dict[str, dict] = {}
    for cfg in configs:
        config_id = cfg["config_id"]
        seed_entries = []
        agg_per_seed = (aggregation.get("configs") or {}).get(config_id, {}).get("per_seed") or []
        for seed in seeds:
            run_id = f"{config_id}_seed_{seed}"
            j = run_judgments.get(run_id)
            if not j:
                continue
            final_ret = 0.0
            for ps in agg_per_seed:
                if int(ps.get("seed", -1)) == seed:
                    final_ret = ps.get("final_return", 0.0)
                    break
            seed_entries.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "intent_score": j.get("intent_score", 0.0),
                    "final_return": final_ret,
                    "best_checkpoint": j.get("best_checkpoint", ""),
                    "diagnosis": j.get("diagnosis", "")[:200],
                    "failure_tags": j.get("failure_tags", []),
                }
            )

        scores = [se["intent_score"] for se in seed_entries]
        returns = [se["final_return"] for se in seed_entries]
        # Use failure tags from the median seed (closest to mean score).
        # Free-form tags are diverse across seeds — frequency thresholds
        # drop most of them. The median seed is most representative.
        mean_score = float(np.mean(scores)) if scores else 0.0
        if seed_entries:
            median_se = min(seed_entries, key=lambda se: abs(se["intent_score"] - mean_score))
            common_tags = list(dict.fromkeys(median_se.get("failure_tags", [])))
        else:
            common_tags = []

        best_idx = int(np.argmax(scores)) if scores else 0
        worst_idx = int(np.argmin(scores)) if scores else 0

        config_judgments[config_id] = {
            "config_id": config_id,
            "num_seeds": len(seed_entries),
            "mean_intent_score": round_metric(float(np.mean(scores))) if scores else 0.0,
            "score_std": round_metric(float(np.std(scores))) if scores else 0.0,
            "mean_final_return": round_metric(float(np.mean(returns))) if returns else 0.0,
            "return_std": round_metric(float(np.std(returns))) if returns else 0.0,
            "best_seed": seed_entries[best_idx]["seed"] if seed_entries else 0,
            "worst_seed": seed_entries[worst_idx]["seed"] if seed_entries else 0,
            "common_failure_tags": common_tags,
            "per_seed": seed_entries,
        }
    return config_judgments


def _kill_process_group(proc: Any) -> None:
    """Kill a subprocess and its entire process group (SIGTERM then SIGKILL)."""
    import os
    import signal

    if proc.poll() is not None:
        return
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass
    sig = signal.SIGTERM
    if pgid is not None and pgid != os.getpgid(os.getpid()):
        try:
            os.killpg(pgid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
            return
        except Exception:
            # Escalate to SIGKILL
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _launch_phase2_worker(
    *,
    iteration_dir: Path,
    config: TrainConfig,
    reward_code_path: str,
) -> Any:
    """Launch a persistent Phase 2 worker subprocess.

    The worker bootstraps Isaac Sim once, creates one rendering env, and
    polls a queue directory for render requests written by training eval
    callbacks.  Videos/trajectories appear during training (like MuJoCo).
    """
    import subprocess

    queue_dir = iteration_dir / ".phase2_queue"
    queue_dir.mkdir(exist_ok=True)

    # Write worker config
    worker_config = {
        "env_id": config.env_id,
        "num_rounds": config.num_eval_rounds,
        "max_steps": config.max_episode_steps or 300,
        "stride": config.trajectory_stride,
        "precision": config.trajectory_precision,
        "device": "cuda:0",
        "reward_code": reward_code_path,
    }
    (queue_dir / "worker_config.json").write_text(json.dumps(worker_config, indent=2))

    from p2p.utils.subprocess_utils import python_cmd

    cmd = [
        "xvfb-run",
        "-a",
        *python_cmd(),
        "-m",
        "p2p.evaluator_isaaclab",
        "--worker-queue",
        str(queue_dir),
    ]

    stderr_log = iteration_dir / "phase2_stderr.log"
    stderr_f = open(stderr_log, "w")  # noqa: SIM115 — kept open for subprocess lifetime
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=stderr_f,
        # Omit start_new_session=True so the worker inherits the session's
        # process group — the process manager's os.killpg() must reach it.
    )
    logger.info("Phase 2 worker launched (pid=%d, queue=%s)", proc.pid, queue_dir)
    return proc


def _stop_phase2_worker(iteration_dir: Path, proc: Any) -> None:
    """Signal the Phase 2 worker to stop and wait for pending renders."""
    import subprocess

    queue_dir = iteration_dir / ".phase2_queue"
    (queue_dir / "stop").touch()
    logger.info("Phase 2 worker: stop signal sent, waiting for pending renders...")

    try:
        proc.wait(timeout=1800)
        if proc.returncode != 0:
            stderr_log = iteration_dir / "phase2_stderr.log"
            tail = stderr_log.read_text()[-500:] if stderr_log.exists() else ""
            logger.warning("Phase 2 worker exited with rc=%d: %s", proc.returncode, tail)
        else:
            logger.info("Phase 2 worker finished")
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        logger.warning("Phase 2 worker timed out after 1800s")
    except (SystemExit, KeyboardInterrupt):
        _kill_process_group(proc)
        raise


_MAX_RENDER_RETRIES = 2
_RENDER_RETRY_TIMEOUT = 600  # seconds per retry attempt
_WORKER_STARTUP_GRACE_SEC = 2  # worker poll interval is ~1s


def _retry_failed_renders(
    *,
    iteration_dir: Path,
    config: object,
    reward_code_path: str,
) -> None:
    """Retry unprocessed Phase 2 render requests with a fresh worker.

    After the Phase 2 worker stops, some ``request_*.json`` files may remain
    unprocessed (worker crashed, invisible robot from USD race, etc.).
    This launches a fresh worker for each retry attempt.  Videos are required
    for VLM judgment — proceeding without them wastes an iteration.
    """
    import subprocess as _sp
    import time as _time

    queue_dir = iteration_dir / ".phase2_queue"
    if not queue_dir.exists():
        return

    for attempt in range(1, _MAX_RENDER_RETRIES + 1):
        n_pending = sum(1 for _ in queue_dir.glob("request_*.json"))
        if n_pending == 0:
            return  # all processed

        logger.warning(
            "Phase 2: %d unprocessed render requests remain (retry %d/%d)",
            n_pending,
            attempt,
            _MAX_RENDER_RETRIES,
        )

        # Remove stale stop file so the new worker enters its poll loop.
        stop_file = queue_dir / "stop"
        stop_file.unlink(missing_ok=True)

        retry_proc = _launch_phase2_worker(
            iteration_dir=iteration_dir,
            config=config,
            reward_code_path=reward_code_path,
        )
        if retry_proc is None:
            logger.error("Phase 2 retry: failed to launch worker")
            break

        # Signal stop after a brief grace period — the worker will drain
        # pending requests before exiting.
        _time.sleep(_WORKER_STARTUP_GRACE_SEC)
        (queue_dir / "stop").touch()

        try:
            retry_proc.wait(timeout=_RENDER_RETRY_TIMEOUT)
        except _sp.TimeoutExpired:
            _kill_process_group(retry_proc)
            logger.warning("Phase 2 retry %d: worker timed out", attempt)
            break
        except OSError as e:
            _kill_process_group(retry_proc)
            logger.warning("Phase 2 retry %d: worker crashed: %s", attempt, e)
            break

        n_remaining = sum(1 for _ in queue_dir.glob("request_*.json"))
        processed = n_pending - n_remaining
        logger.info(
            "Phase 2 retry %d: processed %d/%d requests",
            attempt,
            processed,
            n_pending,
        )

    # Final check
    final_pending = sorted(queue_dir.glob("request_*.json"))
    if final_pending:
        logger.error(
            "Phase 2: %d render requests still unprocessed after %d retries. "
            "Judgment will proceed without videos for these checkpoints.",
            len(final_pending),
            _MAX_RENDER_RETRIES,
        )
