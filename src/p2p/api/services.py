"""File-system based IterationRecord service."""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p2p.api.entity_lifecycle import (
    _get_session_created_at,
    _is_session_dir,
    inject_metadata,
    is_entity_deleted,
)
from p2p.api.process_manager import is_stale
from p2p.api.schemas import (
    EnvInfo,
    IterationDetail,
    IterationSummary,
    LoopIterationSummary,
    MetricsResponse,
    SessionDetail,
)
from p2p.api.session_enrichment_service import (
    schedule_enrichment as _schedule_enrichment,
)
from p2p.contracts import (
    AggregatedMetrics,
    EventDetailRecord,
    IterationRunInfo,
    MeanStdSeries,
    ResourceStatus,
    RunMetricsDetail,
    SessionAnalysis,
    SessionConfig,
    normalize_reward_spec,
    round_metric,
)
from p2p.inference.llm_client import get_client
from p2p.inference.vlm import VLM_FPS
from p2p.session.iteration_record import IterationRecord, SessionRecord, read_json_safe
from p2p.settings import RUNS_DIR, resolve_session_dir
from p2p.training.env_spec import ENV_REGISTRY

logger = logging.getLogger(__name__)


def generate_hp_configs(
    n: int,
    base_config: dict | None = None,
    *,
    env_id: str = "",
    num_envs: int = 1,
) -> list[dict]:
    """Auto-generate N configs: first is baseline, rest perturb key hyperparams.

    Each perturbation picks 1-2 fields and assigns a non-default value.
    Deduplication compares **resolved** configs (base + overrides) so that
    a perturbation whose value already matches the baseline is treated as
    a no-op and retried.  This prevents wasting training slots on configs
    identical to the baseline or to each other.

    Parameters
    ----------
    n:
        Total number of configs to generate (including baseline).
    base_config:
        Base TrainConfig as a dict.  When provided, dedup compares resolved
        values of recipe fields instead of raw override dicts.
    env_id:
        Environment ID.  When *base_config* is not provided but *env_id* is
        non-empty, a zoo-preset baseline is built automatically so that
        resolved-config dedup still works.
    num_envs:
        Number of parallel envs (used for sqrt-scaling when building the
        baseline from *env_id*).
    """
    # Build base_config from zoo preset when not explicitly provided.
    if base_config is None and env_id:
        import json

        from p2p.config import TrainConfig

        base_config = json.loads(
            TrainConfig.from_preset(env_id=env_id, num_envs=num_envs).to_json()
        )
    # Perturbation recipes — values must stay within HP_BOUNDS.
    recipes = [
        ("learning_rate", [2e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3]),
        ("ent_coef", [0.0005, 0.001, 0.005, 0.02, 0.05]),
        ("gamma", [0.95, 0.98, 0.995]),
        ("gae_lambda", [0.8, 0.9, 0.92, 0.98]),
        ("clip_coef", [0.1, 0.3]),
        ("num_steps", [256, 512, 2048, 4096]),
        ("update_epochs", [5, 10, 15]),
        ("target_kl", [0.02, 0.03, 0.05]),
    ]

    recipe_fields = [field for field, _ in recipes]

    def _dedup_key(params: dict) -> str:
        """Build dedup key from the resolved config values of recipe fields."""
        if base_config is None:
            return str(sorted(params.items()))
        merged = {f: base_config.get(f) for f in recipe_fields}
        merged.update(params)
        return str(sorted(merged.items()))

    configs: list[dict] = [
        {"config_id": "baseline", "label": "Baseline (default)", "params": {}},
    ]

    # Baseline resolved key: when base_config is provided, this captures the
    # full resolved values of all recipe fields; when base_config is None
    # (fallback mode), it reduces to "[]" matching any empty-params override.
    used_resolved: set[str] = {_dedup_key({})}
    _MAX_RETRIES = 20

    for i in range(1, n):
        found_unique = False
        resolved_key = ""  # sentinel; always overwritten before use
        params: dict[str, float | int] = {}
        for _attempt in range(_MAX_RETRIES):
            num_fields = random.choice([1, 2])
            chosen = random.sample(recipes, min(num_fields, len(recipes)))
            params = {}
            for field, values in chosen:
                params[field] = random.choice(values)

            resolved_key = _dedup_key(params)
            if resolved_key not in used_resolved:
                found_unique = True
                break  # unique resolved config found

        if not found_unique:
            logger.warning(
                "Could not find unique HP perturbation for config_%d after %d attempts; skipping",
                i,
                _MAX_RETRIES,
            )
            continue

        used_resolved.add(resolved_key)
        label_parts = [f"{k}={v}" for k, v in params.items()]
        configs.append(
            {
                "config_id": f"config_{i}",
                "label": ", ".join(label_parts),
                "params": params,
            }
        )

    return configs


def _enrich_reward_spec(reward_spec: dict, reward_source: str) -> dict:
    """Normalize reward_spec to structured format and fix LaTeX escaping.

    1. Converts legacy ``terms: dict[str, str]`` to ``list[RewardTerm]``.
    2. Re-parses from source when available to fix double-escaped backslashes
       (e.g. ``\\\\cdot`` → ``\\cdot``) caused by the escape-sanitizer.
    """
    reward_spec = normalize_reward_spec(reward_spec)

    if not reward_source:
        return reward_spec

    from p2p.training.reward_loader import _parse_docstring

    latex, terms = _parse_docstring(lambda *a: None, reward_source)
    if terms:
        reward_spec["terms"] = terms
    if latex:
        reward_spec["latex"] = latex
    return reward_spec


def _get_created_at(iteration_dir: Path) -> str:
    """Get creation time from iteration directory (config.json mtime)."""
    config_path = iteration_dir / "config.json"
    if config_path.exists():
        ts = config_path.stat().st_mtime
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat()
    return ""


def _is_iteration_dir(candidate: Path) -> bool:
    """Check if *candidate* is a valid iteration directory.

    Accepts both single-run iterations (top-level ``config.json``) and
    multi-config iterations (sub-run dirs with ``config.json`` or
    ``best_run.json`` at the top level).
    """
    if not candidate.is_dir():
        return False
    if (candidate / "config.json").exists():
        return True
    # Multi-config: has best_run.json or sub-run dirs with config.json
    if (candidate / "best_run.json").exists():
        return True
    # Multi-config in progress: no best_run.json yet, but sub-run dirs exist
    try:
        return any(
            (d / "config.json").exists()
            for d in candidate.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        )
    except OSError:
        return False


def _find_iteration_path(iteration_id: str, session_id: str | None = None) -> Path | None:
    """Find an iteration directory by iteration_id.

    When *session_id* is given, looks directly at ``RUNS_DIR/session_id/iteration_id``
    (O(1)).  Otherwise falls back to scanning all sessions (O(N)) — this is
    only safe when iteration_id is globally unique (legacy timestamp-based IDs).
    """
    if not RUNS_DIR.exists():
        return None
    if session_id is not None:
        candidate = resolve_session_dir(session_id) / iteration_id
        if _is_iteration_dir(candidate):
            return candidate
        return None
    # Fallback: scan all sessions (ambiguous for iter_N names)
    for session_dir in RUNS_DIR.iterdir():
        if not session_dir.is_dir() or not _is_session_dir(session_dir):
            continue
        candidate = session_dir / iteration_id
        if _is_iteration_dir(candidate):
            return candidate
    return None


def _static_prefix(rec: IterationRecord) -> str:
    """Build static URL prefix from the iteration record's path.

    Handles both flat (``runs/session_xxx/iter_1``) and nested benchmark
    layouts (``runs/bm_xxx/case0/iter_1``).
    """
    try:
        rel = rec.path.resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        # Path outside RUNS_DIR (e.g. custom runs_dir) — best-effort fallback
        session_id = rec.path.parent.name
        return f"/static/runs/{session_id}/{rec.iteration_id}"
    return f"/static/runs/{rel.as_posix()}"


def _find_iteration_record(iteration_id: str, session_id: str | None = None) -> IterationRecord:
    """Find an IterationRecord by iteration_id or raise FileNotFoundError."""
    iteration_path = _find_iteration_path(iteration_id, session_id=session_id)
    if iteration_path is None:
        msg = f"Iteration not found: {iteration_id}"
        raise FileNotFoundError(msg)
    return IterationRecord(iteration_path)


def list_iterations() -> list[IterationSummary]:
    if not RUNS_DIR.exists():
        return []
    iterations = []
    for session_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir() or not _is_session_dir(session_dir):
            continue
        for iteration_dir in sorted(session_dir.iterdir(), reverse=True):
            if not iteration_dir.is_dir() or iteration_dir.name.startswith("."):
                continue
            rec = IterationRecord(iteration_dir)
            config = rec.read_config()
            if config is None:
                continue
            reward_spec = rec.read_reward_spec()
            summary = rec.read_summary()
            prefix = _static_prefix(rec)
            iterations.append(
                IterationSummary(
                    iteration_id=rec.iteration_id,
                    session_id=session_dir.name,
                    env_id=config.get("env_id", ""),
                    status=rec.derive_status(),
                    created_at=_get_created_at(rec.path),
                    total_timesteps=config.get("total_timesteps", 0),
                    final_episodic_return=(
                        summary.get("final_episodic_return") if summary else None
                    ),
                    reward_latex=reward_spec.get("latex", ""),
                    reward_description=reward_spec.get("description", ""),
                    video_urls=[f"{prefix}/videos/{v}" for v in rec.video_filenames()],
                    progress=rec.compute_progress(),
                )
            )
    return iterations


def _find_best_sub_run(iteration_path: Path) -> Path | None:
    """For multi-config iterations, find the best sub-run directory.

    Reads ``best_run.json`` to determine the best run, otherwise picks the
    first sub-directory that has a ``config.json``.
    """
    best_run_data = read_json_safe(iteration_path / "best_run.json")
    if best_run_data and best_run_data.get("best_run_id"):
        best = iteration_path / best_run_data["best_run_id"]
        if best.is_dir() and (best / "config.json").exists():
            return best
    # Fallback: first sub-dir with config.json
    for d in sorted(iteration_path.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            return d
    return None


def get_iteration(iteration_id: str, session_id: str | None = None) -> IterationDetail | None:
    iteration_path = _find_iteration_path(iteration_id, session_id=session_id)
    if iteration_path is None:
        return None
    rec = IterationRecord(iteration_path)
    config = rec.read_config()

    # Multi-config: no top-level config.json — delegate to best sub-run
    if config is None:
        sub_run = _find_best_sub_run(iteration_path)
        if sub_run is None:
            return None
        sub_rec = IterationRecord(sub_run)
        config = sub_rec.read_config()
        if config is None:
            return None
        reward_spec = sub_rec.read_reward_spec()
        summary = sub_rec.read_summary()
        training, eval_results = sub_rec.parse_scalars()
        sub_prefix = f"{_static_prefix(rec)}/{sub_run.name}"
        # Judgment is at the iteration root level
        judgment = rec.read_judgment()
        # Reward source from iteration root (shared across configs)
        reward_path = iteration_path / "reward_fn.py"
        reward_source = reward_path.read_text() if reward_path.exists() else ""
        reward_spec = _enrich_reward_spec(reward_spec, reward_source)
        return IterationDetail(
            iteration_id=iteration_id,
            session_id=iteration_path.parent.name,
            env_id=config.get("env_id", ""),
            status=sub_rec.derive_status(),
            created_at=_get_created_at(iteration_path),
            total_timesteps=config.get("total_timesteps", 0),
            final_episodic_return=summary.get("final_episodic_return") if summary else None,
            reward_latex=reward_spec.get("latex", ""),
            reward_description=reward_spec.get("description", ""),
            video_urls=[f"{sub_prefix}/videos/{v}" for v in sub_rec.video_filenames()],
            progress=sub_rec.compute_progress(),
            config=config,
            reward_spec=reward_spec,
            reward_source=reward_source,
            summary=summary,
            eval_results=eval_results,
            judgment=judgment,
            training=training,
        )

    reward_spec = rec.read_reward_spec()
    reward_source = rec.read_reward_source()
    reward_spec = _enrich_reward_spec(reward_spec, reward_source)
    summary = rec.read_summary()
    training, eval_results = rec.parse_scalars()
    prefix = _static_prefix(rec)
    judgment = rec.read_judgment()
    return IterationDetail(
        iteration_id=iteration_id,
        session_id=iteration_path.parent.name,
        env_id=config.get("env_id", ""),
        status=rec.derive_status(),
        created_at=_get_created_at(iteration_path),
        total_timesteps=config.get("total_timesteps", 0),
        final_episodic_return=summary.get("final_episodic_return") if summary else None,
        reward_latex=reward_spec.get("latex", ""),
        reward_description=reward_spec.get("description", ""),
        video_urls=[f"{prefix}/videos/{v}" for v in rec.video_filenames()],
        progress=rec.compute_progress(),
        config=config,
        reward_spec=reward_spec,
        reward_source=reward_source,
        summary=summary,
        eval_results=eval_results,
        judgment=judgment,
        training=training,
    )


def get_metrics(iteration_id: str, session_id: str | None = None) -> MetricsResponse | None:
    iteration_path = _find_iteration_path(iteration_id, session_id=session_id)
    if iteration_path is None:
        return None
    rec = IterationRecord(iteration_path)
    training, evaluation = rec.parse_scalars()
    return MetricsResponse(training=training, evaluation=evaluation)


# ---------------------------------------------------------------------------
# Session support (loop_history.json based)
# ---------------------------------------------------------------------------


def _get_session_record(session_id: str) -> SessionRecord:
    return SessionRecord(resolve_session_dir(session_id))


def _read_last_elapsed(scalars_path: Path) -> float | None:
    """Read the last elapsed_time from a scalars.jsonl file (cheap tail read)."""
    if not scalars_path.exists():
        return None
    try:
        with open(scalars_path, "rb") as f:
            # Seek to end, read last ~4KB to find last line
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(max(0, size - read_size))
            tail = f.read().decode("utf-8", errors="replace")
        lines = tail.strip().splitlines()
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if "elapsed_time" in entry:
                    return float(entry["elapsed_time"])
            except (json.JSONDecodeError, ValueError):
                continue
    except OSError:
        pass
    return None


def _get_live_elapsed(iteration_dir: Path, is_multi_config: bool) -> float | None:
    """Get elapsed time for an in-progress iteration.

    Single-config: reads from ``metrics/scalars.jsonl``.
    Multi-config: takes max elapsed_time across all sub-run scalars.
    """
    if not iteration_dir.exists():
        return None
    if not is_multi_config:
        return _read_last_elapsed(iteration_dir / "metrics" / "scalars.jsonl")
    # Multi-config: max across sub-runs
    max_elapsed: float | None = None
    for sub in iteration_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith(".") or sub.name.startswith("__"):
            continue
        val = _read_last_elapsed(sub / "metrics" / "scalars.jsonl")
        if val is not None:
            max_elapsed = max(max_elapsed or 0.0, val)
    return max_elapsed


def _resolve_preview_urls(video_urls: list[str], filenames: dict[str, str]) -> dict[str, str]:
    """Convert preview filenames to full static URLs using the video directory prefix."""
    prefix = video_urls[0].rsplit("/", 1)[0] if video_urls else ""
    return {k: f"{prefix}/{fn}" for k, fn in filenames.items()} if prefix else {}


def _iteration_to_summary(it: dict, *, lightweight: bool = False) -> LoopIterationSummary:
    judgment = it.get("judgment", {})
    summary = it.get("summary", {})

    if lightweight:
        # Minimal fields for session list — skip heavy rollout/diagnosis data
        iteration_dir_str = it.get("iteration_dir") or it.get("run_dir", "")
        is_multi_config = False
        human_label = None
        if iteration_dir_str:
            p = Path(iteration_dir_str)
            is_multi_config = (p / "aggregation.json").exists() or (p / "best_run.json").exists()
            human_label = read_json_safe(p / "human_label.json")
        return LoopIterationSummary(
            iteration=it.get("iteration", 0),
            iteration_dir=iteration_dir_str,
            intent_score=judgment.get("intent_score"),
            final_return=summary.get("final_episodic_return"),
            is_multi_config=is_multi_config,
            diagnosis="",
            failure_tags=[],
            reward_code="",
            based_on=it.get("based_on", 0),
            vlm_fps=VLM_FPS,
            human_label=human_label,
        )

    # Extract per-checkpoint scores and diagnoses from checkpoint_judgments
    checkpoint_scores: dict[str, float] = {}
    checkpoint_diagnoses: dict[str, str] = {}
    checkpoint_code_diagnoses: dict[str, str] = {}
    checkpoint_vlm_diagnoses: dict[str, str] = {}
    checkpoint_code_scores: dict[str, float] = {}
    checkpoint_vlm_scores: dict[str, float] = {}
    rollout_scores: dict[str, float] = {}
    rollout_diagnoses: dict[str, str] = {}
    rollout_code_diagnoses: dict[str, str] = {}
    rollout_vlm_diagnoses: dict[str, str] = {}
    rollout_code_scores: dict[str, float] = {}
    rollout_vlm_scores: dict[str, float] = {}
    rollout_synthesis_traces: dict[str, list] = {}
    rollout_vlm_preview_filenames: dict[str, str] = {}
    _found_vlm_criteria: list[str] = []  # mutable container for closure
    _found_criteria_scores: list[list] = []
    rollout_criteria_scores: dict[str, list] = {}

    def _extract_checkpoint(step: str, cj: dict) -> None:
        score = cj.get("intent_score")
        if score is not None:
            checkpoint_scores[step] = score
        diag = cj.get("diagnosis", "")
        if diag:
            checkpoint_diagnoses[step] = diag
        code_diag = cj.get("code_diagnosis", "")
        vlm_diag = cj.get("vlm_diagnosis", "")
        # When code judge is used but code_diagnosis wasn't stored separately
        # (code_judge+llm_synthesis), treat diagnosis as code output.
        # Do NOT include multi_rollout — its aggregate diagnosis is a statistical
        # summary, not code judge output.
        sm = cj.get("scoring_method", "")
        if not code_diag and diag and "code_judge" in sm:
            code_diag = diag
        if code_diag:
            checkpoint_code_diagnoses[step] = code_diag
        if vlm_diag:
            checkpoint_vlm_diagnoses[step] = vlm_diag
        if cj.get("code_score") is not None:
            checkpoint_code_scores[step] = cj["code_score"]
        if cj.get("vlm_score") is not None:
            checkpoint_vlm_scores[step] = cj["vlm_score"]

    def _extract_rollouts(cj: dict, step: str) -> None:
        for rj in cj.get("rollout_judgments", []):
            label = rj.get("rollout_label")
            if label:
                key = f"{step}_{label}"
            else:
                key = f"{step}_ep{rj['episode_idx']}"
            rollout_scores[key] = rj.get("intent_score", 0.0)
            rollout_diagnoses[key] = rj.get("diagnosis", "")
            if rj.get("code_diagnosis"):
                rollout_code_diagnoses[key] = rj["code_diagnosis"]
            if rj.get("vlm_diagnosis"):
                rollout_vlm_diagnoses[key] = rj["vlm_diagnosis"]
            if rj.get("code_score") is not None:
                rollout_code_scores[key] = rj["code_score"]
            if rj.get("vlm_score") is not None:
                rollout_vlm_scores[key] = rj["vlm_score"]
            if rj.get("synthesis_tool_calls"):
                rollout_synthesis_traces[key] = rj["synthesis_tool_calls"]
            if rj.get("vlm_preview_filename"):
                rollout_vlm_preview_filenames[key] = rj["vlm_preview_filename"]
            if rj.get("criteria_scores"):
                rollout_criteria_scores[key] = rj["criteria_scores"]
                if not _found_criteria_scores:
                    _found_criteria_scores.append(rj["criteria_scores"])
            if not _found_vlm_criteria and rj.get("vlm_criteria"):
                _found_vlm_criteria.append(rj["vlm_criteria"])
        # Single-video checkpoint: vlm_preview_filename at checkpoint level
        if not cj.get("rollout_judgments") and cj.get("vlm_preview_filename"):
            rollout_vlm_preview_filenames[step] = cj["vlm_preview_filename"]

    for step, cj in judgment.get("checkpoint_judgments", {}).items():
        _extract_checkpoint(step, cj)
        _extract_rollouts(cj, step)

    # Fallback: streaming judgments have rollout_judgments at top level
    if not rollout_scores and judgment.get("rollout_judgments"):
        step = judgment.get("best_checkpoint", "live")
        _extract_checkpoint(step, judgment)
        _extract_rollouts(judgment, step)

    # Multi-config: iteration-level rollouts may lack synthesis traces.
    # Fall back to the best run's per-checkpoint rollouts which have them.
    if not rollout_synthesis_traces:
        _best_run = judgment.get("config_judgments", {})
        if not _best_run:
            _best_run = judgment.get("all_run_judgments", {})
        # Pick whichever run has the highest score
        _best_run_data: dict | None = None
        _best_score = -1.0
        for _rdata in _best_run.values():
            if isinstance(_rdata, dict):
                _rs = _rdata.get("intent_score", 0.0) or 0.0
                if _rs > _best_score:
                    _best_score = _rs
                    _best_run_data = _rdata
        if _best_run_data:
            for step, cj in _best_run_data.get("checkpoint_judgments", {}).items():
                _extract_rollouts(cj, step)

    iteration_dir_str = it.get("iteration_dir") or it.get("run_dir", "")

    # Multi-config iteration fields (computed BEFORE video selection)
    is_multi_config = False
    aggregation = None
    best_config_id = ""
    best_run_id = ""
    if iteration_dir_str:
        iteration_dir = Path(iteration_dir_str)
        agg_path = iteration_dir / "aggregation.json"
        best_path = iteration_dir / "best_run.json"
        if agg_path.exists():
            is_multi_config = True
            raw_agg = read_json_safe(agg_path)
            # aggregation.json wraps per-config data under "configs" key
            aggregation = raw_agg.get("configs", raw_agg) if raw_agg else None
        else:
            # Detect in-progress multi-config from subdirectory names
            _run_pat = re.compile(r"^(.+)_seed_(\d+)$")
            config_ids_found: set[str] = set()
            if iteration_dir.exists():
                for sub in iteration_dir.iterdir():
                    if sub.is_dir():
                        m = _run_pat.match(sub.name)
                        if m:
                            config_ids_found.add(m.group(1))
            if config_ids_found:
                is_multi_config = True
                # Build a live aggregation stub from per-seed summary.json
                # and streaming judgments (for best-run selection during training)
                live_agg: dict = {}
                for cid in sorted(config_ids_found):
                    scores: list[float] = []
                    returns: list[float] = []
                    judge_scores: list[float] = []
                    per_seed: list[dict] = []
                    for sub in iteration_dir.iterdir():
                        m = _run_pat.match(sub.name)
                        if m and m.group(1) == cid:
                            seed = int(m.group(2))
                            sm = read_json_safe(sub / "summary.json")
                            bs, fr = 0.0, 0.0
                            if sm:
                                fallback = sm.get("final_episodic_return", 0.0)
                                bs = sm.get("best_episodic_return", fallback) or 0.0
                                fr = sm.get("final_episodic_return", 0.0) or 0.0
                            # Read best streaming judge score for this run
                            best_judge: float | None = None
                            sj_dir = sub / "streaming_judgments"
                            if sj_dir.is_dir():
                                for sj_file in sj_dir.glob("*.json"):
                                    sj = read_json_safe(sj_file)
                                    if sj:
                                        js = sj.get("intent_score")
                                        if js is not None:
                                            val = float(js)
                                            if best_judge is None or val > best_judge:
                                                best_judge = val
                            scores.append(bs)
                            returns.append(fr)
                            if best_judge is not None:
                                judge_scores.append(best_judge)
                            seed_entry: dict = {
                                "seed": seed,
                                "best_score": bs,
                                "final_return": fr,
                            }
                            if best_judge is not None:
                                seed_entry["best_judge_score"] = best_judge
                            per_seed.append(seed_entry)
                    arr_s = np.array(scores) if scores else np.array([0.0])
                    arr_r = np.array(returns) if returns else np.array([0.0])
                    cid_agg: dict = {
                        "mean_best_score": round_metric(float(arr_s.mean())),
                        "std_best_score": round_metric(float(arr_s.std())),
                        "mean_final_return": round_metric(float(arr_r.mean())),
                        "std_final_return": round_metric(float(arr_r.std())),
                        "per_seed": per_seed,
                    }
                    if judge_scores:
                        arr_j = np.array(judge_scores)
                        cid_agg["mean_best_judge_score"] = round_metric(float(arr_j.mean()))
                    live_agg[cid] = cid_agg
                aggregation = live_agg
        if best_path.exists():
            best_info = read_json_safe(best_path)
            if best_info:
                best_config_id = best_info.get("best_config_id", "")
                best_run_id = best_info.get("best_run_id", "")
        elif agg_path.exists() and not best_config_id:
            # Fall back to aggregation.json top-level fields
            raw_agg = read_json_safe(agg_path)
            if raw_agg:
                best_config_id = raw_agg.get("best_config_id", "")
                best_run_id = raw_agg.get("best_run_id", "")

        # Live: derive best_config/best_run from aggregation when best_run.json
        # doesn't exist yet (training still in progress).
        # Prefer judge score when available, fall back to episodic return.
        if not best_config_id and aggregation:
            # Check if any config has judge data (key present from streaming judgments)
            _has_judge = any("mean_best_judge_score" in cdata for cdata in aggregation.values())
            _rank_key = "mean_best_judge_score" if _has_judge else "mean_final_return"
            _seed_key = "best_judge_score" if _has_judge else "final_return"

            _best_val = float("-inf")
            for cid, cdata in aggregation.items():
                val = cdata.get(_rank_key)
                if val is None:
                    continue
                if val > _best_val:
                    _best_val = val
                    best_config_id = cid
            if best_config_id:
                _best_seed_val = float("-inf")
                for ps in aggregation[best_config_id].get("per_seed", []):
                    _ps_val = ps.get(_seed_key, 0.0)
                    if _ps_val > _best_seed_val:
                        _best_seed_val = _ps_val
                        best_run_id = f"{best_config_id}_seed_{int(ps['seed'])}"

    # Build video URLs from iteration_dir (fall back to run_dir for backward compat)
    video_urls: list[str] = []
    video_source_run_id = ""
    video_source_return: float | None = None
    if iteration_dir_str:
        iteration_dir = Path(iteration_dir_str)
        rec = IterationRecord(iteration_dir)
        prefix = _static_prefix(rec)
        # Check iteration-level videos first
        video_urls = [f"{prefix}/videos/{v}" for v in rec.video_filenames()]
        # Fall back to sub-run directories (multi-config/seed layout)
        if not video_urls and iteration_dir.exists():
            target_sub: Path | None = None

            def _has_videos(d: Path) -> bool:
                vdir = d / "videos"
                return vdir.is_dir() and any(vdir.glob("*.mp4"))

            # Use best_run_id (from best_run.json or live aggregation)
            if best_run_id:
                candidate = iteration_dir / best_run_id
                if candidate.is_dir() and _has_videos(candidate):
                    target_sub = candidate
            # Fall back to first sub-run with videos
            if target_sub is None:
                for sub in sorted(iteration_dir.iterdir()):
                    if sub.is_dir() and _has_videos(sub):
                        target_sub = sub
                        break
            if target_sub is not None:
                sub_rec = IterationRecord(target_sub)
                sub_videos = sub_rec.video_filenames()
                if sub_videos:
                    sub_prefix = f"{prefix}/{target_sub.name}"
                    video_urls = [f"{sub_prefix}/videos/{v}" for v in sub_videos]
                    video_source_run_id = target_sub.name
                    # Read return from summary.json
                    sub_summary = read_json_safe(target_sub / "summary.json")
                    if sub_summary:
                        video_source_return = sub_summary.get("final_episodic_return")

    # Fall back to streaming judgments (written during training by StreamingJudge)
    # For multi-config, check the video source run's streaming_judgments dir;
    # for single-config, check the iteration-level dir.
    # Always scan: the single-judgment fallback above (lines 574-578) uses a
    # "live" step key that won't match video URLs.  The disk scan uses the
    # actual checkpoint step from the filename, producing correct keys.
    if iteration_dir_str:
        sj_candidates = [Path(iteration_dir_str) / "streaming_judgments"]
        if video_source_run_id:
            sj_candidates.insert(
                0, Path(iteration_dir_str) / video_source_run_id / "streaming_judgments"
            )
        for sj_dir in sj_candidates:
            if not sj_dir.is_dir():
                continue
            for f in sj_dir.glob("*.json"):
                step = f.stem
                try:
                    cj = read_json_safe(f)
                    if cj:
                        _extract_checkpoint(step, cj)
                        _extract_rollouts(cj, step)
                except Exception:
                    logger.debug("Failed to parse streaming judgment entry", exc_info=True)
            if checkpoint_scores:
                break  # found scores, no need to check fallback

    # Elapsed time: completed → summary, in-progress → last scalars entry
    elapsed_time_s = summary.get("training_time_s")
    if elapsed_time_s is None and iteration_dir_str:
        elapsed_time_s = _get_live_elapsed(Path(iteration_dir_str), is_multi_config)

    # Human label status
    human_label = None
    if iteration_dir_str:
        human_label = read_json_safe(Path(iteration_dir_str) / "human_label.json")

    rollout_vlm_preview_urls = _resolve_preview_urls(video_urls, rollout_vlm_preview_filenames)
    # Derive motion trail URLs from VLM preview filenames (_vlm -> _motion)
    rollout_motion_preview_filenames = {
        k: v.replace("_vlm.mp4", "_motion.mp4") for k, v in rollout_vlm_preview_filenames.items()
    }
    rollout_motion_preview_urls = _resolve_preview_urls(
        video_urls, rollout_motion_preview_filenames
    )

    return LoopIterationSummary(
        iteration=it.get("iteration", 0),
        iteration_dir=iteration_dir_str,
        intent_score=judgment.get("intent_score"),
        best_checkpoint=judgment.get("best_checkpoint", ""),
        checkpoint_scores=checkpoint_scores,
        checkpoint_diagnoses=checkpoint_diagnoses,
        checkpoint_code_diagnoses=checkpoint_code_diagnoses,
        checkpoint_vlm_diagnoses=checkpoint_vlm_diagnoses,
        checkpoint_code_scores=checkpoint_code_scores,
        checkpoint_vlm_scores=checkpoint_vlm_scores,
        rollout_scores=rollout_scores,
        rollout_diagnoses=rollout_diagnoses,
        rollout_code_diagnoses=rollout_code_diagnoses,
        rollout_vlm_diagnoses=rollout_vlm_diagnoses,
        rollout_code_scores=rollout_code_scores,
        rollout_vlm_scores=rollout_vlm_scores,
        rollout_synthesis_traces=rollout_synthesis_traces,
        rollout_vlm_preview_urls=rollout_vlm_preview_urls,
        rollout_motion_preview_urls=rollout_motion_preview_urls,
        rollout_criteria_scores=rollout_criteria_scores,
        vlm_fps=VLM_FPS,
        diagnosis=judgment.get("diagnosis", ""),
        failure_tags=judgment.get("failure_tags", []),
        reward_code=it.get("reward_code", ""),
        reward_diff_summary=it.get("reward_diff_summary", ""),
        final_return=summary.get("final_episodic_return"),
        video_urls=video_urls,
        is_multi_config=is_multi_config,
        aggregation=aggregation,
        best_config_id=best_config_id,
        best_run_id=best_run_id,
        video_source_run_id=video_source_run_id,
        video_source_return=video_source_return,
        elapsed_time_s=elapsed_time_s,
        # Revise agent output
        reward_reasoning=it.get("reward_reasoning", ""),
        hp_reasoning=it.get("hp_reasoning", ""),
        hp_changes=it.get("hp_changes", {}),
        training_dynamics=it.get("training_dynamics", ""),
        revise_diagnosis=it.get("revise_diagnosis", ""),
        based_on=it.get("based_on", 0),
        # Per-judge raw outputs
        code_diagnosis=judgment.get("code_diagnosis", "")
        or (
            judgment.get("diagnosis", "")
            if "code_judge" in judgment.get("scoring_method", "")
            else ""
        ),
        code_score=judgment.get("code_score"),
        vlm_diagnosis=judgment.get("vlm_diagnosis", ""),
        vlm_score=judgment.get("vlm_score"),
        vlm_criteria=judgment.get("vlm_criteria", "")
        or (_found_vlm_criteria[0] if _found_vlm_criteria else ""),
        criteria_scores=judgment.get("criteria_scores")
        or (_found_criteria_scores[0] if _found_criteria_scores else []),
        scoring_method=judgment.get("scoring_method", ""),
        human_label=human_label,
    )


def _get_session_config_field(history: dict | None, field: str, default: object = "") -> object:
    """Extract a field from the first iteration's config.json.

    For multi-config iterations the config.json lives inside a sub-run
    directory (e.g. ``iter_1/baseline_seed_1/config.json``), so we also
    check one level deeper.
    """
    if history is None:
        return default
    for it in history.get("iterations", []):
        iteration_dir_str = it.get("iteration_dir") or it.get("run_dir", "")
        if not iteration_dir_str:
            continue
        iteration_dir = Path(iteration_dir_str)
        # Direct config.json (single-config iteration)
        config_path = iteration_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                return config.get(field, default)
            except (json.JSONDecodeError, OSError):
                pass
        # Multi-config: check first sub-run directory
        if iteration_dir.is_dir():
            for sub in sorted(iteration_dir.iterdir()):
                if not sub.is_dir():
                    continue
                sub_config = sub / "config.json"
                if sub_config.exists():
                    try:
                        config = json.loads(sub_config.read_text())
                        return config.get(field, default)
                    except (json.JSONDecodeError, OSError):
                        pass
    return default


def _get_session_env_id(history: dict | None) -> str:
    # Prefer top-level env_id (persisted at session start, always available).
    # Fall back to iteration config.json for backward compatibility.
    if history is not None and history.get("env_id") is not None:
        return str(history["env_id"])
    return str(_get_session_config_field(history, "env_id", ""))


def _discover_in_progress_iterations(
    session_dir: Path,
    known_iterations: list[dict],
) -> list[dict]:
    """Discover iter_N/ directories on disk that aren't yet in loop_history.

    Returns synthetic iteration dicts for in-progress iterations so the
    frontend can show them while training is still running.
    """
    known_nums = {it.get("iteration", 0) for it in known_iterations}
    extra: list[dict] = []
    for d in sorted(session_dir.iterdir()):
        m = re.match(r"iter_(\d+)$", d.name)
        if not m or not d.is_dir():
            continue
        iter_num = int(m.group(1))
        if iter_num in known_nums:
            continue
        # Build a minimal iteration dict from what's on disk
        # Try to read latest streaming judgment for in-progress data
        best_sj: dict = {}
        for sj_dir in sorted(d.glob("*/streaming_judgments")):
            for sj_file in sorted(sj_dir.glob("*.json"), reverse=True):
                try:
                    best_sj = json.loads(sj_file.read_text())
                    break
                except Exception:
                    pass
            if best_sj:
                break
        # Strip rollout_judgments / checkpoint_judgments so that
        # _iteration_to_summary() skips the top-level rollout_judgments
        # fallback (which produces wrong "live"-keyed scores) and instead
        # falls through to the streaming-judgment file reader (which uses
        # correct step-based keys matching video filenames).
        sj_metadata = {
            k: v
            for k, v in best_sj.items()
            if k not in ("rollout_judgments", "checkpoint_judgments")
        }
        it_dict: dict = {
            "iteration": iter_num,
            "iteration_dir": str(d),
            "judgment": sj_metadata,
            "summary": {},
            "reward_code": "",
        }
        # Try to read reward_fn.py if it exists
        reward_fn = d / "reward_fn.py"
        if reward_fn.exists():
            try:
                it_dict["reward_code"] = reward_fn.read_text()
            except OSError:
                pass
        extra.append(it_dict)
    return extra


def get_session_config(session_id: str) -> SessionConfig | None:
    """Return the saved start config for a session, or None if not found."""
    sr = _get_session_record(session_id)
    return sr.read_session_config()


def get_session(session_id: str) -> SessionDetail | None:
    sr = _get_session_record(session_id)
    status_data = sr.read_status()
    history = sr.read_history()
    session_dir = resolve_session_dir(session_id)
    created_at = _get_session_created_at(session_dir)

    stale = is_stale(status_data, session_id)
    result: SessionDetail | None = None

    if history is None:
        # Session dir exists but loop_history.json not written yet — the
        # subprocess is likely still generating the initial reward function.
        # Use status.json to detect early running state; fall back to
        # checking session dir existence.
        iterations_on_disk = (
            _discover_in_progress_iterations(session_dir, []) if session_dir.exists() else []
        )
        # Fall back to session_config.json for prompt when loop_history is missing
        prompt = ""
        config_path = session_dir / "session_config.json"
        if config_path.exists():
            try:
                import json

                cfg = json.loads(config_path.read_text())
                prompt = cfg.get("prompt", "")
            except Exception:
                pass
        if status_data:
            result = SessionDetail(
                session_id=session_id,
                prompt=prompt,
                status=status_data["status"],
                best_iteration=0,
                best_score=0.0,
                iterations=[_iteration_to_summary(it) for it in iterations_on_disk],
                created_at=created_at,
                is_stale=stale,
            )
        elif session_dir.exists():
            result = SessionDetail(
                session_id=session_id,
                prompt=prompt,
                status="running",
                best_iteration=0,
                best_score=0.0,
                iterations=[_iteration_to_summary(it) for it in iterations_on_disk],
                created_at=created_at,
            )
    else:
        # Enrich diff summaries in background (non-blocking)
        _schedule_enrichment(session_id)

        # status.json is the single source of truth; fall back to loop_history
        status = status_data["status"] if status_data else history.get("status", "running")
        env_id = _get_session_env_id(history)
        total_ts = int(_get_session_config_field(history, "total_timesteps", 0))

        known_iterations = history.get("iterations", [])
        # Discover in-progress iterations not yet written to loop_history
        extra = _discover_in_progress_iterations(session_dir, known_iterations)
        all_iterations = known_iterations + extra

        # Fallback: if history had no iterations yet, scan discovered ones
        if total_ts == 0 and extra:
            fake_history = {"iterations": extra}
            total_ts = int(_get_session_config_field(fake_history, "total_timesteps", 0))
            if not env_id:
                env_id = str(_get_session_config_field(fake_history, "env_id", ""))

        result = SessionDetail(
            session_id=history["session_id"],
            prompt=history.get("prompt", ""),
            status=status,
            best_iteration=history.get("best_iteration", 0),
            best_score=history.get("best_score", 0.0),
            iterations=[_iteration_to_summary(it) for it in all_iterations],
            error=history.get("error"),
            env_id=env_id,
            created_at=created_at,
            total_timesteps=total_ts,
            pass_threshold=float(history.get("pass_threshold", 0.7)),
            is_stale=stale,
        )

    if result is not None:
        inject_metadata(result, session_dir)
    return result


def get_session_iterations(session_id: str) -> list[LoopIterationSummary] | None:
    session_dir = resolve_session_dir(session_id)
    history = _get_session_record(session_id).read_history()
    known = history.get("iterations", []) if history else []
    extra = _discover_in_progress_iterations(session_dir, known) if session_dir.exists() else []
    all_iterations = known + extra
    if not history and not all_iterations:
        return None
    return [_iteration_to_summary(it) for it in all_iterations]


def _list_session_lightweight(session_id: str) -> SessionDetail | None:
    """Build SessionDetail for listing without heavy side-effects.

    Skips _ensure_diff_summaries (which makes LLM API calls)
    to keep the list endpoint fast.
    """
    sr = _get_session_record(session_id)
    status_data = sr.read_status()
    history = sr.read_history()
    session_dir = resolve_session_dir(session_id)
    created_at = _get_session_created_at(session_dir)
    stale = is_stale(status_data, session_id)

    if history is None:
        if status_data:
            return SessionDetail(
                session_id=session_id,
                prompt="",
                status=status_data["status"],
                best_iteration=0,
                best_score=0.0,
                iterations=[],
                created_at=created_at,
                is_stale=stale,
            )
        if session_dir.exists():
            return SessionDetail(
                session_id=session_id,
                prompt="",
                status="running",
                best_iteration=0,
                best_score=0.0,
                iterations=[],
                created_at=created_at,
            )
        return None

    status = status_data["status"] if status_data else history.get("status", "running")
    env_id = _get_session_env_id(history)

    return SessionDetail(
        session_id=history["session_id"],
        prompt=history.get("prompt", ""),
        status=status,
        best_iteration=history.get("best_iteration", 0),
        best_score=history.get("best_score", 0.0),
        iterations=[
            _iteration_to_summary(it, lightweight=True) for it in history.get("iterations", [])
        ],
        error=history.get("error"),
        env_id=env_id,
        created_at=created_at,
        is_stale=stale,
    )


def list_sessions(include_deleted: bool = False) -> list[SessionDetail]:
    """List all sessions, sorted by created_at descending (newest first)."""
    if not RUNS_DIR.exists():
        return []
    sessions = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        if not _is_session_dir(d):
            continue
        if not include_deleted and is_entity_deleted(d):
            continue
        session = _list_session_lightweight(d.name)
        if session is not None:
            inject_metadata(session, d)
            sessions.append(session)
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions


# ---------------------------------------------------------------------------
# Iteration sub-runs (multi-config × seeds)
# ---------------------------------------------------------------------------


def get_iteration_runs(
    session_id: str,
    iter_num: int,
) -> list[IterationRunInfo] | None:
    """List sub-runs within a multi-config iteration.

    Scans ``iter_{iter_num}/`` for ``{config_id}_seed_{seed}/`` directories.
    """
    iter_dir = resolve_session_dir(session_id) / f"iter_{iter_num}"
    if not iter_dir.is_dir():
        return None

    pattern = re.compile(r"^(.+)_seed_(\d+)$")
    runs: list[IterationRunInfo] = []
    for d in sorted(iter_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        config_id, seed_str = m.group(1), m.group(2)
        rec = IterationRecord(d)
        summary = rec.read_summary()
        prefix = _static_prefix(rec)
        video_urls = [f"{prefix}/videos/{v}" for v in rec.video_filenames()]
        # Read intent_score from judgment or streaming judgments
        intent_score: float | None = None
        judgment_path = d / "judgment.json"
        if judgment_path.exists():
            jdata = read_json_safe(judgment_path)
            if jdata:
                intent_score = jdata.get("intent_score")
        if intent_score is None:
            sj_dir = d / "streaming_judgments"
            if sj_dir.is_dir():
                best_sj = -1.0
                for sj_file in sj_dir.glob("*.json"):
                    sj_data = read_json_safe(sj_file)
                    if sj_data and sj_data.get("intent_score") is not None:
                        best_sj = max(best_sj, sj_data["intent_score"])
                if best_sj >= 0:
                    intent_score = best_sj

        runs.append(
            IterationRunInfo(
                config_id=config_id,
                seed=int(seed_str),
                run_id=d.name,
                status=rec.derive_status(),
                final_return=summary.get("final_episodic_return") if summary else None,
                intent_score=intent_score,
                video_urls=video_urls,
            )
        )
    return runs


def get_run_metrics(
    session_id: str,
    iter_num: int,
    run_id: str,
) -> RunMetricsDetail | None:
    """Return training/eval metrics and video URLs for a single sub-run."""
    run_dir = resolve_session_dir(session_id) / f"iter_{iter_num}" / run_id
    if not run_dir.is_dir():
        return None

    rec = IterationRecord(run_dir)
    training, evaluation = rec.parse_scalars()
    run_config = rec.read_config()

    # Parse config_id and seed from run_id
    m = re.match(r"^(.+)_seed_(\d+)$", run_id)
    config_id = m.group(1) if m else run_id
    seed = int(m.group(2)) if m else 0

    prefix = _static_prefix(rec)
    video_urls = [f"{prefix}/videos/{v}" for v in rec.video_filenames()]

    # Per-checkpoint judgment scores and diagnoses (from run-level judgment.json)
    checkpoint_scores: dict[str, float] = {}
    checkpoint_diagnoses: dict[str, str] = {}
    checkpoint_code_diagnoses: dict[str, str] = {}
    checkpoint_vlm_diagnoses: dict[str, str] = {}
    checkpoint_code_scores: dict[str, float] = {}
    checkpoint_vlm_scores: dict[str, float] = {}
    rollout_scores: dict[str, float] = {}
    rollout_diagnoses: dict[str, str] = {}
    rollout_code_diagnoses: dict[str, str] = {}
    rollout_vlm_diagnoses: dict[str, str] = {}
    rollout_code_scores: dict[str, float] = {}
    rollout_vlm_scores: dict[str, float] = {}
    rollout_synthesis_traces: dict[str, list] = {}
    rollout_vlm_preview_filenames: dict[str, str] = {}
    best_checkpoint = ""
    intent_score: float | None = None
    diagnosis = ""

    def _extract_run_checkpoint(step: str, cj: dict) -> None:
        sc = cj.get("intent_score")
        if sc is not None:
            checkpoint_scores[step] = sc
        dg = cj.get("diagnosis", "")
        if dg:
            checkpoint_diagnoses[step] = dg
        code_diag = cj.get("code_diagnosis", "")
        vlm_diag = cj.get("vlm_diagnosis", "")
        sm = cj.get("scoring_method", "")
        if not code_diag and dg and "code_judge" in sm:
            code_diag = dg
        if code_diag:
            checkpoint_code_diagnoses[step] = code_diag
        if vlm_diag:
            checkpoint_vlm_diagnoses[step] = vlm_diag
        if cj.get("code_score") is not None:
            checkpoint_code_scores[step] = cj["code_score"]
        if cj.get("vlm_score") is not None:
            checkpoint_vlm_scores[step] = cj["vlm_score"]

    def _extract_run_rollouts(cj: dict, step: str) -> None:
        for rj in cj.get("rollout_judgments", []):
            label = rj.get("rollout_label")
            if label:
                key = f"{step}_{label}"
            else:
                key = f"{step}_ep{rj['episode_idx']}"
            rollout_scores[key] = rj.get("intent_score", 0.0)
            rollout_diagnoses[key] = rj.get("diagnosis", "")
            if rj.get("code_diagnosis"):
                rollout_code_diagnoses[key] = rj["code_diagnosis"]
            if rj.get("vlm_diagnosis"):
                rollout_vlm_diagnoses[key] = rj["vlm_diagnosis"]
            if rj.get("code_score") is not None:
                rollout_code_scores[key] = rj["code_score"]
            if rj.get("vlm_score") is not None:
                rollout_vlm_scores[key] = rj["vlm_score"]
            if rj.get("synthesis_tool_calls"):
                rollout_synthesis_traces[key] = rj["synthesis_tool_calls"]
            if rj.get("vlm_preview_filename"):
                rollout_vlm_preview_filenames[key] = rj["vlm_preview_filename"]
        if not cj.get("rollout_judgments") and cj.get("vlm_preview_filename"):
            rollout_vlm_preview_filenames[step] = cj["vlm_preview_filename"]

    judgment_path = run_dir / "judgment.json"
    if judgment_path.exists():
        jdata = read_json_safe(judgment_path)
        if jdata:
            intent_score = jdata.get("intent_score")
            best_checkpoint = jdata.get("best_checkpoint", "")
            diagnosis = jdata.get("diagnosis", "")
            for step, cj in jdata.get("checkpoint_judgments", {}).items():
                _extract_run_checkpoint(step, cj)
                _extract_run_rollouts(cj, step)

    # Fall back to streaming judgments
    sj_dir = run_dir / "streaming_judgments"
    if not checkpoint_scores and sj_dir.is_dir():
        for f in sj_dir.glob("*.json"):
            step = f.stem
            try:
                cj = read_json_safe(f)
                if cj:
                    _extract_run_checkpoint(step, cj)
                    _extract_run_rollouts(cj, step)
            except Exception:
                logger.debug("Failed to parse streaming judgment entry", exc_info=True)

    rollout_vlm_preview_urls = _resolve_preview_urls(video_urls, rollout_vlm_preview_filenames)

    return RunMetricsDetail(
        run_id=run_id,
        config_id=config_id,
        seed=seed,
        training=training,
        evaluation=evaluation,
        video_urls=video_urls,
        checkpoint_scores=checkpoint_scores,
        checkpoint_diagnoses=checkpoint_diagnoses,
        checkpoint_code_diagnoses=checkpoint_code_diagnoses,
        checkpoint_vlm_diagnoses=checkpoint_vlm_diagnoses,
        checkpoint_code_scores=checkpoint_code_scores,
        checkpoint_vlm_scores=checkpoint_vlm_scores,
        rollout_scores=rollout_scores,
        rollout_diagnoses=rollout_diagnoses,
        rollout_code_diagnoses=rollout_code_diagnoses,
        rollout_vlm_diagnoses=rollout_vlm_diagnoses,
        rollout_code_scores=rollout_code_scores,
        rollout_vlm_scores=rollout_vlm_scores,
        rollout_synthesis_traces=rollout_synthesis_traces,
        rollout_vlm_preview_urls=rollout_vlm_preview_urls,
        vlm_fps=VLM_FPS,
        best_checkpoint=best_checkpoint,
        intent_score=intent_score,
        diagnosis=diagnosis,
        config=run_config,
    )


def aggregate_scalars(
    seed_scalars: list[tuple[int, list[dict]]],
    config_id: str,
) -> AggregatedMetrics | None:
    """Compute mean±std metrics from per-seed scalar entries.

    Args:
        seed_scalars: List of (seed_number, training_entries) tuples.
        config_id: Identifier for this config group.

    Returns:
        AggregatedMetrics with interpolated mean/std, or None if no data.
    """
    if not seed_scalars:
        return None

    # Collect all metric keys (exclude non-numeric fields)
    skip_keys = {"global_step", "iteration", "type"}
    all_keys: set[str] = set()
    for _, entries in seed_scalars:
        for entry in entries:
            for k, v in entry.items():
                if k not in skip_keys and isinstance(v, int | float):
                    all_keys.add(k)

    if not all_keys:
        return None

    # Build union of all global_steps
    all_steps: set[int] = set()
    for _, entries in seed_scalars:
        for e in entries:
            step = e.get("global_step")
            if step is not None:
                all_steps.add(int(step))
    grid = sorted(all_steps)

    if not grid:
        return None

    # For each seed, interpolate each metric onto the grid
    # Then compute mean/std at each grid point using available seeds
    metric_arrays: dict[str, list[list[float | None]]] = {k: [] for k in all_keys}

    grid_arr = np.array(grid, dtype=np.float64)

    for _, entries in seed_scalars:
        if not entries:
            continue
        steps = [e.get("global_step", 0) for e in entries]
        max_step = max(steps) if steps else 0
        steps_arr = np.array(steps, dtype=np.float64)
        # Boolean mask: grid points within this seed's range
        mask = grid_arr <= max_step

        for key in all_keys:
            values = [float(e.get(key, 0) or 0) for e in entries]
            if len(steps) < 2:
                # Single data point: fill matching steps with the value
                val = values[0] if values else None
                interp_vals: list[float | None] = [val if m else None for m in mask]
                metric_arrays[key].append(interp_vals)
            else:
                # Vectorized interpolation over the entire grid at once
                values_arr = np.array(values, dtype=np.float64)
                full_interp = np.interp(grid_arr, steps_arr, values_arr)
                interp_vals = [
                    float(full_interp[i]) if mask[i] else None for i in range(len(grid))
                ]
                metric_arrays[key].append(interp_vals)

    # Compute mean/std at each step using available seeds
    result_metrics: dict[str, MeanStdSeries] = {}
    for key in sorted(all_keys):
        means: list[float] = []
        stds: list[float] = []
        for i in range(len(grid)):
            vals = [arr[i] for arr in metric_arrays[key] if arr[i] is not None]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(0.0)
                stds.append(0.0)
        result_metrics[key] = MeanStdSeries(mean=means, std=stds)

    # Downsample to max 200 points
    max_points = 200
    if len(grid) > max_points:
        step_size = len(grid) / max_points
        indices = [int(i * step_size) for i in range(max_points)]
        if indices[-1] != len(grid) - 1:
            indices.append(len(grid) - 1)
        grid = [grid[i] for i in indices]
        for key in result_metrics:
            result_metrics[key] = MeanStdSeries(
                mean=[result_metrics[key]["mean"][i] for i in indices],
                std=[result_metrics[key]["std"][i] for i in indices],
            )

    # Round after downsampling to avoid wasted work on discarded points
    for key in result_metrics:
        result_metrics[key] = MeanStdSeries(
            mean=[round_metric(v) for v in result_metrics[key]["mean"]],
            std=[round_metric(v) for v in result_metrics[key]["std"]],
        )

    return AggregatedMetrics(
        config_id=config_id,
        seeds=[s for s, _ in seed_scalars],
        available_metrics=sorted(all_keys),
        global_steps=grid,
        metrics=result_metrics,
    )


def get_aggregated_metrics(
    session_id: str,
    iter_num: int,
    config_id: str,
) -> AggregatedMetrics | None:
    """Compute mean±std metrics across seeds for a given config.

    Reads ``scalars.jsonl`` from each ``{config_id}_seed_*/`` directory,
    aligns by global_step, and computes per-step mean and std.
    Returns None if the iteration or config is not found.
    """
    iter_dir = resolve_session_dir(session_id) / f"iter_{iter_num}"
    if not iter_dir.is_dir():
        return None

    # Find all seed directories for this config
    pattern = re.compile(rf"^{re.escape(config_id)}_seed_(\d+)$")
    seed_dirs: list[tuple[int, Path]] = []
    for d in sorted(iter_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            seed_dirs.append((int(m.group(1)), d))

    if not seed_dirs:
        return None

    # Parse scalars from each seed — prefer training, fall back to eval
    seed_data: list[tuple[int, list[dict]]] = []
    for seed, d in seed_dirs:
        rec = IterationRecord(d)
        training, evaluation = rec.parse_scalars()
        if training:
            seed_data.append((seed, training))
        elif evaluation:
            # Flatten eval entries: extract reward_terms to top-level,
            # map total_reward → episodic_return for chart compatibility
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
            seed_data.append((seed, flat))

    return aggregate_scalars(seed_data, config_id)


# ---------------------------------------------------------------------------
# Environment presets
# ---------------------------------------------------------------------------


def _available_engines() -> frozenset[str]:
    """Return engines whose runtime is installed.

    MuJoCo is always available (hard dep). IsaacLab is available when
    the ``isaacsim`` pip package is installed (Isaac Sim 5.1+).
    Note: IsaacLab envs register with Gymnasium only after SimulationApp
    is initialized, which happens in training subprocesses, not the API.
    """
    engines: set[str] = {"mujoco"}
    try:
        import isaacsim  # noqa: F401

        engines.add("isaaclab")
    except ImportError:
        pass
    return frozenset(engines)


_ENGINES = _available_engines()

# IsaacLab envs that cannot run in our GPU-based pipeline.
# - SurfaceGripper envs: CPU-only (isaaclab/assets/surface_gripper raises on GPU)
# - PickPlace / GR1T2: blacklisted in IsaacLab __init__.py (pinocchio incompatibility)
_EXCLUDED_ENVS: frozenset[str] = frozenset(
    {
        # SurfaceGripper — CPU-only
        "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
        "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
        "Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0",
        # Blacklisted upstream — pinocchio incompatibility
        "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
        "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
        "Isaac-PickPlace-GR1T2-Abs-v0",
        "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
        "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
        "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
    }
)


def list_envs() -> list[EnvInfo]:
    """Return registered environment presets whose engine is installed.

    IsaacLab envs are only included when ``isaaclab`` is importable,
    preventing the dashboard from offering envs that would crash on launch.
    Envs in ``_EXCLUDED_ENVS`` are also filtered out (GPU-incompatible).
    """
    from p2p.training.hp_presets import get_preset

    result: list[EnvInfo] = []
    for spec in ENV_REGISTRY.values():
        if spec.engine not in _ENGINES:
            continue
        if spec.env_id in _EXCLUDED_ENVS:
            continue
        preset = get_preset(spec.env_id) or {}
        result.append(
            EnvInfo(
                env_id=spec.env_id,
                name=spec.name,
                obs_dim=spec.obs_dim,
                action_dim=spec.action_dim,
                info_keys=spec.info_keys,
                description=spec.description,
                engine=spec.engine,
                zoo_num_envs=preset.get("_zoo_n_envs", 1),
            )
        )
    return result


def get_resource_status() -> ResourceStatus:
    """Return current CPU resource status."""
    from p2p.training.cpu_manager import get_cpu_manager

    return get_cpu_manager().status()


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------


def get_cached_analysis(session_id: str) -> SessionAnalysis | None:
    """Return cached analysis.json for a session, or None."""
    sr = _get_session_record(session_id)
    return sr.read_analysis()


def run_analysis(
    session_id: str,
    on_status: Callable[[str], None] | None = None,
) -> SessionAnalysis:
    """Run agentic analysis and cache the result."""
    from p2p.agents.session_analyzer import analyze_session

    client = get_client()
    result = analyze_session(
        session_id,
        client=client,
        runs_dir=RUNS_DIR,
        on_status=on_status,
    )

    # Cache to analysis.json
    sr = _get_session_record(session_id)
    sr.save_analysis(dict(result))

    return result


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


def list_events(session_id: str) -> list[dict]:
    """Read events.jsonl for a session, with truncated content."""
    from p2p.event_log import read_events

    session_dir = resolve_session_dir(session_id)
    if not session_dir.exists():
        return []
    return read_events(session_dir, truncate=True)


def get_event_detail(session_id: str, seq: int) -> EventDetailRecord | None:
    """Read a single event by sequence number (full content)."""
    from p2p.event_log import read_event_by_seq

    session_dir = resolve_session_dir(session_id)
    if not session_dir.exists():
        return None
    return read_event_by_seq(session_dir, seq)
