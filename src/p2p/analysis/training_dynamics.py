"""Training dynamics analysis: pure data analysis extracted from revise_agent.

Computes summary statistics from scalars.jsonl and formats them for LLM consumption.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from p2p.config import TrainConfig
from p2p.contracts import TrainingDynamics


def _empty_dynamics() -> TrainingDynamics:
    """Return a TrainingDynamics with all fields at their zero/default values."""
    return TrainingDynamics(
        entropy_initial=0.0,
        entropy_final=0.0,
        entropy_trend="flat",
        entropy_decay_rate=0.0,
        entropy_too_fast=False,
        entropy_too_high=False,
        value_loss_initial=0.0,
        value_loss_final=0.0,
        value_loss_trend="flat",
        value_loss_stability=0.0,
        value_loss_diverging=False,
        policy_loss_initial=0.0,
        policy_loss_final=0.0,
        policy_loss_trend="flat",
        approx_kl_mean=0.0,
        approx_kl_max=0.0,
        approx_kl_spike_count=0,
        clip_fraction_mean=0.0,
        clip_fraction_trend="flat",
        explained_variance_final=0.0,
        explained_variance_good=False,
        episodic_return_trend="flat",
        episodic_return_final=0.0,
        episodic_return_max=0.0,
        episodic_return_converged=False,
        episodic_return_improvement_pct=0.0,
        sps_mean=0.0,
        reward_term_stats={},
        reward_term_avg_window=25,
        num_entries=0,
    )


# ---------------------------------------------------------------------------
# Math helpers (mirrors trajectory_metrics.py)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _compute_trend(values: list[float]) -> str:
    if len(values) < 4:
        return "flat"
    mid = len(values) // 2
    first_half = _mean(values[:mid])
    second_half = _mean(values[mid:])
    diff = second_half - first_half
    scale = max(abs(first_half), abs(second_half), 1e-6)
    ratio = diff / scale
    if ratio > 0.1:
        return "increasing"
    if ratio < -0.1:
        return "decreasing"
    return "flat"


# ---------------------------------------------------------------------------
# Training dynamics analysis
# ---------------------------------------------------------------------------


def analyze_training_curves(scalars_path: Path) -> TrainingDynamics:
    """Parse scalars.jsonl and compute per-metric summary statistics."""
    dyn = _empty_dynamics()

    if not scalars_path.exists():
        return dyn

    entries: list[dict] = []
    for line in scalars_path.read_text().strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("type") == "eval":
            continue
        entries.append(entry)

    if not entries:
        return dyn

    dyn["num_entries"] = len(entries)

    # Collect per-metric series
    entropy_vals = [e["entropy"] for e in entries if "entropy" in e]
    value_loss_vals = [e["value_loss"] for e in entries if "value_loss" in e]
    policy_loss_vals = [e["policy_loss"] for e in entries if "policy_loss" in e]
    approx_kl_vals = [e["approx_kl"] for e in entries if "approx_kl" in e]
    clip_frac_vals = [e["clip_fraction"] for e in entries if "clip_fraction" in e]
    expl_var_vals = [e["explained_variance"] for e in entries if "explained_variance" in e]
    ep_return_vals = [e["episodic_return"] for e in entries if "episodic_return" in e]
    sps_vals = [e["sps"] for e in entries if "sps" in e]

    # -- Entropy --
    if entropy_vals:
        dyn["entropy_initial"] = entropy_vals[0]
        dyn["entropy_final"] = entropy_vals[-1]
        dyn["entropy_trend"] = _compute_trend(entropy_vals)
        if abs(dyn["entropy_initial"]) > 1e-8:
            decay_rate = 1.0 - abs(dyn["entropy_final"]) / abs(dyn["entropy_initial"])
            dyn["entropy_decay_rate"] = decay_rate
            dyn["entropy_too_fast"] = decay_rate > 0.9
            dyn["entropy_too_high"] = decay_rate < 0.2

    # -- Value loss --
    if value_loss_vals:
        dyn["value_loss_initial"] = value_loss_vals[0]
        dyn["value_loss_final"] = value_loss_vals[-1]
        dyn["value_loss_trend"] = _compute_trend(value_loss_vals)
        # Stability: CV of last 20%
        tail_start = max(0, len(value_loss_vals) - len(value_loss_vals) // 5)
        tail = value_loss_vals[tail_start:]
        m = _mean(tail)
        if abs(m) > 1e-8:
            dyn["value_loss_stability"] = _std(tail) / abs(m)
        dyn["value_loss_diverging"] = (
            dyn["value_loss_trend"] == "increasing"
            and dyn["value_loss_final"] > dyn["value_loss_initial"] * 5
        )

    # -- Policy loss --
    if policy_loss_vals:
        dyn["policy_loss_initial"] = policy_loss_vals[0]
        dyn["policy_loss_final"] = policy_loss_vals[-1]
        dyn["policy_loss_trend"] = _compute_trend(policy_loss_vals)

    # -- KL divergence --
    if approx_kl_vals:
        dyn["approx_kl_mean"] = _mean(approx_kl_vals)
        dyn["approx_kl_max"] = max(approx_kl_vals)
        dyn["approx_kl_spike_count"] = sum(1 for v in approx_kl_vals if v > 0.02)

    # -- Clip fraction --
    if clip_frac_vals:
        dyn["clip_fraction_mean"] = _mean(clip_frac_vals)
        dyn["clip_fraction_trend"] = _compute_trend(clip_frac_vals)

    # -- Explained variance --
    if expl_var_vals:
        dyn["explained_variance_final"] = expl_var_vals[-1]
        dyn["explained_variance_good"] = expl_var_vals[-1] > 0.5

    # -- Episodic return --
    if ep_return_vals:
        dyn["episodic_return_trend"] = _compute_trend(ep_return_vals)
        dyn["episodic_return_final"] = ep_return_vals[-1]
        dyn["episodic_return_max"] = max(ep_return_vals)
        # Convergence: last 20% within 5% of max
        tail_start = max(0, len(ep_return_vals) - len(ep_return_vals) // 5)
        tail = ep_return_vals[tail_start:]
        tail_mean = _mean(tail)
        if abs(dyn["episodic_return_max"]) > 1e-8:
            dyn["episodic_return_converged"] = (
                abs(tail_mean - dyn["episodic_return_max"]) / abs(dyn["episodic_return_max"])
                < 0.05
            )
        # Improvement %
        if len(ep_return_vals) >= 2 and abs(ep_return_vals[0]) > 1e-8:
            dyn["episodic_return_improvement_pct"] = (
                (ep_return_vals[-1] - ep_return_vals[0]) / abs(ep_return_vals[0]) * 100
            )

    # -- SPS --
    if sps_vals:
        dyn["sps_mean"] = _mean(sps_vals)

    # -- Per-term reward metrics --
    # Collect series for keys starting with "reward_term/"
    term_series: dict[str, list[float]] = {}
    for e in entries:
        for key, val in e.items():
            if key.startswith("reward_term/") or key.startswith("reward_term_"):
                term_series.setdefault(key, []).append(float(val))

    if term_series:
        n = dyn["reward_term_avg_window"]
        reward_term_stats: dict[str, dict] = {}
        for key, vals in term_series.items():
            stats: dict[str, float | str] = {}
            stats["mean_last_n"] = _mean(vals[-n:])
            stats["mean_first_n"] = _mean(vals[:n])
            stats["final"] = vals[-1]
            stats["trend"] = _compute_trend(vals)
            reward_term_stats[key] = stats
        dyn["reward_term_stats"] = reward_term_stats

    return dyn


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_training_dynamics(dyn: TrainingDynamics) -> str:
    """Convert TrainingDynamics into human-readable text with WARNING annotations."""
    if dyn["num_entries"] == 0:
        return "No training dynamics data available."

    lines = ["## Training Dynamics Analysis", f"({dyn['num_entries']} log entries)", ""]

    # Entropy
    ent = f"{dyn['entropy_initial']:.4f} -> {dyn['entropy_final']:.4f}"
    lines.append(f"**Entropy**: {ent} ({dyn['entropy_trend']})")
    lines.append(f"  Decay rate: {dyn['entropy_decay_rate']:.0%}")
    if dyn["entropy_too_fast"]:
        lines.append("  WARNING: Entropy collapsed too fast (>90% decay). Policy may be stuck.")
    if dyn["entropy_too_high"]:
        lines.append(
            "  WARNING: Entropy barely decreased (<20% decay). Policy may not be learning."
        )

    # Value loss
    vl = f"{dyn['value_loss_initial']:.4f} -> {dyn['value_loss_final']:.4f}"
    lines.append(f"**Value loss**: {vl} ({dyn['value_loss_trend']})")
    lines.append(f"  Stability (CV last 20%): {dyn['value_loss_stability']:.2f}")
    if dyn["value_loss_diverging"]:
        lines.append(
            "  WARNING: Value loss is diverging. Consider lowering vf_coef or learning_rate."
        )

    # Policy loss
    pl = f"{dyn['policy_loss_initial']:.4f} -> {dyn['policy_loss_final']:.4f}"
    lines.append(f"**Policy loss**: {pl} ({dyn['policy_loss_trend']})")

    # KL
    kl = f"mean={dyn['approx_kl_mean']:.5f}, max={dyn['approx_kl_max']:.5f}"
    lines.append(f"**Approx KL**: {kl}")
    lines.append(f"  Spikes (>0.02): {dyn['approx_kl_spike_count']}")
    if dyn["approx_kl_spike_count"] > 5:
        lines.append(
            "  WARNING: Frequent KL spikes. Consider lowering learning_rate or adding target_kl."
        )

    # Clip fraction
    cf = f"mean={dyn['clip_fraction_mean']:.3f}"
    lines.append(f"**Clip fraction**: {cf} ({dyn['clip_fraction_trend']})")

    # Explained variance
    lines.append(f"**Explained variance**: {dyn['explained_variance_final']:.3f}")
    if not dyn["explained_variance_good"]:
        lines.append(
            "  WARNING: Low explained variance (<0.5). Value function is a poor predictor."
        )

    # Episodic return
    lines.append(
        f"**Episodic return**: final={dyn['episodic_return_final']:.1f}, "
        f"max={dyn['episodic_return_max']:.1f} ({dyn['episodic_return_trend']})"
    )
    lines.append(f"  Improvement: {dyn['episodic_return_improvement_pct']:+.0f}%")
    if dyn["episodic_return_converged"]:
        lines.append("  Converged (last 20% within 5% of max).")

    # SPS
    lines.append(f"**Throughput**: {dyn['sps_mean']:.0f} steps/sec")

    # Per-term reward metrics (episodic means from last-10-episode window)
    if dyn["reward_term_stats"]:
        lines.append("")
        n = dyn["reward_term_avg_window"]
        lines.append(f"**Per-Term Reward Means (window={n} episodes)**:")
        total = sum(abs(s["mean_last_n"]) for s in dyn["reward_term_stats"].values())
        for key, stats in sorted(dyn["reward_term_stats"].items()):
            trend = stats.get("trend", "flat")
            final = stats.get("final", 0.0)
            first_n = stats.get("mean_first_n", 0.0)
            last_n = stats.get("mean_last_n", 0.0)
            share = abs(last_n) / total * 100 if total > 1e-8 else 0
            lines.append(
                f"  {key}: mean_last={last_n:.4f}, "
                f"mean_first={first_n:.4f}, "
                f"final={final:.4f}, "
                f"trend={trend}, share={share:.0f}%"
            )

    return "\n".join(lines)


def _get_iter_field(it: object, name: str, default: object = "") -> object:
    """Get a field from an IterationData (object or dict)."""
    if hasattr(it, name):
        return getattr(it, name)
    if isinstance(it, dict):
        return it.get(name, default)
    return default


def _first_sentences(text: str, n: int = 2) -> str:
    """Return the first *n* sentences from *text*."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n])


def _extract_reward_formula(reward_code: str) -> str:
    """Extract LaTeX formula and per-term descriptions from reward code docstring."""
    from p2p.training.reward_loader import _parse_docstring

    latex, terms = _parse_docstring(lambda *a: None, reward_code)
    if not latex and not terms:
        return ""

    parts = []
    if latex:
        parts.append(f"  LaTeX: {latex}")
    if terms:
        parts.append("  Terms:")
        for term in terms:
            desc = term.get("description", "")
            parts.append(f"    {term['name']}: {desc}")
            term_latex = term.get("latex", "")
            if term_latex:
                parts.append(f"      {term_latex}")
    return "\n".join(parts)


def _extract_dynamics_summary(dynamics_text: str) -> str:
    """Extract a compact 1-line summary from the full training dynamics text."""
    if not dynamics_text:
        return ""
    parts: list[str] = []
    for line in dynamics_text.splitlines():
        line_s = line.strip()
        if line_s.startswith("**Entropy**:"):
            parts.append(line_s.removeprefix("**Entropy**: "))
        elif line_s.startswith("**Explained variance**:"):
            parts.append(f"expl_var={line_s.split(':', 1)[1].strip()}")
        elif line_s.startswith("**Episodic return**:"):
            parts.append(line_s.removeprefix("**Episodic return**: "))
    return " | ".join(parts) if parts else ""


def _format_iteration_block(
    it: object,
    *,
    best_iteration: int = 0,
    include_full_code: bool = False,
) -> str:
    """Format a single iteration into a thin text block.

    Includes: score, failure tags, HP changes, 1-sentence failure
    analysis. Omits reward formula (use tools for full code).
    When *include_full_code* is True, appends the full reward code
    (used only for the best iteration reference).
    """
    num = _get_iter_field(it, "iteration", "?")
    judgment = _get_iter_field(it, "judgment", {})
    hp_changes = _get_iter_field(it, "hp_changes", {})
    reward_code = _get_iter_field(it, "reward_code", "")

    score = judgment.get("intent_score", 0) if isinstance(judgment, dict) else 0
    judge_diagnosis = judgment.get("diagnosis", "") if isinstance(judgment, dict) else ""
    failure_tags = judgment.get("failure_tags", []) if isinstance(judgment, dict) else []

    marker = " (best)" if num == best_iteration else ""
    tags_str = ", ".join(failure_tags) if failure_tags else "none"
    hp_str = ", ".join(f"{k}={v}" for k, v in hp_changes.items()) if hp_changes else "none"

    config_judgments = judgment.get("config_judgments", {}) if isinstance(judgment, dict) else {}

    block = [f"### Iter {num}{marker} — score {score:.2f}"]
    block.append(f"- Failure tags: {tags_str}")
    block.append(f"- HP changes: {hp_str}")

    # Multi-config breakdown
    if config_judgments:
        for cid, cj in config_judgments.items():
            ms = cj.get("mean_intent_score", 0)
            ss = cj.get("score_std", 0)
            mr = cj.get("mean_final_return", 0)
            ctags = ", ".join(cj.get("common_failure_tags", [])) or "none"
            block.append(f"  - {cid}: score={ms:.2f}±{ss:.2f}, ret={mr:.0f}, tags=[{ctags}]")

    # Failure analysis (1-2 sentences)
    if judge_diagnosis:
        block.append(f"- Failure analysis: {_first_sentences(judge_diagnosis, 2)}")

    # Full code (only for best iteration section)
    if include_full_code and reward_code:
        block.append(f"\n```python\n{reward_code}\n```")

    return "\n".join(block)


def format_iteration_history(
    iterations: list,
    *,
    best_iteration: int = 0,
    best_score: float = 0.0,
) -> tuple[str, str]:
    """Format ALL iteration history and best iteration section.

    Returns (history_text, best_section_text).
    ``history_text`` contains every iteration in a thin format (score,
    failure tags, HP changes, 1-sentence failure
    analysis).  ``best_section_text`` contains the best iteration with
    full reward code.
    """
    if not iterations:
        return "No previous iterations.", ""

    # --- Score trend ---
    scores: list[float] = []
    for it in iterations:
        j = _get_iter_field(it, "judgment", {})
        s = j.get("intent_score", 0) if isinstance(j, dict) else 0
        scores.append(s)

    trend = " → ".join(f"{s:.2f}" for s in scores)

    # --- All iterations in thin format ---
    header = f"## Iteration History ({len(iterations)} iterations)"
    if best_iteration > 0:
        header += f"\nBest: iter {best_iteration}, score {best_score:.2f}"
    header += f"\nScore trend: {trend}"

    sections = [header]
    has_multi_config = False
    for it in iterations:
        sections.append(_format_iteration_block(it, best_iteration=best_iteration))
        j = _get_iter_field(it, "judgment", {})
        if isinstance(j, dict) and j.get("config_judgments"):
            has_multi_config = True

    # Tools hint
    tools_hint = "\nUse tools to inspect full reward code, training dynamics, or judgment."
    if has_multi_config:
        tools_hint += "\nUse get_config_comparison for detailed cross-config analysis."
    if best_iteration > 0:
        sections.append(tools_hint)

    history_text = "\n\n".join(sections)

    # --- Best iteration section (with full code) ---
    best_section = ""
    if best_iteration > 0:
        best_it = None
        for it in iterations:
            num = _get_iter_field(it, "iteration", None)
            if num == best_iteration:
                best_it = it
                break
        if best_it is not None:
            best_section = "## Best Iteration Reference\n" + _format_iteration_block(
                best_it,
                best_iteration=best_iteration,
                include_full_code=True,
            )

    return history_text, best_section


def format_current_config(config: TrainConfig) -> str:
    """List tunable HPs with current values and short explanations."""
    descriptions = {
        "learning_rate": "Adam learning rate",
        "ent_coef": "Entropy bonus. Higher = more exploration",
        "vf_coef": "Value function loss coefficient",
        "clip_coef": "PPO clip range. Lower = more conservative updates",
        "max_grad_norm": "Gradient clipping threshold",
        "gae_lambda": "GAE lambda. Higher = lower bias, higher variance",
        "gamma": "Discount factor. Higher = longer horizon",
        "num_steps": "Rollout length per env per update",
        "update_epochs": "PPO epochs per update. Higher = more sample reuse",
        "num_minibatches": "Number of minibatches per update",
        "target_kl": "Early-stop KL threshold (None = disabled)",
        "total_timesteps": "Total training steps",
        "normalize_obs": "Observation normalization",
        "normalize_reward": "Reward normalization",
        "reward_clip": "Reward clip range after normalization",
        "obs_clip": "Observation clip range after normalization",
    }

    lines = ["## Current Hyperparameters", ""]
    for key in sorted(TrainConfig._TUNABLE_KEYS):
        val = getattr(config, key, None)
        desc = descriptions.get(key, "")
        lines.append(f"  {key}: {val}  # {desc}")
    return "\n".join(lines)
