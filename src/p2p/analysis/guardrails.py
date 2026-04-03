"""Guardrails for detecting reward hacking and training plateaus."""

from __future__ import annotations

import json
from pathlib import Path


def detect_reward_hacking(terms: dict[str, float]) -> str | None:
    """Warn if a single reward term dominates (>90% of total magnitude).

    Args:
        terms: Reward term name -> value mapping from a single step.

    Returns:
        Warning message if hacking detected, None otherwise.
    """
    if not terms:
        return None
    magnitudes = {k: abs(v) for k, v in terms.items()}
    total = sum(magnitudes.values())
    if total == 0:
        return None
    for name, mag in magnitudes.items():
        ratio = mag / total
        if ratio > 0.9:
            return (
                f"Reward hacking suspected: '{name}' accounts for "
                f"{ratio:.0%} of total reward magnitude"
            )
    return None


def check_training_plateau(
    scalars_path: str | Path,
    window: int = 100_000,
    min_improvement: float = 1.0,
) -> bool:
    """Check if training has plateaued over the last `window` steps.

    Compares mean episodic return in the first and second half of the
    window. Returns True if improvement is below `min_improvement`.

    Args:
        scalars_path: Path to metrics/scalars.jsonl.
        window: Number of recent timesteps to analyze.
        min_improvement: Minimum absolute improvement to not be a plateau.

    Returns:
        True if plateau detected, False otherwise.
    """
    scalars_path = Path(scalars_path)
    if not scalars_path.exists():
        return False

    entries = []
    for line in scalars_path.read_text().strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("type") == "eval":
            continue
        if "episodic_return" in entry:
            entries.append(entry)

    if len(entries) < 2:
        return False

    # Filter to last `window` steps
    max_step = entries[-1]["global_step"]
    cutoff = max_step - window
    recent = [e for e in entries if e["global_step"] > cutoff]

    if len(recent) < 2:
        return False

    mid = len(recent) // 2
    first_half = [e["episodic_return"] for e in recent[:mid]]
    second_half = [e["episodic_return"] for e in recent[mid:]]

    if not first_half or not second_half:
        return False

    mean_first = sum(first_half) / len(first_half)
    mean_second = sum(second_half) / len(second_half)

    return abs(mean_second - mean_first) < min_improvement
