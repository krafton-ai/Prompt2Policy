"""Reward term analysis from trajectory data.

Reads trajectory NPZ (or legacy JSONL) and computes per-reward-term
breakdown (mean, std, fraction of total, trend).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_trajectory(path: Path) -> list[dict]:
    """Load trajectory into a list of step dicts.

    Supports both the current compressed NPZ format and the legacy
    JSONL format.  The returned structure is identical regardless of
    on-disk format so all consumers remain unchanged.
    """
    if path.suffix == ".npz":
        return _load_trajectory_npz(path)
    return _load_trajectory_jsonl(path)


def _load_trajectory_jsonl(path: Path) -> list[dict]:
    """Legacy JSONL loader."""
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def _load_trajectory_npz(path: Path) -> list[dict]:
    """Reconstruct list[dict] from columnar NPZ arrays."""
    data = np.load(str(path), allow_pickle=False)

    bool_fields = set(data["_bool_fields"].tolist()) if "_bool_fields" in data else set()
    int_fields = set(data["_int_fields"].tolist()) if "_int_fields" in data else set()
    term_names: list[str] | None = (
        data["_reward_term_names"].tolist() if "_reward_term_names" in data else None
    )

    fields = [k for k in data.files if not k.startswith("_")]
    if not fields:
        return []

    # Extract arrays once to avoid repeated NpzFile cache lookups
    columns = {key: data[key] for key in fields}
    n_steps = len(columns[fields[0]])

    entries: list[dict] = []
    for i in range(n_steps):
        entry: dict = {}
        for key, arr in columns.items():
            val = arr[i]
            if key == "reward_terms" and term_names is not None:
                entry["reward_terms"] = {name: float(v) for name, v in zip(term_names, val)}
            elif key in bool_fields:
                entry[key] = bool(val)
            elif key in int_fields:
                entry[key] = int(val)
            elif val.ndim == 0:
                entry[key] = float(val)
            else:
                entry[key] = val.tolist()
        entries.append(entry)

    return entries


def resolve_trajectory_path(directory: Path, stem: str) -> Path | None:
    """Find a trajectory file by *stem*, preferring .npz over legacy .jsonl."""
    for ext in (".npz", ".jsonl"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def analyze_trajectory(
    trajectory: list[dict],
) -> dict[str, Any]:
    """Compute per-reward-term statistics from trajectory data.

    Args:
        trajectory: List of per-step dicts from trajectory.jsonl.

    Returns:
        Dict mapping reward term names to stat dicts
        (mean, std, trend, fraction_of_total). Empty dict if no data.
    """
    if not trajectory:
        return {}

    return _compute_reward_term_analysis(trajectory)


# ---------------------------------------------------------------------------
# Internal computations
# ---------------------------------------------------------------------------


def _compute_reward_term_analysis(trajectory: list[dict]) -> dict[str, Any]:
    """Analyze each reward term: mean, std, fraction of total, trend."""
    if not trajectory:
        return {}

    # Collect all term values across steps
    all_terms: dict[str, list[float]] = {}
    for t in trajectory:
        terms = t.get("reward_terms", {})
        for name, val in terms.items():
            if name not in all_terms:
                all_terms[name] = []
            all_terms[name].append(float(val))

    if not all_terms:
        return {}

    # Compute stats per term
    analysis = {}
    means = {}
    for name, values in all_terms.items():
        m = _mean(values)
        means[name] = abs(m)
        analysis[name] = {
            "mean": m,
            "std": _std(values),
            "trend": _compute_trend(values),
        }

    # Compute fraction of total
    total_magnitude = sum(means.values())
    for name in analysis:
        if total_magnitude > 0:
            analysis[name]["fraction_of_total"] = means[name] / total_magnitude
        else:
            analysis[name]["fraction_of_total"] = 0.0

    return analysis


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _compute_trend(values: list[float]) -> str:
    """Simple trend detection: compare first-half mean to second-half mean."""
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
