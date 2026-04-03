"""Tests for guardrails module."""

from __future__ import annotations

import json
from pathlib import Path

from p2p.analysis.guardrails import (
    check_training_plateau,
    detect_reward_hacking,
)

# --- detect_reward_hacking ---


def test_detect_reward_hacking_no_dominance():
    terms = {"forward": 5.0, "energy": -3.0, "alive": 2.0}
    assert detect_reward_hacking(terms) is None


def test_detect_reward_hacking_single_term_dominates():
    terms = {"forward": 100.0, "energy": -0.5, "alive": 1.0}
    result = detect_reward_hacking(terms)
    assert result is not None
    assert "forward" in result


def test_detect_reward_hacking_empty_terms():
    assert detect_reward_hacking({}) is None


def test_detect_reward_hacking_all_zeros():
    assert detect_reward_hacking({"a": 0.0, "b": 0.0}) is None


def test_detect_reward_hacking_negative_dominance():
    terms = {"penalty": -50.0, "bonus": 0.1}
    result = detect_reward_hacking(terms)
    assert result is not None
    assert "penalty" in result


# --- check_training_plateau ---


def _write_scalars(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries))


def test_check_training_plateau_detects_flat_returns(tmp_path):
    scalars = tmp_path / "scalars.jsonl"
    entries = [
        {"global_step": i * 10_000, "iteration": i, "episodic_return": 100.0} for i in range(1, 21)
    ]
    _write_scalars(scalars, entries)
    assert check_training_plateau(scalars, window=200_000) is True


def test_check_training_plateau_not_triggered_when_improving(tmp_path):
    scalars = tmp_path / "scalars.jsonl"
    entries = [
        {"global_step": i * 10_000, "iteration": i, "episodic_return": float(i * 10)}
        for i in range(1, 21)
    ]
    _write_scalars(scalars, entries)
    assert check_training_plateau(scalars, window=200_000) is False


def test_check_training_plateau_missing_file(tmp_path):
    assert check_training_plateau(tmp_path / "missing.jsonl") is False


def test_check_training_plateau_ignores_eval_entries(tmp_path):
    scalars = tmp_path / "scalars.jsonl"
    entries = [
        {"global_step": i * 10_000, "iteration": i, "episodic_return": 100.0} for i in range(1, 11)
    ]
    entries.append({"global_step": 100_000, "type": "eval", "total_reward": 500.0})
    _write_scalars(scalars, entries)
    assert check_training_plateau(scalars, window=200_000) is True


def test_check_training_plateau_too_few_entries(tmp_path):
    scalars = tmp_path / "scalars.jsonl"
    _write_scalars(scalars, [{"global_step": 1000, "iteration": 1, "episodic_return": 50.0}])
    assert check_training_plateau(scalars) is False
