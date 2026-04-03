"""Tests for p2p.analysis.trajectory_metrics module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from p2p.analysis.trajectory_metrics import (
    _compute_reward_term_analysis,
    _compute_trend,
    _mean,
    _std,
    analyze_trajectory,
    load_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _step(
    *,
    reward: float = 1.0,
    reward_terms: dict | None = None,
) -> dict:
    """Build a single trajectory step dict for reward-term analysis tests."""
    d: dict = {
        "reward": reward,
    }
    if reward_terms is not None:
        d["reward_terms"] = reward_terms
    return d


def _make_trajectory(n: int = 100, **overrides) -> list[dict]:
    """Create a simple n-step trajectory with optional per-field overrides."""
    steps = []
    for i in range(n):
        kw = {k: (v[i] if isinstance(v, list) else v) for k, v in overrides.items()}
        steps.append(_step(**kw))
    return steps


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestMean:
    def test_normal_values(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_list(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([5.0]) == 5.0


class TestStd:
    def test_normal_values(self):
        # population std of [1, 2, 3] = sqrt(2/3) ~ 0.8165
        assert abs(_std([1.0, 2.0, 3.0]) - math.sqrt(2 / 3)) < 1e-9

    def test_empty_list(self):
        assert _std([]) == 0.0

    def test_single_value(self):
        assert _std([5.0]) == 0.0

    def test_identical_values(self):
        assert _std([3.0, 3.0, 3.0]) == 0.0


class TestComputeTrend:
    def test_increasing(self):
        assert _compute_trend([1.0, 1.0, 5.0, 5.0]) == "increasing"

    def test_decreasing(self):
        assert _compute_trend([5.0, 5.0, 1.0, 1.0]) == "decreasing"

    def test_flat(self):
        assert _compute_trend([3.0, 3.0, 3.0, 3.0]) == "flat"

    def test_too_few_values(self):
        assert _compute_trend([1.0, 2.0]) == "flat"

    def test_empty(self):
        assert _compute_trend([]) == "flat"


# ---------------------------------------------------------------------------
# load_trajectory
# ---------------------------------------------------------------------------


class TestLoadTrajectory:
    def test_loads_jsonl(self, tmp_path: Path):
        p = tmp_path / "traj.jsonl"
        lines = [json.dumps({"step": i, "reward": float(i)}) for i in range(3)]
        p.write_text("\n".join(lines))
        result = load_trajectory(p)
        assert len(result) == 3
        assert result[0]["step"] == 0
        assert result[2]["reward"] == 2.0

    def test_ignores_blank_lines(self, tmp_path: Path):
        p = tmp_path / "traj.jsonl"
        p.write_text('{"a":1}\n\n{"a":2}\n')
        result = load_trajectory(p)
        assert len(result) == 2

    def test_npz_round_trip(self, tmp_path: Path):
        """Save trajectory as NPZ via _save_trajectory, reload via load_trajectory."""
        from p2p.training.sb3_trainer import _save_trajectory

        original = [
            {
                "step": i,
                "timestamp": i * 0.02,
                "obs": [float(x) for x in range(17)],
                "action": [0.1, -0.2, 0.3, 0.0, 0.5, -0.1],
                "next_obs": [float(x) + 0.1 for x in range(17)],
                "reward": 1.5 + i,
                "reward_terms": {"forward": 1.0 + i, "alive": 0.5},
                "terminated": i == 4,
                "truncated": False,
                "qpos": [float(x) * 0.1 for x in range(9)],
                "qvel": [float(x) * 0.01 for x in range(9)],
                "control_cost": 0.05,
            }
            for i in range(5)
        ]
        p = tmp_path / "traj.npz"
        _save_trajectory(original, p)
        assert p.exists()

        loaded = load_trajectory(p)
        assert len(loaded) == 5

        # Scalars
        assert loaded[0]["step"] == 0
        assert loaded[4]["step"] == 4
        assert isinstance(loaded[0]["step"], int)
        assert abs(loaded[2]["reward"] - 3.5) < 0.01
        assert isinstance(loaded[0]["reward"], float)

        # Bools
        assert loaded[0]["terminated"] is False
        assert loaded[4]["terminated"] is True
        assert loaded[0]["truncated"] is False

        # Arrays
        assert len(loaded[0]["obs"]) == 17
        assert len(loaded[0]["action"]) == 6
        assert len(loaded[0]["qpos"]) == 9

        # reward_terms dict
        assert "forward" in loaded[0]["reward_terms"]
        assert "alive" in loaded[0]["reward_terms"]
        assert abs(loaded[0]["reward_terms"]["forward"] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# _compute_reward_term_analysis
# ---------------------------------------------------------------------------


class TestComputeRewardTermAnalysis:
    def test_basic_analysis(self):
        traj = [
            _step(reward_terms={"speed": 3.0, "alive": 1.0}),
            _step(reward_terms={"speed": 5.0, "alive": 1.0}),
            _step(reward_terms={"speed": 7.0, "alive": 1.0}),
            _step(reward_terms={"speed": 9.0, "alive": 1.0}),
        ]
        a = _compute_reward_term_analysis(traj)
        assert "speed" in a
        assert "alive" in a
        assert a["speed"]["mean"] == pytest.approx(6.0)
        assert a["alive"]["mean"] == pytest.approx(1.0)
        # fraction: |6|/(|6|+|1|) = 6/7
        assert a["speed"]["fraction_of_total"] == pytest.approx(6.0 / 7.0)
        assert a["speed"]["trend"] == "increasing"
        assert a["alive"]["trend"] == "flat"

    def test_empty_trajectory(self):
        assert _compute_reward_term_analysis([]) == {}

    def test_no_reward_terms(self):
        traj = [_step(), _step()]  # no reward_terms key
        assert _compute_reward_term_analysis(traj) == {}

    def test_all_zero_terms(self):
        traj = [
            _step(reward_terms={"a": 0.0, "b": 0.0}),
            _step(reward_terms={"a": 0.0, "b": 0.0}),
            _step(reward_terms={"a": 0.0, "b": 0.0}),
            _step(reward_terms={"a": 0.0, "b": 0.0}),
        ]
        a = _compute_reward_term_analysis(traj)
        assert a["a"]["fraction_of_total"] == 0.0
        assert a["b"]["fraction_of_total"] == 0.0


# ---------------------------------------------------------------------------
# analyze_trajectory (integration)
# ---------------------------------------------------------------------------


class TestAnalyzeTrajectory:
    def test_empty_trajectory(self):
        result = analyze_trajectory([])
        assert result == {}

    def test_normal_trajectory(self):
        traj = _make_trajectory(
            n=100,
            reward=1.5,
            reward_terms={"forward": 1.0, "alive": 0.5},
        )
        result = analyze_trajectory(traj)
        assert "forward" in result
        assert result["forward"]["mean"] == pytest.approx(1.0)
