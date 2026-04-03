"""Tests for pure functions in sb3_trainer module."""

from __future__ import annotations

import numpy as np

from p2p.training.sb3_trainer import _round_floats, _tb_tag, build_trajectory_entry

# ---------------------------------------------------------------------------
# build_trajectory_entry
# ---------------------------------------------------------------------------


class _FakeEnvUnwrapped:
    """Minimal stand-in for an env.unwrapped with MuJoCo-like data."""


def _make_entry(
    *,
    step=0,
    reward=0.0,
    info=None,
    terminated=False,
    truncated=False,
    dt=0.01,
    env_id="HalfCheetah-v5",
    obs=None,
    action=None,
    next_obs=None,
):
    """Build a trajectory entry with sensible defaults to reduce test boilerplate."""
    return build_trajectory_entry(
        step=step,
        obs=obs if obs is not None else np.zeros(2),
        action=action if action is not None else np.zeros(1),
        next_obs=next_obs if next_obs is not None else np.zeros(2),
        reward=reward,
        info=info or {},
        terminated=terminated,
        truncated=truncated,
        dt=dt,
        env_id=env_id,
        env_unwrapped=_FakeEnvUnwrapped(),
    )


def test_build_trajectory_entry_basic_fields():
    """Entry contains all expected base fields."""
    entry = _make_entry(
        step=10,
        obs=np.array([1.0, 2.0]),
        action=np.array([0.5]),
        next_obs=np.array([1.1, 2.1]),
        reward=3.14,
        info={"x_position": 1.5, "x_velocity": 0.8},
        dt=0.05,
    )

    assert entry["step"] == 10
    assert entry["timestamp"] == 0.5  # 10 * 0.05
    assert entry["obs"] == [1.0, 2.0]
    assert entry["action"] == [0.5]
    assert entry["next_obs"] == [1.1, 2.1]
    assert entry["reward"] == 3.14
    assert entry["terminated"] is False
    assert entry["truncated"] is False


def test_build_trajectory_entry_reward_terms():
    """Reward terms from info are cast to float."""
    entry = _make_entry(
        reward=1.0,
        info={"reward_terms": {"forward": np.float32(2.5), "ctrl": np.float64(-0.1)}},
        terminated=True,
        env_id="Ant-v5",
    )

    assert entry["reward_terms"] == {"forward": 2.5, "ctrl": -0.1}
    assert all(isinstance(v, float) for v in entry["reward_terms"].values())


def test_build_trajectory_entry_missing_info_keys():
    """Missing reward_terms defaults to empty dict."""
    entry = _make_entry(truncated=True, env_id="Hopper-v5")

    assert entry["reward_terms"] == {}


# ---------------------------------------------------------------------------
# _round_floats
# ---------------------------------------------------------------------------


def test_round_floats_with_plain_float():
    assert _round_floats(3.14159, 2) == 3.14


def test_round_floats_with_numpy_floating():
    assert _round_floats(np.float64(3.14159), 2) == 3.14


def test_round_floats_with_nested_dict():
    data = {"a": 1.23456, "b": {"c": 9.87654}}
    result = _round_floats(data, 3)
    assert result == {"a": 1.235, "b": {"c": 9.877}}


def test_round_floats_with_list():
    result = _round_floats([1.111, 2.222, 3.333], 1)
    assert result == [1.1, 2.2, 3.3]


def test_round_floats_preserves_non_floats():
    data = {"s": "hello", "i": 42, "b": True, "n": None}
    result = _round_floats(data, 2)
    assert result == data


def test_round_floats_with_mixed_nested_structure():
    data = {"vals": [1.999, {"x": 0.001}], "tag": "ok"}
    result = _round_floats(data, 2)
    assert result == {"vals": [2.0, {"x": 0.0}], "tag": "ok"}


# ---------------------------------------------------------------------------
# _tb_tag
# ---------------------------------------------------------------------------


def test_tb_tag_known_key():
    assert _tb_tag("episodic_return") == "return/episodic_return"
    assert _tb_tag("policy_loss") == "loss/clip_loss"
    assert _tb_tag("entropy") == "loss/exploration_loss"


def test_tb_tag_reward_term_prefix():
    assert _tb_tag("reward_term_forward") == "reward_terms/forward"
    assert _tb_tag("reward_term_ctrl") == "reward_terms/ctrl"


def test_tb_tag_unknown_key():
    assert _tb_tag("some_random_metric") == "other/some_random_metric"
