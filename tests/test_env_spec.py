"""Tests for environment spec registry."""

from __future__ import annotations

import pytest

from p2p.training.env_spec import ENV_REGISTRY, get_env_spec


def test_get_env_spec_halfcheetah():
    spec = get_env_spec("HalfCheetah-v5")
    assert spec.name == "HalfCheetah"
    assert spec.obs_dim == 17
    assert spec.action_dim == 6


def test_get_env_spec_ant():
    spec = get_env_spec("Ant-v5")
    assert spec.obs_dim == 27
    assert spec.action_dim == 8


def test_get_env_spec_missing_raises_keyerror():
    with pytest.raises(KeyError):
        get_env_spec("NonExistent-v99")


def test_registry_has_mujoco_envs():
    """All 10 MuJoCo envs must be present."""
    mujoco_expected = {
        "HalfCheetah-v5",
        "Ant-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "Swimmer-v5",
        "Reacher-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
    }
    assert mujoco_expected.issubset(set(ENV_REGISTRY.keys()))


def test_registry_has_isaaclab_envs():
    """Auto-generated IsaacLab envs should be present."""
    isaac_envs = {eid for eid in ENV_REGISTRY if eid.startswith("Isaac-")}
    # At least the core locomotion envs
    assert "Isaac-Velocity-Flat-Anymal-C-v0" in isaac_envs
    assert "Isaac-Velocity-Flat-Unitree-Go2-v0" in isaac_envs
    assert "Isaac-Reach-Franka-v0" in isaac_envs
    assert len(isaac_envs) >= 50  # sync script generates ~90


def test_env_spec_is_frozen():
    spec = get_env_spec("HalfCheetah-v5")
    with pytest.raises(AttributeError):
        spec.obs_dim = 99  # type: ignore[misc]
