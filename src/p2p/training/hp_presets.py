"""Per-environment HP presets.

MuJoCo presets from RL Baselines3 Zoo:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

IsaacLab presets from Isaac Lab examples:
  https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks

These provide well-tuned starting points so the LLM can focus on reward shaping
rather than hyperparameter search.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Zoo-tuned PPO hyperparameters (SB3 param names → TrainConfig field names)
#
# Field mapping:
#   Zoo YAML           → TrainConfig
#   n_steps            → num_steps
#   n_epochs           → update_epochs
#   clip_range         → clip_coef
#   batch_size         → (derived: num_envs * num_steps / num_minibatches)
#
# Note: batch_size in Zoo is a concrete number.  We convert it to
#       num_minibatches = (num_envs * num_steps) / batch_size.
# ---------------------------------------------------------------------------

_COMMON = {
    "normalize_obs": True,
    "normalize_reward": True,
    "net_arch": [256, 256],
    # False: let unhealthy states persist during training (matches API default).
    # MuJoCo envs default to True; we override for more permissive exploration.
    "terminate_when_unhealthy": False,
}

ZOO_PRESETS: dict[str, dict[str, Any]] = {
    # ── MuJoCo (10 tuned envs) ──────────────────────────────────────────
    #
    # HalfCheetah-v4  (batch_size=64, n_envs=1 → minibatches = 512/64 = 8)
    "HalfCheetah-v4": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 2.0633e-05,
        "num_steps": 512,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "update_epochs": 20,
        "clip_coef": 0.1,
        "ent_coef": 0.000401762,
        "vf_coef": 0.58096,
        "max_grad_norm": 0.8,
        "num_minibatches": 8,  # 512 / 64
    },
    # Ant-v4  (commented-out tuned block in Zoo; batch_size=32 → minibatches = 512/32 = 16)
    "Ant-v4": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 1.90609e-05,
        "num_steps": 512,
        "gamma": 0.98,
        "gae_lambda": 0.8,
        "update_epochs": 10,
        "clip_coef": 0.1,
        "ent_coef": 4.9646e-07,
        "vf_coef": 0.677239,
        "max_grad_norm": 0.6,
        "num_minibatches": 16,  # 512 / 32
    },
    # Hopper-v4  (batch_size=32 → minibatches = 512/32 = 16)
    "Hopper-v4": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 9.80828e-05,
        "num_steps": 512,
        "gamma": 0.999,
        "gae_lambda": 0.99,
        "update_epochs": 5,
        "clip_coef": 0.2,
        "ent_coef": 0.00229519,
        "vf_coef": 0.835671,
        "max_grad_norm": 0.7,
        "num_minibatches": 16,  # 512 / 32
    },
    # Walker2d-v4  (batch_size=32 → minibatches = 512/32 = 16)
    "Walker2d-v4": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 5.05041e-05,
        "num_steps": 512,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": 20,
        "clip_coef": 0.1,
        "ent_coef": 0.000585045,
        "vf_coef": 0.871923,
        "max_grad_norm": 1.0,
        "num_minibatches": 16,  # 512 / 32
    },
    # Humanoid-v4  (batch_size=256 → minibatches = 512/256 = 2)
    "Humanoid-v4": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 3.56987e-05,
        "num_steps": 512,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "update_epochs": 5,
        "clip_coef": 0.3,
        "ent_coef": 0.00238306,
        "vf_coef": 0.431892,
        "max_grad_norm": 2.0,
        "num_minibatches": 2,  # 512 / 256
    },
    # HumanoidStandup-v2  (batch_size=32 → minibatches = 512/32 = 16)
    "HumanoidStandup-v2": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 2.55673e-05,
        "num_steps": 512,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "update_epochs": 20,
        "clip_coef": 0.3,
        "ent_coef": 3.62109e-06,
        "vf_coef": 0.430793,
        "max_grad_norm": 0.7,
        "num_minibatches": 16,  # 512 / 32
    },
    # Swimmer-v4  (inherits mujoco-defaults + overrides; batch_size=256, n_envs=4
    #   → rollout = 4*1024 = 4096, minibatches = 4096/256 = 16)
    "Swimmer-v4": {
        **_COMMON,
        "_zoo_n_envs": 4,
        "learning_rate": 6e-04,
        "num_steps": 1024,
        "gamma": 0.9999,
        "gae_lambda": 0.98,
        "num_minibatches": 16,  # 4096 / 256 (with n_envs=4)
    },
    # Reacher-v2  (batch_size=32 → minibatches = 512/32 = 16)
    "Reacher-v2": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 0.000104019,
        "num_steps": 512,
        "gamma": 0.9,
        "gae_lambda": 1.0,
        "update_epochs": 5,
        "clip_coef": 0.3,
        "ent_coef": 7.52585e-08,
        "vf_coef": 0.950368,
        "max_grad_norm": 0.9,
        "num_minibatches": 16,  # 512 / 32
    },
    # InvertedPendulum-v2  (batch_size=64, n_steps=32 → minibatches = 32/64 → 1
    #   because batch_size > rollout for n_envs=1; Zoo runs with n_envs=1)
    "InvertedPendulum-v2": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 0.000222425,
        "num_steps": 32,
        "gamma": 0.999,
        "gae_lambda": 0.9,
        "update_epochs": 5,
        "clip_coef": 0.4,
        "ent_coef": 1.37976e-07,
        "vf_coef": 0.19816,
        "max_grad_norm": 0.3,
        "num_minibatches": 1,
    },
    # InvertedDoublePendulum-v2  (batch_size=512, n_steps=128 → minibatches = 128/512 → 1)
    "InvertedDoublePendulum-v2": {
        **_COMMON,
        "_zoo_n_envs": 1,
        "learning_rate": 0.000155454,
        "num_steps": 128,
        "gamma": 0.98,
        "gae_lambda": 0.8,
        "update_epochs": 10,
        "clip_coef": 0.4,
        "ent_coef": 1.05057e-06,
        "vf_coef": 0.695929,
        "max_grad_norm": 0.5,
        "num_minibatches": 1,
    },
}

# Merge auto-generated IsaacLab presets (from scripts/sync_isaaclab_envs.py)
try:
    from p2p.training._isaaclab_registry import ISAACLAB_PRESETS

    ZOO_PRESETS.update(ISAACLAB_PRESETS)
except ImportError:
    pass  # IsaacLab registry not generated yet

# Merge custom SAR IsaacLab presets (hand-maintained, not auto-generated)
from p2p.training._custom_isaaclab_registry import CUSTOM_ISAACLAB_PRESETS  # noqa: E402

ZOO_PRESETS.update(CUSTOM_ISAACLAB_PRESETS)

# v5 aliases (same dynamics, different Gymnasium API version)
for _base, _alias in [
    ("HalfCheetah-v4", "HalfCheetah-v5"),
    ("Ant-v4", "Ant-v5"),
    ("Hopper-v4", "Hopper-v5"),
    ("Walker2d-v4", "Walker2d-v5"),
    ("Humanoid-v4", "Humanoid-v5"),
    ("HumanoidStandup-v2", "HumanoidStandup-v4"),
    ("Swimmer-v4", "Swimmer-v5"),
    ("Reacher-v2", "Reacher-v4"),
    ("InvertedPendulum-v2", "InvertedPendulum-v4"),
    ("InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4"),
]:
    ZOO_PRESETS[_alias] = ZOO_PRESETS[_base]


def get_preset(env_id: str) -> dict[str, Any] | None:
    """Return Zoo-tuned HP dict for *env_id*, or ``None`` if not available."""
    return ZOO_PRESETS.get(env_id)


def available_envs() -> list[str]:
    """Return list of env_ids that have Zoo presets."""
    return sorted(ZOO_PRESETS.keys())
