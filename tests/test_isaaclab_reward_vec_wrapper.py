"""Tests for IsaacLabRewardVecWrapper — batched _episode_start handling.

Reproduces #360: stateful reward functions that check
``info["_episode_start"]`` crashed with ``ValueError`` when the wrapper
passed a numpy array instead of a torch tensor.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from p2p.training.env import IsaacLabRewardVecWrapper

# ---------------------------------------------------------------------------
# Fixtures: minimal fake VecEnv
# ---------------------------------------------------------------------------


class _FakeVecEnv:
    """Minimal SB3-like VecEnv stub for testing the wrapper."""

    def __init__(self, num_envs: int = 4, obs_dim: int = 8, act_dim: int = 3) -> None:
        self.num_envs = num_envs
        self.observation_space = MagicMock()
        self.observation_space.shape = (obs_dim,)
        self.action_space = MagicMock()
        self.action_space.shape = (act_dim,)
        self.metadata: dict = {}
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        pass

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self._step_count += 1
        obs = np.random.randn(self.num_envs, self._obs_dim).astype(np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        # Env 0 done every 3 steps to test episode boundary handling
        dones = np.zeros(self.num_envs, dtype=bool)
        if self._step_count % 3 == 0:
            dones[0] = True
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if dones[0]:
            infos[0]["episode"] = {"r": 0.0, "l": self._step_count}
        return obs, rewards, dones, infos


@pytest.fixture()
def fake_venv() -> _FakeVecEnv:
    return _FakeVecEnv(num_envs=4, obs_dim=8, act_dim=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpisodeStartIsTensor:
    """Verify that _episode_start is passed as a torch.BoolTensor."""

    def test_episode_start_is_bool_tensor(self, fake_venv: _FakeVecEnv) -> None:
        """The reward function receives _episode_start as a torch.BoolTensor,
        not a numpy array."""
        received: dict[str, Any] = {}

        def capture_reward_fn(obs, action, next_obs, info):
            received["_episode_start"] = info["_episode_start"]
            r = torch.zeros(obs.shape[0])
            return r, {"dummy": r}

        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_fn=capture_reward_fn)
        wrapper._device = torch.device("cpu")  # no GPU needed for test
        wrapper.reset()
        wrapper.step(np.zeros((4, 3), dtype=np.float32))

        starts = received["_episode_start"]
        # Verify public tensor interface (dtype/shape delegated from _BoolBatch)
        assert starts.dtype == torch.bool
        assert starts.shape == (4,)
        # Verify __bool__ works (the whole point of the fix)
        assert bool(starts) is True  # all True after reset

    def test_episode_start_true_after_reset(self, fake_venv: _FakeVecEnv) -> None:
        """All envs should have _episode_start=True on the first step after reset."""
        received: dict[str, Any] = {}

        def capture_reward_fn(obs, action, next_obs, info):
            received["starts"] = info["_episode_start"].clone()
            r = torch.zeros(obs.shape[0])
            return r, {"dummy": r}

        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_fn=capture_reward_fn)
        wrapper._device = torch.device("cpu")
        wrapper.reset()
        wrapper.step(np.zeros((4, 3), dtype=np.float32))

        assert received["starts"].all(), "All envs should be episode_start=True after reset"

    def test_episode_start_per_env_after_done(self, fake_venv: _FakeVecEnv) -> None:
        """Only envs that were done on the previous step should have
        _episode_start=True on the next step."""
        starts_log: list[torch.Tensor] = []

        def logging_reward_fn(obs, action, next_obs, info):
            starts_log.append(info["_episode_start"].clone())
            r = torch.zeros(obs.shape[0])
            return r, {"dummy": r}

        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_fn=logging_reward_fn)
        wrapper._device = torch.device("cpu")
        wrapper.reset()

        actions = np.zeros((4, 3), dtype=np.float32)
        # Step 1: all starts (after reset)
        wrapper.step(actions)
        # Step 2: no dones from step 1 (step_count=1, not divisible by 3)
        wrapper.step(actions)
        # Step 3: env 0 done (step_count=3)
        wrapper.step(actions)
        # Step 4: env 0 should be episode_start (done on step 3)
        wrapper.step(actions)

        # Step 1: all True (after reset)
        assert starts_log[0].all()
        # Step 2: all False (no dones on step 1)
        assert not starts_log[1].any()
        # Step 3: all False (no dones on step 2)
        assert not starts_log[2].any()
        # Step 4: only env 0 True (done on step 3)
        assert starts_log[3][0].item() is True
        assert not starts_log[3][1:].any()


class TestStatefulRewardWithBatchedStarts:
    """Reproduce #360: stateful reward function using _episode_start."""

    STATEFUL_REWARD_CODE = """\
import torch

def _make_reward():
    state = {"prev_obs": None}

    def reward_fn(obs, action, next_obs, info):
        starts = info["_episode_start"]
        if starts.any():
            state["prev_obs"] = obs.clone()
            # Zero reward on reset steps
            r = torch.zeros(obs.shape[0])
            return r, {"delta": r}

        delta = torch.sum((obs - state["prev_obs"]) ** 2, dim=-1)
        state["prev_obs"] = obs.clone()
        return delta, {"delta": delta}

    return reward_fn

reward_fn = _make_reward()
"""

    def test_stateful_reward_produces_nonzero(self, fake_venv: _FakeVecEnv) -> None:
        """A stateful reward function using starts.any() should produce
        non-zero rewards (not silently crash to zeros)."""
        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_code=self.STATEFUL_REWARD_CODE)
        wrapper._device = torch.device("cpu")
        wrapper.reset()

        actions = np.zeros((4, 3), dtype=np.float32)
        # Step 1: episode start, reward = 0
        _, rewards1, _, _ = wrapper.step(actions)
        # Step 2: should produce non-zero delta (random obs changes)
        _, rewards2, _, _ = wrapper.step(actions)

        assert rewards2.sum() > 0, (
            f"Stateful reward should produce non-zero values on step 2, got all zeros: {rewards2}"
        )

    def test_old_scalar_pattern_works_via_boolbatch(self, fake_venv: _FakeVecEnv) -> None:
        """The OLD pattern `if info.get("_episode_start"):` now works
        because _BoolBatch delegates __bool__ to .any()."""
        code = """\
import torch

def reward_fn(obs, action, next_obs, info):
    # This pattern used to crash on multi-element tensors.
    # _BoolBatch makes it work transparently.
    if info.get("_episode_start"):
        pass
    r = torch.ones(obs.shape[0])
    return r, {"alive": r}
"""
        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_code=code)
        wrapper._device = torch.device("cpu")
        wrapper.reset()

        actions = np.zeros((4, 3), dtype=np.float32)
        _, rewards, _, _ = wrapper.step(actions)

        # Should produce non-zero rewards (no crash)
        assert (rewards == 1.0).all(), (
            f"Old scalar pattern should work via _BoolBatch, got {rewards}"
        )
        assert wrapper._reward_fn_err_count == 0


class TestErrorLogging:
    """Verify error counting and escalation behavior."""

    def test_error_counter_resets_on_success(self, fake_venv: _FakeVecEnv) -> None:
        call_count = 0

        def flaky_reward_fn(obs, action, next_obs, info):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient error")
            r = torch.ones(obs.shape[0])
            return r, {"alive": r}

        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_fn=flaky_reward_fn)
        wrapper._device = torch.device("cpu")
        wrapper.reset()

        actions = np.zeros((4, 3), dtype=np.float32)
        # Steps 1-2: errors
        wrapper.step(actions)
        wrapper.step(actions)
        assert wrapper._reward_fn_err_count == 2
        # Step 3: success — counter should reset
        wrapper.step(actions)
        assert wrapper._reward_fn_err_count == 0

    def test_persistent_error_escalates_to_error_log(
        self, fake_venv: _FakeVecEnv, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken_reward_fn(obs, action, next_obs, info):
            raise RuntimeError("always broken")

        wrapper = IsaacLabRewardVecWrapper(fake_venv, reward_fn=broken_reward_fn)
        wrapper._device = torch.device("cpu")
        wrapper.reset()

        actions = np.zeros((4, 3), dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            for _ in range(7):
                wrapper.step(actions)

        assert wrapper._reward_fn_err_count == 7
        error_msgs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("ZERO reward" in r.message for r in error_msgs), (
            "Should log ERROR about ZERO reward after 6 consecutive failures"
        )
