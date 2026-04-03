"""Tests for SimulatorBackend protocol and MuJoCoBackend (#221)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from p2p.config import TrainConfig
from p2p.training.env_spec import EnvSpec
from p2p.training.simulator import MuJoCoBackend, SimulatorBackend, get_simulator


class TestSimulatorProtocol:
    """SimulatorBackend protocol conformance."""

    def test_mujoco_backend_satisfies_protocol(self) -> None:
        backend = MuJoCoBackend()
        assert isinstance(backend, SimulatorBackend)

    def test_mujoco_engine_name(self) -> None:
        backend = MuJoCoBackend()
        assert backend.engine_name == "mujoco"


class TestGetSimulator:
    """Factory function tests."""

    def test_get_mujoco(self) -> None:
        backend = get_simulator("mujoco")
        assert isinstance(backend, MuJoCoBackend)
        assert backend.engine_name == "mujoco"

    def test_get_default(self) -> None:
        backend = get_simulator()
        assert isinstance(backend, MuJoCoBackend)

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown simulator engine 'unknown'"):
            get_simulator("unknown")


class TestTrainConfigEngine:
    """TrainConfig engine field round-trip."""

    def test_default_engine(self) -> None:
        tc = TrainConfig()
        assert tc.engine == "mujoco"

    def test_custom_engine_roundtrip(self) -> None:
        tc = TrainConfig(engine="isaaclab")
        assert tc.engine == "isaaclab"
        # JSON round-trip
        restored = TrainConfig.from_json(tc.to_json())
        assert restored.engine == "isaaclab"

    def test_engine_not_tunable(self) -> None:
        """Engine must not be in _TUNABLE_KEYS -- revise agent cannot change it."""
        assert "engine" not in TrainConfig._TUNABLE_KEYS


class TestEnvSpecEngine:
    """EnvSpec engine field."""

    def test_default_engine(self) -> None:
        spec = EnvSpec(
            env_id="Test-v0",
            name="Test",
            obs_dim=4,
            action_dim=2,
            info_keys={},
            description="test",
        )
        assert spec.engine == "mujoco"

    def test_custom_engine(self) -> None:
        spec = EnvSpec(
            env_id="Test-v0",
            name="Test",
            obs_dim=4,
            action_dim=2,
            info_keys={},
            description="test",
            engine="isaaclab",
        )
        assert spec.engine == "isaaclab"


class TestMuJoCoBackendMethods:
    """MuJoCoBackend method behavior."""

    def test_has_physics_state_true(self) -> None:
        backend = MuJoCoBackend()
        env = MagicMock(data=MagicMock())
        assert backend.has_physics_state(env) is True

    def test_has_physics_state_false(self) -> None:
        backend = MuJoCoBackend()
        env = MagicMock(spec=[])  # no 'data' attribute
        assert backend.has_physics_state(env) is False

    def test_extract_side_info(self) -> None:
        backend = MuJoCoBackend()
        env = MagicMock()
        result = backend.extract_side_info(env)
        assert "mj_data" in result
        assert "mj_model" in result
        assert result["mj_data"] is env.data
        assert result["mj_model"] is env.model

    def test_inject_reward_namespace(self) -> None:
        backend = MuJoCoBackend()
        ns: dict = {}
        backend.inject_reward_namespace(ns)
        # numpy should always be injected
        assert "np" not in ns  # inject only adds mujoco, caller pre-adds np
        # mujoco may or may not be available, but shouldn't raise
        if "mujoco" in ns:
            import mujoco

            assert ns["mujoco"] is mujoco

    @patch("p2p.training.env_spec.extract_mujoco_body_info", return_value="body layout")
    def test_extract_body_info_delegates(self, mock_fn: MagicMock) -> None:
        backend = MuJoCoBackend()
        result = backend.extract_body_info("HalfCheetah-v5")
        assert result == "body layout"
        mock_fn.assert_called_once_with("HalfCheetah-v5")

    @patch("p2p.training.env_spec.extract_joint_semantics", return_value="joint semantics")
    def test_extract_joint_semantics_delegates(self, mock_fn: MagicMock) -> None:
        backend = MuJoCoBackend()
        result = backend.extract_joint_semantics("HalfCheetah-v5")
        assert result == "joint semantics"
        mock_fn.assert_called_once_with("HalfCheetah-v5")
