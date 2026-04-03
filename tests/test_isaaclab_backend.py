"""Tests for IsaacLabBackend (#221, Phase 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from p2p.config import TrainConfig
from p2p.training.env_spec import ENV_REGISTRY
from p2p.training.isaaclab_backend import IsaacLabBackend
from p2p.training.simulator import SimulatorBackend, get_simulator


class TestIsaacLabProtocol:
    """IsaacLabBackend protocol conformance."""

    def test_satisfies_protocol(self) -> None:
        backend = IsaacLabBackend()
        assert isinstance(backend, SimulatorBackend)

    def test_engine_name(self) -> None:
        backend = IsaacLabBackend()
        assert backend.engine_name == "isaaclab"


class TestGetSimulator:
    """Factory function returns IsaacLabBackend for 'isaaclab'."""

    def test_get_isaaclab(self) -> None:
        backend = get_simulator("isaaclab")
        assert isinstance(backend, IsaacLabBackend)
        assert backend.engine_name == "isaaclab"


class TestHasPhysicsState:
    """has_physics_state detection."""

    def test_with_scene_and_robot(self) -> None:
        backend = IsaacLabBackend()
        env = MagicMock()
        env.scene = {"robot": MagicMock()}
        assert backend.has_physics_state(env) is True

    def test_without_scene(self) -> None:
        backend = IsaacLabBackend()
        env = MagicMock(spec=[])  # no 'scene' attribute
        assert backend.has_physics_state(env) is False

    def test_scene_without_robot(self) -> None:
        backend = IsaacLabBackend()
        env = MagicMock()
        env.scene = {"gripper": MagicMock()}
        assert backend.has_physics_state(env) is False


class TestExtractSideInfo:
    """extract_side_info returns robot_data."""

    def test_returns_robot_data(self) -> None:
        backend = IsaacLabBackend()
        robot_mock = MagicMock()
        env = MagicMock()
        env.scene = {"robot": robot_mock}
        result = backend.extract_side_info(env)
        assert "robot_data" in result
        assert result["robot_data"] is robot_mock.data


class TestBuildTrajectoryFields:
    """build_trajectory_fields extracts IsaacLab state."""

    def _make_mock_env(self, *, with_object: bool = False) -> MagicMock:
        """Build a mock env with IsaacLab-like batched tensor data.

        Args:
            with_object: If True, add manipulated object attributes
                (object_pos, object_rot, goal_pos, goal_rot) to simulate
                a dexterous manipulation environment.
        """
        import torch

        data = MagicMock()
        data.joint_pos = torch.tensor([[0.1, 0.2, 0.3]])
        data.joint_vel = torch.tensor([[1.0, 2.0, 3.0]])
        data.body_pos_w = torch.tensor([[[0.0, 0.0, 0.5], [0.1, 0.0, 0.3]]])
        data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        data.root_pos_w = torch.tensor([[1.0, 2.0, 0.5]])
        data.root_quat_w = torch.tensor([[0.707, 0.707, 0.0, 0.0]])
        data.applied_torque = torch.tensor([[0.5, -0.5, 0.1]])
        data.body_acc_w = torch.tensor([[[0.0, 0.0, -9.8], [0.1, 0.0, 0.0]]])

        robot = MagicMock()
        robot.data = data

        scene = MagicMock()
        scene.__getitem__ = lambda self, key: robot if key == "robot" else None
        scene.env_origins = torch.tensor([[0.0, 0.0, 0.0]])

        env = MagicMock(spec=["scene"])
        env.scene = scene

        if with_object:
            # Pen at (0.0, -0.39, 0.6) in local frame
            env.object_pos = torch.tensor([[0.0, -0.39, 0.6]])
            env.object_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            env.goal_pos = torch.tensor([[-0.2, -0.45, 0.68]])
            env.goal_rot = torch.tensor([[0.707, 0.707, 0.0, 0.0]])

        return env

    def test_fields_present(self) -> None:
        backend = IsaacLabBackend()
        env = self._make_mock_env()
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Velocity-Flat-Anymal-C-v0",
            [0.1, 0.2],
            {},
        )
        assert "joint_pos" in fields
        assert "joint_vel" in fields
        assert "body_pos_w" in fields
        assert "body_quat_w" in fields
        assert "root_pos_w" in fields
        assert "root_quat_w" in fields
        assert "control_cost" in fields
        assert "applied_torque" in fields
        assert "body_acc_w" in fields

    def test_root_pos_from_root_pos_w(self) -> None:
        backend = IsaacLabBackend()
        env = self._make_mock_env()
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Velocity-Flat-Anymal-C-v0",
            [0.0],
            {},
        )
        assert abs(fields["root_pos_w"][2] - 0.5) < 1e-5

    def test_root_quat_from_root_quat_w(self) -> None:
        backend = IsaacLabBackend()
        env = self._make_mock_env()
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Velocity-Flat-Anymal-C-v0",
            [0.0],
            {},
        )
        # Mock sets root_quat_w to [0.707, 0.707, 0.0, 0.0] (90-deg pitch)
        assert len(fields["root_quat_w"]) == 4
        assert abs(fields["root_quat_w"][0] - 0.707) < 1e-3

    def test_no_object_fields_for_locomotion(self) -> None:
        """Locomotion envs (no manipulated object) should not have object fields."""
        backend = IsaacLabBackend()
        env = self._make_mock_env(with_object=False)
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Velocity-Flat-Anymal-C-v0",
            [0.0],
            {},
        )
        assert "object_pos" not in fields
        assert "object_rot" not in fields
        assert "goal_pos" not in fields
        assert "goal_rot" not in fields

    def test_object_fields_for_dexterous(self) -> None:
        """Dexterous envs record manipulated object position and orientation."""
        backend = IsaacLabBackend()
        env = self._make_mock_env(with_object=True)
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Spin-Pen-Shadow-Direct-v0",
            [0.0] * 20,
            {},
        )
        assert "object_pos" in fields
        assert "object_rot" in fields
        assert "goal_pos" in fields
        assert "goal_rot" in fields
        # Verify values (env_origins is [0,0,0] so world == local)
        assert len(fields["object_pos"]) == 3
        assert abs(fields["object_pos"][2] - 0.6) < 1e-5
        assert len(fields["object_rot"]) == 4
        assert abs(fields["object_rot"][0] - 1.0) < 1e-5

    def test_object_pos_world_frame_with_origin_offset(self) -> None:
        """Object positions are converted to world frame using env_origins."""
        import torch

        backend = IsaacLabBackend()
        env = self._make_mock_env(with_object=True)
        # Shift env_origins to test frame conversion
        env.scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
        fields = backend.build_trajectory_fields(
            env,
            "Isaac-Spin-Pen-Shadow-Direct-v0",
            [0.0] * 20,
            {},
        )
        # object_pos local = [0.0, -0.39, 0.6], origin = [10, 20, 0]
        assert abs(fields["object_pos"][0] - 10.0) < 1e-5
        assert abs(fields["object_pos"][1] - 19.61) < 1e-5
        assert abs(fields["object_pos"][2] - 0.6) < 1e-5
        # Rotations should NOT have origin offset
        assert abs(fields["object_rot"][0] - 1.0) < 1e-5


class TestInjectRewardNamespace:
    """inject_reward_namespace adds torch (callers pre-seed numpy)."""

    def test_numpy_not_injected_by_backend(self) -> None:
        """Numpy is the caller's responsibility, not the backend's."""
        backend = IsaacLabBackend()
        ns: dict = {}
        backend.inject_reward_namespace(ns)
        assert "np" not in ns  # callers pre-seed this

    def test_torch_injected(self) -> None:
        backend = IsaacLabBackend()
        ns: dict = {"np": np, "numpy": np}  # callers pre-seed numpy
        backend.inject_reward_namespace(ns)
        # torch should be available in our test environment
        import torch

        assert ns["torch"] is torch


class TestExtractBodyInfo:
    """extract_body_info returns static layout for IsaacLab envs."""

    def test_known_env(self) -> None:
        backend = IsaacLabBackend()
        info = backend.extract_body_info("Isaac-Velocity-Flat-Anymal-C-v0")
        assert "joint_pos" in info
        assert "body_pos_w" in info
        assert "resting" in info.lower()  # resting height info

    def test_unknown_env(self) -> None:
        backend = IsaacLabBackend()
        info = backend.extract_body_info("NonExistent-v0")
        assert "No body info" in info


class TestJointSemantics:
    """Joint semantics for IsaacLab."""

    def test_extract_joint_semantics(self) -> None:
        backend = IsaacLabBackend()
        info = backend.extract_joint_semantics("Isaac-Velocity-Flat-Anymal-C-v0")
        assert info is not None
        assert "right-hand rule" in info
        assert "root_quat_w" in info


class TestEnvSpecRegistry:
    """IsaacLab env specs in ENV_REGISTRY."""

    @pytest.mark.parametrize(
        "env_id",
        [
            "Isaac-Velocity-Flat-Anymal-C-v0",
            "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "Isaac-Reach-Franka-v0",
        ],
    )
    def test_registered(self, env_id: str) -> None:
        spec = ENV_REGISTRY[env_id]
        assert spec.engine == "isaaclab"
        assert spec.env_id == env_id

    def test_anymal_spec_details(self) -> None:
        spec = ENV_REGISTRY["Isaac-Velocity-Flat-Anymal-C-v0"]
        assert "Anymal" in spec.name
        assert spec.engine == "isaaclab"
        assert spec.uses_quaternion_orientation is True

    def test_franka_spec_details(self) -> None:
        spec = ENV_REGISTRY["Isaac-Reach-Franka-v0"]
        assert "Franka" in spec.name
        assert spec.engine == "isaaclab"


class TestBodyInfoSceneData:
    """Verify scene/object data visibility in extract_body_info output.

    Ensures that existing envs with scene data (cabinet, dexterous object)
    still include it, and locomotion envs never get scene sections.
    """

    @pytest.mark.parametrize(
        "env_id",
        [
            "Isaac-Franka-Cabinet-Direct-v0",
            "Isaac-Open-Drawer-Franka-v0",
            "Isaac-Repose-Cube-Shadow-Direct-v0",
        ],
    )
    def test_existing_scene_data_preserved(self, env_id: str) -> None:
        backend = IsaacLabBackend()
        info = backend.extract_body_info(env_id)
        assert "Scene object:" in info, f"{env_id} lost its scene section"

    @pytest.mark.parametrize(
        "env_id",
        [
            "Isaac-Velocity-Flat-Anymal-C-v0",
            "Isaac-Ant-Direct-v0",
            "Isaac-Reach-Franka-v0",
        ],
    )
    def test_no_scene_for_locomotion(self, env_id: str) -> None:
        backend = IsaacLabBackend()
        info = backend.extract_body_info(env_id)
        assert "Scene object:" not in info
        assert "Manipulated object" not in info

    def test_env_object_attrs_rendering(self) -> None:
        """When env_object_attrs is in the cache, it should appear in output."""
        from unittest.mock import patch

        fake_cache = {
            "Isaac-Test-Stack-v0": {
                "joints": [{"index": 0, "name": "j0", "resting_pos": 0.0}],
                "bodies": [{"index": 0, "name": "b0", "x": 0.0, "y": 0.0, "z": 0.5}],
                "root_resting_z": 0.5,
                "source": "runtime_introspection",
                "env_object_attrs": {
                    "object_pos": [0.3, 0.0, 0.02],
                    "goal_pos": [0.3, 0.1, 0.05],
                },
            }
        }
        with patch("p2p.training.isaaclab_backend._load_body_cache", return_value=fake_cache):
            backend = IsaacLabBackend()
            info = backend.extract_body_info("Isaac-Test-Stack-v0")
        assert "Manipulated object(s)" in info
        assert "info['object_pos']" in info
        assert "info['goal_pos']" in info
        assert "step['object_pos']" in info


class TestHPPresets:
    """IsaacLab HP presets."""

    def test_anymal_preset_exists(self) -> None:
        from p2p.training.hp_presets import get_preset

        preset = get_preset("Isaac-Velocity-Flat-Anymal-C-v0")
        assert preset is not None
        assert preset["_zoo_n_envs"] == 4096
        assert preset["num_steps"] == 24

    def test_franka_preset_exists(self) -> None:
        from p2p.training.hp_presets import get_preset

        preset = get_preset("Isaac-Reach-Franka-v0")
        assert preset is not None

    def test_train_config_from_preset(self) -> None:
        tc = TrainConfig.from_preset(env_id="Isaac-Velocity-Flat-Anymal-C-v0", engine="isaaclab")
        assert tc.env_id == "Isaac-Velocity-Flat-Anymal-C-v0"
        assert tc.engine == "isaaclab"
