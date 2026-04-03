"""Simulator backend abstraction for multi-engine support (#221).

Defines the ``SimulatorBackend`` protocol and ``MuJoCoBackend`` implementation.
Callsites throughout the pipeline use ``get_simulator(spec.engine)`` to obtain
an engine-appropriate backend instead of calling MuJoCo-specific functions directly.

Design sketch::

    SimulatorBackend (Protocol)
      +-- MuJoCoBackend      (delegates to env_spec.py functions)
      +-- IsaacLabBackend    (isaaclab_backend.py)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SimulatorBackend(Protocol):
    """Protocol for simulator engine backends.

    Each backend encapsulates engine-specific operations: body introspection,
    trajectory field extraction, reward namespace injection, and rotation
    convention detection.  The loop and agents interact with this interface
    instead of calling MuJoCo-specific functions directly.
    """

    @property
    def engine_name(self) -> str:
        """Canonical engine identifier (e.g. 'mujoco', 'isaaclab')."""
        ...

    def has_physics_state(self, env_unwrapped: Any) -> bool:
        """Return True if the unwrapped env has engine-specific state available."""
        ...

    def extract_side_info(self, env_unwrapped: Any) -> dict[str, Any]:
        """Extract engine-specific side information from an unwrapped env.

        For MuJoCo: returns {"mj_data": ..., "mj_model": ...}.
        """
        ...

    def build_trajectory_fields(
        self,
        env_unwrapped: Any,
        env_id: str,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Build engine-specific trajectory fields for logging.

        For MuJoCo: returns qpos, qvel, xpos, xquat, cvel, cfrc_ext, plus
        derived fields (z_height, torso_angle, etc.) based on the EnvSpec.
        """
        ...

    def inject_reward_namespace(self, namespace: dict[str, Any]) -> None:
        """Inject engine-specific modules into reward function namespace.

        For MuJoCo: adds ``mujoco`` module and ``numpy`` to namespace.
        """
        ...

    def extract_body_info(self, env_id: str) -> str:
        """Extract body layout description for LLM prompts."""
        ...

    def extract_joint_semantics(self, env_id: str) -> str:
        """Extract joint axis and sign convention info for LLM prompts.

        Returns a structured text with body frame definition, free joint
        angular velocity semantics, and per-hinge axis + range.
        """
        ...

    def get_camera_description(self, env_id: str) -> str:
        """Return a human-readable camera viewpoint description for VLM prompts."""
        ...


class MuJoCoBackend:
    """MuJoCo simulator backend -- delegates to existing env_spec.py functions.

    This is a thin wrapper that satisfies the ``SimulatorBackend`` protocol.
    """

    @property
    def engine_name(self) -> str:
        return "mujoco"

    def has_physics_state(self, env_unwrapped: Any) -> bool:
        return hasattr(env_unwrapped, "data")

    def extract_side_info(self, env_unwrapped: Any) -> dict[str, Any]:
        return {
            "mj_data": env_unwrapped.data,
            "mj_model": env_unwrapped.model,
        }

    def build_trajectory_fields(
        self,
        env_unwrapped: Any,
        env_id: str,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        import numpy as np

        data = env_unwrapped.data
        action_np = np.asarray(action).flatten()
        fields: dict[str, Any] = {
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "control_cost": float(np.sum(np.square(action_np))),
        }
        # Body-level kinematics used by code-based judge functions
        fields["xpos"] = data.xpos.tolist()
        fields["xquat"] = data.xquat.tolist()
        fields["cvel"] = data.cvel.tolist()
        fields["cfrc_ext"] = data.cfrc_ext.tolist()
        return fields

    def inject_reward_namespace(self, namespace: dict[str, Any]) -> None:
        try:
            import mujoco

            namespace["mujoco"] = mujoco
        except ImportError:
            logger.debug("mujoco not available for reward namespace")

    def extract_body_info(self, env_id: str) -> str:
        from p2p.training.env_spec import extract_mujoco_body_info

        return extract_mujoco_body_info(env_id)

    def extract_joint_semantics(self, env_id: str) -> str:
        from p2p.training.env_spec import (
            extract_joint_semantics as _extract_joint_semantics,
        )

        return _extract_joint_semantics(env_id)

    def get_camera_description(self, env_id: str) -> str:
        return (
            "The camera views the scene from the side. "
            "The agent's forward direction (+x axis) is to the RIGHT of the screen."
        )


def _build_backends() -> dict[str, type]:
    """Build the backend registry. IsaacLabBackend import is deferred."""
    backends: dict[str, type] = {"mujoco": MuJoCoBackend}
    try:
        from p2p.training.isaaclab_backend import IsaacLabBackend

        backends["isaaclab"] = IsaacLabBackend
    except ImportError:
        logger.debug("IsaacLabBackend not available (import failed)")
    return backends


_BACKENDS: dict[str, type] = _build_backends()
_INSTANCES: dict[str, SimulatorBackend] = {}


def get_simulator(engine: str = "mujoco") -> SimulatorBackend:
    """Factory: return a cached SimulatorBackend instance for the given engine.

    Backends are stateless, so a single instance per engine is reused.
    Raises ValueError for unknown engines.
    """
    cached = _INSTANCES.get(engine)
    if cached is not None:
        return cached
    cls = _BACKENDS.get(engine)
    if cls is None:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown simulator engine {engine!r}. Supported: {supported}")
    instance = cls()
    _INSTANCES[engine] = instance
    return instance
