"""IsaacLab simulator backend (#221, Phase 3).

Implements ``SimulatorBackend`` for NVIDIA IsaacLab (Isaac Sim) environments.
All IsaacLab/torch imports are lazy — importing this module does not require
IsaacLab to be installed.

Key differences from MuJoCo:
- State tensors are batched: ``(num_envs, *)`` — we extract ``[0]`` for
  single-env trajectory recording.
- Data lives on GPU as ``torch.Tensor`` — we call ``.cpu().numpy()`` before
  serialization.
- Robot state is accessed via ``scene["robot"].data`` (an ``ArticulationData``
  object) rather than ``mj_data``.

State mapping:
    MuJoCo (mj_data.*)       →  IsaacLab (robot.data.*)
    qpos                     →  joint_pos  (batched)
    qvel                     →  joint_vel  (batched)
    xpos                     →  body_pos_w (batched)
    xquat                    →  body_quat_w (batched, wxyz)
    cfrc_ext                 →  body_acc_w / contact sensor (batched)
    ctrl                     →  applied_torque (batched)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Env attributes for manipulated objects (dexterous envs).
# Used by both IsaacLabRewardVecWrapper (env.py) and build_trajectory_fields.
OBJ_POS_ATTRS = frozenset({"object_pos", "goal_pos", "in_hand_pos"})
OBJ_ROT_ATTRS = frozenset({"object_rot", "goal_rot"})

# Human-readable descriptions for env-level object attributes.
# Keys must stay in sync with OBJ_POS_ATTRS | OBJ_ROT_ATTRS.
_OBJECT_ATTR_DESCS: dict[str, str] = {
    "object_pos": "info['object_pos'] (num_envs, 3) — object position",
    "object_rot": "info['object_rot'] (num_envs, 4) — orientation (wxyz)",
    "goal_pos": "info['goal_pos'] (num_envs, 3) — target position",
    "goal_rot": "info['goal_rot'] (num_envs, 4) — target orientation",
    "in_hand_pos": "info['in_hand_pos'] (num_envs, 3) — in-hand position",
}
if set(_OBJECT_ATTR_DESCS.keys()) != OBJ_POS_ATTRS | OBJ_ROT_ATTRS:
    raise RuntimeError("_OBJECT_ATTR_DESCS keys out of sync with OBJ_POS_ATTRS | OBJ_ROT_ATTRS")


class IsaacLabBackend:
    """IsaacLab simulator backend.

    Satisfies the ``SimulatorBackend`` protocol for IsaacLab environments.
    All heavy imports (torch, omni, isaaclab) are deferred to method calls.
    """

    @property
    def engine_name(self) -> str:
        return "isaaclab"

    def has_physics_state(self, env_unwrapped: Any) -> bool:
        """Check if env has an IsaacLab scene with a robot articulation."""
        if not hasattr(env_unwrapped, "scene"):
            return False
        try:
            env_unwrapped.scene["robot"]
            return True
        except KeyError:
            return False

    def extract_side_info(self, env_unwrapped: Any) -> dict[str, Any]:
        """Extract robot and scene articulation data from the IsaacLab scene.

        Returns ``robot_data`` (the ``ArticulationData`` object) so reward
        functions can access joint positions, velocities, body poses, etc.

        For manipulation envs with scene objects (e.g. cabinets), also returns
        ``scene`` — a dict mapping articulation names to their data objects.
        """
        robot = env_unwrapped.scene["robot"]
        result: dict[str, Any] = {"robot_data": robot.data}
        scene = getattr(env_unwrapped, "scene", None)
        if scene is not None:
            scene_data: dict[str, Any] = {}
            for key, art in getattr(scene, "articulations", {}).items():
                if key == "robot":
                    continue
                scene_data[key] = art.data
            for key, obj in getattr(scene, "rigid_objects", {}).items():
                scene_data[key] = obj.data
            if scene_data:
                result["scene"] = scene_data
        return result

    def build_trajectory_fields(
        self,
        env_unwrapped: Any,
        env_id: str,
        action: Any,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Build trajectory fields from IsaacLab robot state.

        Extracts the first environment's data (index 0) from batched tensors
        and converts to CPU numpy for JSON serialization.

        For dexterous manipulation envs (Shadow Hand, etc.), also records
        manipulated object position/orientation so the code judge can
        evaluate object-centric metrics.
        """
        import numpy as np

        robot = env_unwrapped.scene["robot"]
        data = robot.data

        def _to_list(tensor: Any) -> list:
            arr = tensor[0].detach().cpu().numpy()
            return arr.tolist()

        action_np = np.asarray(action).flatten()
        fields: dict[str, Any] = {
            "joint_pos": _to_list(data.joint_pos),
            "joint_vel": _to_list(data.joint_vel),
            "body_pos_w": _to_list(data.body_pos_w),
            "body_quat_w": _to_list(data.body_quat_w),
            "control_cost": float(np.sum(np.square(action_np))),
        }

        # Not yet verified whether root_pos_w always equals body_pos_w[:, 0, :].
        if hasattr(data, "root_pos_w"):
            fields["root_pos_w"] = _to_list(data.root_pos_w)
        if hasattr(data, "root_quat_w"):
            fields["root_quat_w"] = _to_list(data.root_quat_w)

        if hasattr(data, "applied_torque"):
            fields["applied_torque"] = _to_list(data.applied_torque)

        # Contact proxy — equivalent to MuJoCo cfrc_ext
        if hasattr(data, "body_acc_w"):
            fields["body_acc_w"] = _to_list(data.body_acc_w)

        # Object attrs live on env (not robot.data) in local frame;
        # shift by env_origins for world-frame consistency with body_pos_w.
        scene = getattr(env_unwrapped, "scene", None)
        origins_0 = None
        if scene is not None and hasattr(scene, "env_origins"):
            origins_0 = scene.env_origins[0].detach().cpu().numpy()

        for attr in (*OBJ_POS_ATTRS, *OBJ_ROT_ATTRS):
            val = getattr(env_unwrapped, attr, None)
            if val is None:
                continue
            arr = val[0].detach().cpu().numpy()
            if attr in OBJ_POS_ATTRS and origins_0 is not None:
                arr = arr + origins_0
            fields[attr] = arr.tolist()

        # Scene articulations (non-robot) — e.g. cabinet doors/drawers
        if scene is not None:
            for key, art in getattr(scene, "articulations", {}).items():
                if key == "robot":
                    continue
                prefix = f"scene_{key}"
                fields[f"{prefix}_joint_pos"] = _to_list(art.data.joint_pos)
                bp = art.data.body_pos_w[0].detach().cpu().numpy()
                if origins_0 is not None:
                    bp = bp + origins_0
                fields[f"{prefix}_body_pos_w"] = bp.tolist()
            for key, obj in getattr(scene, "rigid_objects", {}).items():
                prefix = f"scene_{key}"
                rp = obj.data.root_pos_w[0].detach().cpu().numpy()
                if origins_0 is not None:
                    rp = rp + origins_0
                fields[f"{prefix}_body_pos_w"] = [rp.tolist()]

        return fields

    def inject_reward_namespace(self, namespace: dict[str, Any]) -> None:
        """Inject IsaacLab-relevant modules into the reward function namespace.

        Adds ``torch`` — IsaacLab reward functions typically operate on GPU
        tensors directly. Callers pre-seed numpy (np/numpy) before calling.
        """
        try:
            import torch

            namespace["torch"] = torch
        except ImportError:
            logger.debug("torch not available for reward namespace")

    def extract_body_info(self, env_id: str) -> str:
        """Return body layout description for IsaacLab environments.

        Reads from a pre-computed cache (``_isaaclab_body_cache.json``,
        generated by ``scripts/cache_isaaclab_body_info.py``). Falls back
        to a generic template if the cache is missing or the env is not
        in the cache.
        """
        info = _load_body_cache().get(env_id)
        if info:
            return _format_cached_body_info(env_id, info)
        return _generic_body_info(env_id)

    def extract_joint_semantics(self, env_id: str) -> str:
        """Return joint and orientation semantics for IsaacLab environments."""
        return (
            "Joint Semantics (IsaacLab):\n"
            "  All joints are body-frame (angle between child and parent body).\n"
            "  All rotations follow the right-hand rule about the joint axis.\n"
            "  Use np.cross(axis, offset) to compute rotation direction in code.\n"
            "\n"
            "Root orientation (quaternion-based, world frame):\n"
            "  robot_data.root_quat_w — quaternion (w, x, y, z)\n"
            "  robot_data.root_ang_vel_b — angular velocity (roll, pitch, yaw) "
            "in body frame\n"
            "    roll (about x): positive = left side up\n"
            "    pitch (about y): positive = nose down\n"
            "    yaw (about z): positive = counter-clockwise from above\n"
            "  robot_data.body_quat_w — all body orientations "
            "(num_envs, num_bodies, 4)\n"
            "  To compute yaw from quaternion (w,x,y,z): "
            "atan2(2(wz+xy), 1-2(y^2+z^2))\n"
            "  For rotation intents, track cumulative yaw change, not "
            "absolute values."
        )

    def get_camera_description(self, env_id: str) -> str:
        # Must stay in sync with _configure_phase2_camera in evaluator_isaaclab.py.
        if "Shadow" in env_id:
            return (
                "The camera views the scene from the front and slightly above, "
                "looking down at the palm and fingers. "
                "The hand is roughly centered in the frame."
            )
        return (
            "The camera views the scene from the side. "
            "The agent's forward direction (+x axis) is to the RIGHT of the screen."
        )


def _load_body_cache() -> dict:
    """Load the IsaacLab body info cache (lazy, singleton)."""
    global _body_cache  # noqa: PLW0603
    if _body_cache is not None:
        return _body_cache

    import json
    from pathlib import Path

    cache_path = Path(__file__).parent / "_isaaclab_body_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            _body_cache = json.load(f)
    else:
        _body_cache = {}
        logger.debug("IsaacLab body cache not found at %s", cache_path)
    return _body_cache


_body_cache: dict | None = None


def _format_cached_body_info(env_id: str, info: dict) -> str:
    """Format cached body/joint info into a string for LLM prompts.

    Supports two cache formats:
    - Static (from config parsing): joint_patterns, init_joint_pos, resting_z
    - Runtime (from introspection): joints, bodies with concrete indices
    """
    lines: list[str] = []
    resting_z = info.get("resting_z", info.get("root_resting_z", 0))
    lines.append(f"Root resting height: z ≈ {resting_z:.3f} meters")
    lines.append("")

    # --- Runtime format (concrete indices from introspection) ---
    if "joints" in info:
        joints = info["joints"]
        lines.append(f"robot_data.joint_pos — shape (num_envs, {len(joints)}).")
        for j in joints:
            lines.append(
                f"  joint_pos[{j['index']}] → {j['name']} (resting: {j['resting_pos']:.4f})"
            )
        lines.append("")
        lines.append(f"robot_data.joint_vel — shape (num_envs, {len(joints)}).")
        for j in joints:
            lines.append(f"  joint_vel[{j['index']}] → {j['name']}")
        bodies = info.get("bodies", [])
        if bodies:
            lines.append("")
            lines.append("Body indices (resting z = height above ground):")
            for b in bodies:
                lines.append(f"  body[{b['index']}] = {b['name']}  (resting z ≈ {b['z']:.3f})")
            lines.append("")
            nb = len(bodies)
            lines.append(f"robot_data.body_pos_w — shape (num_envs, {nb}, 3).")
            lines.append("  body_pos_w[env_idx][body_idx] = [x, y, z] in world frame")
            for b in bodies:
                lines.append(
                    f"  body_pos_w[0][{b['index']}] → {b['name']} "
                    f"(resting ≈ [{b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f}])"
                )

    # --- Static format (regex patterns from asset config) ---
    elif "joint_patterns" in info or "init_joint_pos" in info:
        patterns = info.get("joint_patterns", [])
        init_pos = info.get("init_joint_pos", {})
        # Use init_joint_pos keys as joint patterns if joint_patterns is empty
        effective_patterns = patterns or list(init_pos.keys())
        if effective_patterns:
            lines.append("Joint naming convention (regex patterns from asset config):")
            for pat in effective_patterns:
                lines.append(f"  {pat}")
        if init_pos:
            lines.append("")
            lines.append("Initial joint positions:")
            for pat, val in init_pos.items():
                lines.append(f"  joints matching '{pat}' → {val}")
        lines.append("")
        lines.append("robot_data.joint_pos — joint positions (num_envs, num_joints).")
        lines.append("robot_data.joint_vel — joint velocities (num_envs, num_joints).")
        lines.append("robot_data.body_pos_w — body positions (num_envs, num_bodies, 3).")
        lines.append("  body_pos_w[env_idx][body_idx] = [x, y, z] in world frame")
        lines.append(
            "robot_data.body_quat_w — body orientations (wxyz) (num_envs, num_bodies, 4)."
        )

    # Object info (manipulated object separate from robot articulation)
    obj = info.get("object_info")
    if obj:
        lines.append("")
        lines.append("Manipulated object (separate rigid body, NOT part of robot articulation):")
        lines.append(f"  Name: {obj.get('name', 'unknown')}")
        if "init_pos" in obj:
            p = obj["init_pos"]
            lines.append(f"  Initial position: [{p[0]}, {p[1]}, {p[2]}]")
        if "note" in obj:
            lines.append(f"  IMPORTANT: {obj['note']}")
        lines.append("  Reward fn: info['object_pos'] (num_envs, 3) — WORLD frame")
        lines.append("  Reward fn: info['object_rot'] (num_envs, 4) — quaternion (wxyz)")
        lines.append("  Trajectory: step['object_pos'] list[float] — [x,y,z] WORLD frame")
        lines.append("  Trajectory: step['object_rot'] list[float] — [w,x,y,z] quaternion")
        lines.append("  Goal:   step['goal_pos'] / info['goal_pos'] — target position")
        lines.append("  Goal:   step['goal_rot'] / info['goal_rot'] — target quaternion")
        lines.append(
            "  All positions (object_pos, body_pos_w) are in the SAME world"
            " frame. Compute distances directly — no frame conversion needed."
        )

    # Scene articulations (non-robot objects in the scene, e.g. cabinet)
    scene_arts = info.get("scene_articulations", {})
    for sa_key, sa_info in scene_arts.items():
        sa_type = sa_info.get("type", "articulation")
        lines.append("")
        lines.append(f'Scene object: "{sa_key}" ({sa_type})')
        if sa_type == "rigid_object":
            sa_bodies = sa_info.get("bodies", [])
            if sa_bodies:
                b = sa_bodies[0]
                lines.append(f"  Initial position: [{b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f}]")
            lines.append(f"  Reward fn: info['scene']['{sa_key}'].root_pos_w (num_envs, 3)")
            lines.append(f"  Trajectory: step['scene_{sa_key}_body_pos_w'] [[x,y,z]] world frame")
        else:
            sa_joints = sa_info.get("joints", [])
            sa_bodies = sa_info.get("bodies", [])
            if sa_joints:
                lines.append(f"  Joints ({len(sa_joints)}):")
                for j in sa_joints:
                    lines.append(
                        f"    [{j['index']}] {j['name']} (resting: {j['resting_pos']:.4f})"
                    )
            if sa_bodies:
                lines.append(f"  Bodies ({len(sa_bodies)}):")
                for b in sa_bodies:
                    lines.append(
                        f"    [{b['index']}] {b['name']}"
                        f" (pos ~ [{b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f}])"
                    )
            nj = len(sa_joints)
            nb = len(sa_bodies)
            lines.append(f"  Reward fn: info['scene']['{sa_key}'].joint_pos (num_envs, {nj})")
            lines.append(f"  Reward fn: info['scene']['{sa_key}'].body_pos_w (num_envs, {nb}, 3)")
            lines.append(f"  Trajectory: step['scene_{sa_key}_joint_pos'] list[float]")
            lines.append(
                f"  Trajectory: step['scene_{sa_key}_body_pos_w'] list[[x,y,z]] world frame"
            )

    # Env-level object attributes (manager-based manipulation envs).
    # These are separate from scene_articulations — some envs expose objects
    # as env attributes (object_pos, goal_pos) rather than scene rigid objects.
    env_obj = info.get("env_object_attrs", {})
    if env_obj:
        lines.append("")
        lines.append("Manipulated object(s) — env-level attributes:")
        for attr, vals in env_obj.items():
            lines.append(f"  Initial {attr}: {vals}")
        lines.append("")
        lines.append("  Access in reward fn (batched tensors, all WORLD frame):")
        for attr in env_obj:
            desc = _OBJECT_ATTR_DESCS.get(attr, f"info['{attr}']")
            lines.append(f"    {desc}")
        lines.append("  Access in trajectory/judge:")
        for attr in env_obj:
            lines.append(f"    step['{attr}'] — list[float]")
        lines.append("  All positions are in WORLD frame — compute distances directly.")

    # Common access patterns
    lines.append("")
    lines.append("robot_data.root_pos_w — root body position (num_envs, 3).")
    lines.append(f"  root_pos_w[:, 2] = root height for all envs (resting ≈ {resting_z:.3f})")
    lines.append("robot_data.root_quat_w — root orientation (wxyz) (num_envs, 4).")
    lines.append("robot_data.root_lin_vel_b — root linear vel (body frame) (num_envs, 3).")
    lines.append("robot_data.root_ang_vel_b — root angular vel (body frame) (num_envs, 3).")
    lines.append("robot_data.applied_torque — joint torques (num_envs, num_joints).")
    lines.append("")
    lines.append(
        "NOTE: All tensors are batched (first dim = num_envs). Use [:, idx] slicing"
        " to operate on all envs at once. Return (num_envs,) shaped reward tensors."
    )

    return "\n".join(lines)


def _generic_body_info(env_id: str) -> str:
    """Fallback when cache is not available."""
    from p2p.training.env_spec import ENV_REGISTRY

    spec = ENV_REGISTRY.get(env_id)
    if spec is None:
        return f"No body info available for {env_id}."

    lines = [
        f"Environment: {spec.env_id} ({spec.description})",
        "",
        "IsaacLab robot state access (via robot_data = info['robot_data']):",
        "  robot_data.joint_pos    — joint positions, shape (num_envs, num_joints)",
        "  robot_data.joint_vel    — joint velocities, shape (num_envs, num_joints)",
        "  robot_data.body_pos_w   — body positions in world frame, "
        "shape (num_envs, num_bodies, 3)",
        "  robot_data.body_quat_w  — body orientations (wxyz), shape (num_envs, num_bodies, 4)",
        "  robot_data.root_pos_w   — root body position, shape (num_envs, 3)",
        "  robot_data.root_quat_w  — root body orientation (wxyz), shape (num_envs, 4)",
        "  robot_data.root_lin_vel_b — root linear velocity (body frame), shape (num_envs, 3)",
        "  robot_data.root_ang_vel_b — root angular velocity (body frame), shape (num_envs, 3)",
        "  robot_data.applied_torque — applied joint torques, shape (num_envs, num_joints)",
        "",
        "NOTE: All tensors are batched (first dim = num_envs) and live on GPU.",
        "Use tensor[env_idx] to index a single environment.",
        "",
        "(Detailed joint/body layout not available — run "
        "scripts/cache_isaaclab_body_info.py to generate the cache.)",
    ]
    return "\n".join(lines)
