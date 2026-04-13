"""Environment specification presets for MuJoCo and IsaacLab tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    """Immutable description of a simulation environment."""

    env_id: str
    name: str
    obs_dim: int
    action_dim: int
    info_keys: dict[str, str]
    description: str
    # qpos/qvel index mapping for trajectory recording (None = not applicable)
    z_height_qpos_idx: int | None = None
    torso_angle_qpos_idx: int | None = None
    torso_angvel_qvel_idx: int | None = None
    # State layout reference for side_info mode (engine-neutral)
    state_ref: str = ""
    # True for 3D envs (Ant, Humanoid) that use quaternion orientation qpos[3:7]
    uses_quaternion_orientation: bool = False
    # Simulator engine identifier (e.g. "mujoco", "isaaclab")
    engine: str = "mujoco"
    # Simulation timestep (seconds per step = model.opt.timestep * frame_skip).
    # Used to compute max_episode_steps from a real-time duration target.
    # 0.0 means unknown (will fall back to TrainConfig default).
    dt: float = 0.0


ENV_REGISTRY: dict[str, EnvSpec] = {
    # --- Locomotion ---
    "HalfCheetah-v5": EnvSpec(
        env_id="HalfCheetah-v5",
        name="HalfCheetah",
        obs_dim=17,
        action_dim=6,
        info_keys={
            "x_velocity": "forward speed",
            "x_position": "horizontal displacement",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
        },
        description="6-joint planar cheetah",
        z_height_qpos_idx=1,
        torso_angle_qpos_idx=2,
        torso_angvel_qvel_idx=2,
        state_ref=(
            "qpos[0]=x, qpos[1]=z_torso, qpos[2]=angle_torso, "
            "qpos[3:9]=joint_angles (bthigh, bshin, bfoot, fthigh, fshin, ffoot); "
            "qvel[0]=vx, qvel[1]=vz, qvel[2]=ang_vel_torso, qvel[3:9]=joint_vels"
        ),
        dt=0.05,
    ),
    "Ant-v5": EnvSpec(
        env_id="Ant-v5",
        name="Ant",
        obs_dim=27,
        action_dim=8,
        info_keys={
            "x_velocity": "forward speed",
            "y_velocity": "lateral speed",
            "x_position": "x displacement",
            "y_position": "y displacement",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
            "reward_survive": "survival bonus",
            "reward_contact": "contact penalty",
        },
        description="4-legged 3D ant",
        z_height_qpos_idx=2,
        # 3D free body: orientation is quaternion (qpos[3:7]), not a single angle
        state_ref=(
            "qpos[0:3]=xyz_torso, qpos[3:7]=quaternion_torso, "
            "qpos[7:15]=joint_angles (hip1-4, ankle1-4); "
            "qvel[0:3]=v_xyz, qvel[3:6]=ang_vel_torso, qvel[6:14]=joint_vels"
        ),
        uses_quaternion_orientation=True,
        dt=0.05,
    ),
    "Hopper-v5": EnvSpec(
        env_id="Hopper-v5",
        name="Hopper",
        obs_dim=11,
        action_dim=3,
        info_keys={
            "x_velocity": "forward speed",
            "x_position": "horizontal displacement",
            "z_distance_from_origin": "height",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
            "reward_survive": "survival bonus",
        },
        description="single-legged planar hopper",
        z_height_qpos_idx=1,
        torso_angle_qpos_idx=2,
        torso_angvel_qvel_idx=2,
        state_ref=(
            "qpos[0]=x, qpos[1]=z_torso, qpos[2]=angle_torso, "
            "qpos[3:6]=joint_angles (thigh, leg, foot); "
            "qvel[0]=vx, qvel[1]=vz, qvel[2]=ang_vel_torso, qvel[3:6]=joint_vels"
        ),
        dt=0.008,
    ),
    "Walker2d-v5": EnvSpec(
        env_id="Walker2d-v5",
        name="Walker2d",
        obs_dim=17,
        action_dim=6,
        info_keys={
            "x_velocity": "forward speed",
            "x_position": "horizontal displacement",
            "z_distance_from_origin": "height",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
            "reward_survive": "survival bonus",
        },
        description="bipedal planar walker",
        z_height_qpos_idx=1,
        torso_angle_qpos_idx=2,
        torso_angvel_qvel_idx=2,
        state_ref=(
            "qpos[0]=x, qpos[1]=z_torso, qpos[2]=angle_torso, "
            "qpos[3:9]=joint_angles (thigh_r, leg_r, foot_r, thigh_l, leg_l, foot_l); "
            "qvel[0]=vx, qvel[1]=vz, qvel[2]=ang_vel_torso, qvel[3:9]=joint_vels"
        ),
        dt=0.008,
    ),
    "Humanoid-v5": EnvSpec(
        env_id="Humanoid-v5",
        name="Humanoid",
        obs_dim=348,
        action_dim=17,
        info_keys={
            "x_velocity": "forward speed",
            "y_velocity": "lateral speed",
            "x_position": "x displacement",
            "y_position": "y displacement",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
            "reward_survive": "survival bonus",
            "reward_contact": "contact penalty",
        },
        description="21-dof 3D humanoid",
        z_height_qpos_idx=2,
        # 3D free body: orientation is quaternion (qpos[3:7]), not a single angle
        state_ref=(
            "qpos[0:3]=xyz_torso, qpos[3:7]=quaternion_torso, "
            "qpos[7:28]=joint_angles (21 joints); "
            "qvel[0:3]=v_xyz, qvel[3:6]=ang_vel_torso, qvel[6:27]=joint_vels"
        ),
        uses_quaternion_orientation=True,
        dt=0.015,
    ),
    "HumanoidStandup-v5": EnvSpec(
        env_id="HumanoidStandup-v5",
        name="HumanoidStandup",
        obs_dim=348,
        action_dim=17,
        info_keys={
            "x_position": "x displacement",
            "y_position": "y displacement",
            "z_distance_from_origin": "torso height",
            "reward_linup": "stand-up reward",
            "reward_quadctrl": "quadratic control penalty",
            "reward_impact": "impact penalty",
        },
        description="21-dof humanoid stand-up task",
        z_height_qpos_idx=2,
        # 3D free body: orientation is quaternion (qpos[3:7]), not a single angle
        state_ref=(
            "qpos[0:3]=xyz_torso, qpos[3:7]=quaternion_torso, "
            "qpos[7:28]=joint_angles (21 joints); "
            "qvel[0:3]=v_xyz, qvel[3:6]=ang_vel_torso, qvel[6:27]=joint_vels"
        ),
        uses_quaternion_orientation=True,
        dt=0.015,
    ),
    "Swimmer-v5": EnvSpec(
        env_id="Swimmer-v5",
        name="Swimmer",
        obs_dim=8,
        action_dim=2,
        info_keys={
            "x_velocity": "forward speed",
            "y_velocity": "lateral speed",
            "x_position": "x displacement",
            "y_position": "y displacement",
            "reward_ctrl": "control penalty",
            "reward_forward": "forward reward",
        },
        description="3-link planar swimmer",
        # Planar swimmer on x-y plane — no z-height concept
        torso_angle_qpos_idx=2,
        torso_angvel_qvel_idx=2,
        state_ref=(
            "qpos[0]=x, qpos[1]=y, qpos[2]=angle_torso, "
            "qpos[3:5]=joint_angles (rot2, rot3); "
            "qvel[0]=vx, qvel[1]=vy, qvel[2]=ang_vel_torso, qvel[3:5]=joint_vels"
        ),
        dt=0.04,
    ),
    # --- Control ---
    "Reacher-v5": EnvSpec(
        env_id="Reacher-v5",
        name="Reacher",
        obs_dim=10,
        action_dim=2,
        info_keys={
            "reward_dist": "distance-to-target penalty",
            "reward_ctrl": "control penalty",
        },
        description="2-joint planar reacher",
        state_ref=("qpos[0:2]=joint_angles, qpos[2:4]=target_xy; qvel[0:2]=joint_vels"),
        dt=0.02,
    ),
    "InvertedPendulum-v5": EnvSpec(
        env_id="InvertedPendulum-v5",
        name="InvertedPendulum",
        obs_dim=4,
        action_dim=1,
        info_keys={
            "reward_survive": "survival bonus",
        },
        description="inverted pendulum balance",
        state_ref="qpos[0]=cart_pos, qpos[1]=pole_angle; qvel[0]=cart_vel, qvel[1]=pole_vel",
        dt=0.04,
    ),
    "InvertedDoublePendulum-v5": EnvSpec(
        env_id="InvertedDoublePendulum-v5",
        name="InvertedDoublePendulum",
        obs_dim=9,
        action_dim=1,
        info_keys={
            "reward_survive": "survival bonus",
            "distance_penalty": "tip distance penalty",
            "velocity_penalty": "angular velocity penalty",
        },
        description="inverted double pendulum balance",
        state_ref=(
            "qpos[0]=cart_pos, qpos[1]=pole1_angle, qpos[2]=pole2_angle; "
            "qvel[0]=cart_vel, qvel[1]=pole1_vel, qvel[2]=pole2_vel"
        ),
        dt=0.05,
    ),
}

# Merge auto-generated IsaacLab envs (from scripts/sync_isaaclab_envs.py)
try:
    from p2p.training._isaaclab_registry import ISAACLAB_ENV_SPECS

    ENV_REGISTRY.update(ISAACLAB_ENV_SPECS)
except ImportError:
    pass  # IsaacLab registry not generated yet

# Merge custom SAR IsaacLab envs (hand-maintained, not auto-generated)
from p2p.training._custom_isaaclab_registry import CUSTOM_ISAACLAB_ENV_SPECS  # noqa: E402

ENV_REGISTRY.update(CUSTOM_ISAACLAB_ENV_SPECS)


def max_steps_for_duration(env_id: str, duration_s: float) -> int | None:
    """Compute max_episode_steps from a real-time duration target.

    Returns ``round(duration_s / dt)`` for envs with a known ``dt``,
    or ``None`` if the env is not in the registry or has ``dt == 0``.
    Uses ``round()`` instead of ``int()`` to avoid off-by-one from
    floating-point truncation (e.g. ``int(99.999...)`` → 99).
    """
    spec = ENV_REGISTRY.get(env_id)
    if spec is None or spec.dt <= 0:
        return None
    return round(duration_s / spec.dt)


def get_env_spec(env_id: str) -> EnvSpec:
    """Look up an environment spec by env_id. Raises KeyError if not found."""
    return ENV_REGISTRY[env_id]


def get_spec_by_name(name: str) -> EnvSpec | None:
    """Look up an environment spec by display name (e.g. ``"HalfCheetah"``).

    Also accepts a full env_id (e.g. ``"HalfCheetah-v5"``).
    Returns ``None`` if no match is found.
    """
    # Direct env_id match first
    if name in ENV_REGISTRY:
        return ENV_REGISTRY[name]
    # Search by display name
    for spec in ENV_REGISTRY.values():
        if spec.name == name:
            return spec
    return None


_ENGINE_DISPLAY_NAMES: dict[str, str] = {
    "mujoco": "MuJoCo",
    "isaaclab": "IsaacLab",
}


def engine_display_name(engine: str) -> str:
    """Return properly capitalized display name for an engine identifier."""
    return _ENGINE_DISPLAY_NAMES.get(engine, engine)


# ---------------------------------------------------------------------------
# MuJoCo body layout extraction (lazy + cached)
# ---------------------------------------------------------------------------

_mujoco_body_cache: dict[str, str] = {}
_mujoco_body_lock = __import__("threading").Lock()


def extract_mujoco_body_info(env_id: str) -> str:
    """Extract body layout with concrete access patterns from a MuJoCo model.

    Resets the env once to obtain resting Cartesian positions so the LLM
    can write meaningful height / distance thresholds.  Cached per env_id.
    Thread-safe via ``_mujoco_body_lock``.
    """
    with _mujoco_body_lock:
        if env_id in _mujoco_body_cache:
            return _mujoco_body_cache[env_id]

    # Lazy import — gymnasium is heavy and side_info is off by default
    import gymnasium as gym
    import mujoco

    env = gym.make(env_id)
    try:
        if not hasattr(env.unwrapped, "model"):
            raise ValueError(f"{env_id} is not a MuJoCo environment")
        env.reset()
        model = env.unwrapped.model
        data = env.unwrapped.data
        # Ensure Cartesian positions (xpos etc.) are up to date
        mujoco.mj_forward(model, data)

        # Collect bodies with resting positions (derive label for unnamed bodies)
        bodies: list[tuple[int, str, float, float, float]] = []
        for i in range(model.nbody):
            name = model.body(i).name
            if name == "world":
                continue
            if not name:
                # Derive label from joint attached to this body
                for j in range(model.njnt):
                    if model.jnt_bodyid[j] == i:
                        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                        name = f"{jname}_body"
                        break
                else:
                    name = f"body{i}"
            x, y, z = (float(v) for v in data.xpos[i])
            bodies.append((i, name, x, y, z))

        # --- qpos / qvel (generalized coordinates) ---
        # Detect root joint type to add frame clarification
        root_jtype = model.jnt_type[0] if model.njnt > 0 else -1
        has_slide_root = root_jtype == 2  # slide joint
        has_free_root = root_jtype == 0  # free joint

        lines: list[str] = [
            f"mj_data.qpos — generalized positions, shape ({model.nq},).",
        ]
        if has_free_root:
            lines.append("  (Free joint: qpos position values are in world frame.)")
        elif has_slide_root:
            body_id = model.jnt_bodyid[0]
            bpos = model.body_pos[body_id]
            lines.append(
                f"  (Slide joints: qpos values are DISPLACEMENTS from the body's"
                f" XML origin [{bpos[0]:.2f}, {bpos[1]:.2f}, {bpos[2]:.2f}],"
                f" NOT world-frame positions."
                f" Use xpos for world-frame position/height checks.)"
            )

        for i in range(model.njnt):
            name = model.joint(i).name
            jtype = model.jnt_type[i]
            adr = model.jnt_qposadr[i]
            if jtype == 0:  # free joint: 3 pos + 4 quat
                lines.append(
                    f"  mj_data.qpos[{adr}:{adr + 3}] → {name} position [x, y, z] (world frame)"
                )
                lines.append(
                    f"  mj_data.qpos[{adr + 3}:{adr + 7}] → {name} quaternion [w, x, y, z]"
                )
            elif jtype == 1:  # ball joint: 4 quat
                lines.append(f"  mj_data.qpos[{adr}:{adr + 4}] → {name} quaternion [w, x, y, z]")
            elif jtype == 2:  # slide joint
                axis = model.jnt_axis[i]
                axis_label = "xyz"[int(max(range(3), key=lambda k: abs(float(axis[k]))))]
                body_id = model.jnt_bodyid[i]
                offset = model.body_pos[body_id]["xyz".index(axis_label)]
                lines.append(
                    f"  mj_data.qpos[{adr}]  → {name}"
                    f" (displacement along {axis_label} in meters,"
                    f" world {axis_label} = qpos[{adr}] + {offset:.2f})"
                )
            else:  # hinge joint
                lines.append(f"  mj_data.qpos[{adr}]  → {name} (angle in radians)")

        lines.append("")
        lines.append(f"mj_data.qvel — generalized velocities, shape ({model.nv},).")
        for i in range(model.njnt):
            name = model.joint(i).name
            jtype = model.jnt_type[i]
            adr = model.jnt_dofadr[i]
            if jtype == 0:  # free joint: 3 linear + 3 angular
                lines.append(
                    f"  mj_data.qvel[{adr}:{adr + 3}]"
                    f" → {name} linear velocity [vx, vy, vz] (m/s, world frame)"
                )
                lines.append(
                    f"  mj_data.qvel[{adr + 3}:{adr + 6}]"
                    f" → {name} angular velocity [wx, wy, wz] (rad/s, world frame)"
                )
            elif jtype == 1:  # ball joint: 3 angular
                lines.append(
                    f"  mj_data.qvel[{adr}:{adr + 3}]"
                    f" → {name} angular velocity [wx, wy, wz] (rad/s)"
                )
            elif jtype == 2:  # slide joint
                axis = model.jnt_axis[i]
                axis_label = "xyz"[int(max(range(3), key=lambda k: abs(float(axis[k]))))]
                lines.append(
                    f"  mj_data.qvel[{adr}]  → {name}"
                    f" (linear velocity along {axis_label}, m/s, world frame)"
                )
            else:  # hinge joint
                lines.append(f"  mj_data.qvel[{adr}]  → {name} (angular velocity, rad/s)")

        # --- Per-body Cartesian kinematics ---
        lines.append("")
        lines.append("Body indices:")
        for idx, name, _rx, _ry, rz in bodies:
            lines.append(f"  body[{idx}] = {name}  (resting z ≈ {rz:.2f})")

        # Concrete access patterns — the LLM can copy-paste these directly
        lines.append("")
        lines.append("mj_data.xpos — Cartesian body positions in world frame, shape (nbody, 3).")
        lines.append(
            '  (NOTE: "x" in xpos means "Cartesian", not "x-axis". It contains all 3 world axes.)'
        )
        lines.append("  xpos[body_index][0] = world x")
        lines.append("  xpos[body_index][1] = world y")
        lines.append("  xpos[body_index][2] = world z (height)")
        lines.append("")
        for idx, name, rx, ry, rz in bodies:
            lines.append(
                f"  mj_data.xpos[{idx}]     → {name} [x, y, z]"
                f"  (resting ≈ [{rx:.2f}, {ry:.2f}, {rz:.2f}])"
            )
            lines.append(f"  mj_data.xpos[{idx}][0]  → {name} x")
            lines.append(f"  mj_data.xpos[{idx}][1]  → {name} y")
            lines.append(f"  mj_data.xpos[{idx}][2]  → {name} z (height)")

        # Rotation (quaternion)
        lines.append("")
        lines.append(
            "mj_data.xquat — body orientation quaternion in world frame,"
            " shape (nbody, 4) = [w, x, y, z]."
        )
        for idx, name, *_ in bodies:
            lines.append(f"  mj_data.xquat[{idx}]    → {name} orientation [w, x, y, z]")

        # Velocity (linear + angular)
        lines.append("")
        lines.append(
            "mj_data.cvel — body velocity in world frame,"
            " shape (nbody, 6) = [angular(3), linear(3)]."
        )
        for idx, name, *_ in bodies:
            lines.append(
                f"  mj_data.cvel[{idx}]     → {name} velocity [ang_x, ang_y, ang_z, vx, vy, vz]"
            )
            lines.append(f"  mj_data.cvel[{idx}][:3] → {name} angular velocity")
            lines.append(f"  mj_data.cvel[{idx}][3:] → {name} linear velocity")

        # Contact forces
        lines.append("")
        lines.append(
            "mj_data.cfrc_ext — external contact forces per body in world frame,"
            " shape (nbody, 6) = [torque(3), force(3)]."
        )
        for idx, name, *_ in bodies:
            lines.append(f"  mj_data.cfrc_ext[{idx}] → forces on {name}")

        result = "\n".join(lines)
    finally:
        env.close()

    with _mujoco_body_lock:
        _mujoco_body_cache[env_id] = result
    return result


# ---------------------------------------------------------------------------
# Body geometry extraction (lazy + cached)
# ---------------------------------------------------------------------------

_body_geometry_cache: dict[str, str] = {}

# MuJoCo type constants
_JNT_TYPE_NAMES: dict[int, str] = {
    0: "free",
    1: "ball",
    2: "slide",
    3: "hinge",
}

_GEOM_TYPE_NAMES: dict[int, str] = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


def extract_body_geometry(env_id: str) -> str:
    """Extract static body geometry from a MuJoCo model.

    Returns a structured text description of the robot's physical structure
    including body hierarchy, masses, geom shapes/sizes, segment lengths,
    and joint ranges.  All data comes from ``mj_model`` (no simulation
    state).  Useful for reward and judge prompt construction so the LLM
    can reason about physical dimensions and achievable poses.

    Cached per *env_id*.  Thread-safe via ``_mujoco_body_lock``.
    """
    with _mujoco_body_lock:
        if env_id in _body_geometry_cache:
            return _body_geometry_cache[env_id]

    import gymnasium as gym
    import numpy as np

    env = gym.make(env_id)
    try:
        if not hasattr(env.unwrapped, "model"):
            raise ValueError(f"{env_id} is not a MuJoCo environment")
        model = env.unwrapped.model

        # -- Collect body names --
        body_names: dict[int, str] = {}
        for i in range(model.nbody):
            name = model.body(i).name
            if name:
                body_names[i] = name

        # -- Build parent-child tree --
        children: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
        for i in range(1, model.nbody):
            pid = int(model.body_parentid[i])
            children[pid].append(i)

        # -- Map joints and geoms to bodies --
        joint_by_body: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
        for j in range(model.njnt):
            joint_by_body[int(model.jnt_bodyid[j])].append(j)

        geom_by_body: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
        for g in range(model.ngeom):
            geom_by_body[int(model.geom_bodyid[g])].append(g)

        # -- Helpers --
        def _geom_desc(g: int) -> str:
            gtype = int(model.geom_type[g])
            gname = _GEOM_TYPE_NAMES.get(gtype, f"type{gtype}")
            size = model.geom_size[g]
            gobj_name = model.geom(g).name or ""
            label = f'"{gobj_name}" ' if gobj_name else ""
            if gtype == 2:  # sphere
                shape = f"{label}{gname} (radius={size[0]:.4f})"
            elif gtype in (3, 5):  # capsule / cylinder
                shape = f"{label}{gname} (radius={size[0]:.4f}, half-length={size[1]:.4f})"
            elif gtype == 6:  # box
                shape = (
                    f"{label}{gname} (half-extents=[{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}])"
                )
            else:
                sz_str = ", ".join(f"{s:.4f}" for s in size if s != 0.0)
                shape = f"{label}{gname} (size=[{sz_str}])"

            # Geom position within body (if non-zero offset)
            gpos = model.geom_pos[g]
            pos_norm = float(np.linalg.norm(gpos))
            if pos_norm > 1e-4:
                shape += f" at [{gpos[0]:.2f}, {gpos[1]:.2f}, {gpos[2]:.2f}] on body"

            return shape

        def _joint_desc(j: int) -> str:
            jtype = int(model.jnt_type[j])
            jname = model.joint(j).name
            tname = _JNT_TYPE_NAMES.get(jtype, f"type{jtype}")
            limited = bool(model.jnt_limited[j])
            if limited and jtype in (2, 3):
                lo, hi = model.jnt_range[j]
                unit = "m" if jtype == 2 else "rad"
                lo_deg = f" ({np.degrees(lo):.1f}\u00b0)" if jtype == 3 else ""
                hi_deg = f" ({np.degrees(hi):.1f}\u00b0)" if jtype == 3 else ""
                return f"{jname} ({tname}, range=[{lo:.3f}{lo_deg}, {hi:.3f}{hi_deg}] {unit})"
            return f"{jname} ({tname})"

        # -- Build output --
        lines: list[str] = [f"Body Geometry for {env_id}"]
        lines.append("=" * len(lines[0]))
        lines.append("At zero-configuration, all body-local frames align with world frame:")
        lines.append("  x = forward, y = left, z = up.")
        lines.append("Attachment points and geom positions are in parent body's local frame.")

        # 1. Body hierarchy with mass, joints, geoms
        lines.append("")
        lines.append("Body Hierarchy (parent \u2192 child):")
        lines.append("-" * 35)

        def _walk(bid: int, depth: int) -> None:
            name = body_names.get(bid, f"body{bid}")
            if name == "world":
                lines.append(f"{'  ' * depth}world")
                for child in children[bid]:
                    _walk(child, depth + 1)
                return

            mass = float(model.body_mass[bid])
            indent = "  " * depth

            # body_pos: fixed attachment point on parent (parent's local frame)
            # Skip for root bodies (parent=world) — that's spawn placement
            pid = int(model.body_parentid[bid])
            attach_str = ""
            if pid != 0:
                bp = model.body_pos[bid]
                if float(np.linalg.norm(bp)) > 1e-6:
                    attach_str = f", attached at [{bp[0]:.2f}, {bp[1]:.2f}, {bp[2]:.2f}] on parent"
            lines.append(f"{indent}{name} (mass={mass:.3f} kg{attach_str})")

            for j in joint_by_body[bid]:
                lines.append(f"{indent}  joint: {_joint_desc(j)}")
            for g in geom_by_body[bid]:
                lines.append(f"{indent}  geom: {_geom_desc(g)}")

            for child in children[bid]:
                _walk(child, depth + 1)

        _walk(0, 0)

        # 2. Body origin distances (norm of model.body_pos — fixed model property)
        lines.append("")
        lines.append("Body Origin Distances (between connected bodies):")
        lines.append("-" * 50)
        for bid in range(1, model.nbody):
            pid = int(model.body_parentid[bid])
            if pid == 0:  # parent is world — spawn placement, not geometry
                continue
            dist = float(np.linalg.norm(model.body_pos[bid]))
            if dist > 1e-6:
                name = body_names.get(bid, f"body{bid}")
                pname = body_names.get(pid, f"body{pid}")
                lines.append(f"  {pname} \u2192 {name}: {dist:.4f} m")

        # 3. Total mass
        total_mass = float(np.sum(model.body_mass[1:]))  # exclude world
        lines.append("")
        lines.append(f"Total robot mass: {total_mass:.3f} kg")

        result = "\n".join(lines)
    finally:
        env.close()

    with _mujoco_body_lock:
        _body_geometry_cache[env_id] = result
    return result


# ---------------------------------------------------------------------------
# Joint semantics extraction (lazy + cached)
# ---------------------------------------------------------------------------

_joint_semantics_cache: dict[str, str] = {}


def extract_joint_semantics(env_id: str) -> str:
    """Extract joint axis and sign convention information for all joints.

    For free joints: reports qpos/qvel layout with angular velocity
    semantics (roll/pitch/yaw with physical meaning).
    For slide joints: reports the slide axis direction.
    For hinge joints: reports the rotation axis in the body's local frame.

    The LLM can compute the rotation direction using the right-hand rule
    and ``np.cross(axis, offset)`` in its code.  No pre-computed direction
    labels are provided — the axis + body frame definition + right-hand
    rule give the LLM everything it needs.

    Uses only ``mj_model`` (no simulation, no perturbation).
    Cached per *env_id*.  Thread-safe via ``_mujoco_body_lock``.
    """
    with _mujoco_body_lock:
        if env_id in _joint_semantics_cache:
            return _joint_semantics_cache[env_id]

    import gymnasium as gym
    import numpy as np

    env = gym.make(env_id)
    try:
        if not hasattr(env.unwrapped, "model"):
            raise ValueError(f"{env_id} is not a MuJoCo environment")
        model = env.unwrapped.model

        # -- Analyze each joint, collecting free/slide and hinge separately --
        free_lines: list[str] = []
        slide_lines: list[str] = []
        hinge_lines: list[str] = []

        for j in range(model.njnt):
            jname = model.joint(j).name
            jtype = int(model.jnt_type[j])
            axis = model.jnt_axis[j]

            if jtype == 0:  # free joint
                adr = model.jnt_qposadr[j]
                dof = model.jnt_dofadr[j]
                free_lines.append(
                    f"{jname} (free, world frame):\n"
                    f"  qpos[{adr}:{adr + 3}] = [x, y, z] position\n"
                    f"  qpos[{adr + 3}:{adr + 7}] = [w, x, y, z] quaternion\n"
                    f"  qvel[{dof}:{dof + 3}] = [vx, vy, vz] linear velocity\n"
                    f"  qvel[{dof + 3}:{dof + 6}] = [wx, wy, wz] angular velocity\n"
                    f"    wx (roll about x): positive = left side up\n"
                    f"    wy (pitch about y): positive = nose down\n"
                    f"    wz (yaw about z): positive = counter-clockwise from above"
                )
                continue

            if jtype == 1:  # ball joint
                body_name = model.body(model.jnt_bodyid[j]).name or f"body{model.jnt_bodyid[j]}"
                hinge_lines.append(f"  {jname} (ball, on {body_name}): 3-DOF rotation")
                continue

            if jtype == 2:  # slide joint
                ax_str = f"[{axis[0]:.1f}, {axis[1]:.1f}, {axis[2]:.1f}]"
                slide_lines.append(f"{jname} (slide, axis={ax_str})")
                continue

            # -- Hinge joint: report axis and range --
            ax_str = f"[{axis[0]:.1f}, {axis[1]:.1f}, {axis[2]:.1f}]"
            limited = bool(model.jnt_limited[j])
            if limited:
                lo, hi = model.jnt_range[j]
                range_str = f", range=[{np.degrees(lo):.0f}, {np.degrees(hi):.0f}] deg"
            else:
                range_str = ""
            hinge_lines.append(f"  {jname} (hinge, axis={ax_str}{range_str})")

        # -- Assemble output --
        lines: list[str] = [f"Joint Semantics for {env_id}"]
        lines.append("=" * len(lines[0]))
        lines.append("Free joint: world frame (position, orientation, velocities).")
        lines.append("Hinge joints: body frame (angle between child and parent body).")
        lines.append("At zero-configuration, body frames align with world frame:")
        lines.append("  x = forward, y = left, z = up.")

        if free_lines:
            lines.append("")
            lines.extend(free_lines)

        if slide_lines:
            lines.append("")
            lines.extend(slide_lines)

        if hinge_lines:
            lines.append("")
            lines.append("Hinge joints (body frame):")
            lines.append("  All rotations follow the right-hand rule about the joint axis.")
            lines.append("  Use np.cross(axis, offset) to compute rotation direction in code.")
            lines.extend(hinge_lines)

        result = "\n".join(lines)
    finally:
        env.close()

    with _mujoco_body_lock:
        _joint_semantics_cache[env_id] = result
    return result
