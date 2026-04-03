"""Shared prompt building blocks used across multiple prompt modules."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2p.training.env_spec import EnvSpec

# ---------------------------------------------------------------------------
# Shared reward function contract
# ---------------------------------------------------------------------------


def _reward_sig(side_info: bool) -> str:  # noqa: ARG001
    """Return the reward_fn signature string."""
    return "obs, action, next_obs, info"


def _import_note(side_info: bool, engine: str) -> str:
    """Return the import rule note."""
    if side_info:
        if engine == "isaaclab":
            return " torch also available."
        return " mujoco also available."
    return " No other imports."


def build_reward_contract(env: EnvSpec, *, side_info: bool = False) -> str:
    """Return the reward function contract block shared by author and reviser."""
    info_keys_list = ", ".join(env.info_keys.keys())
    sig = _reward_sig(side_info)
    import_note = _import_note(side_info, engine=env.engine)
    obs_s = str(env.obs_dim) if env.obs_dim else "variable"
    act_s = str(env.action_dim) if env.action_dim else "variable"

    side_info_section = ""
    if side_info:
        from p2p.training.env_spec import extract_body_geometry
        from p2p.training.simulator import get_simulator

        backend = get_simulator(env.engine)
        body_info_raw = backend.extract_body_info(env.env_id)
        # Indent body info to match the surrounding 4-space indent level
        body_info = textwrap.indent(body_info_raw, "    ")

        # Joint semantics (axis, range, right-hand rule, angular velocity)
        joint_sem_raw = backend.extract_joint_semantics(env.env_id)
        if joint_sem_raw:
            body_info += "\n\n" + textwrap.indent(joint_sem_raw, "    ")

        # Body geometry (MuJoCo only — static model data)
        if env.engine == "mujoco":
            body_geo_raw = extract_body_geometry(env.env_id)
            body_info += "\n\n" + textwrap.indent(body_geo_raw, "    ")

        # Build in parts to avoid multi-line f-string interpolation misalign
        if env.engine == "isaaclab":
            header = textwrap.dedent(f"""\

                **IsaacLab side_info mode is ENABLED (vectorized).**
                Access robot state via the ``info`` dict:
                - ``robot_data = info["robot_data"]`` — post-step robot data object
                - You may ``import torch`` (it is available in the namespace).
                - **IMPORTANT**: All tensors are batched with shape ``(num_envs, ...)``.
                  Use ``[:, idx]`` slicing (NOT ``[0, idx]``). The reward function is
                  called ONCE for all environments. Return a **tensor** of shape
                  ``(num_envs,)`` as the total reward (not a float), and each term in
                  the dict must also be ``(num_envs,)`` shaped.
                - Extract at the top: ``robot_data = info["robot_data"]``

                Useful robot_data attributes (all batched ``(num_envs, ...)``):
                - ``robot_data.joint_pos``   — joint positions ({env.env_id}-specific layout)
                - ``robot_data.joint_vel``   — joint velocities
                - ``robot_data.body_pos_w``  — body positions (num_envs, num_bodies, 3)
                - ``robot_data.body_quat_w`` — body orientations (wxyz) (num_envs, num_bodies, 4)
                - ``robot_data.body_acc_w``  — body accelerations (contact proxy)
                - ``robot_data.applied_torque`` — applied joint torques
                - ``scene_data = info.get("scene", {{}})`` — dict of non-robot articulation
                  data objects (e.g. ``scene_data["cabinet"].joint_pos`` for cabinet
                  door angles). Empty dict for locomotion envs.
            """)
        else:
            header = textwrap.dedent("""\

                **MuJoCo side_info mode is ENABLED.**
                Access MuJoCo state via the ``info`` dict:
                - ``mj_data = info["mj_data"]``  — post-step ``mujoco.MjData`` object
                - ``mj_model = info["mj_model"]`` — ``mujoco.MjModel`` object
                - You may ``import mujoco`` (it is available in the namespace).
                - Recommend extracting at the top of reward_fn for convenience:
                  ``mj_data, mj_model = info["mj_data"], info["mj_model"]``

                Useful mj_data attributes:
                - ``mj_data.qpos``        — generalized positions (layout below)
                - ``mj_data.qvel``        — generalized velocities
                - ``mj_data.xpos``        — Cartesian body positions (nbody, 3) — world frame
                - ``mj_data.xmat``        — body rotation matrices (nbody, 9)
                - ``mj_data.subtree_com`` — center of mass per subtree (nbody, 3)
                - ``mj_data.cfrc_ext``    — external contact forces per body (nbody, 6)
                - ``mj_data.sensordata``  — sensor readings (if model defines sensors)
                - ``mj_data.ctrl``        — control inputs
            """)
        side_info_section = header + "\n" + body_info + "\n"

    height_var = "root" if env.engine == "isaaclab" else "torso"

    # Episode-start example code and rules are engine-specific:
    # MuJoCo: scalar bool per env (each env gets its own closure).
    # IsaacLab: bool tensor (num_envs,) — one closure shared across all envs.
    _code_indent = " " * 16  # aligns with reward_fn body inside _make_reward
    if env.engine == "isaaclab":
        episode_start_example = textwrap.indent(
            textwrap.dedent("""\
            # Episode boundary: bool tensor (num_envs,)
            starts = info["_episode_start"]
            if starts.any():
        """),
            _code_indent,
        )
        episode_start_rules = (
            '- Episode resets: ``info["_episode_start"]`` is a bool-like\n'
            "          object of shape ``(num_envs,)``.\n"
            "          One closure is shared across all envs (batched tensors).\n"
            '          **NEVER** use ``if info.get("_episode_start"):`` —\n'
            "          this raises ``ValueError`` on multi-element tensors.\n"
            '          Use ``starts = info["_episode_start"]`` then:\n'
            "          ``if starts.any():`` for global checks, or per-env\n"
            '          masking like ``state["prev"][starts] = curr[starts].clone()``.'
        )
    else:
        episode_start_example = textwrap.indent(
            textwrap.dedent("""\
            # Reset state at episode boundary (provided by the environment):
            if info.get("_episode_start"):
        """),
            _code_indent,
        )
        episode_start_rules = (
            '- Episode resets: check ``info.get("_episode_start")`` — it is\n'
            "          ``True`` on the first step after ``env.reset()``. Always reset\n"
            "          ALL state fields when this flag is True.\n"
            "          Each parallel env gets its own closure (exec'd per env)."
        )

    return side_info_section + textwrap.dedent(f"""\
        **Option A: Stateless** (simple tasks — run, balance, reach):
        ```python
        def reward_fn({sig}):
            \"\"\"LaTeX: r = r_{{height}} + r_{{speed}} - r_{{ctrl}}
            Terms:
              height: reward for torso height above 1.0m
                r_{{height}} = 2.0 \\cdot \\max(0, z - 1.0)
              speed: reward for forward velocity
                r_{{speed}} = 1.0 \\cdot v_x
              ctrl: control penalty
                r_{{ctrl}} = 0.05 \\cdot \\|a\\|^2
            \"\"\"
            return total_reward, {{"height": r_height, "speed": r_speed, "ctrl": -r_ctrl}}
        ```

        **Option B: Stateful closure** (sequential/phased tasks — tumble,
        flip-then-run, jump-and-land, any task needing temporal ordering):
        ```python
        def _make_reward():
            state = {{"step": 0, "max_val": 0.0, "phase_done": False}}

            def reward_fn({sig}):
                \"\"\"LaTeX: r = r_{{height}} + r_{{rotation}} + r_{{flip\\_bonus}} - r_{{ctrl}}
                Terms:
                  height: reward for gaining height during jump
                    r_{{height}} = 3.0 \\cdot \\max(0, z_{{{height_var}}} - 0.7)
                  rotation: reward for backward rotation while airborne
                    r_{{rotation}} = 2.0 \\cdot \\Delta\\theta_{{back}}
                  flip_bonus: one-time bonus for completing full back rotation
                    r_{{flip\\_bonus}} = 50.0 \\text{{ if }} \\theta_{{cumul}} \\geq 2\\pi
                  ctrl: control penalty
                    r_{{ctrl}} = 0.05 \\cdot \\|a\\|^2
                \"\"\"
{episode_start_example}                    state["step"] = 0
                    state["max_val"] = 0.0
                    state["phase_done"] = False

                state["step"] += 1
                # ... phase-gated rewards, one-time bonuses, etc ...
                return total_reward, {{"height": r_height, "rotation": r_rot, ...}}

            return reward_fn

        reward_fn = _make_reward()
        ```

        Use Option B when the task requires:
        - Sequential sub-goals (first X, then Y)
        - One-time milestone bonuses (collected once per episode)
        - Phase-gated rewards (reward Y only after X is achieved)
        - Monotonic progress tracking (only reward new records)

        If the goal contains "then", "after", "land successfully",
        "transition into", or describes ordered phases, use Option B.

        Key rules for stateful rewards:
        {episode_start_rules}
        - Use monotonic max-tracking to prevent oscillation exploits.
        - Use one-time boolean flags for milestone bonuses.

        Rules (both options):
        - Only use numpy (imported as np).{import_note}
        - obs/next_obs: shape ({obs_s},), action: shape ({act_s},).
        - info is a dict with keys: {info_keys_list}.
        - Return (float, dict[str, float]) — total reward and named terms.
          The terms dict must ONLY contain values that are summed into the
          total reward. Do NOT add diagnostic/observation values (e.g.
          ``d_torso_z``, ``d_rotation_deg``) — they pollute reward analysis.
        - Include a docstring with:
          (1) Overall LaTeX formula: MUST be the explicit sum of ALL
              individual term variables (e.g. ``r = r_{{height}} + r_{{rot}}
              + r_{{bonus}} - r_{{ctrl}}``). Never use vague groupings like
              ``r_{{phase}}`` or ``r_{{bonus}}`` that collapse multiple terms.
          (2) Terms section: each term has a short description AND a proper
              LaTeX equation showing how it is computed (not prose).
        - ALWAYS return ALL terms in the dict, even when a term is inactive
          (phase-gated or conditional) — use 0.0 for inactive terms.
        - Keep it simple and numerically stable.""")
