"""Prompt templates for code-based judge generation."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2p.training.env_spec import EnvSpec

# ---------------------------------------------------------------------------
# Trajectory entry schema templates
# ---------------------------------------------------------------------------

_TRAJECTORY_BASE_SCHEMA = """\
Each trajectory entry is a dict with these fields (all optional except step):
  step: int              — timestep index (0, 1, 2, ...)
  timestamp: float       — simulation time in seconds (dt={dt}s per step)
  obs: list[float]       — observation vector (dim={obs_dim})
  action: list[float]    — action vector (dim={action_dim}), torque commands
  reward: float          — scalar reward from the reward function
  reward_terms: dict[str, float] — named reward components (e.g. {{"forward": 0.8, "alive": 0.5}})
  terminated: bool       — episode ended by failure (e.g. fall, out of bounds)
  truncated: bool        — episode ended by time limit (max_episode_steps reached)
"""

_TRAJECTORY_MUJOCO_SCHEMA = """\
  qpos: list[float]      — generalized positions (per-joint). For slide joints,
      values are displacements from the body's XML-defined position, NOT world-frame
      positions. For free joints, values ARE in world frame.
      See the Body Layout section below for per-joint details and offsets.
  qvel: list[float]      — generalized velocities (per-joint)
  control_cost: float    — control penalty (sum of squared actions)
  xpos: list[list[float]]    — Cartesian body positions in world frame,
      shape (nbody, 3) [x, y, z] per body. Index 0 is the world body;
      real bodies start at index 1.
  xquat: list[list[float]]   — body orientation quaternions, shape (nbody, 4) [w, x, y, z] per body
  cvel: list[list[float]]    — body velocities, shape (nbody, 6) [ang(3), lin(3)] per body
  cfrc_ext: list[list[float]] — external contact forces, (nbody, 6) [torque(3), force(3)]
"""

_TRAJECTORY_ISAACLAB_SCHEMA = """\
  joint_pos: list[float]     — joint positions
  joint_vel: list[float]     — joint velocities
  body_pos_w: list[list[float]] — body positions in world frame (num_bodies, 3)
  body_quat_w: list[list[float]] — body orientations (wxyz) (num_bodies, 4)
  root_pos_w: list[float]    — root body position [x, y, z] (meters)
  control_cost: float        — sum of squared actions
  applied_torque: list[float] — applied joint torques
  body_acc_w: list[list[float]] — body accelerations (contact proxy)
  object_pos: list[float]    — object position [x,y,z] world frame (dexterous envs only)
  object_rot: list[float]    — object orientation [w,x,y,z] (dexterous envs only)
  in_hand_pos: list[float]   — in-hand position [x,y,z] world frame (dexterous envs only)
  goal_pos: list[float]      — target position [x,y,z] world frame (dexterous envs only)
  goal_rot: list[float]      — target orientation [w,x,y,z] (dexterous envs only)
  scene_<name>_joint_pos: list[float]       — scene object joints (manipulation envs)
  scene_<name>_body_pos_w: list[list[float]] — scene object bodies, world frame (manip.)
"""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_judge_system_prompt(intent: str, env: EnvSpec, *, max_episode_steps: int = 1000) -> str:
    """Build system prompt for judge code generation.

    Auto-detects environments via ``env.state_ref`` and appends
    body layout + extended trajectory schema when applicable.

    Parameters
    ----------
    max_episode_steps:
        The actual episode length limit from TrainConfig.  Exposed in the
        prompt so the LLM generates correct survival-fraction calculations
        instead of hardcoding 1000.
    """
    info_keys_str = ", ".join(f"{k} ({v})" for k, v in env.info_keys.items())

    has_physics = bool(env.state_ref) or env.engine == "isaaclab"

    _3d_rotation_hint = (
        "Use robot_data.root_quat_w (wxyz) for orientation "
        "and robot_data.root_ang_vel_b for angular velocity."
        if env.engine == "isaaclab"
        else "Compute yaw from quaternion qpos[3:7] using "
        "atan2(2*(w*z+x*y), 1-2*(y**2+z**2)), and use qvel[5] for yaw rate."
    )

    # Engine-specific rotation rules for the judge system prompt
    if env.engine == "mujoco":
        _rotation_rules = (
            "- For 2D MuJoCo envs, use ``torso_angle`` (pitch in radians) for rotation.\n"
            "        - **3D environments** (Ant, Humanoid, quadrupeds): These do NOT\n"
            "          have ``torso_angle``. " + _3d_rotation_hint + "\n"
            "          For cumulative rotation, unwrap the yaw angles."
        )
    elif env.engine == "isaaclab":
        _rotation_rules = (
            "- For IsaacLab envs: " + _3d_rotation_hint + "\n"
            "          For cumulative rotation, unwrap the yaw angles."
        )
    else:
        _rotation_rules = (
            "- Check cumulative rotation using the engine's rotation fields.\n"
            "          For cumulative rotation, unwrap the yaw angles."
        )

    # Role
    if env.engine == "isaaclab":
        role_prefix = "IsaacLab robotics"
    elif env.engine == "mujoco" and has_physics:
        role_prefix = "MuJoCo robotics"
    else:
        role_prefix = "RL trajectory"
    role = (
        f"You are an expert {role_prefix} evaluator. You write Python judge\n"
        "functions that rigorously assess whether a reinforcement learning\n"
        "agent's trajectory demonstrates a specific intended behavior."
    )

    # Trajectory schema
    dt = env.dt if env.dt > 0 else 0.02
    entry_schema = _TRAJECTORY_BASE_SCHEMA.format(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        dt=dt,
    )
    if env.engine == "mujoco" and has_physics:
        entry_schema += _TRAJECTORY_MUJOCO_SCHEMA
    elif env.engine == "isaaclab":
        entry_schema += _TRAJECTORY_ISAACLAB_SCHEMA

    # Body layout (auto-detected from physics engine)
    body_layout_section = ""
    if has_physics:
        from p2p.training.env_spec import (
            engine_display_name,
            extract_body_geometry,
        )
        from p2p.training.simulator import get_simulator

        backend = get_simulator(env.engine)
        body_info_raw = backend.extract_body_info(env.env_id)
        body_info = textwrap.indent(body_info_raw, "        ")
        elabel = engine_display_name(env.engine)
        body_layout_section = textwrap.dedent(f"""\

            ## {elabel} Body Layout & Kinematics
            Use the mapping below to find the correct index for each body.

{body_info}
        """)

        # Joint semantics (axis, range, right-hand rule, angular velocity)
        joint_sem_raw = backend.extract_joint_semantics(env.env_id)
        if joint_sem_raw:
            joint_sem_indented = textwrap.indent(joint_sem_raw, "        ")
            body_layout_section += textwrap.dedent(f"""\

                ## Joint Semantics
{joint_sem_indented}
            """)

        # Body geometry (MuJoCo only — static model data)
        if env.engine == "mujoco":
            body_geo_raw = extract_body_geometry(env.env_id)
            body_geo_indented = textwrap.indent(body_geo_raw, "        ")
            body_layout_section += textwrap.dedent(f"""\

                ## Body Geometry
{body_geo_indented}
            """)

    return textwrap.dedent(f"""\
        {role}

        ## Environment: {env.env_id}
        - Name: {env.name}
        - Description: {env.description}
        - Observation dim: {env.obs_dim or "variable"}
        - Action dim: {env.action_dim or "variable"} (torque-controlled joints)
        - max_episode_steps: {max_episode_steps}
        - info keys: {info_keys_str}

        ## Trajectory Entry Schema
        {entry_schema}
{body_layout_section}
        ## Intent to Evaluate
        "{intent}"

        ## Function Contract

        Write a single Python function with this EXACT signature:

        ```python
        def judge_fn(trajectory: list[dict], summary: dict) -> dict:
        ```

        **Inputs:**
        - `trajectory`: list of dicts (one per timestep), each matching the
          schema above.  The episode length limit is **{max_episode_steps}**
          steps — use this value (not a hardcoded constant) whenever you
          need to normalize by episode length (e.g. survival fraction =
          `len(trajectory) / {max_episode_steps}`).
          MAY BE EMPTY — handle gracefully (return score 0.0).
        - `summary`: dict — currently empty (reserved for future use).
          Do NOT rely on any summary keys. Judge behavior purely from
          the trajectory data.

        **Output:** a dict with these EXACT keys:
        - "intent_score": float in [0.0, 1.0]
        - "diagnosis": str — structured sub-task achievement report (see format below)
        - "failure_tags": list[str] — descriptive, intent-specific tags

        **Diagnosis format** — the "diagnosis" string has four parts:

        1. **Structure** — dependency graph using:
           - `1 -> 2` for sequential (2 depends on 1)
           - `(1, 2)` for independent (evaluated in parallel)
           - Dot notation for sub-tasks: `1.1`, `1.2` are sub-tasks
             of `1`; `1.1.2` is a sub-task of `1.1`
        2. **Legend** — maps each number to its sub-task name
        3. **Definitions** — maps short variable names used in
           measured values to their exact meaning and trajectory
           data source. This section is critical for downstream
           consumers (e.g., synthesizer) who see only the diagnosis
           output, not the judge code.
        4. **Score details** — each sub-task with score/weight,
           PASS/FAIL/SKIP status, and measured value.

        ## Failure Tags
        Generate descriptive, intent-specific failure tags that capture
        actual failure modes observed. Do not use generic tags — describe
        what specifically went wrong relative to the stated intent (e.g.,
        "rotating_wrong_direction", "no_height_gain_before_flip",
        "gait_asymmetric").

        Example (for intent "walk to a red cone and stop"):
        ```
        Structure:
        1 -> 2
        1 = (1.1, 1.2)
        2 = 2.1 -> 2.2

        Legend:
        1. Locomotion Quality
          1.1 Forward Progress
          1.2 Upright Posture
        2. Goal Completion
          2.1 Reach Target
          2.2 Full Stop

        Scoring Criteria:
          1.1 Forward Progress
              metric: mean(vx) over all steps
              threshold: >= 0.5 m/s
              partial: linear from 0.0 to threshold
          1.2 Upright Posture
              metric: mean(torso_up_z) over all steps
              threshold: >= 0.85
              partial: linear from 0.5 to threshold
          2.1 Reach Target
              metric: min(dist_to_target) over all steps
              threshold: <= 0.3m
              partial: linear from 2.0m to threshold
          2.2 Full Stop
              metric: mean(|vx|) over last 20 steps
              threshold: <= 0.1 m/s
              partial: linear from 1.0 to threshold

        Definitions:
          vx = obs[N] — forward velocity (m/s)
          torso_up_z = z-component of torso local up-vector (1.0 = upright)
          dist_to_target = Euclidean distance to target position

        intent_score: 0.42

        1. Locomotion Quality (0.38/0.500)
          1.1 Forward Progress (PASS) [0.250/0.250]
              measured: mean(vx) = 0.82 m/s (threshold: >= 0.5 m/s)
          1.2 Upright Posture (FAIL) [0.13/0.250]
              measured: mean(torso_up_z) = 0.72 (threshold: >= 0.85)
        2. Goal Completion (0.04/0.500)
          2.1 Reach Target (FAIL) [0.04/0.250]
              measured: min(dist) = 1.5m (threshold: <= 0.3m)
          2.2 Full Stop (SKIP) [0.0/0.250]
              Gated by 2.1 (score 0.04 < 50% of 0.250 = 0.125)
        ```

        ## Core Scoring Method: Per-Sub-Task Partial Credit

        Break the intent into sub-tasks and score each on a continuous
        scale. The total intent_score is the sum of all per-sub-task scores.

        **Step 1 — Hierarchical decomposition:**
        Decompose the intent into groups and sub-tasks. Sub-tasks within
        a group can be **sequential** (ordered phases where each depends
        on the previous) or **independent** (parallel quality criteria
        evaluated regardless of each other). Groups themselves can also
        be sequential or independent. Scale the number of sub-tasks to
        match the intent's complexity. Each sub-task must have a
        concrete, measurable criterion based on trajectory data.

        **No naive step-count survival.** Do NOT add sub-tasks
        based on episode length or step count unless the intent
        explicitly requests a duration. Physics-based survival
        criteria (upright posture, balance) are fine when relevant.

        **Step 2 — Scoring rules:**
        1. **Hierarchical weight allocation**: Top-level groups receive
           equal share of the total score (1.0). Each group's budget is
           distributed among its sub-tasks. This prevents a group with
           more internal decomposition steps from dominating the score.
        2. For each sub-task, compute a **continuous score** in
           [0, sub-task weight]:
           - 0 = no progress toward this sub-task
           - partial credit = some progress but below threshold
           - full weight = threshold met or exceeded
           The sub-task is considered **succeeded** (True) when the FULL
           threshold is met — NOT a fraction of it. NEVER use
           `value >= threshold * 0.6` or similar multipliers to determine
           success. Partial credit gives lower scores for partial progress;
           the success flag must be binary at the actual threshold.
           **Score and success MUST be consistent**: use the SAME metric
           and threshold for both. If score == full weight, success must
           be True. If success is False, score must be < full weight.
           Never compute score from one metric (e.g. mean height) and
           success from a different metric (e.g. min height).
        3. **Sequential gating** (within a group): Sub-task N is only
           evaluated if sub-task N-1 scores above 50% of its weight.
           This prevents crediting later phases when earlier phases
           clearly failed, while avoiding harsh cutoffs from minor
           threshold misses (e.g., 1.44m vs 1.5m should not zero out
           everything downstream).
           If phases must occur in order (e.g., launch before rotation
           before landing), they MUST be sequential. Making them
           independent would give credit for behaviors that never
           actually happened, producing an incorrect score.
        4. **Independent groups**: Each group is evaluated independently.
           Failure in one group does NOT affect other groups.
        5. intent_score = sum of all per-sub-task scores (naturally in
           [0.0, 1.0]).

        **Step 3 — Per-trial evaluation:**
        For sequential sub-tasks, all sub-tasks in a group MUST be
        evaluated within a SINGLE continuous attempt (trial). The agent
        may make multiple attempts within one episode — evaluate each
        trial independently and use the one with the highest total
        score. Do NOT combine sub-tasks achieved across different
        trials.

        Scoring must be MONOTONIC: more progress = higher score.
        Partial credit ensures the score is informative even for failed
        sub-tasks.

        The judge's role is correctness, not generosity. Do not lower
        thresholds, relax measurement windows, or use lenient metrics
        (e.g., mean height instead of min height) to accommodate
        expected agent limitations. An incorrect judge that gives
        undeserved credit misleads the training loop into reinforcing
        wrong behaviors.

        Do NOT simplify, relax, or approximate evaluation criteria for
        any reason — implementation convenience, edge case handling, or
        expected agent limitations. Every simplification that makes the
        evaluation less accurate is a bug. If the intent says "rotation
        while airborne," measure rotation only during airborne timesteps.
        If the intent says "no hopping," check every timestep. If phases
        are sequential, make them sequential. The judge must be a
        faithful implementation of what the intent literally requires.
        Every implementation choice — thresholds, measurement windows,
        data filtering, segment merging — is a potential source of
        incorrect scores. Before adding any robustness logic, verify
        it cannot allow wrong behaviors to score high.

        ## Additional Analysis Techniques

        Combine sub-task scoring with failure mode detection.
        Actively check for:
        - Misattributed success: motion that superficially resembles the
          intent but fails on closer inspection (e.g. wrong rotation
          direction, horizontal drift instead of vertical jump)
        - Degenerate policies: near-constant actions, no variation
        - Oscillation: high-frequency vibration without net progress
        - Early termination: episode ended much shorter than expected

        ## Rules
        - Only use `numpy` (available as `np`) and `math` — no other imports.
        - Access ALL dict keys with `.get()` to avoid KeyError on missing fields.
        - Handle empty trajectory: return score 0.0 with a descriptive tag
          (e.g. "no_trajectory_data").
        - The function must be self-contained — no global state, no side effects.
        - Use descriptive variable names that reflect physics quantities.
        - **Rotation direction**: When the intent specifies a direction
          (e.g. "back tumble", "forward flip"), you MUST check the SIGN of
          cumulative rotation change — do NOT use abs() on rotation, as
          an agent rotating the wrong direction must NOT score on rotation
          sub-tasks. When no direction is specified (e.g. "do a flip"),
          using abs() is fine since either direction satisfies the intent.
        {_rotation_rules}
        - Return ONLY the Python function in a ```python block.
    """)


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

FIX_JUDGE_CODE_TEMPLATE = """\
The following judge function code has an error:

```python
{code}
```

Error: {error}

Fix the code and return ONLY the corrected Python function in a ```python block.
"""

_REVIEW_CODE_BUGS_HINTS: dict[str, str] = {
    "mujoco": (
        "wrong trajectory field access (e.g. xpos[0] vs\n"
        "   xpos[1] for torso), missing .get(), off-by-one, division by zero,\n"
        "   cumulative angle tracking errors"
    ),
    "isaaclab": (
        "wrong trajectory field access (e.g. wrong body_pos_w or\n"
        "   body_quat_w index), missing .get(), off-by-one, division by zero,\n"
        "   cumulative angle tracking errors"
    ),
}
_REVIEW_CODE_BUGS_DEFAULT = (
    "wrong trajectory field access, missing .get(), off-by-one,\n"
    "   division by zero, cumulative angle tracking errors"
)


def build_review_judge_code_prompt(
    intent: str, code: str, engine: str, *, include_code: bool = True
) -> str:
    """Build the adversarial judge code review prompt with engine-specific hints.

    When *include_code* is False, the code block is omitted (for follow-up
    rounds where the code is already in the conversation history).
    """
    code_bugs_hint = _REVIEW_CODE_BUGS_HINTS.get(engine, _REVIEW_CODE_BUGS_DEFAULT)

    if include_code:
        code_section = f"\n## Judge function to review\n```python\n{code}\n```\n"
    else:
        code_section = (
            "\n## Judge function to review\n"
            "Review the latest version of the judge function in this conversation.\n"
        )

    return f"""\
You are an adversarial reviewer for RL judge functions.
Your goal: find loopholes where the judge gives the wrong answer.

## Intent to evaluate
"{intent}"
{code_section}
## Review process

The task decomposition (structure, sub-tasks, metrics, thresholds) is
already approved and fixed. Your job is to verify the code faithfully
implements it.

For EACH leaf sub-task in the decomposition, verify:

1. **Metric implementation** — Does the code compute the exact metric
   defined in the Scoring Criteria? Check for:
   - {code_bugs_hint}
   - Variables not reset between attempts/iterations
   - Accumulation errors (wrong loop range, off-by-one boundaries)
   - Filtering/windowing that doesn't match the Definitions section
   - Wrong sign convention, logical errors

2. **Threshold and partial credit** — Does the code apply the exact
   threshold and partial credit formula from the Scoring Criteria?
   Check for:
   - Hardcoded values that don't match the defined threshold
   - Inverted comparison direction (>= vs <=)
   - Partial credit formula that doesn't scale as specified

3. **Gating and weights** — Does the code apply sequential gating
   and hierarchical weight allocation as defined in the Structure?

## Output format

For each sub-task, state what the decomposition defines, what the
code actually does, and whether they match:

  Sub-task: <name>
  Spec: <what the decomposition defines>
  Code: <what the code actually does>
  Verdict: <Match or Bug — explain the discrepancy>

BE STRICT. If you find ANY bug, you MUST fix it. Do NOT rationalize
it away as unlikely or edge case.

Decision rule:
- If you found ANY loophole or bug → output the corrected code
  in a ```python block. Do NOT say LGTM.
- If and ONLY if you found zero issues → respond with exactly: LGTM
"""


DECOMPOSE_MSG = """\
Decompose this intent into task groups and sub-tasks for evaluation:
"{intent}"

Follow the hierarchical decomposition method described above.
For each leaf sub-task, define:
- metric: what is measured from trajectory data
- threshold: what value means full credit (pass)
- partial: how scores scale below the threshold

Output ONLY the structure, legend, scoring criteria, and definitions:
```
Structure:
<top-level dependency graph>
<sub-task expansions>

Legend:
<number-to-name mapping for each sub-task>

Scoring Criteria:
  <sub-task number> <sub-task name>
      metric: <what is measured, using short variable names>
      threshold: <pass condition>
      partial: <how partial credit is computed>
  ...

Definitions:
  <short_name> = <full description and trajectory data source>
  ...
```
"""

DECOMPOSE_REVIEW_MSG = """\
Review the decomposition you just produced for the intent:
"{intent}"

You have {remaining} review round(s) remaining. Focus on significant
issues only — minor threshold adjustments can be handled during
implementation. Do NOT simplify or remove details from Scoring
Criteria or Definitions — metric definitions, thresholds, and
measurement specifics must remain precise and unambiguous.

Analyze your decomposition and check for:
1. **Completeness** — Does every requirement in the intent have a
   corresponding sub-task? Is anything missing?
2. **Structure** — Are sequential/independent relationships correct?
   Phases that must occur in order MUST be sequential. Parallel
   quality criteria should be independent.
3. **No naive step-count survival** — Remove any sub-task that
   measures survival purely by episode length or step count, UNLESS
   the intent explicitly requests a duration. Physics-based criteria
   (upright posture, balance, torso height) are fine when relevant.
4. **Measurement design** — For each leaf sub-task, consider:
   - Could a wrong behavior PASS this check? (false positive)
   - Could a correct behavior FAIL this check? (false negative)
   If yes, the metric or threshold needs revision.

For each check, reference the specific sub-task and explain your
reasoning. If you find a concrete issue, describe it clearly and
then output the corrected decomposition in a ``` block.

If you found NO issues, respond with exactly: LGTM
"""

IMPLEMENT_MSG = """\
Now implement the `judge_fn` based on the approved decomposition above.

Write a single Python function with this EXACT signature:
```python
def judge_fn(trajectory: list[dict], summary: dict) -> dict:
```

The function must implement the scoring criteria exactly as specified
in the decomposition. Include the Structure, Legend, Scoring Criteria,
and Definitions sections in the diagnosis output string.

Return ONLY the Python function in a ```python block.
"""
