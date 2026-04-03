"""Prompt templates for reward function generation and revision."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from p2p.prompts._shared import build_reward_contract
from p2p.training.env_spec import engine_display_name

if TYPE_CHECKING:
    from p2p.training.env_spec import EnvSpec


def build_system_prompt(env: EnvSpec, *, side_info: bool = False) -> str:
    """Build a system prompt tailored to the given environment."""
    info_keys_str = ", ".join(f"{k} ({v})" for k, v in env.info_keys.items())
    reward_contract = build_reward_contract(env, side_info=side_info)
    engine_label = engine_display_name(env.engine)
    obs_str = f"{env.obs_dim}-dim" if env.obs_dim else "variable dim"
    act_str = f"{env.action_dim}-dim" if env.action_dim else "variable dim"
    return textwrap.dedent(f"""\
        You are an expert reward engineer for {engine_label} reinforcement learning.

        Environment: {env.env_id}
        - Observation space: {obs_str} (joint angles, velocities, etc.)
        - Action space: {act_str} (torque-controlled joints)
        - info dict keys: {info_keys_str}

        You must write Python code that defines `reward_fn` at module level.

        {reward_contract}

        Before writing, consider: what would a perfectly rational agent
        do to maximize this reward? If that behavior differs from the
        intent, redesign the terms. Match the reward structure to the
        intent: sustained states/rates → per-step reward;
        cumulative achievements → progress tracking (monotonic max);
        sequential tasks → phased sub-goals with gated milestones.
    """)


GENERATE_TEMPLATE = textwrap.dedent("""\
    Write a reward function for the following goal:

    {prompt}

    Return ONLY the Python function, no explanation.
""")

REVISE_TEMPLATE = textwrap.dedent("""\
    The previous reward function did not achieve the desired behavior.

    ## Goal
    {prompt}

    ## Previous reward function
    ```python
    {previous_code}
    ```

    ## Training metrics
    - Final episodic return: {final_return}
    - Total timesteps: {total_timesteps}
    - Training time: {training_time:.1f}s

    ## Current hyperparameters
    {current_config}

    ## VLM Judgment
    - Intent score: {intent_score}/1.0
    - Diagnosis: {diagnosis}
    - Failure tags: {failure_tags}

    Revise the reward function to address these issues.

    You may ALSO suggest hyperparameter changes if the training dynamics indicate \
    a need (e.g. unstable training → lower learning_rate, premature convergence → \
    higher ent_coef, slow learning → higher learning_rate).

    Tunable hyperparameters: learning_rate, ent_coef, gamma, gae_lambda, \
    clip_coef, vf_coef, max_grad_norm, num_steps, update_epochs, num_minibatches.

    Return the revised Python function in a ```python block.
    If you want to change hyperparameters, add a ```json block with ONLY the \
    fields to change. Example:
    ```json
    {{"learning_rate": 0.001, "ent_coef": 0.02}}
    ```
    If no hyperparameter changes are needed, omit the json block entirely.
""")

FIX_CODE_TEMPLATE = textwrap.dedent("""\
    The following Python reward function code has an error:

    ```python
    {code}
    ```

    Error: {error}

    Fix the code and return ONLY the corrected Python function. No explanation.
""")

_SIGN_INDEX_HINTS: dict[str, str] = {
    "mujoco": "wrong sign convention, wrong obs/xpos indices, wrong qpos/qvel indices",
    "isaaclab": "wrong sign convention, wrong obs indices, wrong body_pos_w/joint_pos indices",
}
_SIGN_INDEX_DEFAULT = "wrong sign convention, wrong observation or state variable indices"


def build_review_code_prompt(prompt: str, code: str, engine: str) -> str:
    """Build the reward code review prompt with engine-specific index hints."""
    sign_index_hint = _SIGN_INDEX_HINTS.get(engine, _SIGN_INDEX_DEFAULT)

    return textwrap.dedent(f"""\
        You are a strict code reviewer for RL reward functions.
        Your goal: ensure the reward function is free of bugs for its given
        LaTeX equation and reward term descriptions in the docstring.

        ## Goal (what the agent should learn)
        {prompt}

        ## Reward function to review
        ```python
        {code}
        ```

        Check for these categories of bugs:

        1. **Dead terms** — reward terms described in the docstring but always
           0.0 in every code path (never assigned a non-zero value)
        2. **State initialization** — state variables initialized to wrong
           values (e.g. ``prev_angle = 0.0`` instead of reading from
           ``obs``/``info`` at episode start)
        3. **Incomplete tracking** — cumulative quantities only tracked in
           some phases but needed across all (e.g. rotation only accumulated
           in phase 1, missing phase 0 rotation)
        4. **Sign/index errors** — {sign_index_hint}
        5. **Unreachable conditions** — conditions that can never be True,
           off-by-one errors, division by zero

        ## Output format

        For each item you investigate, first state what you are checking,
        then reason through the code, and ONLY THEN conclude whether it is
        a bug or not. Put the verdict AFTER your reasoning, not before.

        Example:
          Check 1: r_airborne term liveness
          The docstring describes r_airborne but looking at the code,
          r_airborne is initialized to 0.0 and only assigned in phase 1
          line 120. In phases 0, 2, 3 it stays 0.0. This is expected —
          airborne reward only applies during the flip phase. → OK

          Check 2: cumulative rotation tracking in phase 0
          In phase 0 (lines 121-123), cumulative_rotation is only updated
          when delta_angle < 0. Forward rotation during launch is lost.
          In phase 1 (line 136), all rotation is tracked. This means
          backward rotation during launch is counted but forward rotation
          is silently dropped. → Bug

        BE STRICT. If you identify ANY confirmed bug, you MUST fix it — do NOT
        rationalize it away or dismiss it as "minor" or "edge case".

        IMPORTANT constraints:
        - ONLY fix confirmed bugs. Do NOT add new reward terms, change the
          overall reward design, or restructure the code beyond what is
          needed to fix the identified issues.
        - Do NOT change thresholds, weights, or bonuses unless they are
          clearly wrong (e.g. reward for wrong direction).

        Decision rule:
        - If you found ANY confirmed bug → output the corrected code in a
          ```python block. Do NOT say LGTM.
        - If and ONLY if you found zero bugs → respond with exactly: LGTM
    """)
