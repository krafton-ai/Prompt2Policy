"""Prompt templates for the revise agent (reward + HP revision)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from p2p.prompts._shared import _reward_sig, build_reward_contract
from p2p.training.env_spec import engine_display_name

if TYPE_CHECKING:
    from p2p.training.env_spec import EnvSpec


def _build_change_constraint(is_plateau: bool) -> str:
    """Return the change-scope constraint block for the system prompt.

    During normal operation, enforces the 1-2 targeted changes rule.
    During plateau (many iterations without improvement from the same
    parent), lifts that constraint and encourages structural rewrites.
    """
    if is_plateau:
        return textwrap.dedent("""\
            ━━━ PLATEAU MODE: STRUCTURAL REWRITE ALLOWED ━━━
            The 1-2 change limit is SUSPENDED because many iterations
            have failed to improve from the same best iteration.
            Incremental tweaks are exhausted.

            You are encouraged to:
            - Redesign the reward from scratch with a different philosophy.
            - Change multiple terms, structure, and coefficients at once.
            - You may demote STRONG lessons to SOFT via
              update_experiment_lessons(index=N, tier='SOFT', reason='...')
              before violating them. HARD lessons remain enforced.
            - If you find lessons that are wrong or counterproductive, retire
              them via update_experiment_lessons(index=N, tier='RETIRED',
              reason='...'). Retired lessons stay visible to prevent
              rediscovering the same mistake.

            Hard prohibitions still apply:
            - NEVER switch a monotonic reward (e.g. linear forward velocity)
              to a non-monotonic one (e.g. Gaussian peak) — this removes the
              gradient signal at low velocities and the agent cannot bootstrap.
            - NEVER add a penalty term whose typical magnitude exceeds the
              dominant positive reward term — this inverts the reward landscape
              and teaches the agent to stay still.
            - NEVER multiply the forward reward by an uprightness gate in
              early iterations — this creates a chicken-and-egg problem where
              the agent needs to be upright to get forward reward but needs
              forward reward signal to learn how to stay upright.""")

    return textwrap.dedent("""\
        ━━━ MANDATORY: 1-2 CHANGES ONLY ━━━
        You MUST make at most 1-2 targeted changes per revision.
        This is not a suggestion — it is a hard constraint.

        Why: If you change 5 things and the score drops, you cannot tell
        which change caused the regression. One change at a time preserves
        causal attribution and enables monotonic improvement.

        ━━━ LESSON TIERS ━━━
        Lessons have graduated tiers that tell you how strictly to follow them:
        - [HARD]: Catastrophic failure proven. You MUST respect these absolutely.
        - [STRONG]: Confirmed principle. Follow unless you have specific evidence
          they are wrong. To violate, first demote to SOFT via set_tier.
        - [SOFT]: Context-specific or aged out. Challenge freely if you have a
          good reason.
        - [RETIRED]: No longer active, shown for reference only.

        You can promote any lesson to HARD via update_experiment_lessons(
        index=N, tier='HARD', reason='...') when you discover a lesson
        is truly catastrophic.

        Rules:
        - Identify the SINGLE most impactful failure mode from your diagnosis.
        - Fix ONLY that one thing (or at most two closely related things).
        - Leave every other term, coefficient, and structure UNCHANGED.
        - NEVER simultaneously: change reward structure + add new penalties
          + modify coefficients + alter alive bonus. Pick ONE axis.

        Hard prohibitions (these have caused catastrophic regressions):
        - NEVER switch a monotonic reward (e.g. linear forward velocity)
          to a non-monotonic one (e.g. Gaussian peak) — this removes the
          gradient signal at low velocities and the agent cannot bootstrap.
        - NEVER add a penalty term whose typical magnitude exceeds the
          dominant positive reward term — this inverts the reward landscape
          and teaches the agent to stay still.
        - NEVER multiply the forward reward by an uprightness gate in
          early iterations — this creates a chicken-and-egg problem where
          the agent needs to be upright to get forward reward but needs
          forward reward signal to learn how to stay upright.""")


def build_revise_system_prompt(
    env: EnvSpec,
    *,
    side_info: bool = False,
    has_tools: bool = False,
    hp_tuning: bool = True,
    is_plateau: bool = False,
    two_phase: bool = False,
) -> str:
    """System prompt for the revise agent."""
    info_keys_str = ", ".join(f"{k} ({v})" for k, v in env.info_keys.items())
    reward_contract = build_reward_contract(env, side_info=side_info)
    sig = _reward_sig(side_info)

    side_info_reminder = ""
    if side_info:
        engine_label = engine_display_name(env.engine)
        if env.engine == "isaaclab":
            side_info_reminder = textwrap.dedent("""\

                ⚠️ CRITICAL: IsaacLab side_info is ENABLED (vectorized).
                Your revised reward_fn MUST access robot state via the info dict:
                  ``robot_data = info["robot_data"]``
                All tensors are batched (num_envs, ...). Use ``[:, idx]`` slicing
                (NOT ``[0, idx]``). Return a **(num_envs,)** shaped reward tensor
                and **(num_envs,)** shaped term tensors.
                DO NOT drop robot_data — it is required for body position, rotation,
                and joint state calculations.
            """)
        else:
            side_info_reminder = textwrap.dedent("""\

                ⚠️ CRITICAL: MuJoCo side_info is ENABLED for this environment.
                Your revised reward_fn MUST access MuJoCo state via the info dict:
                  ``mj_data, mj_model = info["mj_data"], info["mj_model"]``
                DO NOT drop these — they are required for body position, rotation,
                and contact force calculations. If the current reward uses mj_data
                or mj_model, your revision MUST preserve that access.
            """)

    tools_section = ""
    if has_tools:
        _required_sections = (
            "Diagnosis, Lesson, Based On, Planned Changes"
            if two_phase
            else "Diagnosis, Lesson, Based On, Reward Reasoning, Revised Reward Function"
        )
        tools_section = textwrap.dedent(f"""\
        ━━━ TOOLS ━━━
        You have tools to inspect past iterations and current evaluation rollouts:

        **Score timeline & best iteration:**
        - get_iteration_scores(top_k?) — compact score timeline across ALL past
          iterations with trend analysis (improving/declining/plateau/erratic).
          Returns the best iteration's reward code so you can build on it.
          Call this FIRST to understand the trajectory before diving deeper.

        **Past iterations (drill-down):**
        - get_iteration_reward_code(iteration) — full reward code
        - get_iteration_training_dynamics(iteration, run_id?) — training curve analysis.
          In multi-config mode, use run_id (e.g. 'baseline_seed_1')
          to query a specific run; defaults to the best run.
        - get_iteration_judgment_detail(iteration) — full judgment details
        - compare_iterations(iter_a, iter_b) — diff reward code + scores

        **Current iteration rollout judgments:**
        - get_checkpoint_judgments(step, run_id?, detail?) — rollout judgments for a
          checkpoint. run_id selects which config×seed run (default: best). detail
          levels: 'aggregate' (default, stats only), 'summary' (aggregate +
          best/worst rollout), 'all' (every rollout's full diagnosis).
          Start with 'aggregate' or 'summary' to save tokens.
        - get_rollout_judgment(step, rollout_label?, episode_idx?) — single
          rollout detail. Use rollout_label='p10'/'median'/'p90' for parallel
          eval, or episode_idx (integer) for sequential eval.

        **Multi-config comparison (only in multi-config mode):**
        - get_config_comparison(detail?) — cross-config comparison showing how each
          config×seed combination performed. detail levels: 'aggregate' (default,
          per-config mean score/return), 'summary' (+ best/worst seed per config),
          'all' (+ every seed's full judgment).

        **Strategic overview:**
        - get_strategy_summary() — strategic 'tree of thoughts' summary of the FULL
          iteration history. Groups iterations into score-trend phases, logs every
          regression and its cause, surfaces persistent failure patterns, and detects
          HP/reward oscillation. Call this to avoid repeating failed approaches and
          to ensure monotonic improvement without oscillation.

        **Experiment lineage (if available):**
        - get_experiment_lineage() — the experiment tree for this session with
          scores and accumulated lessons. Each node has a lesson learned from
          its outcome. Call this FIRST to review what has been tried and what
          lessons have been distilled. This prevents you from repeating
          experiments that already failed.

        **Recommended workflow:**
        1. START with get_experiment_lineage() (if available) to see the
           experiment tree and accumulated lessons.
        2. Call get_strategy_summary() to see phases, regressions, persistent
           failures, and oscillation warnings within the current session.
        3. Call get_iteration_scores() to see the full score timeline and get
           the best iteration's reward code.
        4. In multi-config mode, call get_config_comparison(detail='summary')
           to see which configs worked and which failed.
        5. Call get_checkpoint_judgments on the LAST checkpoint with detail='summary'
           to see aggregate stats + best/worst cases for the best run.
        6. Optionally query non-best runs with run_id to understand WHY they failed.
        7. Check success_rate and common_failure_tags — these are more reliable
           than any single rollout's score.

        After exploring, produce your final revision using the output format
        described in the system prompt. Your final message MUST contain all
        required sections ({_required_sections}, and HP sections if
        applicable).
        """)
    tools_block = f"\n{tools_section}" if tools_section else ""

    # HP tuning sections (conditionally included)
    if hp_tuning:
        hp_tuning_section = textwrap.dedent("""\
            ━━━ HP TUNING ━━━
            **HP–reward interactions**:
            - Scale up reward → lower learning_rate
            - New sparse term → raise ent_coef temporarily
            - Remove dominant term → lower learning_rate (policy shock)
            - Major restructure + normalize_reward → noisy early training,
              consider more total_timesteps

            **Decision rules** (condition → action):
            - explained_var < 0.3 → increase update_epochs (2×) or vf_coef (1.5×)
            - explained_var 0.3-0.5 → increase update_epochs by 50%
            - value_loss diverging → halve learning_rate
            - approx_kl > 0.02 → halve learning_rate or add target_kl=0.03
            - approx_kl spikes (>0.04) frequent → add target_kl=0.02
            - clip_fraction > 0.3 → tighten clip_coef
            - clip_fraction < 0.05 → can increase learning_rate
            - Entropy decay > 90% → increase ent_coef 2-5×
            - Entropy decay < 20% → reduce ent_coef or more timesteps
            - Phased tasks: high early entropy is OK — worry only if it
              collapses before agent discovers the full sequence
            - Return plateau + still improving → more total_timesteps
            - Return flat 30%+ → reward or exploration problem, not budget
            - num_steps too short for task horizon → cover ≥2 full episodes
            - Long horizon (>500 steps) → gamma=0.995
            - High return variance → gae_lambda toward 0.9
        """)
        cross_iteration_hp_note = (
            "- 3+ flat iterations despite different rewards → likely HP problem.\n"
            "- Scale drifted 10×+ → value function may be unstable."
        )
        hp_output_format = textwrap.dedent("""\

            ## HP Reasoning
            <training dynamics analysis and HP changes, if any>

            ## HP Changes
            ```json
            {"param_name": [old_value, new_value]}
            ```
            Each entry is ``[current_value, proposed_value]`` so reviewers can
            see what changed at a glance. Example:
            ```json
            {"num_steps": [1024, 2048], "gamma": [0.99, 0.995]}
            ```
            If no HP changes are needed, return an empty object ``{}``.
            Read the current values from the "Current Hyperparameters" section above.
        """)
        diagnostic_step_6 = "6. **Coordinated proposal**: Reward + HP changes together."
        hp_contract_note = (
            "Only change HPs when training dynamics clearly indicate a problem.\n"
            "Return an empty JSON object {} if no HP changes needed."
        )
        n_sections = 7
    else:
        hp_tuning_section = ""
        cross_iteration_hp_note = ""
        hp_output_format = ""
        diagnostic_step_6 = "6. **Proposal**: Formulate your revised reward function."
        hp_contract_note = (
            "Hyperparameters are FIXED (pre-tuned). Focus ONLY on reward design.\n"
            "Do NOT suggest HP changes."
        )
        n_sections = 5

    # Two-phase mode: drop the code section from the output format
    if two_phase:
        _changes_heading = "Planned Changes"
        _no_change_instruction = (
            "write 'NO_CHANGE: keeping best iteration's code -- [reason]' "
            "in your Planned Changes section and set Based On to the "
            "best iteration number."
        )
        n_sections -= 1
        _based_on_extra = (
            "  The system will mechanically fetch this iteration's "
            "verbatim reward code for you to edit."
        )
        _reasoning_section = textwrap.dedent("""\
            ## Planned Changes
            <specific changes and why, referencing diagnosis>
            Do NOT write code. Describe your planned changes in natural
            language. The system will fetch the base code and apply your
            changes in a separate step.""")
    else:
        _changes_heading = "Reward Reasoning"
        _no_change_instruction = (
            "copy the best iteration's code exactly into your "
            '"## Revised Reward Function" section and write '
            '"NO_CHANGE: keeping best iteration\'s code -- [reason]" '
            "in your Reward Reasoning."
        )
        _based_on_extra = ""
        _reasoning_section = textwrap.dedent(f"""\
            ## Reward Reasoning
            <specific changes and why, referencing diagnosis>

            ## Revised Reward Function
            ```python
            def reward_fn({sig}):
                ...
            ```""")

    engine_label = engine_display_name(env.engine)
    obs_str = f"{env.obs_dim}-dim" if env.obs_dim else "variable"
    act_str = f"{env.action_dim}-dim" if env.action_dim else "variable"

    return textwrap.dedent(f"""\
        You are a senior RL engineer specializing in reward shaping and PPO
        hyperparameter tuning for {engine_label} continuous-control tasks.

        Environment: {env.env_id}
        - Observation: {obs_str}, Action: {act_str} (torques)
        - info keys: {info_keys_str}
        {tools_block}
        ━━━ DIAGNOSTIC PROCESS (6 steps, in order) ━━━
        1. **Behavior gap**: What is the agent doing vs. what it should do?
        2. **Checkpoint & rollout consistency**: Review ALL checkpoint judgments
           AND their per-rollout statistics together.
           - Each checkpoint has multiple eval rollouts. Check success_rate and
             score_std — these matter more than any single rollout score.
           - High mean_score + low success_rate = inconsistent behavior, not
             reliable. High mean_score + high success_rate = truly learned.
           - If only 1 checkpoint scores high but others are low, the reward
             may produce lucky outliers — not reliable learned behavior.
           - Consistent improvement across checkpoints = reward is working.
           - A single spike with low neighbors = likely noise or overfitting.
           - Cross-reference intent scores with trajectory metrics:
             * Score rising but distance/rotation/height flat → judge noise
             * Metrics improving but score flat → reward is learning the
               right behavior, judge may be miscalibrated — keep the reward
             * Score high but episodic return dropping → unstable policy
             * Return rising + score rising + metrics improving → strong
               signal that reward is effective and valid
           - Look at the score trend across checkpoints (early→late):
             monotonically improving = good training signal from reward.
             Erratic = reward may be noisy or have exploitable local optima.
        3. **Reward root cause**: Which term(s) caused wrong behavior?
           Dominant term? Sparse/ignored term? Conflicting terms?
           High return + low intent_score = reward hacking — find the exploit.
        4. **Training dynamics**: Check value loss, explained variance,
           KL, clip fraction, entropy. Reward problem vs. optimization problem?
        5. **Cross-iteration patterns**: What was tried before? Oscillating?
           Upward trend? ALWAYS build on the best-scoring iteration's
           approach. Do not randomly explore — make targeted, incremental
           improvements to the best reward code. If the current iteration
           regressed from the best, revert toward the best version and
           apply smaller, focused edits.
        {diagnostic_step_6}

        ━━━ REWARD DESIGN ━━━

        **1. Optimal policy alignment**
        Before writing code, answer: "What would a perfectly rational
        agent do to maximize this reward over a discounted horizon?"
        If that optimal behavior differs from the intent, the reward
        is wrong BY DESIGN — no amount of training will fix it.
        Fix the reward, not the training.

        **2. Classify the intent → choose reward structure**
        The right reward structure depends on what the intent asks for:

        - **Sustained state/rate** ("run fast", "stay balanced",
          "track a target"): Instantaneous per-step reward is correct.
          The intended behavior IS the per-step optimum.
          E.g., reward = x_velocity for "run forward."

        - **Cumulative achievement** ("rotate 360°", "reach a
          position", "flip"): Instantaneous rate is DANGEROUS —
          policy can oscillate (each half-cycle scores, net = 0).
          Use cumulative progress tracking with monotonic max:
          ``delta = max(0, current - state["best"])``.

        - **Sequential/phased task** ("jump, tumble, then land"):
          Decompose into sub-goals. Gate later phases behind earlier
          completion. One-time milestone bonuses at transitions.
          Stabilization penalties after final phase.

        When revising, first check: does the current reward structure
        match the intent type? A structural mismatch is the most
        common root cause of reward hacking.

        **3. Discount-aware shaping**
        The policy maximizes Σ γ^t r_t. With γ < 1, future rewards
        are discounted. Implications:
        - Sparse end-of-episode rewards need γ ≈ 1 or dense per-step
          shaping to propagate the learning signal backward.
        - Sequential tasks: each sub-goal must provide LOCAL reward
          at the steps where it happens, not only at completion.
        - Long horizon tasks may need γ = 0.995+.

        **4. Closing the intent-reward gap**
        Reward hacking = the optimal policy under your reward differs
        from intent. For each term, ask: "Can the policy score high
        on this WITHOUT doing what I want?" Common gaps:

        - **Unconditional quantities**: Rewarding X without requiring
          related Y → cheapest path to X (e.g., rotation without
          height → ground-spin). Fix: cross-condition (A × B).
        - **Repeated bonuses**: Milestone fires every step the
          condition holds. Fix: one-time boolean flag per episode.
        - **Post-completion drift**: Policy keeps earning reward after
          goal is done. Fix: stabilization penalty gated on completion.
        - **Scale dominance**: One term >> others → policy ignores all
          else. Fix: no term > 60% of typical total.

        **5. Scale & stability**
        - Keep total magnitude within 2-3× of previous iteration.
        - New terms start at LOW weight (0.1-0.3× of main signal).
        - Reward scale restructuring makes value estimates stale.

        {hp_tuning_section}
        {_build_change_constraint(is_plateau)}

        In your "## {_changes_heading}" section, you MUST explicitly list:
        1. What EXACTLY you are changing (max 2 items)
        2. What you are deliberately leaving unchanged and why

        ━━━ CROSS-ITERATION STRATEGY ━━━
        - Never revert to a failed structure without a new hypothesis.
        - Build incrementally on the best-scoring iteration.
        - Track what was tried: if a modification appeared in past iterations
          and did not improve the score, do NOT re-try it. Try something new.
        - After 3+ iterations without beating the best score, coefficient
          tuning on existing terms is unlikely to help. Consider structural
          changes: new reward signals, different decompositions, or
          phase/gating redesign.
        {cross_iteration_hp_note}

        ━━━ NO_CHANGE OPTION ━━━
        If you have exhausted all promising modifications and cannot identify
        a genuinely new, untried improvement, you MAY return the best
        iteration's reward code UNCHANGED. This is strictly better than
        cycling through previously failed changes.
        To use NO_CHANGE: {_no_change_instruction}

        ━━━ REWARD FUNCTION CONTRACT ━━━
        {reward_contract}

        The code MUST define `reward_fn` at module level (directly or
        via `reward_fn = _make_reward()`).

        {hp_contract_note}
        {side_info_reminder}
        ━━━ OUTPUT FORMAT (strict, all {n_sections} sections) ━━━

        ## Diagnosis
        <6-step reasoning from the diagnostic process above>

        ## Lesson
        Retrospective: what did you learn from THIS iteration's training
        result? What went wrong or right, and what should future iterations
        avoid repeating or build upon?
        Format: <title> — <what happened and why>. <Actionable rule>.
        Example: "Height-gating rotation prevents ground rolling — adding
        height > 0.3 gate before rotation reward eliminated ground-roll
        exploit. Without the gate the agent finds it easier to spin on the
        ground than jump. Always gate airborne rewards on minimum height."
        2-4 sentences. Be specific to what actually happened, not generic.

        **Multi-config mode**: Your lesson MUST compare ALL config variants.
        State which HP settings performed best/worst, WHY (from training
        dynamics and judgment), and what HP directions to explore next.
        Example: "Higher lr (1e-3) outperformed lower lr (3e-4) by 20%,
        suggesting the reward landscape is smooth enough for aggressive
        updates. Low entropy (0.005) caused premature convergence in
        config_1. Next iteration should keep lr=1e-3 and explore
        ent_coef in [0.01, 0.03]."

        ## Based On
        <iteration number whose reward code you are building on>
        Write ONLY the integer. If you are modifying the current iteration's
        code, write the current iteration number. If you are reverting to
        a previous best iteration's code and editing from there, write that
        iteration number. Example: ``2`` means your revised code is based
        on iteration 2's reward function.{_based_on_extra}

        {_reasoning_section}
        {hp_output_format}
    """)


def build_phase2_prompt(
    base_code: str,
    planned_changes: str,
    env: EnvSpec,
    *,
    side_info: bool = False,
) -> tuple[str, str]:
    """Build system and user prompts for Phase 2 (code generation).

    Phase 2 receives verbatim base code from the ``Based On`` iteration
    plus the planned changes from Phase 1, and produces the complete
    revised reward function.

    Returns
    -------
    (system_prompt, user_prompt)
    """
    reward_contract = build_reward_contract(env, side_info=side_info)
    sig = _reward_sig(side_info)
    engine_label = engine_display_name(env.engine)

    system_prompt = textwrap.dedent(f"""\
        You are a reward function code editor for {engine_label} RL environments.
        Your ONLY job is to apply planned changes to an existing reward
        function precisely and completely.

        Rules:
        - Output a COMPLETE ``reward_fn`` (or ``_make_reward`` closure).
        - Apply the planned changes exactly as described.
        - Do NOT add, remove, or modify anything beyond what the planned
          changes specify.
        - The code MUST define ``reward_fn`` at module level (directly or
          via ``reward_fn = _make_reward()``).
        - Return ``(float, dict[str, float])`` -- total reward and named terms.

        ━━━ REWARD FUNCTION CONTRACT ━━━
        {reward_contract}

        ━━━ OUTPUT FORMAT ━━━
        ## Revised Reward Function
        ```python
        def reward_fn({sig}):
            ...
        ```
    """)

    user_prompt = textwrap.dedent(f"""\
        ## Base Reward Function
        ```python
        {base_code}
        ```

        ## Planned Changes
        {planned_changes}

        Apply the planned changes to the base reward function above.
        Output the complete revised function.
    """)

    return system_prompt, user_prompt


def _format_checkpoint_judgments(judgment: dict) -> str:
    """Format all per-checkpoint judgments into a readable section.

    Shows score progression and trajectory metrics across training checkpoints
    so the revise agent can assess consistency vs. lucky outliers and
    cross-reference scores with physical metrics.
    """
    cp_judgments = judgment.get("checkpoint_judgments", {})
    if not cp_judgments:
        # No per-checkpoint data — fall back to single judgment display
        score = judgment.get("intent_score", "N/A")
        diag = judgment.get("diagnosis", "N/A")
        tags = ", ".join(judgment.get("failure_tags", [])) or "none"
        return (
            "## Judgment\n"
            f"- Intent score: {score}/1.0\n"
            f"- Diagnosis: {diag}\n"
            f"- Failure tags: {tags}"
        )

    if len(cp_judgments) == 1:
        # Single checkpoint — show it without consistency analysis
        step = next(iter(cp_judgments))
        cp = cp_judgments[step]
        score = cp.get("intent_score", "N/A")
        diag = cp.get("diagnosis", "N/A")
        tags = ", ".join(cp.get("failure_tags", [])) or "none"
        return (
            "## Judgment (single checkpoint)\n"
            f"- Step: {step}\n"
            f"- Intent score: {score}/1.0\n"
            f"- Diagnosis: {diag}\n"
            f"- Failure tags: {tags}"
        )

    def _safe_int(s: str) -> tuple[int, str]:
        """Sort key: numeric steps first, then lexicographic fallback."""
        try:
            return (0, int(s))  # type: ignore[return-value]
        except (ValueError, TypeError):
            return (1, s)  # type: ignore[return-value]

    def _num(val: object, default: float = 0.0) -> float:
        """Coerce a value to float, falling back to *default* on None/error."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    sorted_steps = sorted(cp_judgments.keys(), key=_safe_int)
    scores = [_num(cp_judgments[s].get("intent_score")) for s in sorted_steps]
    best_step = judgment.get("best_checkpoint", "")

    # Aggregate failure tags across all checkpoints (dedupe, preserve order)
    all_tags: list[str] = []
    for s in sorted_steps:
        cp = cp_judgments[s]
        if not isinstance(cp, dict):
            continue
        for t in cp.get("failure_tags", []):
            if t not in all_tags:
                all_tags.append(t)

    # Check if any checkpoint has per-rollout data
    has_rollout_data = any(
        isinstance(cp_judgments[s], dict)
        and (
            cp_judgments[s].get("rollout_judgments") or cp_judgments[s].get("checkpoint_aggregate")
        )
        for s in sorted_steps
    )

    # Score + diagnosis table
    if has_rollout_data:
        lines = [
            "## All Checkpoint Judgments (multi-rollout)",
            f"({len(sorted_steps)} checkpoints, each with multiple eval rollouts)",
            f"Common failure tags: {', '.join(all_tags) or 'none'}",
            "",
            "| Step | Mean | SucRate | Std | N | Common Failures | Diagnosis |",
            "|------|------|---------|-----|---|-----------------|-----------|",
        ]
        for step in sorted_steps:
            cp = cp_judgments[step]
            if not isinstance(cp, dict):
                lines.append(f"| {step} | N/A | - | - | - | - | (invalid data) |")
                continue
            agg = cp.get("checkpoint_aggregate", {})
            if agg:
                mean_s = _num(agg.get("mean_intent_score"))
                sr = agg.get("success_rate", 0)
                std = _num(agg.get("score_std"))
                n = len(agg.get("rollout_judgments", []))
                ctags = ", ".join(agg.get("common_failure_tags", [])) or "-"
                diag = agg.get("aggregate_diagnosis", "") or ""
                diag_short = diag[:60] + "..." if len(diag) > 60 else (diag or "-")
                marker = " (best)" if step == best_step else ""
                lines.append(
                    f"| {step}{marker} | {mean_s:.3f} | {sr:.0%} | {std:.3f} "
                    f"| {n} | {ctags} | {diag_short} |"
                )
            else:
                # Fallback: single-rollout checkpoint
                score = _num(cp.get("intent_score"))
                tags = ", ".join(cp.get("failure_tags", [])) or "-"
                diag = cp.get("diagnosis", "") or ""
                diag_short = diag[:60] + "..." if len(diag) > 60 else (diag or "-")
                marker = " (best)" if step == best_step else ""
                lines.append(
                    f"| {step}{marker} | {score:.3f} | - | - | 1 | {tags} | {diag_short} |"
                )

        lines.append("")
        lines.append(
            "Use `get_checkpoint_judgments(step, detail='summary')` to see aggregate "
            "stats + best/worst rollouts for a checkpoint. Focus on the LAST checkpoint."
        )
    else:
        lines = [
            "## All Checkpoint Judgments",
            f"({len(sorted_steps)} checkpoints evaluated — "
            f"look for consistency across training progress)",
            f"Common failure tags: {', '.join(all_tags) or 'none'}",
            "",
            "| Step | Score | Failure Tags | Diagnosis (excerpt) |",
            "|------|-------|--------------|---------------------|",
        ]

        for step in sorted_steps:
            cp = cp_judgments[step]
            if not isinstance(cp, dict):
                lines.append(f"| {step} | N/A | - | (invalid data) |")
                continue
            score = _num(cp.get("intent_score"))
            tags = ", ".join(cp.get("failure_tags", [])) or "-"
            diag = cp.get("diagnosis", "") or ""
            diag_short = diag[:80] + "..." if len(diag) > 80 else (diag or "-")
            marker = " (best)" if step == best_step else ""
            lines.append(f"| {step}{marker} | {score:.3f} | {tags} | {diag_short} |")

    # Consistency summary
    if len(scores) >= 2:
        avg = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        lines.append("")
        lines.append(
            f"**Score stats**: avg={avg:.3f}, spread={spread:.3f} "
            f"(max={max(scores):.3f}, min={min(scores):.3f})"
        )

        # Score trend (monotonically improving?)
        improving = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
        declining = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        if improving:
            lines.append(
                "**Score trend**: monotonically improving across checkpoints "
                "— reward is producing consistent training signal."
            )
        elif declining:
            lines.append(
                "**Score trend**: monotonically declining across checkpoints "
                "— policy may be degrading over training or reward is being exploited."
            )
        elif spread > 0.3:
            lines.append(
                "**WARNING**: Large spread + non-monotonic scores — "
                "the high score may be an outlier, not reliable behavior."
            )

    return "\n".join(lines)


def _format_reward_spec(reward_spec: dict | None) -> str:
    """Format structured reward_spec as a concise text section."""
    if not reward_spec:
        return ""

    terms = reward_spec.get("terms", [])
    if not terms:
        return ""

    lines = ["## Reward Spec"]
    latex = reward_spec.get("latex", "")
    if latex:
        lines.append(f"LaTeX: {latex}")
    lines.append("Terms:")
    for term in terms:
        if isinstance(term, dict):
            name = term.get("name", "?")
            desc = term.get("description", "")
            tlatex = term.get("latex", "")
            weight = term.get("weight")
            parts = [f"  - {name}: {desc}"]
            if tlatex:
                parts.append(f"    equation: {tlatex}")
            if weight is not None:
                parts.append(f"    weight: {weight}")
            lines.extend(parts)
    return "\n".join(lines)


def build_revise_user_prompt(
    prompt: str,
    reward_code: str,
    judgment: dict,
    summary: dict,
    dynamics_text: str,
    history_text: str,
    config_text: str,
    best_code_section: str = "",
    hp_tuning: bool = True,
    reward_spec: dict | None = None,
    stagnation_warning: str = "",
) -> str:
    """Assemble the user prompt for the revise agent LLM call."""
    # Per-checkpoint judgment breakdown (all eval checkpoints)
    checkpoint_section = _format_checkpoint_judgments(judgment)

    # Guardrail warnings (if available)
    guardrail = summary.get("guardrail_warning", "")
    guardrail_section = ""
    if guardrail:
        guardrail_section = f"## Guardrail Warnings\n{guardrail}"

    # Multi-config context (if available)
    multi_config_section = ""
    config_judgments = judgment.get("config_judgments")
    all_run_judgments = judgment.get("all_run_judgments")
    if config_judgments:
        run_ids = sorted(all_run_judgments.keys()) if all_run_judgments else []
        mc_lines = [
            "## Multi-Config Training",
            f"- Configs: {len(config_judgments)} ({', '.join(sorted(config_judgments.keys()))})",
            f"- Total runs: {len(run_ids)}",
            f"- Available run_ids: {', '.join(run_ids)}" if run_ids else "",
        ]
        mc_lines.append("- Per-config scores:")
        for cid, cj in sorted(config_judgments.items()):
            ms = cj.get("mean_intent_score", 0)
            ss = cj.get("score_std", 0)
            mc_lines.append(f"  - {cid}: {ms:.3f} ± {ss:.3f}")
        multi_config_section = "\n".join(line for line in mc_lines if line)

    best_block = f"\n{best_code_section}" if best_code_section.strip() else ""
    stagnation_block = f"\n{stagnation_warning}\n" if stagnation_warning else ""

    # Structured reward spec section (concise term overview)
    spec_section = _format_reward_spec(reward_spec)

    return textwrap.dedent(f"""\
        ## Goal
        {prompt}

        ## Current Reward Function
        ```python
        {reward_code}
        ```
        {best_block}
        {spec_section}

        ## Training Summary
        - Final episodic return: {summary.get("final_episodic_return", "N/A")}
        - Total timesteps: {summary.get("total_timesteps", "N/A")}
        - Training time: {summary.get("training_time_s", 0):.1f}s
        - Total episodes: {summary.get("total_episodes", "N/A")}

        {checkpoint_section}

        {guardrail_section}

        {multi_config_section}

        {dynamics_text}

        {history_text}

        {config_text}
        {stagnation_block}
        Follow the 6-step diagnostic process (behavior gap → checkpoint
        consistency → reward root-cause → training dynamics → cross-iteration
        patterns → proposal). Pay special attention to cross-checkpoint
        consistency: if only one checkpoint scores high while others are low,
        the reward may be producing lucky outliers rather than reliable
        behavior.{
        " If you changed reward scale significantly, address"
        " the implications for the value function and"
        " whether HP adjustments are needed."
        if hp_tuning
        else " Focus exclusively on reward function improvements."
    }

        {
        ""
        if stagnation_warning
        else (
            "MANDATORY: Make at most 1-2 targeted changes. In your Reward Reasoning, "
            "you MUST explicitly state: (1) what you are changing, (2) what you are "
            "keeping unchanged. If the current iteration regressed, revert to the "
            "best iteration's code and make at most 1-2 smaller changes."
        )
    }

        **Multi-rollout evaluation**: Each checkpoint was evaluated with multiple
        rollouts (not just one). Use `get_checkpoint_judgments(step, detail='summary')`
        on the LAST checkpoint to see aggregate stats + best/worst cases.
        success_rate and common_failure_tags across rollouts are far more reliable
        than any single rollout's score.
        A high mean score with high success rate = consistent behavior. A high
        mean score with low success rate = inconsistent, may be lucky outliers.
    """)


def build_generate_user_prompt(
    prompt: str,
    config_text: str,
) -> str:
    """Assemble the user prompt for the first iteration (initial reward generation).

    Uses the same 5-section output format as revision so that parsing is
    identical.  The "Diagnosis" section becomes an intent analysis instead.
    """
    return textwrap.dedent(f"""\
        ## Goal
        {prompt}

        This is the **first iteration** — there is no previous reward function,
        no training history, and no judgment to revise from. Your task is to
        design a reward function from scratch.

        Analyze the goal carefully before writing code:
        1. What specific behavior does this goal require?
        2. Is this a sustained state/rate, a cumulative achievement, or a
           sequential/phased task? Choose the appropriate reward structure.
        3. What would a perfectly rational agent do to maximize your reward?
           If that differs from the intent, fix the design before writing code.

        {config_text}

        Use the same output format:

        ## Diagnosis
        <Intent analysis: what behavior is needed and which reward structure
        (per-step / cumulative progress / phased milestones) fits best>

        ## Lesson
        <Initial design hypothesis — what reward structure you chose and why,
        what you expect to work or need watching. 2-4 sentences.>

        ## Based On
        0

        ## Reward Reasoning
        <Design rationale: which terms you chose and why>

        ## Revised Reward Function
        ```python
        def reward_fn(...):
            ...
        ```

        ## HP Reasoning
        <Whether default HPs are appropriate, or if the task needs adjustments>

        ## HP Changes
        ```json
        {{}}
        ```
    """)
