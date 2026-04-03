"""Revise Agent: LLM-driven reward & HP revision.

Uses training dynamics analysis from ``training_dynamics`` module, reviews
iteration history, and asks Claude for both revised reward code **and**
hyperparameter adjustments.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from p2p.agents.revise_tool_dispatch import ReviseToolDispatch
from p2p.agents.reward_author import _sanitize_unicode
from p2p.agents.reward_author import revise as _simple_revise
from p2p.analysis.training_dynamics import (
    analyze_training_curves,
    format_current_config,
    format_iteration_history,
    format_training_dynamics,
)
from p2p.config import BOOL_HP_KEYS, HP_BOUNDS, TrainConfig
from p2p.contracts import Phase1Result, ReviseResult, TrainingDynamics
from p2p.inference.agent_tools import (
    dispatch_tool_calls,
    emit_conversation,
    serialize_assistant_response,
)
from p2p.inference.llm_client import create_message, extract_response_text
from p2p.prompts.revise_agent import (
    build_generate_user_prompt,
    build_phase2_prompt,
    build_revise_system_prompt,
    build_revise_user_prompt,
)
from p2p.settings import LLM_MODEL

if TYPE_CHECKING:
    import anthropic

    from p2p.training.env_spec import EnvSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HP validation
# ---------------------------------------------------------------------------


def _normalize_hp_values(changes: dict) -> dict:
    """Convert ``[old, new]`` pairs to plain new values.

    The revise agent is instructed to output ``{"param": [old, new]}``.
    This extracts the new value so downstream code sees a flat dict.
    Also handles the case where the LLM outputs plain values (backward compat).
    """
    out: dict[str, Any] = {}
    for k, v in changes.items():
        if isinstance(v, list) and len(v) == 2:
            out[k] = v[1]  # [old, new] -> new
        else:
            out[k] = v
    return out


def validate_hp_changes(changes: dict) -> dict:
    """Strip unsafe keys, clamp values to bounds. Returns cleaned dict."""
    tunable = TrainConfig._TUNABLE_KEYS
    cleaned: dict[str, Any] = {}
    for k, v in changes.items():
        if k not in tunable:
            log.warning("Stripping non-tunable HP: %s", k)
            continue
        if k in BOOL_HP_KEYS:
            if not isinstance(v, bool):
                log.warning("Stripping non-bool value for %s: %s", k, v)
                continue
        elif k in HP_BOUNDS:
            lo, hi = HP_BOUNDS[k]
            if isinstance(v, (int, float)):
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                # Preserve int type for int-valued params
                if k in (
                    "num_steps",
                    "update_epochs",
                    "num_minibatches",
                    "total_timesteps",
                    "max_episode_steps",
                ):
                    v = int(v)
        cleaned[k] = v
    return cleaned


def _extract_section(text: str, header: str, stop_before: str) -> str:
    """Extract text between ``## header`` and ``## stop_before``."""
    m = re.search(
        rf"##\s*{header}\s*\n(.*?)(?=##\s*(?:{stop_before}))",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _extract_code_block(text: str, header: str, lang: str = "python") -> str | None:
    """Extract a fenced code block under ``## header``. Returns None if absent."""
    m = re.search(
        rf"##\s*{header}\s*\n.*?```(?:{lang})?\s*\n(.+?)```",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _parse_hp_changes(text: str) -> tuple[dict[str, Any], str]:
    """Extract and validate HP reasoning + HP changes JSON from *text*."""
    hp_reasoning = _extract_section(text, "HP Reasoning", "HP Changes")
    raw = _extract_code_block(text, "HP Changes", "json")
    hp_changes: dict[str, Any] = {}
    if raw:
        try:
            hp_changes = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Failed to parse HP changes JSON, ignoring")
    if hp_changes:
        hp_changes = validate_hp_changes(_normalize_hp_values(hp_changes))
    return hp_changes, hp_reasoning


def _parse_preamble(text: str, next_section: str) -> tuple[str, str, int]:
    """Extract diagnosis, lesson, and based_on from structured LLM response.

    *next_section* is the section that follows Based On (e.g.
    ``"Reward Reasoning"`` or ``"Planned Changes"``).

    Returns ``(diagnosis, lesson, based_on)``.
    """
    diagnosis = _extract_section(text, "Diagnosis", f"Lesson|Based On|{next_section}")
    lesson = _extract_section(text, "Lesson", f"Based On|{next_section}")
    based_on_raw = _extract_section(text, "Based On", next_section)
    based_on = 0
    if based_on_raw:
        m = re.search(r"\d+", based_on_raw)
        if m:
            based_on = int(m.group())
    return diagnosis, lesson, based_on


def _extract_and_sanitize_reward_code(text: str, header: str = "Revised Reward Function") -> str:
    """Extract, sanitize, and fix up reward code from LLM response text.

    Tries the named *header* code block first, then falls back to any
    fenced Python block.  Applies Unicode sanitization and appends the
    ``reward_fn = _make_reward()`` assignment when missing.

    Raises ``ValueError`` if no valid reward code is found.
    """
    code = _extract_code_block(text, header)
    if not code:
        fallback = re.search(r"```(?:python)?\s*\n(.+?)```", text, re.DOTALL)
        code = fallback.group(1).strip() if fallback else None
    if not code:
        msg = f"No reward function code found under '## {header}'"
        raise ValueError(msg)

    code = _sanitize_unicode(code)

    if "def reward_fn" not in code:
        msg = "Extracted code does not contain 'def reward_fn'"
        raise ValueError(msg)

    # Stateful pattern: if _make_reward is defined but reward_fn = _make_reward()
    # is missing, append it so the module-level name exists after exec().
    if "_make_reward" in code and "reward_fn = _make_reward()" not in code:
        code += "\n\nreward_fn = _make_reward()\n"

    return code


def _parse_revise_response(text: str) -> ReviseResult:
    """Parse structured LLM response into ReviseResult.

    NOTE: NO_CHANGE responses need no special handling — the agent puts
    the best iteration's code unchanged into ``## Revised Reward Function``,
    which this parser extracts normally.
    """
    diagnosis, lesson, based_on = _parse_preamble(text, "Reward Reasoning")
    reward_reasoning = _extract_section(text, "Reward Reasoning", "Revised Reward Function")
    reward_code = _extract_and_sanitize_reward_code(text)
    hp_changes, hp_reasoning = _parse_hp_changes(text)

    return ReviseResult(
        reward_code=reward_code,
        reward_reasoning=reward_reasoning,
        hp_changes=hp_changes,
        hp_reasoning=hp_reasoning,
        diagnosis=diagnosis,
        lesson=lesson,
        based_on=based_on,
    )


class Phase2Error(ValueError):
    """Raised when Phase 2 code generation fails after retries."""


# ---------------------------------------------------------------------------
# Two-phase revision helpers
# ---------------------------------------------------------------------------


def _parse_phase1_response(text: str) -> Phase1Result:
    """Parse Phase 1 output (diagnosis + plan, no code).

    Extracts all fields except reward code.  Raises ``ValueError`` if
    ``planned_changes`` is empty (signals the LLM did not follow the
    two-phase format and the caller should fall back to single-phase).
    """
    diagnosis, lesson, based_on = _parse_preamble(text, "Planned Changes")
    planned_changes = _extract_section(text, "Planned Changes", "HP Reasoning|HP Changes")
    if not planned_changes:
        msg = "Phase 1 response missing '## Planned Changes' section"
        raise ValueError(msg)

    hp_changes, hp_reasoning = _parse_hp_changes(text)

    return Phase1Result(
        diagnosis=diagnosis,
        lesson=lesson,
        based_on=based_on,
        planned_changes=planned_changes,
        hp_changes=hp_changes,
        hp_reasoning=hp_reasoning,
    )


def _resolve_base_code(based_on: int, iterations: list, current_code: str) -> str:
    """Fetch the verbatim reward code from a prior iteration.

    Falls back to *current_code* when *based_on* is non-positive or no
    matching iteration is found.
    """
    if based_on <= 0:
        return current_code
    for it in iterations:
        num = it.get("iteration", 0) if isinstance(it, dict) else getattr(it, "iteration", 0)
        if num == based_on:
            code = (
                it.get("reward_code", "")
                if isinstance(it, dict)
                else getattr(it, "reward_code", "")
            )
            if code:
                return code
    log.warning(
        "based_on=%d not found in %d iterations, falling back to current code",
        based_on,
        len(iterations),
    )
    return current_code


def _run_phase2(
    base_code: str,
    planned_changes: str,
    *,
    env: "EnvSpec",
    side_info: bool,
    client: "anthropic.Anthropic",
    model: str,
    thinking_effort: str = "",
) -> str:
    """Execute Phase 2: apply planned changes to base code via LLM.

    Makes a single ``create_message`` call (no tools).  Retries once on
    parse failure.  Raises ``Phase2Error`` if the code cannot be extracted
    after the retry.
    """
    system_prompt, user_prompt = build_phase2_prompt(
        base_code,
        planned_changes,
        env,
        side_info=side_info,
    )

    prev_text = ""
    last_error = ""
    for attempt in range(2):
        if attempt == 0:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            # Retry with correction prompt
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": prev_text},
                {
                    "role": "user",
                    "content": (
                        f"Your response could not be parsed: {last_error}\n\n"
                        "Please output the complete revised function inside:\n"
                        "## Revised Reward Function\n```python\n...\n```"
                    ),
                },
            ]

        response = create_message(
            client,
            model=model,
            thinking_effort=thinking_effort,
            system=system_prompt,
            messages=messages,
        )
        prev_text = extract_response_text(response)

        try:
            return _extract_and_sanitize_reward_code(prev_text)
        except ValueError as exc:
            last_error = str(exc)
            log.warning("Phase 2 attempt %d failed: %s", attempt + 1, last_error)

    raise Phase2Error(f"Phase 2 code extraction failed after 2 attempts: {last_error}")


# ---------------------------------------------------------------------------
# Tool definitions for the revise agent
# ---------------------------------------------------------------------------

REVISE_TOOLS = [
    {
        "name": "get_iteration_reward_code",
        "description": (
            "Get the full reward function code from a past iteration. "
            "In multi-config mode, the reward function is shared across all "
            "configs within an iteration, so no run_id is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number to retrieve.",
                }
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "get_iteration_training_dynamics",
        "description": (
            "Get training dynamics analysis (entropy, value loss, KL, "
            "clip fraction, explained variance, return trend) for a past iteration. "
            "In multi-config mode, use run_id to query a specific config×seed run "
            "(default: best run)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number to retrieve.",
                },
                "run_id": {
                    "type": "string",
                    "description": (
                        "Config×seed run ID (format: '{config_id}_seed_{seed}', "
                        "e.g. 'baseline_seed_1', 'config_0_seed_2'). "
                        "Defaults to 'best' (the best run)."
                    ),
                },
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "compare_iterations",
        "description": (
            "Compare two iterations: shows score diff, failure tag diff, "
            "and a unified diff of their reward code. "
            "In multi-config mode, also includes per-config score breakdown "
            "for each iteration. The reward diff is always meaningful since "
            "reward code is shared across configs within an iteration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iter_a": {
                    "type": "integer",
                    "description": "First iteration number.",
                },
                "iter_b": {
                    "type": "integer",
                    "description": "Second iteration number.",
                },
            },
            "required": ["iter_a", "iter_b"],
        },
    },
    {
        "name": "get_iteration_judgment_detail",
        "description": (
            "Get the judgment for a past iteration: score, "
            "diagnosis, failure_tags. "
            "By default returns the best run's judgment. "
            "In multi-config mode, use run_id to query a specific "
            "config×seed run, or omit for best. Also includes "
            "per-config score summary when multi-config data is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number to retrieve.",
                },
                "run_id": {
                    "type": "string",
                    "description": (
                        "Config×seed run ID (format: '{config_id}_seed_{seed}', "
                        "e.g. 'baseline_seed_1', 'config_0_seed_2'). "
                        "Defaults to 'best'. Only relevant in multi-config mode."
                    ),
                },
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "get_checkpoint_judgments",
        "description": (
            "Get rollout judgments and aggregate statistics for a specific "
            "eval checkpoint step in the CURRENT iteration. "
            "In multi-config mode, use run_id to query a specific run "
            "(default: best run). "
            "detail='aggregate' (default): aggregate stats only (mean score, "
            "success rate, std, common failure tags). Most token-efficient. "
            "detail='summary': aggregate + best, median, and worst rollout diagnoses. "
            "detail='all': aggregate + every rollout's full diagnosis. "
            "Start with 'aggregate' or 'summary'; use 'all' only when needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "Checkpoint step label (e.g., '245760').",
                },
                "run_id": {
                    "type": "string",
                    "description": (
                        "Config×seed run ID (format: '{config_id}_seed_{seed}', "
                        "e.g. 'baseline_seed_1', 'config_0_seed_2'). "
                        "Defaults to 'best'. Only relevant in multi-config mode."
                    ),
                },
                "detail": {
                    "type": "string",
                    "enum": ["aggregate", "summary", "all"],
                    "description": (
                        "Level of detail: 'aggregate' (default, stats only), "
                        "'summary' (aggregate + best/median/worst rollouts), "
                        "'all' (aggregate + every rollout)."
                    ),
                },
            },
            "required": ["step"],
        },
    },
    {
        "name": "get_rollout_judgment",
        "description": (
            "Get the detailed judgment for a single rollout at a specific "
            "checkpoint in the CURRENT iteration. Includes intent_score, diagnosis, "
            "failure_tags, and eval_return. "
            "Look up by rollout_label (e.g. 'p10', 'median', 'p90') for parallel "
            "eval, or by episode_idx (integer) for sequential eval. "
            "In multi-config mode, use run_id to query a specific config×seed run "
            "(default: best run)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "Checkpoint step label.",
                },
                "rollout_label": {
                    "type": "string",
                    "description": (
                        "Rollout label (e.g. 'p10', 'median', 'p90'). "
                        "Use this for parallel eval rollouts."
                    ),
                },
                "episode_idx": {
                    "type": "integer",
                    "description": (
                        "Episode index (0-based). Use this for sequential eval rollouts."
                    ),
                },
                "run_id": {
                    "type": "string",
                    "description": (
                        "Config×seed run ID (format: '{config_id}_seed_{seed}', "
                        "e.g. 'baseline_seed_1', 'config_0_seed_2'). "
                        "Defaults to 'best'. Only relevant in multi-config mode."
                    ),
                },
            },
            "required": ["step"],
        },
    },
    {
        "name": "get_config_comparison",
        "description": (
            "Get cross-config comparison for multi-config training in the "
            "CURRENT iteration. Shows how each config×seed combination performed. "
            "detail='aggregate' (default): per-config mean score, std, return. "
            "detail='summary': aggregate + best/worst seed per config. "
            "detail='all': aggregate + every seed's full judgment. "
            "Returns error in single-config mode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["aggregate", "summary", "all"],
                    "description": (
                        "Level of detail: 'aggregate' (default), "
                        "'summary' (+ best/worst seed), "
                        "'all' (+ every seed judgment)."
                    ),
                },
            },
        },
    },
    {
        "name": "get_iteration_scores",
        "description": (
            "Get a compact score timeline across ALL past iterations with trend "
            "analysis. Highlights the best-scoring iteration and returns its "
            "reward code so you can build on what worked. "
            "Use this FIRST to understand the improvement trajectory and anchor "
            "your revision on the best-performing iteration. "
            "Optional top_k returns only the top-K scoring iterations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Return only the top-K iterations by score. Default: all iterations."
                    ),
                },
            },
        },
    },
    {
        "name": "get_strategy_summary",
        "description": (
            "Strategic 'tree of thoughts' summary of the FULL iteration history. "
            "Groups iterations into score-trend phases, logs every regression "
            "and its cause, surfaces persistent failure patterns, and detects "
            "HP/reward oscillation. Call this to avoid repeating mistakes or "
            "losing the big picture. Token-efficient: returns structured "
            "analysis, not raw data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_experiment_lineage",
        "description": (
            "Get the experiment lineage tree and accumulated lessons for this session. "
            "Shows all iterations as a tree structure with each node's score, "
            "what was tried, and the lesson learned from its outcome. "
            "The 'lessons' list contains distilled insights from the session. "
            "Call this FIRST to see what has been tried and avoid repeating "
            "failed approaches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "update_experiment_lessons",
        "description": (
            "Change one lesson's tier. Use the 1-based index from the lessons "
            "display and the target tier. A reason is required to maintain an "
            "audit trail.\n\n"
            "Tier semantics:\n"
            "- HARD: Catastrophic failure, always enforced. Only you can set this.\n"
            "- STRONG: Confirmed principle, should respect.\n"
            "- SOFT: Context-specific, may challenge freely.\n"
            "- RETIRED: No longer active, kept for reference. Use this for "
            "lessons proven wrong — they stay visible so the same mistake is "
            "not rediscovered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": (
                        "1-based index of the lesson to change. "
                        "Matches the numbering in the lessons display."
                    ),
                },
                "tier": {
                    "type": "string",
                    "enum": ["HARD", "STRONG", "SOFT", "RETIRED"],
                    "description": "New tier for the lesson.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why this tier change is warranted. "
                        "E.g. 'Proven wrong: obj_pos is world frame, not local.'"
                    ),
                },
            },
            "required": ["index", "tier", "reason"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch (delegated to ReviseToolDispatch)
# ---------------------------------------------------------------------------


def _get_judgment(it: Any) -> dict:
    """Extract judgment dict from an iteration (dict or dataclass)."""
    if hasattr(it, "judgment"):
        return it.judgment if isinstance(it.judgment, dict) else {}
    return it.get("judgment", {})


def _get_field(it: Any, field: str, default: Any = "") -> Any:
    """Generic accessor for iteration fields (dict or dataclass)."""
    if hasattr(it, field):
        return getattr(it, field)
    return it.get(field, default) if isinstance(it, dict) else default


def _compute_stagnation_warning(
    iterations: list,
    best_iteration: int,
    best_score: float,
) -> str:
    """Detect plateau/oscillation and return a warning string for the prompt.

    Auto-injected into the user prompt so the agent always sees it,
    regardless of whether it calls ``get_strategy_summary()``.
    Returns an empty string when no stagnation is detected.

    When many iterations have branched from the same best iteration
    without improvement, the warning escalates: the normal 1-2 change
    constraint is explicitly lifted and the agent is encouraged to make
    structural rewrites rather than incremental tweaks.
    """
    if len(iterations) < 3:
        return ""

    # Build per-iteration timeline
    timeline: list[dict] = []
    for it in iterations:
        num = it.iteration if hasattr(it, "iteration") else it.get("iteration", 0)
        j = _get_judgment(it)
        score = j.get("intent_score", 0) if isinstance(j, dict) else 0
        reasoning = _get_field(it, "reward_reasoning", "")
        timeline.append({"iter": num, "score": round(score, 3), "reasoning": reasoning})
    timeline.sort(key=lambda e: e["iter"])

    # --- Plateau detection: how many iterations since best score was beaten? ---
    iters_since_best = len(timeline) - best_iteration
    is_plateau = iters_since_best >= 3

    # --- Tried changes ledger: extract what was tried and outcome ---
    tried: list[str] = []
    for i, entry in enumerate(timeline):
        if entry["iter"] <= 1 or not entry["reasoning"]:
            continue
        prev_score = timeline[i - 1]["score"] if i > 0 else 0
        delta = entry["score"] - prev_score
        if delta > 0.02:
            outcome = "improved"
        elif delta < -0.02:
            outcome = "regressed"
        else:
            outcome = "no effect"
        # Truncate reasoning to first line / 120 chars for compactness
        reason_short = entry["reasoning"].split("\n")[0][:120]
        tried.append(f"  iter {entry['iter']}: [{outcome}] {reason_short}")

    if not is_plateau:
        return ""

    # Build warning
    lines = [
        "## STAGNATION WARNING (auto-detected)",
        f"Best score {best_score:.3f} was achieved at iteration {best_iteration}.",
        f"{iters_since_best} iterations have passed without improvement.",
        "",
    ]

    if tried:
        lines.append("### Changes tried so far and their outcomes:")
        lines.extend(tried[-15:])  # Last 15 to keep prompt size reasonable
        lines.append("")

    # Escalation: after many failed attempts from the same parent,
    # lift the 1-2 change constraint and push for structural novelty.
    lines.extend(
        [
            "### Anti-stagnation rules:",
            "- Do NOT re-try a change that previously had 'no effect' or 'regressed'.",
            "- If the same type of modification (same coefficient, same penalty term)",
            "  appears multiple times above, it is EXHAUSTED — try something different.",
            "",
            "### ESCALATION: 1-2 change limit is LIFTED",
            "The normal 1-2 change constraint does NOT apply during plateau.",
            "You have been branching from the same best iteration for many",
            "attempts with only incremental tweaks — all failed.  Incremental",
            "changes are exhausted.",
            "",
            "You are now REQUIRED to make a STRUCTURAL rewrite:",
            "- Redesign the reward decomposition from scratch (new term structure,",
            "  not just swapping one component of the existing structure).",
            "- Try a fundamentally different reward philosophy: phase-based rewards,",
            "  curriculum/staged rewards, contact-force-based signals, imitation-style",
            "  reference tracking, or energy-based formulations.",
            "- You may challenge STRONG lessons — demote them to SOFT or RETIRED via",
            "  update_experiment_lessons(index=N, tier='SOFT', reason='...').",
            "  HARD lessons remain enforced even during plateau.",
            "- If you find lessons that are WRONG or counterproductive, retire them via",
            "  update_experiment_lessons(index=N, tier='RETIRED', reason='...').",
            "  Retired lessons stay visible so the same mistake is not rediscovered.",
            "- You may change multiple terms, coefficients, and structure simultaneously.",
            "  Causal attribution matters less than escaping the local optimum.",
            "",
            "If you genuinely cannot identify a structurally novel approach, output",
            "NO_CHANGE: return the best iteration's reward code UNCHANGED.",
            "This is better than cycling through the same failed modifications.",
        ]
    )

    return "\n".join(lines)


def _build_tool_dispatch(
    iterations: list,
    current_judgment: dict | None = None,
    lineage: dict | None = None,
    session_dir: str | Path | None = None,
) -> dict:
    """Build a dispatch table mapping tool names to handler functions."""
    return ReviseToolDispatch(
        iterations, current_judgment, lineage=lineage, session_dir=session_dir
    ).build_dispatch()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


MAX_PARSE_RETRIES = 3


def generate(
    prompt: str,
    *,
    config: TrainConfig,
    client: anthropic.Anthropic,
    env: EnvSpec,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> str:
    """Generate the initial reward function using the ReviseAgent prompt.

    This replaces ``reward_author.generate()`` so that the richer reward
    design guidance in the ReviseAgent system prompt is used from the start.
    Returns the reward code string (same contract as ``reward_author.generate``).
    """
    system_prompt = build_revise_system_prompt(env, side_info=config.side_info)
    config_text = format_current_config(config)
    user_prompt = build_generate_user_prompt(prompt, config_text)

    log.info("Generating initial reward via ReviseAgent (model=%s)", model)
    response = create_message(
        client,
        model=model,
        thinking_effort=thinking_effort,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = extract_response_text(response)

    for parse_attempt in range(MAX_PARSE_RETRIES):
        try:
            result = _parse_revise_response(text)
            log.info(
                "Initial reward generated (attempt %d): %d HP changes",
                parse_attempt + 1,
                len(result["hp_changes"]),
            )
            return result["reward_code"]
        except ValueError as exc:
            if parse_attempt == MAX_PARSE_RETRIES - 1:
                break
            log.warning(
                "Generate parsing failed (attempt %d/%d: %s), retrying",
                parse_attempt + 1,
                MAX_PARSE_RETRIES,
                exc,
            )
            response = create_message(
                client,
                model=model,
                thinking_effort=thinking_effort,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Your response could not be parsed: {exc}\n\n"
                            "Please rewrite your response using the exact "
                            "section format:\n"
                            "## Diagnosis\n## Lesson\n## Based On\n## Reward Reasoning\n"
                            "## Revised Reward Function\n```python\n...\n```\n"
                            "## HP Reasoning\n## HP Changes\n```json\n...\n```"
                        ),
                    },
                ],
            )
            text = extract_response_text(response)

    # Fallback: use simple reward_author.generate
    log.warning("ReviseAgent generate failed after %d attempts, falling back", MAX_PARSE_RETRIES)
    from p2p.agents.reward_author import generate as _simple_generate

    return _simple_generate(
        prompt,
        model=model,
        client=client,
        env=env,
        side_info=config.side_info,
        thinking_effort=thinking_effort,
    )


_serialize_assistant = serialize_assistant_response


def _build_shared_context(
    prompt: str,
    reward_code: str,
    judgment: dict,
    summary: dict,
    *,
    config: TrainConfig,
    iterations: list,
    scalars_path: Path,
    env: EnvSpec,
    best_iteration: int = 0,
    best_score: float = 0.0,
    has_tools: bool = False,
    hp_tuning: bool = True,
    two_phase: bool | None = None,
) -> tuple[str, str, str, TrainingDynamics]:
    """Compute shared analysis context for revise calls (dynamics, prompts).

    *two_phase* defaults to ``True`` when iterations exist (and the caller
    supports it).  Pass ``False`` explicitly to disable.

    Returns (system_prompt, user_prompt, dynamics_text, dynamics).
    """
    dynamics = analyze_training_curves(scalars_path)
    dynamics_text = format_training_dynamics(dynamics)
    history_text, best_code_section = format_iteration_history(
        iterations, best_iteration=best_iteration, best_score=best_score
    )
    config_text = format_current_config(config) if hp_tuning else ""

    # Auto-detect stagnation/oscillation and inject warning
    stagnation_warning = _compute_stagnation_warning(iterations, best_iteration, best_score)
    is_plateau = bool(stagnation_warning)

    # Enable two-phase revision when there is prior iteration history
    # so Phase 2 can mechanically fetch the real base code.
    if two_phase is None:
        two_phase = bool(iterations)

    system_prompt = build_revise_system_prompt(
        env,
        side_info=config.side_info,
        has_tools=has_tools,
        hp_tuning=hp_tuning,
        is_plateau=is_plateau,
        two_phase=two_phase,
    )
    user_prompt = build_revise_user_prompt(
        prompt,
        reward_code,
        judgment,
        summary,
        dynamics_text,
        history_text,
        config_text,
        best_code_section=best_code_section,
        hp_tuning=hp_tuning,
        stagnation_warning=stagnation_warning,
    )
    return system_prompt, user_prompt, dynamics_text, dynamics


def _call_primary_revise(
    system_prompt: str,
    user_prompt: str,
    dynamics_text: str,
    *,
    prompt: str,
    reward_code: str,
    judgment: dict,
    summary: dict,
    client: anthropic.Anthropic,
    env: EnvSpec,
    side_info: bool,
    model: str,
    thinking_effort: str = "",
) -> tuple[ReviseResult, str]:
    """Call LLM for primary revision. Returns (result, raw_response_text)."""
    log.info("Calling revise agent for primary revision (model=%s)", model)
    response = create_message(
        client,
        model=model,
        thinking_effort=thinking_effort,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = extract_response_text(response)

    try:
        result = _parse_revise_response(text)
        result["training_dynamics"] = dynamics_text
        log.info(
            "Revise parsed: %d HP changes, reasoning=%d chars",
            len(result["hp_changes"]),
            len(result["reward_reasoning"]),
        )
        return result, text
    except ValueError as exc:
        log.warning("Revise parsing failed (%s), falling back to simple revision", exc)
        fallback_code, fallback_hp = _simple_revise(
            prompt,
            reward_code,
            judgment,
            summary,
            model=model,
            client=client,
            env=env,
            side_info=side_info,
        )
        validated_hp = validate_hp_changes(_normalize_hp_values(fallback_hp))
        result = ReviseResult(
            reward_code=fallback_code,
            reward_reasoning="",
            hp_changes=validated_hp,
            hp_reasoning="",
            training_dynamics=dynamics_text,
            diagnosis="",
            lesson="",
            based_on=0,
        )
        # Reconstruct text for variant context
        fallback_text = (
            f"## Revised Reward Function\n```python\n{fallback_code}\n```\n\n"
            f"## HP Changes\n```json\n{json.dumps(validated_hp, indent=2)}\n```"
        )
        return result, fallback_text


def revise(
    prompt: str,
    reward_code: str,
    judgment: dict,
    summary: dict,
    *,
    config: TrainConfig,
    iterations: list,
    scalars_path: Path,
    client: anthropic.Anthropic,
    env: EnvSpec,
    model: str = LLM_MODEL,
    best_iteration: int = 0,
    best_score: float = 0.0,
    max_tool_rounds: int = 32,
    hp_tuning: bool = True,
    session_dir: str | Path | None = None,
    thinking_effort: str = "",
) -> ReviseResult:
    """Run the full revise pipeline: analyze → format → LLM → parse.

    Falls back to simple revision (reward code only) if parsing fails.
    When *hp_tuning* is False, HP sections are omitted from the prompt
    and any HP changes in the response are discarded (reward-only mode).
    When *session_dir* is provided, loads the session experiment lineage
    and exposes it via the tool set.
    """
    # Load session lineage if available (exposed via tool, not injected)
    lineage = None
    if session_dir is not None:
        from p2p.session.lineage import load_lineage

        lineage = load_lineage(session_dir)
        if not lineage["iterations"]:
            lineage = None

    use_tools = bool(iterations)

    system_prompt, user_prompt, dynamics_text, _ = _build_shared_context(
        prompt,
        reward_code,
        judgment,
        summary,
        config=config,
        iterations=iterations,
        scalars_path=scalars_path,
        env=env,
        best_iteration=best_iteration,
        best_score=best_score,
        has_tools=use_tools,
        hp_tuning=hp_tuning,
    )

    # Lineage is available via the get_experiment_lineage tool (not injected
    # into the prompt) so the agent can query it on demand without bloating
    # every request with the full tree.  At iteration 1 (no tools), lineage
    # is irrelevant — the agent is generating the initial reward function.

    # Call Claude (with tools if we have iteration history)
    log.info(
        "Calling revise agent (model=%s, iterations=%d, lineage=%d nodes)",
        model,
        len(iterations),
        len(lineage["iterations"]) if lineage else 0,
    )

    call_kwargs: dict[str, Any] = {
        "model": model,
        "thinking_effort": thinking_effort,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if use_tools:
        call_kwargs["tools"] = REVISE_TOOLS

    response = create_message(client, **call_kwargs)

    # Tool-use loop (if tools are enabled)
    # Build conversation_log for full debugging visibility
    conversation_log: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        _serialize_assistant(response),
    ]

    if use_tools:
        tool_dispatch = _build_tool_dispatch(
            iterations,
            current_judgment=judgment,
            lineage=lineage,
            session_dir=session_dir,
        )
        messages = call_kwargs["messages"].copy()

        for _round_num in range(max_tool_rounds):
            if response.stop_reason == "tool_use":
                tool_results = dispatch_tool_calls(response, tool_dispatch, agent_name="revise")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                conversation_log.append({"role": "tool_results", "content": tool_results})

                response = create_message(
                    client,
                    model=model,
                    thinking_effort=thinking_effort,
                    system=system_prompt,
                    tools=REVISE_TOOLS,
                    messages=messages,
                )
                conversation_log.append(_serialize_assistant(response))
            elif response.stop_reason == "end_turn":
                break

        # Forced final turn: if the model still wants tools after
        # exhausting max_tool_rounds, call once more WITHOUT tools
        # so it is forced to produce the text output.
        if response.stop_reason == "tool_use":
            log.warning(
                "Revise agent exhausted %d tool rounds, forcing final turn",
                max_tool_rounds,
            )
            tool_results = dispatch_tool_calls(response, tool_dispatch, agent_name="revise")
            if tool_results:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                conversation_log.append({"role": "tool_results", "content": tool_results})
            else:
                log.warning("tool_use stop_reason but no tool_use blocks found")

            # Final call WITHOUT tools — forces end_turn with text
            response = create_message(
                client,
                model=model,
                thinking_effort=thinking_effort,
                system=system_prompt,
                messages=messages,
            )
            conversation_log.append(_serialize_assistant(response))
            if response.stop_reason != "end_turn":
                log.warning("Final forced turn stop_reason: %s", response.stop_reason)

    emit_conversation("revise", model, conversation_log)

    # Extract text and parse
    text = extract_response_text(response)

    # --- Two-phase parsing: try Phase 1 format first ---
    if use_tools:
        try:
            phase1 = _parse_phase1_response(text)
        except ValueError as parse_exc:
            log.info(
                "Two-phase format not detected (%s), falling back to single-phase",
                parse_exc,
            )
            phase1 = None

        if phase1 is not None:
            # Phase 1 parsed — run Phase 2 (errors here are real failures,
            # not format mismatches, so let Phase2Error propagate).
            base_code = _resolve_base_code(phase1["based_on"], iterations, reward_code)
            revised_code = _run_phase2(
                base_code,
                phase1["planned_changes"],
                env=env,
                side_info=config.side_info,
                client=client,
                model=model,
                thinking_effort=thinking_effort,
            )
            result = ReviseResult(
                reward_code=revised_code,
                reward_reasoning=phase1["planned_changes"],
                hp_changes=phase1["hp_changes"],
                hp_reasoning=phase1["hp_reasoning"],
                training_dynamics=dynamics_text,
                diagnosis=phase1["diagnosis"],
                lesson=phase1["lesson"],
                based_on=phase1["based_on"],
            )
            if not hp_tuning:
                result["hp_changes"] = {}
                result["hp_reasoning"] = ""
            log.info(
                "Two-phase revise succeeded: based_on=%d, %d HP changes",
                phase1["based_on"],
                len(result["hp_changes"]),
            )
            return result

    # --- Single-phase parsing (original path / fallback) ---
    try:
        result = _parse_revise_response(text)
        result["training_dynamics"] = dynamics_text
        # Reward-only mode: discard any HP changes the LLM returned
        if not hp_tuning:
            result["hp_changes"] = {}
            result["hp_reasoning"] = ""
        log.info(
            "Revise parsed: %d HP changes, reasoning=%d chars",
            len(result["hp_changes"]),
            len(result["reward_reasoning"]),
        )
        return result
    except ValueError as exc:
        log.warning(
            "Revise parsing failed (%s), falling back to simple revision",
            exc,
        )
        fallback_code, fallback_hp = _simple_revise(
            prompt,
            reward_code,
            judgment,
            summary,
            model=model,
            client=client,
            env=env,
            side_info=config.side_info,
        )
        validated_hp = validate_hp_changes(_normalize_hp_values(fallback_hp)) if hp_tuning else {}
        return ReviseResult(
            reward_code=fallback_code,
            reward_reasoning="",
            hp_changes=validated_hp,
            hp_reasoning="",
            training_dynamics=dynamics_text,
            diagnosis="",
            lesson="",
            based_on=0,
        )


# ---------------------------------------------------------------------------
# Multi-config HP variant revision
# ---------------------------------------------------------------------------

_HP_VARIANT_DIRECTIVES = [
    (
        "Suggest a FUNDAMENTALLY DIFFERENT HP strategy. "
        "If the primary uses high learning rate, try low. "
        "If it uses high entropy, try low. Explore a different region of HP space."
    ),
    (
        "Suggest a CONSERVATIVE HP variant. "
        "Use smaller learning rate, moderate entropy, and standard values. "
        "Prioritize training stability over speed."
    ),
    (
        "Suggest an AGGRESSIVE HP variant. "
        "Use higher learning rate, more exploration (higher ent_coef), "
        "and longer rollouts. Prioritize fast learning even at risk of instability."
    ),
    (
        "Suggest a BALANCED HP variant focusing on sample efficiency. "
        "Optimize update_epochs, num_minibatches, and GAE lambda "
        "for better gradient estimates per sample."
    ),
]


def _parse_hp_only_response(text: str) -> dict:
    """Parse HP-only response (no reward code)."""
    hp_changes, hp_reasoning = _parse_hp_changes(text)
    return {"hp_changes": hp_changes, "hp_reasoning": hp_reasoning}


def revise_multi(
    n_variants: int,
    prompt: str,
    reward_code: str,
    judgment: dict,
    summary: dict,
    *,
    config: TrainConfig,
    iterations: list,
    scalars_path: Path,
    client: anthropic.Anthropic,
    env: EnvSpec,
    model: str = LLM_MODEL,
    best_iteration: int = 0,
    best_score: float = 0.0,
    hp_tuning: bool = True,
    session_dir: str | Path | None = None,
    thinking_effort: str = "",
) -> list[ReviseResult]:
    """Generate 1 shared reward + N HP variants for multi-config training.

    The reward function is shared across all variants (from the primary
    revision).  Each variant gets a different set of hyperparameter changes
    to explore the HP space while keeping returns directly comparable.

    The primary revision uses the full tool-enabled revise() so the LLM
    can query cross-config comparison data via get_config_comparison.

    When *hp_tuning* is False, HP sections are omitted and all variants
    share the same (unchanged) hyperparameters (reward-only mode).
    """
    # 1. Primary revision with full tool access (can query cross-config data)
    primary = revise(
        prompt,
        reward_code,
        judgment,
        summary,
        config=config,
        iterations=iterations,
        scalars_path=scalars_path,
        client=client,
        env=env,
        model=model,
        best_iteration=best_iteration,
        best_score=best_score,
        session_dir=session_dir,
        thinking_effort=thinking_effort,
    )

    if n_variants <= 1:
        return [primary]

    # Build context for HP variants (need system/user prompts for continuation).
    # Use single-phase format since HP variants don't run Phase 2.
    system_prompt, user_prompt, dynamics_text, _ = _build_shared_context(
        prompt,
        reward_code,
        judgment,
        summary,
        config=config,
        iterations=iterations,
        scalars_path=scalars_path,
        env=env,
        best_iteration=best_iteration,
        best_score=best_score,
        hp_tuning=hp_tuning,
        two_phase=False,
    )

    # Reconstruct primary text for variant continuation context
    primary_text = (
        f"## Reward Reasoning\n{primary['reward_reasoning']}\n\n"
        f"## Revised Reward Function\n```python\n{primary['reward_code']}\n```\n\n"
        f"## HP Reasoning\n{primary['hp_reasoning']}\n\n"
        f"## HP Changes\n```json\n{json.dumps(primary['hp_changes'], indent=2)}\n```"
    )

    # Reward-only mode: discard HP changes from primary
    if not hp_tuning:
        primary["hp_changes"] = {}
        primary["hp_reasoning"] = ""

    if n_variants <= 1 or not hp_tuning:
        # In reward-only mode, all variants share the same reward + no HP changes
        return [primary] * n_variants if not hp_tuning and n_variants > 1 else [primary]

    results = [primary]

    # 2. Generate N-1 HP variants (same reward, different HPs)
    for i in range(n_variants - 1):
        directive = _HP_VARIANT_DIRECTIVES[i % len(_HP_VARIANT_DIRECTIVES)]
        variant_msg = (
            f"The reward function above is FINAL and shared across all "
            f"configurations. Do NOT suggest a new reward function.\n\n"
            f"Now suggest variant {i + 2} of {n_variants} — "
            f"an ALTERNATIVE set of hyperparameter changes "
            f"(different from the primary).\n\n"
            f"{directive}\n\n"
            f"Output ONLY:\n"
            f"## HP Reasoning\n(your reasoning)\n\n"
            f"## HP Changes\n```json\n{{...}}\n```"
        )

        log.info("Calling revise agent for HP variant %d/%d", i + 2, n_variants)
        try:
            resp = create_message(
                client,
                model=model,
                thinking_effort=thinking_effort,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": primary_text},
                    {"role": "user", "content": variant_msg},
                ],
            )
            variant_text = extract_response_text(resp)
            hp_result = _parse_hp_only_response(variant_text)
            variant = ReviseResult(
                reward_code=primary["reward_code"],
                reward_reasoning=primary["reward_reasoning"],
                hp_changes=hp_result["hp_changes"],
                hp_reasoning=hp_result["hp_reasoning"],
                training_dynamics=dynamics_text,
                diagnosis=primary["diagnosis"],
                lesson=primary["lesson"],
                based_on=primary["based_on"],
            )
            results.append(variant)
            log.info(
                "HP variant %d parsed: %d changes",
                i + 2,
                len(variant["hp_changes"]),
            )
        except Exception as exc:
            log.warning(
                "HP variant %d generation failed (%s), reusing primary HPs",
                i + 2,
                exc,
            )
            results.append(primary)

    return results
