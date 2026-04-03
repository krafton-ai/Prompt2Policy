"""Intent elicitation: decompose a vague intent into structured behavioral criteria."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import anthropic

from p2p.inference.llm_client import create_message, extract_response_text
from p2p.settings import LLM_MODEL

if TYPE_CHECKING:
    from p2p.contracts import IntentCriterion

logger = logging.getLogger(__name__)


def elaborate_intent(
    prompt: str,
    env_id: str,
    *,
    client: anthropic.Anthropic | None,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> list[IntentCriterion]:
    """Call LLM to decompose intent into 5-10 behavioral criteria."""
    from p2p.training.env_spec import (
        extract_body_geometry,
        get_env_spec,
    )
    from p2p.training.simulator import get_simulator

    env = get_env_spec(env_id)
    backend = get_simulator(env.engine)

    # Build env context
    env_context = (
        f"Environment: {env.env_id}\n"
        f"Name: {env.name}\n"
        f"Description: {env.description}\n"
        f"Observation dim: {env.obs_dim}, Action dim: {env.action_dim}\n"
    )

    if env.state_ref or env.engine == "isaaclab":
        body_info = backend.extract_body_info(env.env_id)
        env_context += f"\nBody layout:\n{body_info}\n"
        joint_sem = backend.extract_joint_semantics(env.env_id)
        if joint_sem:
            env_context += f"\nJoint semantics:\n{joint_sem}\n"
        if env.engine == "mujoco":
            body_geo = extract_body_geometry(env.env_id)
            env_context += f"\nBody geometry:\n{body_geo}\n"

    system_prompt = f"""\
You are an expert RL behavior specification assistant. Given a user's behavioral
intent for a simulated robot, decompose it into 5-10 specific, measurable
behavioral criteria that a judge should check.

{env_context}

Each criterion should be:
- Specific and physically measurable (not vague)
- Categorized as one of: gait, posture, dynamics, stability, efficiency
- Marked with default_on=true if it is essential to the intent, false if optional

Return a JSON array of objects with keys: title, description, category, default_on.
- title: a SHORT plain-English label shown in a checkbox UI. MUST be under 8 words.
  NEVER include variable names, array indices, numbers, units, or parentheticals.
  Good examples: "Alternate legs symmetrically", "Torso stays upright", "Minimal lateral drift"
  Bad examples: "mj_data.qvel[0] > 0.4 m/s", "Torso height (xpos[1][2]) above 1.0m",
    "Forward velocity of at least 0.3 m/s", "body_pos_w[:, 2] > 0.5",
    "joint_pos[3] exceeds limit"
- description: full technical detail with measurable thresholds. Reference joints and bodies
  by NAME (e.g. "right hip pitch", "torso height") rather than array indices or code variables.
  Include specific numeric thresholds where possible (e.g. "forward velocity > 0.3 m/s",
  "torso pitch angle within +/-15 degrees"). Do NOT use mj_data, qpos, qvel, joint_pos,
  body_pos_w, or any code-level references. This is shown on demand, not by default.
Return ONLY the JSON array, no other text."""

    user_msg = f'Intent: "{prompt}"\n\nDecompose this into behavioral criteria.'

    response = create_message(
        client,
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        thinking_effort=thinking_effort,
    )

    text = extract_response_text(response).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else len(text)
        text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")].rstrip()

    try:
        criteria: list[dict] = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON for intent criteria: %s", exc)
        return []

    # Validate structure
    validated: list[IntentCriterion] = []
    for c in criteria:
        if not isinstance(c, dict):
            continue
        if "description" not in c or "category" not in c:
            continue
        validated.append(
            {
                "title": str(c.get("title", c["description"].split(":")[0])),
                "description": str(c["description"]),
                "category": str(c.get("category", "dynamics")),
                "default_on": bool(c.get("default_on", True)),
            }
        )

    return validated


def build_elaborated_text(
    original_prompt: str,
    selected_descriptions: list[str],
    custom_criteria: list[str] | None = None,
) -> str:
    """Reconstruct intent string from original prompt + selected criteria.

    Returns '{prompt}. Behavioral criteria:\\n- ...' or just the original
    prompt if no criteria are selected.
    """
    all_criteria = list(selected_descriptions)
    if custom_criteria:
        all_criteria.extend(custom_criteria)

    if not all_criteria:
        return original_prompt

    criteria_text = "\n".join(f"- {c}" for c in all_criteria)
    return f"{original_prompt}\n\nBehavioral criteria:\n{criteria_text}"
