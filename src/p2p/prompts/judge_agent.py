"""Prompt templates for behavior judgment (VLM + LLM stages)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from p2p.training.env_spec import engine_display_name, get_spec_by_name
from p2p.training.simulator import get_simulator

if TYPE_CHECKING:
    from p2p.training.env_spec import EnvSpec

logger = logging.getLogger(__name__)


def _resolve_spec(env_name: str, engine: str) -> EnvSpec | None:
    """Find an EnvSpec by display name, preferring the requested engine."""
    from p2p.training.env_spec import ENV_REGISTRY

    for s in ENV_REGISTRY.values():
        if s.name == env_name and s.engine == engine:
            return s
    spec = get_spec_by_name(env_name)
    if spec is not None and spec.engine != engine:
        logger.warning(
            "Requested engine %s for env %s, but resolved spec uses engine %s",
            engine,
            env_name,
            spec.engine,
        )
    return spec


def _rotation_context_for_env(env_name: str, engine: str) -> str | None:
    """Look up rotation convention for VLM prompts.

    VLMs need torso rotation direction to distinguish forward vs backward
    rotation in videos. Returns joint semantics which includes the
    right-hand rule and angular velocity sign conventions.
    """
    spec = _resolve_spec(env_name, engine)
    if spec is not None:
        backend = get_simulator(spec.engine)
        return backend.extract_joint_semantics(spec.env_id)
    return None


def _camera_context_for_env(env_name: str, engine: str) -> str:
    """Look up camera viewpoint description by display name.

    Falls back to a generic side-view description for unknown envs.
    """
    spec = _resolve_spec(env_name, engine)
    if spec is not None:
        backend = get_simulator(spec.engine)
        return backend.get_camera_description(spec.env_id)
    return (
        f"The default {engine_display_name(engine)} camera views the scene from the side. "
        "The agent's forward direction (+x axis) is to the RIGHT of the screen."
    )


# ---------------------------------------------------------------------------
# Two-turn VLM judging (agreement bias mitigation)
# ---------------------------------------------------------------------------


def build_vlm_expectations_prompt(
    intent: str,
    env_name: str,
    engine: str,
    *,
    initial_frame: bool = False,
) -> str:
    """Build Turn 1 prompt: asks VLM to pre-commit visual criteria.

    When initial_frame=True, a first-frame JPEG is appended as a separate Part
    by the caller.  The prompt includes an "Initial Frame" section so the VLM
    can ground its criteria in the actual scene.
    """
    engine_label = engine_display_name(engine)

    rotation_note = ""
    rot_conv = _rotation_context_for_env(env_name, engine=engine)
    if rot_conv:
        rotation_note = (
            f"\n## Rotation Direction (from physics engine)\n"
            f"{rot_conv}\n"
            f"NOTE: VLMs often cannot reliably distinguish forward vs backward "
            f"rotation in {engine_label} videos. The synthesis stage will cross-reference "
            f"your visual assessment with physics data to resolve ambiguity.\n"
        )

    camera_note = _camera_context_for_env(env_name, engine)

    return (
        f"You are evaluating a {engine_label} {env_name} simulation.\n"
        f"\n"
        f"## Task\n"
        f'The robot should: "{intent}"\n'
        f"\n"
        f"## Camera Orientation\n"
        f"{camera_note}\n"
        f"{rotation_note}"
        f"\n"
        f"## Instructions\n"
        f"Before seeing the video, describe what specific visual evidence would "
        f"indicate the task was successfully completed. Consider:\n"
        f"- Body positions, orientations, and postures at key moments\n"
        f"- Expected motion sequences (what happens first, then next)\n"
        f"- Signs of failure or partial completion\n"
        f"\n"
        f"List 3-5 concrete visual criteria, ordered by importance."
        + (
            "\n\n## Initial Frame\n"
            "Below is the first frame of the episode, showing the "
            "agent's starting pose and the environment. Use this "
            "to ground your visual criteria in the actual scene."
            if initial_frame
            else ""
        )
    )


_VLM_CRITERIA_JSON_FORMAT = """\
Reply with ONLY this JSON:
{{"criteria": [{{"criterion": "<criterion text>", \
"assessment": "<what you observed for this criterion>", \
"status": "<met | partially_met | not_met>"}}, ...], \
"intent_score": <float 0.0 to 1.0 in 0.1 steps>, \
"diagnosis": "<2-3 sentences overall assessment>", \
"failure_tags": ["<tag>", ...]}}"""

_TRACKED_CAMERA_NOTE = """\
## Tracked Camera
The camera tracks the agent. When the agent moves forward, \
the background (e.g. checkered floor) moves in the opposite \
direction. Do NOT mistake background motion for agent motion. \
Judge the agent's behavior relative to its own body, not the background."""

_MOTION_TRAIL_DUAL_NOTE = """\
## Motion Trail
You have been given two videos of the same rollout.
- **Video 1** is a standard 10fps recording.
- **Video 2** includes motion trails — translucent ghosts of previous \
frames are blended onto each frame to show movement continuity.

Use BOTH videos to assess: Video 1 shows actual poses clearly; \
Video 2 reveals motion direction and speed via ghosting."""


def build_vlm_scoring_prompt(
    mode: Literal["image", "video"],
    intent: str,
    env_name: str,
    engine: str,
    *,
    criteria_diagnosis: bool = False,
    motion_trail_dual: bool = False,
) -> str:
    """Build Turn 2 prompt: shown with media, scores against pre-committed criteria."""
    engine_label = engine_display_name(engine)
    if mode == "image":
        media_desc = "composite image (START / PEAK ACTION / END snapshots)"
    else:
        media_desc = "video"

    trail_note = (
        f"\n{_MOTION_TRAIL_DUAL_NOTE}\n\n{_TRACKED_CAMERA_NOTE}\n" if motion_trail_dual else ""
    )

    if criteria_diagnosis:
        criteria_instruction = (
            f"For EACH criterion you listed above, describe what you "
            f"actually observed in the {media_desc} — note whether it "
            f"was met, partially met, or not met, and provide specific "
            f"visual evidence. For each criterion, first write the assessment, "
            f"then set the status field to exactly one of: "
            f"met, partially_met, or not_met.\n\n"
            f"After assessing all criteria, assign a SINGLE overall "
            f"intent_score that reflects how well the agent fulfilled "
            f'the stated intent: "{intent}"\n\n'
            f"Do NOT simply average the criteria. Use holistic judgment — "
            f"some criteria matter more than others depending on the "
            f"intent.\n\n"
        )
        json_format = _VLM_CRITERIA_JSON_FORMAT
    else:
        criteria_instruction = (
            "Score the behavior by evaluating each criterion you described above.\n"
            "For each, note whether it was met, partially met, or not met.\n\n"
        )
        json_format = _VLM_JSON_FORMAT

    return (
        f"Now examine this {media_desc} of the actual {engine_label} {env_name} rollout.\n"
        f"{trail_note}\n"
        f"{criteria_instruction}"
        f"{VLM_SCORING_RUBRIC}\n\n"
        f"{json_format}"
    )


VLM_SCORING_RUBRIC = """\
## Scoring (report in 0.1 increments from 0.0 to 1.0)
0.0 = no evidence of the intended behavior
0.2 = minimal or weak signal (barely relevant, very incomplete, no meaningful progress)
0.4 = partial attempt (some relevant elements, but far from the goal)
0.6 = mostly achieved with notable flaws
0.8 = good execution with minor imperfections
1.0 = perfect match to the stated intent
Use intermediate values (0.1, 0.3, 0.5, 0.7, 0.9) when behavior falls between two anchors."""

_VLM_JSON_FORMAT = """\
Reply with ONLY this JSON:
{{"intent_score": <float 0.0 to 1.0 in 0.1 steps>, "diagnosis": "<2-3 sentences>", \
"failure_tags": ["<tag>", ...]}}"""


# ---------------------------------------------------------------------------
# Stage 3: LLM synthesis shared fragments
# ---------------------------------------------------------------------------

_FAILURE_TAG_INSTRUCTIONS = """\
For failure_tags, generate descriptive, intent-specific tags that capture \
the actual failure modes observed. Do NOT use generic tags — describe what \
specifically went wrong relative to the stated intent. Examples:
- "rotating_wrong_direction" (not just "wrong_behavior")
- "no_height_gain_before_flip" (not just "partial_attempt")
- "gait_asymmetric" (not just "unstable_gait")

If a failure from a previous iteration persists, REUSE the exact same tag \
string so it can be tracked across iterations. Only create a new tag when \
a genuinely distinct failure is observed."""

_SYNTHESIS_JSON_FORMAT = """\
Reply with ONLY this JSON:
{{"intent_score": <float 0.0-1.0>, "diagnosis": "<2-3 sentences combining both assessments>", \
"failure_tags": ["<descriptive_tag>", ...]}}"""

# ---------------------------------------------------------------------------
# Stage 3 (Dual): Synthesis with both VLM + code-based judge
# ---------------------------------------------------------------------------

DUAL_JUDGE_SYNTHESIS_PROMPT = f"""\
You are the final judge in an RL training evaluation pipeline. You receive \
two independent judge assessments of an agent's behavior — one from a \
code-based judge (trajectory data analysis) and one from a VLM \
(visual video analysis) — and must synthesize them into a single verdict.

## Intent
"{{intent}}"

## Code-Based Judge (trajectory analysis)
- Score: {{code_score}}
- Diagnosis: {{code_diagnosis}}
- Failure tags: {{code_tags}}

## VLM Judge (visual assessment)
The VLM score was assigned using this rubric:
{VLM_SCORING_RUBRIC}

- Score: {{vlm_score}}
- Diagnosis: {{vlm_diagnosis}}
- Failure tags: {{vlm_tags}}

## Failure Tag History (from previous iterations)
{{tag_history}}

## Instructions
Synthesize both assessments into a final judgment. The code-based judge \
runs custom analysis code against the trajectory data. \
The VLM judge watches the actual video. Each may catch issues the other \
misses — weigh both based on the quality and specificity of their evidence.

When the two judges disagree significantly, explain why in your diagnosis \
and lean toward the judge whose evidence is more directly relevant to the \
intent. For rotation direction (forward flip vs backflip), ALWAYS trust \
the code-based judge — it uses exact physics data, while VLMs frequently \
misidentify rotation direction in simulation videos.

{_FAILURE_TAG_INSTRUCTIONS}

{_SYNTHESIS_JSON_FORMAT}\
"""


# ---------------------------------------------------------------------------
# Stage 3 (Agentic): Synthesis with optional tool use
# ---------------------------------------------------------------------------

AGENTIC_SYNTHESIS_SYSTEM = """\
You are the final arbiter in a {engine_label} behavior evaluation pipeline.  You receive \
two independent judge assessments of a {engine_label} agent's behavior and must \
produce a single final verdict.

## Decision process

1. Compare the code-based judge score and VLM judge score.
2. If the judges AGREE (score difference <= 0.2 AND diagnoses are consistent), \
produce the final JSON verdict immediately.  Do NOT use any tools.
3. If the judges DISAGREE (score difference > 0.2, or contradictory diagnoses, \
or one passes while the other fails), you MAY use the provided tools to \
investigate before ruling:
   - `reask_vlm`: Ask a targeted follow-up question about the video.  You can \
specify `start_time`, `end_time` (seconds), and `fps` to zoom into a specific \
segment at higher temporal resolution.  Use this when the disagreement concerns \
a specific moment (e.g., "did the agent lift off in the first 2 seconds?" → \
`start_time=0, end_time=2, fps=10`).  Times are clamped to the video duration. \
Short segments at higher FPS are cheap and more informative than re-watching \
the full video.
   - `run_trajectory_check`: Run a short Python snippet against the trajectory \
data to verify a specific claim.
4. After investigation (or immediately if judges agree), output ONLY the final \
JSON verdict — nothing else.

## Rules
- You have a budget of {max_tool_calls} tool calls.  After each tool call the \
remaining budget is shown — plan accordingly.
- For rotation direction (forward/backward flip), ALWAYS trust the code-based \
judge over the VLM — it uses exact physics data.
- Each tool call should target a specific question raised by the disagreement.
- When using `reask_vlm`, always narrow `start_time`/`end_time` before \
increasing `fps`.  Do NOT increase FPS on the full video — it will be clamped \
and wastes your tool budget.
- Your final message MUST contain ONLY the JSON verdict and no other text.
- For failure_tags, generate descriptive, intent-specific tags (e.g., \
"rotating_wrong_direction", not "wrong_behavior").  If a failure from a \
previous iteration persists, REUSE the exact same tag string.

## JSON format
{{"intent_score": <float 0.0-1.0>, "diagnosis": "<2-3 sentences>", \
"failure_tags": ["<descriptive_tag>", ...]}}\
"""


AGENTIC_SYNTHESIS_USER = """\
## Intent
"{intent}"

## Code-Based Judge (trajectory analysis)
- Score: {code_score}
- Diagnosis: {code_diagnosis}
- Failure tags: {code_tags}

## VLM Judge (visual assessment)
The VLM score was assigned using this rubric:
{vlm_rubric}

- Score: {vlm_score}
- Diagnosis: {vlm_diagnosis}
- Failure tags: {vlm_tags}
- Evaluated at: {vlm_fps} FPS{video_duration_info}

{env_conventions_section}
## Failure Tag History (from previous iterations)
{tag_history}

Produce the final JSON verdict.  If the two judges agree, respond immediately \
with the JSON.  If they conflict, use the tools to investigate first.\
"""
