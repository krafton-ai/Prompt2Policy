"""Reward code utilities: extraction, validation, fixing, and review."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from p2p.config import TrainConfig
from p2p.inference.llm_client import create_message, extract_response_text
from p2p.prompts.reward_author import (
    FIX_CODE_TEMPLATE,
    GENERATE_TEMPLATE,
    REVISE_TEMPLATE,
    build_review_code_prompt,
    build_system_prompt,
)
from p2p.settings import LLM_MODEL
from p2p.training.reward_loader import load_from_code
from p2p.utils.utils import extract_code_block, run_code_review_loop

logger = logging.getLogger(__name__)

MAX_CODE_RETRIES = 3
MAX_CODE_REVIEWS = 3

if TYPE_CHECKING:
    import anthropic

    from p2p.training.env_spec import EnvSpec


_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2192": "->",  # →
    "\u2190": "<-",  # ←
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
    "\u00d7": "*",  # ×
    "\u2014": "--",  # —
    "\u2013": "-",  # –
    "\u201c": '"',  # \u201c
    "\u201d": '"',  # \u201d
    "\u2018": "'",  # \u2018
    "\u2019": "'",  # \u2019
}


def _sanitize_unicode(code: str) -> str:
    """Replace common Unicode characters that cause SyntaxError with ASCII."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        code = code.replace(char, replacement)
    return code


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (fenced or raw).

    Handles both stateless (single ``reward_fn``) and stateful
    (``_make_reward`` closure with ``reward_fn = _make_reward()``) patterns.

    Delegates core extraction to :func:`p2p.utils.utils.extract_code_block`,
    then applies reward-specific post-processing (unicode sanitization
    and ``_make_reward`` closure wiring).
    """
    code = extract_code_block(text, ["def reward_fn", "_make_reward"])
    code = _sanitize_unicode(code)

    # Stateful pattern: if _make_reward is defined but the module-level
    # assignment is missing, append it so exec() produces `reward_fn`.
    if "_make_reward" in code and "reward_fn = _make_reward()" not in code:
        code += "\n\nreward_fn = _make_reward()\n"

    return code


def _validate_with_env(
    reward_fn: Callable,
    env_id: str,
    *,
    side_info: bool = False,
) -> None:
    """Validate reward function against real environment transitions.

    Creates the environment with the same wrapper stack used in training,
    runs a reset + 2 steps, and calls the reward function on each transition
    immediately (before the next step mutates ``mj_data``).

    Raises ``RuntimeError`` if the reward function crashes or returns
    an invalid type.
    """
    try:
        import gymnasium as gym
        import numpy as np

        from p2p.training.env import CustomRewardWrapper
    except Exception:
        logger.debug("Could not import gymnasium/env for runtime validation")
        return

    # Dummy reward fn that just returns 0 — we only need the info dict
    def _dummy_reward(obs: object, action: object, next_obs: object, info: object) -> tuple:
        return 0.0, {}

    from p2p.training.env_spec import ENV_REGISTRY
    from p2p.training.simulator import get_simulator

    spec = ENV_REGISTRY.get(env_id)
    _engine = spec.engine if spec else "mujoco"
    _backend = get_simulator(_engine) if side_info else None

    try:
        env = gym.make(env_id)
        env = CustomRewardWrapper(env, _dummy_reward, side_info=side_info, engine=_engine)
    except Exception:
        logger.debug("Could not create %s for runtime validation", env_id)
        return

    def _check_result(result: object, label: str) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            msg = (
                f"Reward function must return (float, dict), "
                f"got {type(result).__name__} on {label} transition"
            )
            raise RuntimeError(msg)
        _total, terms = result
        if not isinstance(terms, dict):
            msg = (
                f"Reward function terms must be dict, "
                f"got {type(terms).__name__} on {label} transition"
            )
            raise RuntimeError(msg)

    try:
        obs, _info = env.reset()
        action = env.action_space.sample()
        action_np = np.asarray(action).flatten()

        # Step 1: episode start (info["_episode_start"] = True)
        next_obs, _r, _term, _trunc, info_start = env.step(action_np)
        if _backend and _backend.has_physics_state(env.unwrapped):
            info_start.update(_backend.extract_side_info(env.unwrapped))
        # Validate immediately — side info is current for this step
        try:
            result = reward_fn(obs.copy(), action_np.copy(), next_obs.copy(), info_start)
        except Exception as exc:
            msg = f"Reward function runtime error on start transition: {type(exc).__name__}: {exc}"
            raise RuntimeError(msg) from exc
        _check_result(result, "start")

        # Step 2: mid-episode (info["_episode_start"] = False)
        obs2 = next_obs.copy()
        action2 = env.action_space.sample()
        action2_np = np.asarray(action2).flatten()
        next_obs2, _r2, _term2, _trunc2, info_mid = env.step(action2_np)
        if _backend and _backend.has_physics_state(env.unwrapped):
            info_mid.update(_backend.extract_side_info(env.unwrapped))
        # Validate immediately — side info is current for this step
        try:
            result = reward_fn(obs2, action2_np, next_obs2.copy(), info_mid)
        except Exception as exc:
            msg = f"Reward function runtime error on mid transition: {type(exc).__name__}: {exc}"
            raise RuntimeError(msg) from exc
        _check_result(result, "mid")

    except RuntimeError:
        raise
    except Exception:
        logger.debug("Failed runtime validation for %s", env_id, exc_info=True)
    finally:
        env.close()


def validate_reward_code(
    code: str,
    env_id: str | None = None,
    *,
    side_info: bool = False,
    engine: str = "mujoco",
) -> None:
    """Validate that reward code compiles, defines a reward function, and runs.

    Performs up to three levels of validation:
    1. Compile + exec + check ``reward_fn`` exists and is callable.
    2. (If *env_id* provided) Run the reward function against real
       environment transitions to catch runtime errors (KeyError,
       IndexError, TypeError, etc.) and check return type.

    Raises SyntaxError, ValueError, or RuntimeError if any check fails.
    """
    reward_fn = load_from_code(code, engine=engine)

    if env_id is None:
        return

    _validate_with_env(reward_fn, env_id, side_info=side_info)


def load_reward_fn(code: str, engine: str = "mujoco") -> Callable:
    """Dynamically load a reward function from a code string.

    Delegates to ``reward_loader.load_from_code`` which supports both
    ``RewardFunction`` subclasses and legacy ``reward_fn`` functions.
    The returned object is always callable via ``__call__``.
    """
    return load_from_code(code, engine=engine)


def fix_code(
    code: str,
    error: str | Exception,
    *,
    model: str = LLM_MODEL,
    client: anthropic.Anthropic,
    env: EnvSpec,
    thinking_effort: str = "",
) -> str:
    """Ask LLM to fix a syntax error in reward code.

    Used when revised code fails validation — more targeted than re-running
    the full revise agent.
    """
    system = build_system_prompt(env)
    messages = [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": FIX_CODE_TEMPLATE.format(code=code, error=error)},
    ]
    response = create_message(
        client,
        model=model,
        thinking_effort=thinking_effort,
        system=system,
        messages=messages,
    )
    return _extract_code(extract_response_text(response))


def review_reward_code(
    code: str,
    prompt: str,
    *,
    model: str = LLM_MODEL,
    client: anthropic.Anthropic,
    env: EnvSpec,
    side_info: bool = False,
    env_id: str | None = None,
    thinking_effort: str = "",
) -> str:
    """Review reward code for logical bugs and return corrected code.

    Performs up to ``MAX_CODE_REVIEWS`` rounds of LLM review. Each round:
    1. Sends code to LLM reviewer for logical correctness check.
    2. If reviewer responds ``LGTM``, returns code as-is.
    3. If reviewer provides corrected code, validates it (syntax + runtime)
       and uses the corrected version for the next round.

    Returns the (possibly improved) code string.
    """
    return run_code_review_loop(
        code=code,
        system=build_system_prompt(env, side_info=side_info),
        first_msg=build_review_code_prompt(prompt, code, engine=env.engine),
        extract_fn=_extract_code,
        validate_fn=lambda c: validate_reward_code(
            c, env_id, side_info=side_info, engine=env.engine
        ),
        client=client,
        model=model,
        max_rounds=MAX_CODE_REVIEWS,
        label="Reward code",
        thinking_effort=thinking_effort,
    )


def generate(
    prompt: str,
    *,
    model: str = LLM_MODEL,
    client: anthropic.Anthropic,
    env: EnvSpec,
    side_info: bool = False,
    thinking_effort: str = "",
) -> str:
    """Generate a reward function from a natural language prompt.

    Retries up to MAX_CODE_RETRIES times if the generated code has syntax errors.
    """
    system = build_system_prompt(env, side_info=side_info)
    messages = [{"role": "user", "content": GENERATE_TEMPLATE.format(prompt=prompt)}]

    for attempt in range(MAX_CODE_RETRIES):
        response = create_message(
            client,
            model=model,
            thinking_effort=thinking_effort,
            system=system,
            messages=messages,
        )
        text = extract_response_text(response)
        code = _extract_code(text)

        try:
            validate_reward_code(code, engine=env.engine)
            return code
        except (SyntaxError, SyntaxWarning, ValueError) as exc:
            if attempt == MAX_CODE_RETRIES - 1:
                raise
            logger.warning(
                "Generated code failed validation (attempt %d/%d): %s",
                attempt + 1,
                MAX_CODE_RETRIES,
                exc,
            )
            # Ask LLM to fix the error, keeping system prompt context
            messages = [
                {"role": "user", "content": GENERATE_TEMPLATE.format(prompt=prompt)},
                {"role": "assistant", "content": text},
                {"role": "user", "content": FIX_CODE_TEMPLATE.format(code=code, error=exc)},
            ]

    # Unreachable, but satisfies type checker
    msg = "Code generation failed after retries"
    raise RuntimeError(msg)


def _extract_config_overrides(text: str) -> dict:
    """Extract JSON config overrides from LLM response, if present."""
    match = re.search(r"```json\s*\n(.+?)```", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return {}


def revise(
    prompt: str,
    previous_code: str,
    judgment: dict,
    summary: dict,
    *,
    model: str = LLM_MODEL,
    client: anthropic.Anthropic,
    env: EnvSpec,
    current_config: dict | None = None,
    side_info: bool = False,
    thinking_effort: str = "",
) -> tuple[str, dict]:
    """Revise a reward function based on VLM judgment and training metrics.

    Returns
    -------
    (revised_code, config_overrides)
        ``config_overrides`` is a dict of TrainConfig fields to change for
        the next iteration.  Empty dict if no changes suggested.
    """
    # Format current config for the prompt
    config_str = "N/A"
    if current_config:
        tunable = TrainConfig._TUNABLE_KEYS
        config_lines = [f"- {k}: {v}" for k, v in current_config.items() if k in tunable]
        config_str = "\n".join(config_lines) if config_lines else "N/A"

    user_msg = REVISE_TEMPLATE.format(
        prompt=prompt,
        previous_code=previous_code,
        final_return=summary.get("final_episodic_return", "N/A"),
        total_timesteps=summary.get("total_timesteps", "N/A"),
        training_time=summary.get("training_time_s", 0),
        intent_score=judgment.get("intent_score", "N/A"),
        diagnosis=judgment.get("diagnosis", "N/A"),
        failure_tags=", ".join(judgment.get("failure_tags", [])),
        current_config=config_str,
    )
    response = create_message(
        client,
        model=model,
        thinking_effort=thinking_effort,
        system=build_system_prompt(env, side_info=side_info),
        messages=[{"role": "user", "content": user_msg}],
    )
    text = extract_response_text(response)
    code = _extract_code(text)

    # Extract and filter config overrides (only allow tunable fields)
    raw_overrides = _extract_config_overrides(text)
    overrides = {k: v for k, v in raw_overrides.items() if k in TrainConfig._TUNABLE_KEYS}

    return code, overrides
