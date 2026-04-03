"""Shared utility functions."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_log_tail(path: Path, n: int = 20) -> str:
    """Return the last *n* lines of a log file, or ``""`` on failure.

    Handles ``OSError`` (missing/unreadable files) gracefully so callers
    don't need their own try/except blocks.
    """
    try:
        lines = path.read_text().splitlines()
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return ""
    except OSError:
        logger.warning("Failed to read log file: %s", path)
        return ""


def extract_code_block(text: str, fn_names: str | Sequence[str]) -> str:
    """Extract Python code from an LLM response (fenced or raw).

    Looks for a fenced code block (````` ```python ... ``` `````) first, then
    falls back to raw text if any of *fn_names* appear in it.

    Parameters
    ----------
    text:
        Raw LLM response text.
    fn_names:
        One or more function/identifier names whose presence signals valid code.

    Returns
    -------
    str
        Extracted code, stripped of leading/trailing whitespace.

    Raises
    ------
    ValueError
        If no valid code block is found.
    """
    if isinstance(fn_names, str):
        fn_names = [fn_names]

    # Fenced code blocks (```python ... ```) — pick the one containing fn_names
    for match in re.finditer(r"```(?:python)?\s*\n(.+?)```", text, re.DOTALL):
        code = match.group(1).strip()
        if any(name in code for name in fn_names):
            return code

    # Raw fallback: text contains a known function definition
    if any(name in text for name in fn_names):
        return text.strip()

    names = ", ".join(fn_names)
    msg = f"No valid code found in LLM response (expected: {names})"
    raise ValueError(msg)


def run_code_review_loop(
    *,
    code: str,
    system: str,
    first_msg: str,
    extract_fn: Callable[[str], str],
    validate_fn: Callable[[str], None],
    client: Any,
    model: str,
    max_rounds: int = 3,
    label: str = "Code",
    thinking_effort: str = "",
) -> str:
    """Generic multi-turn LLM code review loop.

    Used by :func:`p2p.agents.reward_author.review_reward_code`.
    Each round asks the LLM to review the current code, extracts any
    corrections, validates them, and feeds validation errors back for
    another attempt.

    Parameters
    ----------
    code:
        Initial code to review.
    system:
        System prompt for the LLM.
    first_msg:
        First user message (review request with code).
    extract_fn:
        Callable to extract code from LLM response text.  Should raise
        ``ValueError`` when no code block is found.
    validate_fn:
        Callable to validate extracted code.  Should raise on failure.
    client:
        Anthropic client instance.
    model:
        LLM model identifier.
    max_rounds:
        Maximum number of review rounds.
    label:
        Human-readable label for log messages (e.g. "Code", "Reward code").
    thinking_effort:
        Thinking effort level for the LLM.

    Returns
    -------
    str
        The (possibly improved) code string.
    """
    from p2p.inference.llm_client import create_message, extract_response_text

    messages: list[dict[str, str]] = [{"role": "user", "content": first_msg}]

    for attempt in range(max_rounds):
        response = create_message(
            client,
            model=model,
            system=system,
            messages=messages,
            thinking_effort=thinking_effort,
        )
        text = extract_response_text(response)

        # Append assistant response to conversation history
        messages.append({"role": "assistant", "content": text})

        # Try to extract corrected code from the response.
        # Do this BEFORE checking LGTM because the reviewer may quote
        # code snippets in its reasoning (```), which would make a naive
        # "LGTM and no backticks" check fail.
        try:
            new_code = extract_fn(text)
        except ValueError:
            # No corrected function found — treat as LGTM if present
            if "LGTM" in text:
                logger.info(
                    "%s review passed (round %d/%d)",
                    label,
                    attempt + 1,
                    max_rounds,
                )
                return code
            logger.warning(
                "%s review returned no code and no LGTM, treating as pass. Response: %s",
                label,
                text[:500],
            )
            return code

        # Validate the corrected code
        try:
            validate_fn(new_code)
        except (SyntaxError, SyntaxWarning, ValueError, RuntimeError) as exc:
            logger.warning(
                "%s review: corrected code failed validation (round %d/%d): %s, will retry",
                label,
                attempt + 1,
                max_rounds,
                exc,
            )
            # Feed validation error back in the same conversation
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your corrected code failed validation:\n{exc}\n\n"
                        f"Please fix the error and return the corrected code "
                        f"in a ```python block."
                    ),
                }
            )
            continue

        logger.info(
            "%s review found issues and fixed (round %d/%d)",
            label,
            attempt + 1,
            max_rounds,
        )
        code = new_code

        # Ask for followup review of the corrected code
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your corrected code passed validation. Now review it again: "
                    "did your fixes introduce any NEW bugs? Check ONLY for new "
                    "issues — do not re-raise already-fixed problems.\n\n"
                    "If no new bugs, respond with exactly: LGTM\n"
                    "Otherwise, output corrected code in a ```python block."
                ),
            }
        )

    logger.warning(
        "%s review exhausted %d rounds without LGTM, returning last version",
        label,
        max_rounds,
    )
    return code
