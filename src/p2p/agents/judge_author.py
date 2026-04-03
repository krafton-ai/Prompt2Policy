"""Code-based judge generation and execution.

When ``vlm_model="code-judge"`` is selected, this module generates an
environment-specific Python ``judge_fn`` via the Claude API, then executes
it in a sandboxed namespace against saved trajectory data.  This replaces
the VLM stage (Stage 2) in the judging pipeline.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from p2p.inference.llm_client import create_message, extract_response_text
from p2p.prompts.judge_author import (
    DECOMPOSE_MSG,
    DECOMPOSE_REVIEW_MSG,
    FIX_JUDGE_CODE_TEMPLATE,
    IMPLEMENT_MSG,
    build_judge_system_prompt,
    build_review_judge_code_prompt,
)
from p2p.settings import LLM_MODEL
from p2p.utils.utils import extract_code_block

if TYPE_CHECKING:
    import anthropic

    from p2p.training.env_spec import EnvSpec

logger = logging.getLogger(__name__)

MAX_JUDGE_RETRIES = 5
MAX_JUDGE_REVIEWS = 5
MAX_DECOMPOSE_REVIEWS = 5

# ---------------------------------------------------------------------------
# Code generation via Claude API
# ---------------------------------------------------------------------------


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (fenced or raw)."""
    return extract_code_block(text, "def judge_fn")


def _collect_sample_trajectory(env_id: str, n_steps: int = 5) -> list[dict[str, Any]] | None:
    """Collect a short trajectory with random actions for validation.

    Uses :func:`p2p.training.sb3_trainer.build_trajectory_entry` so the field set
    is guaranteed identical to the real evaluation trajectory.

    Returns ``None`` if the environment cannot be created (e.g. MuJoCo not
    installed), so callers can fall back to empty-input validation.
    """
    try:
        import gymnasium as gym

        from p2p.training.sb3_trainer import build_trajectory_entry
    except Exception:
        return None

    # Look up engine from ENV_REGISTRY
    from p2p.training.env_spec import ENV_REGISTRY

    spec = ENV_REGISTRY.get(env_id)
    engine = spec.engine if spec else "mujoco"

    try:
        env = gym.make(env_id)
    except Exception:
        logger.debug("Could not create %s for sample trajectory", env_id)
        return None

    try:
        obs, _info = env.reset()
        trajectory: list[dict[str, Any]] = []
        dt = getattr(env.unwrapped, "dt", 0.02)

        for step in range(n_steps):
            action = env.action_space.sample()
            action_np = np.asarray(action).flatten()
            next_obs, reward_val, terminated, truncated, info = env.step(action_np)

            entry = build_trajectory_entry(
                step=step,
                obs=obs,
                action=action_np,
                next_obs=next_obs,
                reward=reward_val,
                info=info,
                terminated=terminated,
                truncated=truncated,
                dt=dt,
                env_id=env_id,
                env_unwrapped=env.unwrapped,
                engine=engine,
            )
            trajectory.append(entry)
            obs = next_obs
            if terminated or truncated:
                break

        return trajectory
    except Exception:
        logger.debug("Failed to collect sample trajectory for %s", env_id, exc_info=True)
        return None
    finally:
        env.close()


def validate_judge_code(code: str, env_id: str | None = None) -> None:
    """Validate that judge code compiles, defines ``judge_fn``, and runs correctly.

    Performs three levels of validation:
    1. Compile + exec + check ``judge_fn`` exists and is callable.
    2. Dry-run with empty inputs ``([], {})``.
    3. (If *env_id* provided) Run against a real sample trajectory collected
       from the environment to catch field-access bugs (KeyError, IndexError).

    Raises on syntax errors, missing/non-callable ``judge_fn``, or if any
    run returns a non-dict.
    """
    compiled = compile(code, "<judge_fn>", "exec")
    ns: dict[str, Any] = {"np": np, "numpy": np, "math": math}
    exec(compiled, ns)  # noqa: S102
    fn = ns.get("judge_fn")
    if fn is None or not callable(fn):
        msg = "Code does not define a callable judge_fn"
        raise ValueError(msg)
    result = fn([], {})
    if not isinstance(result, dict):
        msg = f"judge_fn returned {type(result).__name__}, expected dict"
        raise ValueError(msg)

    # Real-data validation: catch field-access bugs before training starts.
    # Wrap in try/except to convert any exception (KeyError, IndexError,
    # TypeError, etc.) from LLM-generated code into RuntimeError so the
    # retry/review loop can catch it and feed the error back to the LLM.
    if env_id is not None:
        sample = _collect_sample_trajectory(env_id)
        if sample:
            try:
                result = fn(sample, {})
            except Exception as exc:
                msg = f"judge_fn runtime error on real trajectory: {type(exc).__name__}: {exc}"
                raise RuntimeError(msg) from exc
            if not isinstance(result, dict):
                rtype = type(result).__name__
                msg = f"judge_fn returned {rtype} on real trajectory, expected dict"
                raise ValueError(msg)


def _strip_thinking(messages: list[dict]) -> list[dict]:
    """Remove thinking blocks from assistant messages.

    Returns a new list with the same messages but assistant content
    stripped of any ``thinking`` blocks.  This prevents the reviewer
    from being anchored by the author's reasoning (confirmation bias).
    """
    stripped = []
    for msg in messages:
        if msg.get("role") != "assistant":
            stripped.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(msg)
            continue
        filtered = [b for b in content if not _is_thinking_block(b)]
        # If all blocks were thinking, keep original to avoid empty content
        # (this is rare — models typically include at least one text block)
        stripped.append({**msg, "content": filtered or content})
    return stripped


def _is_thinking_block(block: object) -> bool:
    """Check if a content block is a thinking block."""
    if isinstance(block, dict):
        return block.get("type") == "thinking"
    return getattr(block, "type", None) == "thinking"


def _extract_decomposition(text: str) -> str:
    """Extract decomposition block (Structure + Legend + Scoring Criteria + Definitions)."""
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.startswith(("Structure:", "structure:")):
                return block
            lines = block.split("\n", 1)
            if len(lines) > 1 and lines[1].strip().startswith(("Structure:", "structure:")):
                return lines[1].strip()
    if "Structure:" in text:
        idx = text.index("Structure:")
        return text[idx:].strip()
    msg = "No decomposition block found in response"
    raise ValueError(msg)


def _decompose_intent(
    intent: str,
    *,
    client: anthropic.Anthropic,
    system_prompt: str,
    model: str,
    thinking_effort: str,
    out_messages: list[dict] | None = None,
) -> str:
    """Decompose intent into sub-tasks and self-review.

    If *out_messages* is provided, appends to it so the caller can
    continue the conversation in the implementation phase.

    Returns the final approved decomposition text.
    """
    messages = out_messages if out_messages is not None else []

    # ── Decompose ──
    messages.append({"role": "user", "content": DECOMPOSE_MSG.format(intent=intent)})

    response = create_message(
        client,
        model=model,
        system=system_prompt,
        messages=messages,
        thinking_effort=thinking_effort,
    )
    text = extract_response_text(response)
    messages.append({"role": "assistant", "content": response.content})
    decomposition = _extract_decomposition(text)

    # ── Self-review ──
    # Strip thinking to prevent confirmation bias — the reviewer should
    # evaluate the decomposition fresh, not be anchored by its own reasoning.
    messages[:] = _strip_thinking(messages)

    for review_round in range(MAX_DECOMPOSE_REVIEWS):
        remaining = MAX_DECOMPOSE_REVIEWS - review_round
        messages.append(
            {
                "role": "user",
                "content": DECOMPOSE_REVIEW_MSG.format(intent=intent, remaining=remaining),
            }
        )

        response = create_message(
            client,
            model=model,
            system=system_prompt,
            messages=messages,
            thinking_effort=thinking_effort,
        )
        text = extract_response_text(response)
        messages.append({"role": "assistant", "content": response.content})

        if "LGTM" in text:
            logger.info(
                "Judge: decomposition approved (round %d/%d)",
                review_round + 1,
                MAX_DECOMPOSE_REVIEWS,
            )
            break

        try:
            decomposition = _extract_decomposition(text)
        except ValueError:
            logger.warning(
                "Judge: decomposition review returned no block and no LGTM, treating as pass"
            )
            break

        logger.info(
            "Judge: decomposition revised (round %d/%d)",
            review_round + 1,
            MAX_DECOMPOSE_REVIEWS,
        )
    else:
        logger.warning(
            "Judge: decomposition review exhausted %d rounds without LGTM",
            MAX_DECOMPOSE_REVIEWS,
        )

    return decomposition


def _implement_judge(
    intent: str,
    messages: list[dict],
    *,
    client: anthropic.Anthropic,
    env: EnvSpec,
    system_prompt: str,
    model: str,
    thinking_effort: str,
) -> str:
    """Implement judge_fn and review for loopholes.

    Continues the conversation from the decomposition phase so the LLM
    retains reasoning context from Phase 1.

    Returns validated source code defining ``judge_fn``.
    """
    # ── Implement ──
    messages.append({"role": "user", "content": IMPLEMENT_MSG})

    code = ""
    for attempt in range(MAX_JUDGE_RETRIES):
        response = create_message(
            client,
            model=model,
            system=system_prompt,
            messages=messages,
            thinking_effort=thinking_effort,
        )
        text = extract_response_text(response)
        messages.append({"role": "assistant", "content": response.content})

        try:
            code = _extract_code(text)
            validate_judge_code(code, env_id=env.env_id)
            break
        except (SyntaxError, SyntaxWarning, ValueError, RuntimeError) as exc:
            if attempt == MAX_JUDGE_RETRIES - 1:
                raise
            logger.warning(
                "Judge: code validation failed (attempt %d/%d): %s",
                attempt + 1,
                MAX_JUDGE_RETRIES,
                exc,
            )
            messages.append(
                {
                    "role": "user",
                    "content": FIX_JUDGE_CODE_TEMPLATE.format(code=code, error=exc),
                }
            )

    # ── Loophole review ──
    # Strip thinking blocks to prevent confirmation bias — the reviewer
    # should evaluate the code on its own merits, not be anchored by the
    # author's reasoning from decomposition and implementation phases.
    messages[:] = _strip_thinking(messages)

    for review_round in range(MAX_JUDGE_REVIEWS):
        review_prompt = build_review_judge_code_prompt(
            intent,
            code,
            engine=env.engine,
            include_code=(review_round == 0),
        )
        messages.append({"role": "user", "content": review_prompt})

        response = create_message(
            client,
            model=model,
            system=system_prompt,
            messages=messages,
            thinking_effort=thinking_effort,
        )
        text = extract_response_text(response)
        messages.append({"role": "assistant", "content": response.content})

        try:
            new_code = _extract_code(text)
        except ValueError:
            if "LGTM" in text:
                logger.info(
                    "Judge: loophole review passed (round %d/%d)",
                    review_round + 1,
                    MAX_JUDGE_REVIEWS,
                )
                break
            logger.warning("Judge: loophole review returned no code and no LGTM, treating as pass")
            break

        try:
            validate_judge_code(new_code, env_id=env.env_id)
        except (SyntaxError, SyntaxWarning, ValueError, RuntimeError) as exc:
            logger.warning(
                "Judge: reviewed code failed validation (round %d/%d): %s",
                review_round + 1,
                MAX_JUDGE_REVIEWS,
                exc,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your corrected code failed validation:\n{exc}\n\n"
                        "Please fix the error and return the corrected code "
                        "in a ```python block."
                    ),
                }
            )
            continue

        logger.info(
            "Judge: loophole review found issues and fixed (round %d/%d)",
            review_round + 1,
            MAX_JUDGE_REVIEWS,
        )
        code = new_code
    else:
        logger.warning(
            "Judge: loophole review exhausted %d rounds without LGTM",
            MAX_JUDGE_REVIEWS,
        )

    # ── Final validation ──
    validate_judge_code(code, env_id=env.env_id)
    return code


def generate_judge_code(
    intent: str,
    *,
    client: anthropic.Anthropic,
    env: EnvSpec | None = None,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
    max_episode_steps: int = 1000,
) -> str:
    """Generate a judge function via a two-phase process in one session.

    Phase 1 (decomposition): Decompose the intent into sub-tasks,
    then self-review for completeness and correctness (up to
    ``MAX_DECOMPOSE_REVIEWS`` rounds).

    Phase 2 (implementation): Implement ``judge_fn`` from the approved
    decomposition, then review for loopholes (up to
    ``MAX_JUDGE_REVIEWS`` rounds).

    All phases share one conversation so the LLM retains context
    from decomposition thinking during implementation.

    Returns the source code string defining ``judge_fn``.
    """
    if env is None:
        from p2p.training.env_spec import get_env_spec

        env = get_env_spec("HalfCheetah-v5")

    system_prompt = build_judge_system_prompt(intent, env, max_episode_steps=max_episode_steps)

    # Phase 1: decompose + review
    logger.info("Judge: Phase 1 — decomposition")
    messages: list[dict] = []
    decomposition = _decompose_intent(
        intent,
        client=client,
        system_prompt=system_prompt,
        model=model,
        thinking_effort=thinking_effort,
        out_messages=messages,
    )
    logger.info("Judge: decomposition finalized:\n%s", decomposition)

    # Phase 2: implement + loophole review (same session)
    logger.info("Judge: Phase 2 — implementation")
    return _implement_judge(
        intent,
        messages,
        client=client,
        env=env,
        system_prompt=system_prompt,
        model=model,
        thinking_effort=thinking_effort,
    )


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

_FALLBACK_RESULT: dict[str, Any] = {
    "intent_score": 0.0,
    "diagnosis": "Code judge execution error",
    "failure_tags": ["code_judge_error"],
}


_JUDGE_TIMEOUT_SEC = 30


def run_sandboxed_fn(
    code: str,
    fn_name: str,
    args: tuple,
    *,
    timeout_sec: int = _JUDGE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Execute *code* in a sandboxed namespace and call *fn_name* with *args*.

    Security measures:
    - ``__builtins__`` restricted to safe subset (no ``open``, ``exec``,
      ``eval``, ``__import__``, ``compile``, ``globals``).
    - Only ``numpy`` and ``math`` imports are allowed.
    - Execution subject to *timeout_sec* wall-clock timeout via threading.

    Returns the dict returned by *fn_name*.  On any error (syntax, missing
    function, timeout, runtime exception, non-dict return) returns
    ``{"error": "<description>"}``.
    """
    import threading

    _allowed_imports = {"numpy", "math"}

    def _safe_import(name: str, *a: Any, **kw: Any) -> Any:
        top_level = name.split(".")[0]
        if top_level not in _allowed_imports:
            msg = f"import of {name!r} is not allowed in sandboxed code"
            raise ImportError(msg)
        return __builtins__["__import__"](name, *a, **kw)  # type: ignore[index]

    _safe_builtins = {
        k: v
        for k, v in __builtins__.items()  # type: ignore[union-attr]
        if k
        not in {
            "open",
            "exec",
            "eval",
            "compile",
            "__import__",
            "globals",
            "locals",
            "breakpoint",
            "exit",
            "quit",
            "input",
            "help",
            "memoryview",
        }
    }
    _safe_builtins["__import__"] = _safe_import

    namespace: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "math": math,
        "__builtins__": _safe_builtins,
    }

    try:
        compiled = compile(code, f"<{fn_name}>", "exec")
        exec(compiled, namespace)  # noqa: S102
    except SyntaxError:
        logger.exception("%s code has syntax errors", fn_name)
        return {"error": f"{fn_name} syntax error"}
    except Exception:
        logger.exception("Failed to compile/exec %s code", fn_name)
        return {"error": f"{fn_name} compilation error"}

    fn = namespace.get(fn_name)
    if fn is None or not callable(fn):
        logger.error("Code does not define a callable %s", fn_name)
        return {"error": f"missing {fn_name} function"}

    result_holder: list[Any] = []
    error_holder: list[tuple] = []  # sys.exc_info() triples to preserve thread traceback

    def _run() -> None:
        import sys

        try:
            result_holder.append(fn(*args))
        except Exception:
            error_holder.append(sys.exc_info())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        logger.error("%s timed out after %ds", fn_name, timeout_sec)
        return {"error": f"{fn_name} timed out ({timeout_sec}s)"}
    if error_holder:
        exc_type, exc_val, _tb = error_holder[0]
        logger.error("%s raised an exception", fn_name, exc_info=error_holder[0])
        exc_name = exc_type.__name__ if exc_type else "Exception"
        exc_msg = str(exc_val) if exc_val else ""
        detail = f"{exc_name}: {exc_msg}" if exc_msg else exc_name
        return {"error": f"{fn_name} runtime error: {detail}"}
    if not result_holder:
        return {"error": f"{fn_name} returned no result"}

    result = result_holder[0]
    if not isinstance(result, dict):
        logger.error("%s returned %s, expected dict", fn_name, type(result).__name__)
        return {"error": f"{fn_name} did not return a dict"}
    return result


def execute_judge_code(
    judge_code: str,
    trajectory: list[dict],
    summary: dict,
    *,
    timeout_sec: int = _JUDGE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Execute generated judge code in a sandboxed namespace.

    Calls ``run_sandboxed_fn`` to execute ``judge_fn(trajectory, summary)``
    in a restricted environment, then validates and clamps the result.

    Returns a dict with keys: intent_score, diagnosis, failure_tags.
    On any error, returns a fallback dict with score 0.0.
    """
    result = run_sandboxed_fn(
        judge_code, "judge_fn", (trajectory, summary), timeout_sec=timeout_sec
    )
    if "error" in result:
        return {**_FALLBACK_RESULT, "diagnosis": result["error"]}

    # Validate and clamp intent_score
    raw_score = result.get("intent_score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    if math.isnan(score) or math.isinf(score):
        logger.warning("judge_fn returned %s intent_score, defaulting to 0.0", score)
        score = 0.0
    if score < 0.0 or score > 1.0:
        logger.warning(
            "judge_fn returned out-of-range intent_score %.4f, clamping to [0, 1]", score
        )
    result["intent_score"] = max(0.0, min(1.0, score))

    # Ensure required keys exist
    result.setdefault("diagnosis", "")
    result.setdefault("failure_tags", [])

    return result
