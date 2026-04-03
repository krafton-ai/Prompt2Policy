"""Agentic LLM analysis of a training session.

The LLM uses readonly tools bound to a single SessionRecord to
selectively read iteration data and produce a structured analysis.
"""

from __future__ import annotations

import difflib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from p2p.contracts import SessionAnalysis, round_metric
from p2p.inference.llm_client import create_message
from p2p.session.iteration_record import IterationRecord, SessionRecord
from p2p.settings import LLM_MODEL_LIGHT

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_session_overview",
        "description": (
            "Get session metadata and a summary of all iterations "
            "(iteration number, intent_score, status, failure_tags)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_iteration_detail",
        "description": (
            "Get detailed information for a specific iteration: "
            "judgment (score, diagnosis, failure_tags), "
            "reward function code, config, and training summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number (0-indexed)",
                },
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "get_iteration_metrics",
        "description": (
            "Get training scalar metrics summary for a specific iteration: "
            "loss curves, returns, entropy, SPS, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number (0-indexed)",
                },
            },
            "required": ["iteration"],
        },
    },
    {
        "name": "compare_reward_code",
        "description": (
            "Compare reward function code between two iterations. Returns a unified diff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iter_a": {
                    "type": "integer",
                    "description": "First iteration number",
                },
                "iter_b": {
                    "type": "integer",
                    "description": "Second iteration number",
                },
            },
            "required": ["iter_a", "iter_b"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations (readonly, session-scoped)
# ---------------------------------------------------------------------------


def _get_iter_record(session: SessionRecord, iteration: int) -> IterationRecord | None:
    return session.get_iteration(f"iter_{iteration}")


def _read_overview(session: SessionRecord) -> dict[str, Any]:
    history = session.read_history() or {}
    iterations_summary = []
    for it in history.get("iterations", []):
        j = it.get("judgment", {})
        iterations_summary.append(
            {
                "iteration": it.get("iteration"),
                "intent_score": round_metric(j.get("intent_score")),
                "passed": j.get("passed"),
                "failure_tags": j.get("failure_tags", []),
                "final_return": round_metric(it.get("summary", {}).get("final_episodic_return")),
            }
        )
    return {
        "session_id": history.get("session_id", session.session_id),
        "prompt": history.get("prompt", ""),
        "status": history.get("status", "unknown"),
        "best_iteration": history.get("best_iteration", 0),
        "best_score": round_metric(history.get("best_score", 0.0)),
        "total_iterations": len(iterations_summary),
        "iterations": iterations_summary,
    }


def _read_detail(session: SessionRecord, iteration: int) -> dict[str, Any]:
    rec = _get_iter_record(session, iteration)
    if rec is None:
        return {"error": f"Iteration {iteration} not found"}
    judgment = rec.read_judgment() or {}
    summary = rec.read_summary() or {}
    config = rec.read_config() or {}
    return {
        "iteration": iteration,
        "judgment": {
            "intent_score": round_metric(judgment.get("intent_score")),
            "passed": judgment.get("passed"),
            "diagnosis": judgment.get("diagnosis", ""),
            "failure_tags": judgment.get("failure_tags", []),
            "best_checkpoint": judgment.get("best_checkpoint"),
        },
        "reward_code": rec.read_reward_source(),
        "config": {
            "total_timesteps": config.get("total_timesteps"),
            "env_id": config.get("env_id"),
            "learning_rate": config.get("learning_rate"),
        },
        "summary": summary,
    }


def _read_metrics(session: SessionRecord, iteration: int) -> dict[str, Any]:
    rec = _get_iter_record(session, iteration)
    if rec is None:
        return {"error": f"Iteration {iteration} not found"}
    training, evaluation = rec.parse_scalars()
    # Summarize rather than dump all raw data
    result: dict[str, Any] = {"iteration": iteration}
    if training:
        last = training[-1]
        first = training[0]
        result["training_summary"] = {
            "total_entries": len(training),
            "first_step": first.get("global_step"),
            "last_step": last.get("global_step"),
            "final_policy_loss": last.get("policy_loss"),
            "final_value_loss": last.get("value_loss"),
            "final_entropy": last.get("entropy"),
            "final_episodic_return": last.get("episodic_return"),
            "final_sps": last.get("sps"),
        }
    if evaluation:
        result["evaluation"] = [
            {
                "step": e.get("global_step"),
                "total_reward": e.get("total_reward"),
                "episode_length": e.get("episode_length"),
            }
            for e in evaluation
        ]
    return result


def _compare_reward(session: SessionRecord, iter_a: int, iter_b: int) -> dict[str, Any]:
    rec_a = _get_iter_record(session, iter_a)
    rec_b = _get_iter_record(session, iter_b)
    code_a = rec_a.read_reward_source() if rec_a else ""
    code_b = rec_b.read_reward_source() if rec_b else ""
    if not code_a and not code_b:
        return {"error": "Neither iteration has reward code"}
    diff = list(
        difflib.unified_diff(
            code_a.splitlines(keepends=True),
            code_b.splitlines(keepends=True),
            fromfile=f"iter_{iter_a}/reward_fn.py",
            tofile=f"iter_{iter_b}/reward_fn.py",
        )
    )
    return {
        "iter_a": iter_a,
        "iter_b": iter_b,
        "diff": "".join(diff) if diff else "(identical)",
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an RL training analysis expert. You are analyzing a session of \
iterative reward function optimization for simulated robotics tasks.

Your job:
1. Use the provided tools to explore the session data.
2. Start with get_session_overview to understand the big picture.
3. Drill into interesting iterations (score drops, breakthroughs, failures).
4. Produce a structured analysis.

Output format (JSON):
{
  "analysis_en": "Analysis paragraph (3-5 sentences)",
  "key_findings": ["finding 1", ...],
  "recommendations": ["recommendation 1", ...]
}

Guidelines:
- Focus on WHY scores changed between iterations
- Identify reward hacking or training instabilities
- Keep findings actionable and specific
- 3-5 key findings, 2-3 recommendations
"""

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def analyze_session(
    session_id: str,
    *,
    client: anthropic.Anthropic,
    model: str = LLM_MODEL_LIGHT,
    runs_dir: Path = Path("runs"),
    max_rounds: int = 1000,
    on_status: Callable[[str], None] | None = None,
) -> SessionAnalysis:
    """Run agentic analysis of a session using tool-use loop.

    Args:
        session_id: Session directory name.
        client: Anthropic API client.
        model: Model to use for analysis.
        runs_dir: Root runs directory.
        max_rounds: Maximum tool-use rounds.
        on_status: Optional callback for streaming status messages.

    Returns:
        SessionAnalysis dict with structured results.
    """
    session = SessionRecord(runs_dir / session_id)
    if not session.path.exists():
        msg = f"Session not found: {session_id}"
        raise FileNotFoundError(msg)

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    # Tool dispatch table
    tool_dispatch: dict[str, Callable[[dict], Any]] = {
        "get_session_overview": lambda _inp: _read_overview(session),
        "get_iteration_detail": lambda inp: _read_detail(session, inp["iteration"]),
        "get_iteration_metrics": lambda inp: _read_metrics(session, inp["iteration"]),
        "compare_reward_code": lambda inp: _compare_reward(session, inp["iter_a"], inp["iter_b"]),
    }

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Analyze session '{session_id}'. "
                "Use the tools to explore the data, then provide your analysis as JSON."
            ),
        },
    ]

    tool_calls_used = 0

    for round_num in range(max_rounds):
        _status(f"Round {round_num + 1}: calling LLM...")

        response = create_message(
            client,
            model=model,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Check if the model wants to use tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    inp_str = json.dumps(tool_input, ensure_ascii=False)[:80]
                    _status(f"Round {round_num + 1}: {tool_name}({inp_str})")

                    handler = tool_dispatch.get(tool_name)
                    if handler:
                        result = handler(tool_input)
                        tool_calls_used += 1
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    result_str = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )
                    logger.info(
                        "[analyze] %s(%s) → %s",
                        tool_name,
                        json.dumps(tool_input, ensure_ascii=False),
                        result_str[:500],
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Extract the final text response
            _status("Parsing analysis result...")
            return _parse_analysis(response, session_id, model, tool_calls_used)

        else:
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

    # If we exhausted rounds, try to parse what we have from the last response
    _status("Max rounds reached, extracting result...")
    return _parse_analysis(response, session_id, model, tool_calls_used)


def _parse_analysis(
    response: Any,
    session_id: str,
    model: str,
    tool_calls_used: int,
) -> SessionAnalysis:
    """Extract SessionAnalysis from the final LLM response."""
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # Try to parse JSON from the response
    logger.info("[analyze] LLM final text (%d chars): %s", len(text), text[:500])
    analysis_data = _extract_json(text)

    return SessionAnalysis(
        session_id=session_id,
        analysis_en=analysis_data.get("analysis_en", text),
        key_findings=analysis_data.get("key_findings", []),
        recommendations=analysis_data.get("recommendations", []),
        tool_calls_used=tool_calls_used,
        model=model,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from text that may contain markdown fences."""
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    import re

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return {}
