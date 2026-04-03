"""Shared tool-use loop infrastructure for LLM agents.

Provides ``run_tool_loop()`` to drive the tool-use conversation cycle,
``dispatch_tool_calls()`` to process tool_use blocks from a response,
and ``serialize_assistant_response()`` for response serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from p2p.event_log import emit as _emit
from p2p.inference.llm_client import create_message

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_assistant_response(response: Any) -> dict[str, Any]:
    """Serialize an Anthropic response into a JSON-safe conversation entry."""
    blocks: list[dict[str, Any]] = []
    for block in response.content or []:
        if hasattr(block, "text") and isinstance(block.text, str):
            blocks.append({"type": "text", "text": block.text})
        elif getattr(block, "type", None) == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input or {},
                }
            )
    return {
        "role": "assistant",
        "content": blocks,
        "stop_reason": response.stop_reason,
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def dispatch_tool_calls(
    response: Any,
    tool_dispatch: dict[str, Callable[[dict], Any]],
    *,
    agent_name: str = "agent",
) -> list[dict[str, Any]]:
    """Process all tool_use blocks in *response* via *tool_dispatch*.

    Returns a list of ``tool_result`` dicts ready to append to messages.
    """
    tool_results: list[dict[str, Any]] = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        handler = tool_dispatch.get(block.name)
        if handler:
            try:
                result_data = handler(block.input or {})
            except Exception as exc:
                logger.exception("[%s] Handler crash for %s", agent_name, block.name)
                result_data = {"error": f"Internal error in {block.name}: {type(exc).__name__}"}
        else:
            result_data = {"error": f"Unknown tool: {block.name}"}

        result_str = json.dumps(result_data, ensure_ascii=False, default=str)
        logger.info(
            "[%s] %s(%s) → %s",
            agent_name,
            block.name,
            json.dumps(block.input or {}, ensure_ascii=False),
            result_str[:500],
        )
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            }
        )
    return tool_results


# ---------------------------------------------------------------------------
# Conversation logging
# ---------------------------------------------------------------------------


def emit_conversation(
    agent_name: str,
    model: str,
    conversation: list[dict[str, Any]],
) -> None:
    """Emit an ``llm.conversation`` event for debugging."""
    _emit(
        "llm.conversation",
        data={
            "agent": agent_name,
            "model": model,
            "rounds": len([m for m in conversation if m["role"] == "assistant"]),
            "conversation": conversation,
        },
    )


# ---------------------------------------------------------------------------
# Tool-use loop
# ---------------------------------------------------------------------------


@dataclass
class ToolLoopResult:
    """Result of ``run_tool_loop``."""

    response: Any
    """Final Anthropic response object."""
    conversation_log: list[dict[str, Any]] = field(default_factory=list)
    """Full conversation log for debugging."""
    tool_calls_used: int = 0
    """Total number of tool calls dispatched."""


def run_tool_loop(
    *,
    client: anthropic.Anthropic | None,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_dispatch: dict[str, Callable[[dict], Any]],
    max_rounds: int = 1000,
    on_status: Callable[[str], None] | None = None,
    agent_name: str = "agent",
    force_final_turn: bool = False,
) -> ToolLoopResult:
    """Run a tool-use conversation loop with an LLM.

    Routes to the correct provider (Anthropic, Gemini, OpenAI) via
    ``create_message``, which auto-detects the provider from the model name.

    Args:
        client: Anthropic API client (None when using non-Anthropic models).
        model: Model identifier.
        system: System prompt.
        tools: Tool schema definitions.
        messages: Initial messages (will be mutated).
        tool_dispatch: Mapping of tool names to handler callables.
        max_rounds: Maximum tool-use rounds before stopping.
        on_status: Optional callback for status messages.
        agent_name: Name used in log messages.
        force_final_turn: If True and max_rounds is exhausted while the model
            still wants tools, make one more call WITHOUT tools to force text.

    Returns:
        ToolLoopResult with the final response, conversation log, and tool
        call count.
    """

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    conversation_log: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": messages[0]["content"] if messages else ""},
    ]
    tool_calls_used = 0

    # First LLM call (always happens, even if max_rounds=0)
    _status("Round 1: calling LLM...")
    response = create_message(
        client,
        model=model,
        system=system,
        tools=tools,
        messages=messages,
        agent_name=agent_name,
    )
    conversation_log.append(serialize_assistant_response(response))

    for round_num in range(max_rounds):
        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            logger.warning("[%s] Unexpected stop_reason: %s", agent_name, response.stop_reason)
            break

        tool_results = dispatch_tool_calls(response, tool_dispatch, agent_name=agent_name)
        tool_calls_used += len(tool_results)

        for block in response.content:
            if block.type == "tool_use":
                inp_str = json.dumps(block.input, ensure_ascii=False)[:80]
                _status(f"Round {round_num + 2}: {block.name}({inp_str})")

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        conversation_log.append({"role": "tool_results", "content": tool_results})

        _status(f"Round {round_num + 2}: calling LLM...")
        response = create_message(
            client,
            model=model,
            system=system,
            tools=tools,
            messages=messages,
            agent_name=agent_name,
        )
        conversation_log.append(serialize_assistant_response(response))

    # Forced final turn: process pending tool calls, then call without tools
    if force_final_turn and response and response.stop_reason == "tool_use":
        logger.warning(
            "[%s] Exhausted %d tool rounds, forcing final turn",
            agent_name,
            max_rounds,
        )
        tool_results = dispatch_tool_calls(response, tool_dispatch, agent_name=agent_name)
        tool_calls_used += len(tool_results)

        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            conversation_log.append({"role": "tool_results", "content": tool_results})
        else:
            logger.warning("[%s] tool_use stop_reason but no tool_use blocks found", agent_name)

        response = create_message(
            client,
            model=model,
            system=system,
            messages=messages,
            agent_name=agent_name,
        )
        conversation_log.append(serialize_assistant_response(response))
        if response.stop_reason != "end_turn":
            logger.warning(
                "[%s] Final forced turn stop_reason: %s",
                agent_name,
                response.stop_reason,
            )

    return ToolLoopResult(
        response=response,
        conversation_log=conversation_log,
        tool_calls_used=tool_calls_used,
    )
