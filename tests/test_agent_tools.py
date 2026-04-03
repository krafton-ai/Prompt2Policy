"""Tests for p2p.inference.agent_tools — tool dispatch infrastructure."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from p2p.inference.agent_tools import (
    ToolLoopResult,
    dispatch_tool_calls,
    emit_conversation,
    run_tool_loop,
    serialize_assistant_response,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for Anthropic response objects
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = "hello"


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = "tu_1"
    name: str = "my_tool"
    input: dict = field(default_factory=dict)
    text: str | None = None  # should NOT be serialized as text


@dataclass
class FakeResponse:
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "end_turn"


# ===================================================================
# serialize_assistant_response
# ===================================================================


class TestSerializeAssistantResponse:
    def test_text_block(self):
        resp = FakeResponse(content=[FakeTextBlock(text="hi")])
        result = serialize_assistant_response(resp)

        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "hi"}

    def test_tool_use_block(self):
        block = FakeToolUseBlock(id="tu_x", name="search", input={"q": "foo"})
        resp = FakeResponse(content=[block])
        result = serialize_assistant_response(resp)

        assert len(result["content"]) == 1
        assert result["content"][0] == {
            "type": "tool_use",
            "id": "tu_x",
            "name": "search",
            "input": {"q": "foo"},
        }

    def test_mixed_blocks(self):
        blocks = [
            FakeTextBlock(text="thinking..."),
            FakeToolUseBlock(id="tu_2", name="calc", input={"x": 1}),
            FakeTextBlock(text="done"),
        ]
        resp = FakeResponse(content=blocks)
        result = serialize_assistant_response(resp)

        assert len(result["content"]) == 3
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][2]["type"] == "text"

    def test_empty_content(self):
        resp = FakeResponse(content=[])
        result = serialize_assistant_response(resp)

        assert result["content"] == []
        assert result["role"] == "assistant"

    def test_none_content(self):
        resp = FakeResponse(content=None)
        result = serialize_assistant_response(resp)

        assert result["content"] == []

    def test_stop_reason_preserved(self):
        resp = FakeResponse(content=[], stop_reason="tool_use")
        result = serialize_assistant_response(resp)

        assert result["stop_reason"] == "tool_use"


# ===================================================================
# dispatch_tool_calls
# ===================================================================


class TestDispatchToolCalls:
    def test_dispatches_known_tool(self):
        block = FakeToolUseBlock(id="tu_1", name="greet", input={"name": "Ada"})
        resp = FakeResponse(content=[block])
        dispatch = {"greet": lambda inp: {"msg": f"Hi {inp['name']}"}}

        results = dispatch_tool_calls(resp, dispatch)

        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tu_1"
        parsed = json.loads(results[0]["content"])
        assert parsed == {"msg": "Hi Ada"}

    def test_handler_exception_returns_error(self):
        """Handler raising an exception returns an error dict instead of crashing."""

        def bad_handler(_inp):
            raise RuntimeError("disk full")

        block = FakeToolUseBlock(id="tu_err", name="bad", input={})
        resp = FakeResponse(content=[block])

        results = dispatch_tool_calls(resp, {"bad": bad_handler})

        assert len(results) == 1
        parsed = json.loads(results[0]["content"])
        assert "error" in parsed
        assert "Internal error in bad" in parsed["error"]
        assert "RuntimeError" in parsed["error"]

    def test_handler_exception_does_not_stop_other_tools(self):
        """An exception in one handler does not prevent dispatching the next."""

        def crash(_inp):
            raise ValueError("oops")

        blocks = [
            FakeToolUseBlock(id="tu_c1", name="crash", input={}),
            FakeToolUseBlock(id="tu_c2", name="ok", input={}),
        ]
        resp = FakeResponse(content=blocks)
        dispatch = {"crash": crash, "ok": lambda _: {"status": "fine"}}

        results = dispatch_tool_calls(resp, dispatch)

        assert len(results) == 2
        assert "error" in json.loads(results[0]["content"])
        assert json.loads(results[1]["content"])["status"] == "fine"

    def test_unknown_tool_returns_error(self):
        block = FakeToolUseBlock(id="tu_2", name="missing_tool", input={})
        resp = FakeResponse(content=[block])

        results = dispatch_tool_calls(resp, {})

        assert len(results) == 1
        parsed = json.loads(results[0]["content"])
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]
        assert "missing_tool" in parsed["error"]

    def test_empty_content(self):
        resp = FakeResponse(content=[])
        results = dispatch_tool_calls(resp, {"foo": lambda x: x})

        assert results == []

    def test_skips_text_blocks(self):
        blocks = [
            FakeTextBlock(text="reasoning"),
            FakeToolUseBlock(id="tu_3", name="calc", input={"n": 42}),
        ]
        resp = FakeResponse(content=blocks)
        dispatch = {"calc": lambda inp: {"result": inp["n"] * 2}}

        results = dispatch_tool_calls(resp, dispatch)

        assert len(results) == 1
        parsed = json.loads(results[0]["content"])
        assert parsed["result"] == 84

    def test_multiple_tool_calls(self):
        blocks = [
            FakeToolUseBlock(id="tu_a", name="add", input={"a": 1, "b": 2}),
            FakeToolUseBlock(id="tu_b", name="add", input={"a": 3, "b": 4}),
        ]
        resp = FakeResponse(content=blocks)
        dispatch = {"add": lambda inp: {"sum": inp["a"] + inp["b"]}}

        results = dispatch_tool_calls(resp, dispatch)

        assert len(results) == 2
        assert json.loads(results[0]["content"])["sum"] == 3
        assert json.loads(results[1]["content"])["sum"] == 7

    def test_handler_returning_non_serializable_uses_default_str(self):
        """Handler returns object with datetime — json.dumps default=str handles it."""
        from datetime import datetime

        block = FakeToolUseBlock(id="tu_4", name="now", input={})
        resp = FakeResponse(content=[block])
        dt = datetime(2025, 1, 1, 12, 0, 0)
        dispatch = {"now": lambda _: {"time": dt}}

        results = dispatch_tool_calls(resp, dispatch)

        parsed = json.loads(results[0]["content"])
        assert "2025" in parsed["time"]

    def test_custom_agent_name_in_logging(self, caplog):
        block = FakeToolUseBlock(id="tu_5", name="ping", input={})
        resp = FakeResponse(content=[block])
        dispatch = {"ping": lambda _: {"pong": True}}

        with caplog.at_level("INFO", logger="p2p.inference.agent_tools"):
            dispatch_tool_calls(resp, dispatch, agent_name="test_agent")

        assert "test_agent" in caplog.text


# ===================================================================
# emit_conversation
# ===================================================================


class TestEmitConversation:
    @patch("p2p.inference.agent_tools._emit")
    def test_emits_correct_event(self, mock_emit):
        conversation = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bye"},
        ]
        emit_conversation("my_agent", "claude-test", conversation)

        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "llm.conversation"
        data = args[1]["data"]
        assert data["agent"] == "my_agent"
        assert data["model"] == "claude-test"
        assert data["rounds"] == 2  # two assistant messages
        assert data["conversation"] is conversation

    @patch("p2p.inference.agent_tools._emit")
    def test_zero_assistant_rounds(self, mock_emit):
        conversation = [{"role": "user", "content": "hello"}]
        emit_conversation("agent", "model", conversation)

        data = mock_emit.call_args[1]["data"]
        assert data["rounds"] == 0


# ===================================================================
# ToolLoopResult
# ===================================================================


class TestToolLoopResult:
    def test_defaults(self):
        resp = FakeResponse()
        result = ToolLoopResult(response=resp)

        assert result.response is resp
        assert result.conversation_log == []
        assert result.tool_calls_used == 0


# ===================================================================
# run_tool_loop
# ===================================================================


class TestRunToolLoop:
    """Tests for the main tool-use loop.

    All tests mock ``create_message`` to avoid real API calls.
    """

    def _make_client(self):
        return MagicMock()

    @patch("p2p.inference.agent_tools.create_message")
    def test_single_turn_end_turn(self, mock_create):
        """Model responds with end_turn immediately — no tool dispatch."""
        mock_create.return_value = FakeResponse(
            content=[FakeTextBlock(text="done")],
            stop_reason="end_turn",
        )

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "hi"}],
            tool_dispatch={},
        )

        assert result.tool_calls_used == 0
        assert mock_create.call_count == 1
        assert result.conversation_log[-1]["stop_reason"] == "end_turn"

    @patch("p2p.inference.agent_tools.create_message")
    def test_one_tool_round_then_end(self, mock_create):
        """Model calls a tool once, then ends."""
        tool_resp = FakeResponse(
            content=[FakeToolUseBlock(id="tu_1", name="ping", input={})],
            stop_reason="tool_use",
        )
        final_resp = FakeResponse(
            content=[FakeTextBlock(text="pong received")],
            stop_reason="end_turn",
        )
        mock_create.side_effect = [tool_resp, final_resp]

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[{"name": "ping"}],
            messages=[{"role": "user", "content": "ping"}],
            tool_dispatch={"ping": lambda _: {"pong": True}},
        )

        assert result.tool_calls_used == 1
        assert mock_create.call_count == 2

    @patch("p2p.inference.agent_tools.create_message")
    def test_multi_round_tool_use(self, mock_create):
        """Model calls tools across multiple rounds."""
        round1 = FakeResponse(
            content=[FakeToolUseBlock(id="tu_1", name="step", input={"n": 1})],
            stop_reason="tool_use",
        )
        round2 = FakeResponse(
            content=[FakeToolUseBlock(id="tu_2", name="step", input={"n": 2})],
            stop_reason="tool_use",
        )
        final = FakeResponse(
            content=[FakeTextBlock(text="all done")],
            stop_reason="end_turn",
        )
        mock_create.side_effect = [round1, round2, final]

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "go"}],
            tool_dispatch={"step": lambda inp: {"ok": inp["n"]}},
        )

        assert result.tool_calls_used == 2
        assert mock_create.call_count == 3

    @patch("p2p.inference.agent_tools.create_message")
    def test_max_rounds_limits_iterations(self, mock_create):
        """Loop stops after max_rounds even if model keeps requesting tools."""
        tool_resp = FakeResponse(
            content=[FakeToolUseBlock(id="tu_1", name="loop", input={})],
            stop_reason="tool_use",
        )
        # Return tool_use forever — loop should stop after max_rounds
        mock_create.return_value = tool_resp

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "go"}],
            tool_dispatch={"loop": lambda _: {"ok": True}},
            max_rounds=3,
        )

        # 1 initial call + 3 rounds = 4 calls total
        assert mock_create.call_count == 4
        assert result.tool_calls_used == 3

    @patch("p2p.inference.agent_tools.create_message")
    def test_force_final_turn(self, mock_create):
        """force_final_turn makes one extra call without tools."""
        tool_resp = FakeResponse(
            content=[FakeToolUseBlock(id="tu_1", name="t", input={})],
            stop_reason="tool_use",
        )
        final = FakeResponse(
            content=[FakeTextBlock(text="forced end")],
            stop_reason="end_turn",
        )
        # max_rounds=1: initial + 1 round (both tool_use), then forced final
        mock_create.side_effect = [tool_resp, tool_resp, final]

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "go"}],
            tool_dispatch={"t": lambda _: {"ok": True}},
            max_rounds=1,
            force_final_turn=True,
        )

        # Initial + 1 round + forced final = 3 calls
        assert mock_create.call_count == 3
        # 1 from round loop + 1 from force_final_turn dispatch
        assert result.tool_calls_used == 2

    @patch("p2p.inference.agent_tools.create_message")
    def test_on_status_callback_called(self, mock_create):
        """on_status callback is invoked during loop execution."""
        mock_create.return_value = FakeResponse(
            content=[FakeTextBlock(text="done")],
            stop_reason="end_turn",
        )
        statuses: list[str] = []

        run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "hi"}],
            tool_dispatch={},
            on_status=statuses.append,
        )

        assert len(statuses) >= 1
        assert "Round 1" in statuses[0]

    @patch("p2p.inference.agent_tools.create_message")
    def test_unexpected_stop_reason_breaks_loop(self, mock_create):
        """Unexpected stop_reason (not end_turn or tool_use) stops the loop."""
        mock_create.return_value = FakeResponse(
            content=[FakeTextBlock(text="weird")],
            stop_reason="max_tokens",
        )

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "hi"}],
            tool_dispatch={},
        )

        assert mock_create.call_count == 1
        assert result.tool_calls_used == 0

    @patch("p2p.inference.agent_tools.create_message")
    def test_conversation_log_structure(self, mock_create):
        """Conversation log contains system, user, and assistant entries."""
        mock_create.return_value = FakeResponse(
            content=[FakeTextBlock(text="hi")],
            stop_reason="end_turn",
        )

        result = run_tool_loop(
            client=self._make_client(),
            model="test",
            system="test system",
            tools=[],
            messages=[{"role": "user", "content": "hello"}],
            tool_dispatch={},
        )

        log = result.conversation_log
        assert log[0]["role"] == "system"
        assert log[0]["content"] == "test system"
        assert log[1]["role"] == "user"
        assert log[1]["content"] == "hello"
        assert log[2]["role"] == "assistant"
