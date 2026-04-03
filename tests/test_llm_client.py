"""Tests for pure functions in llm_client module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import p2p.inference.llm_client
from p2p.inference.llm_client import (
    _extract_tool_results,
    _extract_user_text,
    _serialize_content_blocks,
    extract_response_text,
    extract_thinking_text,
    get_client,
)

# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_llm_singleton():
    p2p.inference.llm_client._client = None
    yield
    p2p.inference.llm_client._client = None


def test_get_client_returns_anthropic_instance():
    """get_client() creates an Anthropic client."""
    with (
        patch("p2p.inference.llm_client.ANTHROPIC_API_KEY", "sk-fake"),
        patch("p2p.inference.llm_client.anthropic.Anthropic") as mock_cls,
    ):
        mock_cls.return_value = MagicMock()
        client = get_client()
        assert client is mock_cls.return_value


def test_get_client_returns_same_instance_on_second_call():
    """get_client() caches the client (singleton)."""
    with (
        patch("p2p.inference.llm_client.ANTHROPIC_API_KEY", "sk-fake"),
        patch("p2p.inference.llm_client.anthropic.Anthropic") as mock_cls,
    ):
        mock_cls.return_value = MagicMock()
        first = get_client()
        second = get_client()
        assert first is second
        mock_cls.assert_called_once()


def test_get_client_passes_api_key_when_set():
    """ANTHROPIC_API_KEY from settings flows to Anthropic(api_key=...)."""
    with (
        patch("p2p.inference.llm_client.ANTHROPIC_API_KEY", "sk-test-key"),
        patch("p2p.inference.llm_client.anthropic.Anthropic") as mock_cls,
    ):
        get_client()
        mock_cls.assert_called_once_with(api_key="sk-test-key")


@pytest.mark.parametrize("key_value", ["", None])
def test_get_client_returns_none_when_key_missing(key_value):
    """Missing ANTHROPIC_API_KEY (empty or None) returns None."""
    with patch("p2p.inference.llm_client.ANTHROPIC_API_KEY", key_value):
        assert get_client() is None


# ---------------------------------------------------------------------------
# _extract_user_text
# ---------------------------------------------------------------------------


def test_extract_user_text_string_content():
    messages = [{"role": "user", "content": "Hello world"}]
    assert _extract_user_text(messages) == "Hello world"


def test_extract_user_text_list_content_with_text_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "image", "source": "..."},
                {"type": "text", "text": "Second part"},
            ],
        }
    ]
    assert _extract_user_text(messages) == "First part Second part"


def test_extract_user_text_returns_last_user_message():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Second"},
    ]
    assert _extract_user_text(messages) == "Second"


def test_extract_user_text_no_user_messages():
    messages = [{"role": "assistant", "content": "Only assistant"}]
    assert _extract_user_text(messages) == ""


def test_extract_user_text_empty_messages():
    assert _extract_user_text([]) == ""


def test_extract_user_text_missing_content_key():
    messages = [{"role": "user"}]
    assert _extract_user_text(messages) == ""


# ---------------------------------------------------------------------------
# _extract_tool_results
# ---------------------------------------------------------------------------


def test_extract_tool_results_with_tool_result_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "id_123",
                    "content": "result data",
                },
                {"type": "text", "text": "some text"},
                {
                    "type": "tool_result",
                    "tool_use_id": "id_456",
                    "content": "other result",
                },
            ],
        }
    ]
    results = _extract_tool_results(messages)
    assert len(results) == 2
    assert results[0] == {"tool_use_id": "id_123", "content": "result data"}
    assert results[1] == {"tool_use_id": "id_456", "content": "other result"}


def test_extract_tool_results_no_tool_results():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "just text"}],
        }
    ]
    assert _extract_tool_results(messages) == []


def test_extract_tool_results_string_content():
    messages = [{"role": "user", "content": "plain string"}]
    assert _extract_tool_results(messages) == []


def test_extract_tool_results_empty_messages():
    assert _extract_tool_results([]) == []


def test_extract_tool_results_missing_fields_default_empty():
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result"}],
        }
    ]
    results = _extract_tool_results(messages)
    assert results == [{"tool_use_id": "", "content": ""}]


# ---------------------------------------------------------------------------
# _serialize_content_blocks
# ---------------------------------------------------------------------------


def _make_response(blocks):
    """Create a minimal response-like object with .content attribute."""
    return SimpleNamespace(content=blocks)


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def test_serialize_content_blocks_text_only():
    response = _make_response([_text_block("Hello"), _text_block("World")])
    result = _serialize_content_blocks(response)
    assert result["response"] == "Hello\nWorld"
    assert result["tool_calls"] == []


def test_serialize_content_blocks_tool_only():
    response = _make_response([_tool_block("t1", "search", {"query": "test"})])
    result = _serialize_content_blocks(response)
    assert result["response"] == ""
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0] == {
        "id": "t1",
        "name": "search",
        "input": {"query": "test"},
    }


def test_serialize_content_blocks_mixed():
    response = _make_response(
        [
            _text_block("Thinking..."),
            _tool_block("t1", "calc", {"expr": "2+2"}),
            _text_block("Done."),
        ]
    )
    result = _serialize_content_blocks(response)
    assert result["response"] == "Thinking...\nDone."
    assert len(result["tool_calls"]) == 1


def test_serialize_content_blocks_empty_content():
    response = _make_response([])
    result = _serialize_content_blocks(response)
    assert result["response"] == ""
    assert result["tool_calls"] == []


def test_serialize_content_blocks_none_content():
    response = _make_response(None)
    result = _serialize_content_blocks(response)
    assert result["response"] == ""
    assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Thinking block helpers
# ---------------------------------------------------------------------------


def _thinking_block(text):
    return SimpleNamespace(type="thinking", thinking=text)


def test_serialize_content_blocks_with_thinking():
    response = _make_response(
        [_thinking_block("Let me reason..."), _text_block("The answer is 42.")]
    )
    result = _serialize_content_blocks(response)
    assert result["response"] == "The answer is 42."
    assert result["thinking"] == "Let me reason..."
    assert result["tool_calls"] == []


def test_serialize_content_blocks_thinking_not_present_when_empty():
    response = _make_response([_text_block("No thinking here.")])
    result = _serialize_content_blocks(response)
    assert "thinking" not in result


def test_extract_response_text_skips_thinking():
    response = _make_response([_thinking_block("internal reasoning"), _text_block("final answer")])
    assert extract_response_text(response) == "final answer"


def test_extract_thinking_text():
    response = _make_response(
        [
            _thinking_block("step 1"),
            _text_block("answer"),
            _thinking_block("step 2"),
        ]
    )
    assert extract_thinking_text(response) == "step 1\nstep 2"


def test_extract_thinking_text_empty_when_no_thinking():
    response = _make_response([_text_block("just text")])
    assert extract_thinking_text(response) == ""


# ---------------------------------------------------------------------------
# create_message — adaptive thinking injection
# ---------------------------------------------------------------------------


def _mock_client_and_response():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [SimpleNamespace(type="text", text="ok")]
    mock_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_create_message_injects_adaptive_thinking_from_effort_kwarg():
    """thinking_effort='high' should inject adaptive thinking + output_config."""
    mock_client = _mock_client_and_response()

    with patch("p2p.settings.THINKING_EFFORT", ""):
        from p2p.inference.llm_client import create_message

        create_message(
            mock_client,
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "test"}],
            thinking_effort="high",
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "high"}


def test_create_message_injects_adaptive_thinking_from_env_default():
    """THINKING_EFFORT='max' should auto-inject adaptive thinking."""
    mock_client = _mock_client_and_response()

    with patch("p2p.settings.THINKING_EFFORT", "max"):
        from p2p.inference.llm_client import create_message

        create_message(
            mock_client,
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "test"}],
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "max"}


def test_create_message_no_thinking_when_effort_empty():
    """THINKING_EFFORT='' means no thinking."""
    mock_client = _mock_client_and_response()

    with patch("p2p.settings.THINKING_EFFORT", ""):
        from p2p.inference.llm_client import create_message

        create_message(
            mock_client,
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "test"}],
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "thinking" not in call_kwargs
    assert "output_config" not in call_kwargs


def test_create_message_strips_temperature_when_thinking():
    """temperature must be removed when thinking is active (API requirement)."""
    mock_client = _mock_client_and_response()

    with patch("p2p.settings.THINKING_EFFORT", ""):
        from p2p.inference.llm_client import create_message

        create_message(
            mock_client,
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "test"}],
            thinking_effort="high",
            temperature=0.7,
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "temperature" not in call_kwargs
    assert "thinking" in call_kwargs
