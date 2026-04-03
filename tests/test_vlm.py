"""Tests for VLM multi-provider routing and video support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# call_vlm_auto routing
# ---------------------------------------------------------------------------


@patch("p2p.inference.vlm.call_vlm_vllm", return_value='{"intent_score": 0.7}')
def test_call_vlm_auto_routes_vllm(mock_vllm):
    from p2p.inference.vlm import call_vlm_auto

    call_vlm_auto(
        "test prompt",
        ["base64img"],
        vlm_model="vllm-Qwen/Qwen3.5-27B",
        video_path=Path("/tmp/test.mp4"),
    )
    mock_vllm.assert_called_once()
    # Verify prefix was stripped
    call_kwargs = mock_vllm.call_args
    assert call_kwargs.kwargs["model"] == "Qwen/Qwen3.5-27B"


@patch("p2p.inference.vlm.call_vlm_gemini", return_value='{"intent_score": 0.8}')
def test_call_vlm_auto_routes_gemini_with_video(mock_gemini):
    from p2p.inference.vlm import call_vlm_auto

    video = Path("/tmp/test.mp4")
    call_vlm_auto(
        "test prompt",
        ["base64img"],
        vlm_model="gemini-3.1-pro-preview",
        video_path=video,
    )
    mock_gemini.assert_called_once()
    call_kwargs = mock_gemini.call_args
    assert call_kwargs.kwargs.get("video_path") == video


@patch("p2p.inference.vlm.call_vlm_anthropic", return_value='{"intent_score": 0.6}')
def test_call_vlm_auto_routes_claude_no_video(mock_claude):
    from p2p.inference.vlm import call_vlm_auto

    call_vlm_auto(
        "test prompt",
        ["base64img"],
        vlm_model="claude-opus-4-6",
        video_path=Path("/tmp/test.mp4"),
    )
    mock_claude.assert_called_once()
    # Claude doesn't receive video_path
    call_args = mock_claude.call_args
    assert "video_path" not in call_args.kwargs


@patch("p2p.inference.vlm.call_vlm", return_value='{"intent_score": 0.5}')
def test_call_vlm_auto_routes_ollama_default(mock_ollama):
    from p2p.inference.vlm import call_vlm_auto

    call_vlm_auto("test prompt", ["base64img"], vlm_model="qwen3.5:27b")
    mock_ollama.assert_called_once()


# ---------------------------------------------------------------------------
# call_vlm_auto backward compat (no video_path)
# ---------------------------------------------------------------------------


@patch("p2p.inference.vlm.call_vlm", return_value='{"intent_score": 0.5}')
def test_call_vlm_auto_no_video_path_backward_compat(mock_ollama):
    """call_vlm_auto works without video_path (backward compatible)."""
    from p2p.inference.vlm import call_vlm_auto

    result = call_vlm_auto("test", ["img"], vlm_model="qwen3.5:27b")
    assert result == '{"intent_score": 0.5}'


# ---------------------------------------------------------------------------
# vllm_health_check
# ---------------------------------------------------------------------------


@patch("p2p.inference.vllm_server.requests.get")
def test_vllm_health_check_success(mock_get):
    from p2p.inference.vllm_server import vllm_health_check

    mock_get.return_value = MagicMock(status_code=200)
    assert vllm_health_check(8100) is True


@patch("p2p.inference.vllm_server.requests.get", side_effect=ConnectionError("refused"))
def test_vllm_health_check_failure(mock_get):
    from p2p.inference.vllm_server import vllm_health_check

    assert vllm_health_check(8100) is False


# ---------------------------------------------------------------------------
# Provider supports video
# ---------------------------------------------------------------------------


def test_provider_supports_video():
    from p2p.agents.judge_agent import _provider_supports_video

    assert _provider_supports_video("vllm-Qwen/Qwen3.5-27B") is True
    assert _provider_supports_video("gemini-3.1-pro-preview") is True
    assert _provider_supports_video("claude-opus-4-6") is False
    assert _provider_supports_video("qwen3.5:27b") is False


# ---------------------------------------------------------------------------
# call_vlm_anthropic config vs API error separation
# ---------------------------------------------------------------------------


@patch("p2p.inference.llm_client.get_client", side_effect=RuntimeError("API key missing"))
def test_call_vlm_anthropic_config_error_not_wrapped(_mock_client):
    """Config errors from get_client() must propagate directly, not as VLMError."""
    import pytest

    from p2p.inference.vlm import call_vlm_anthropic

    with pytest.raises(RuntimeError, match="API key missing"):
        call_vlm_anthropic("prompt", "img_b64")


# ---------------------------------------------------------------------------
# call_vlm_two_turn routing
# ---------------------------------------------------------------------------


@patch("p2p.inference.vlm._two_turn_vllm", return_value=("criteria", '{"intent_score": 0.7}'))
def test_call_vlm_two_turn_routes_vllm(mock_vllm):
    from p2p.inference.vlm import call_vlm_two_turn

    call_vlm_two_turn(
        "turn1",
        "turn2",
        ["base64img"],
        vlm_model="vllm-Qwen/Qwen3.5-27B",
        video_path=Path("/tmp/test.mp4"),
    )
    mock_vllm.assert_called_once()
    call_kwargs = mock_vllm.call_args
    assert call_kwargs.kwargs["model"] == "Qwen/Qwen3.5-27B"


@patch("p2p.inference.vlm._two_turn_gemini", return_value=("criteria", '{"intent_score": 0.8}'))
def test_call_vlm_two_turn_routes_gemini(mock_gemini):
    from p2p.inference.vlm import call_vlm_two_turn

    video = Path("/tmp/test.mp4")
    call_vlm_two_turn(
        "turn1",
        "turn2",
        ["base64img"],
        vlm_model="gemini-3.1-pro-preview",
        video_path=video,
    )
    mock_gemini.assert_called_once()
    call_kwargs = mock_gemini.call_args
    assert call_kwargs.kwargs.get("video_path") == video


@patch("p2p.inference.vlm._two_turn_anthropic", return_value=("criteria", '{"intent_score": 0.6}'))
def test_call_vlm_two_turn_routes_claude(mock_claude):
    from p2p.inference.vlm import call_vlm_two_turn

    call_vlm_two_turn(
        "turn1",
        "turn2",
        ["base64img"],
        vlm_model="claude-opus-4-6",
        video_path=Path("/tmp/test.mp4"),
    )
    mock_claude.assert_called_once()
    # Claude doesn't receive video_path
    call_args = mock_claude.call_args
    assert "video_path" not in call_args.kwargs


@patch("p2p.inference.vlm._two_turn_ollama", return_value=("criteria", '{"intent_score": 0.5}'))
def test_call_vlm_two_turn_routes_ollama(mock_ollama):
    from p2p.inference.vlm import call_vlm_two_turn

    call_vlm_two_turn("turn1", "turn2", ["base64img"], vlm_model="qwen3.5:27b")
    mock_ollama.assert_called_once()
