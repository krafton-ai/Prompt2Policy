"""Tests for reward_author: code extraction, loading, generate/revise."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock

import numpy as np
import pytest

from p2p.agents.reward_author import (
    _extract_code,
    generate,
    load_reward_fn,
    revise,
    validate_reward_code,
)
from p2p.training.env_spec import get_env_spec

# -- _extract_code -----------------------------------------------------------


def test_extract_code_fenced_block():
    text = "Here is the code:\n```python\ndef reward_fn():\n    pass\n```\nDone."
    assert _extract_code(text) == "def reward_fn():\n    pass"


def test_extract_code_fenced_no_lang():
    text = "```\ndef reward_fn():\n    return 0, {}\n```"
    assert _extract_code(text) == "def reward_fn():\n    return 0, {}"


def test_extract_code_raw_python():
    text = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
    assert "def reward_fn" in _extract_code(text)


def test_extract_code_no_code_raises():
    with pytest.raises(ValueError, match="No valid code found"):
        _extract_code("Just some random text with no code")


def test_extract_code_stateful_closure():
    """Stateful _make_reward closure should be extracted and get the module-level assignment."""
    text = textwrap.dedent("""\
        ```python
        def _make_reward():
            state = {"step": 0}
            def reward_fn(obs, action, next_obs, info):
                state["step"] += 1
                return 1.0, {"step": float(state["step"])}
            return reward_fn
        ```
    """)
    code = _extract_code(text)
    assert "_make_reward" in code
    assert "reward_fn = _make_reward()" in code


def test_extract_code_stateful_closure_already_has_assignment():
    """If the LLM already includes reward_fn = _make_reward(), don't duplicate it."""
    text = textwrap.dedent("""\
        ```python
        def _make_reward():
            state = {"step": 0}
            def reward_fn(obs, action, next_obs, info):
                state["step"] += 1
                return 1.0, {"step": float(state["step"])}
            return reward_fn
        reward_fn = _make_reward()
        ```
    """)
    code = _extract_code(text)
    assert code.count("reward_fn = _make_reward()") == 1


# -- load_reward_fn -----------------------------------------------------------


def test_load_reward_fn_valid():
    code = (
        "def reward_fn(obs, action, next_obs, info):\n"
        "    return float(np.sum(obs)), {'sum': float(np.sum(obs))}\n"
    )
    fn = load_reward_fn(code)
    obs = np.ones(17)
    reward, terms = fn(obs, np.zeros(6), obs, {})
    assert reward == 17.0
    assert terms["sum"] == 17.0


def test_load_reward_fn_missing_function():
    code = "x = 42\n"
    with pytest.raises(ValueError, match="does not define"):
        load_reward_fn(code)


def test_load_reward_fn_syntax_error():
    code = "def reward_fn(\n"
    with pytest.raises(SyntaxError):
        load_reward_fn(code)


def test_load_reward_fn_stateful_closure():
    """Stateful _make_reward closure should load and maintain state across calls."""
    code = textwrap.dedent("""\
        def _make_reward():
            state = {"step": 0}
            def reward_fn(obs, action, next_obs, info):
                state["step"] += 1
                return float(state["step"]), {"step": float(state["step"])}
            return reward_fn
        reward_fn = _make_reward()
    """)
    fn = load_reward_fn(code)
    obs = np.ones(17)
    action = np.zeros(6)
    r1, t1 = fn(obs, action, obs, {})
    r2, t2 = fn(obs, action, obs, {})
    assert r1 == 1.0
    assert r2 == 2.0  # state persists across calls


def test_load_reward_fn_invalid_escape_is_sanitized():
    """Code with invalid escape sequences like \\| should be auto-fixed (#53)."""
    code = (
        "def reward_fn(obs, action, next_obs, info):\n"
        '    """LaTeX: r = \\|a\\|"""\n'
        '    return 1.0, {"a": 1.0}\n'
    )
    fn = load_reward_fn(code)
    result, terms = fn(np.zeros(17), np.zeros(6), np.zeros(17), {})
    assert result == 1.0


# -- validate_reward_code -----------------------------------------------------


def test_validate_reward_code_valid():
    code = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
    validate_reward_code(code)  # should not raise


def test_validate_reward_code_syntax_error():
    with pytest.raises(SyntaxError):
        validate_reward_code("def reward_fn(\n")


# -- generate with retry (#66) ------------------------------------------------


def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock anthropic.Anthropic client returning given text."""
    block = MagicMock()
    block.type = "text"
    block.text = response_text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 100
    response.usage.output_tokens = 200
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _make_sequential_mock_client(responses: list[str]) -> MagicMock:
    """Create a mock client that returns different responses on successive calls."""
    client = MagicMock()
    side_effects = []
    for text in responses:
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        response.usage.input_tokens = 100
        response.usage.output_tokens = 200
        side_effects.append(response)
    client.messages.create.side_effect = side_effects
    return client


def test_generate_calls_api_and_extracts_code():
    code = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
    resp = f"```python\n{code}\n```"
    client = _make_mock_client(resp)

    result = generate(
        "run fast", client=client, model="claude-opus-4-6", env=get_env_spec("HalfCheetah-v5")
    )

    assert result == code
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-6"
    assert any("run fast" in m["content"] for m in call_kwargs["messages"])


def test_generate_retries_on_syntax_error():
    """generate() should retry when LLM produces invalid code (#66)."""
    bad_code = "def reward_fn(\n"
    good_code = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
    bad_response = f"Here's the reward function:\n```python\n{bad_code}\n```\nLet me know."
    client = _make_sequential_mock_client(
        [
            bad_response,
            f"```python\n{good_code}\n```",
        ]
    )

    result = generate(
        "run fast", client=client, model="claude-opus-4-6", env=get_env_spec("HalfCheetah-v5")
    )

    assert result == good_code
    assert client.messages.create.call_count == 2
    # Retry should preserve the original LLM response, not just extracted code (#80)
    retry_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    assistant_msg = [m for m in retry_messages if m["role"] == "assistant"][0]
    assert assistant_msg["content"] == bad_response


def test_generate_raises_after_max_retries():
    """generate() should raise after MAX_CODE_RETRIES failed attempts."""
    bad_code = "def reward_fn(\n"
    responses = [f"```python\n{bad_code}\n```"] * 3
    client = _make_sequential_mock_client(responses)

    with pytest.raises(SyntaxError):
        generate(
            "run fast",
            client=client,
            model="claude-opus-4-6",
            env=get_env_spec("HalfCheetah-v5"),
        )


def test_revise_calls_api_with_judgment_and_summary():
    code = "def reward_fn(obs, action, next_obs, info):\n    return 2.0, {}"
    resp = f"```python\n{code}\n```"
    client = _make_mock_client(resp)

    judgment = {
        "intent_score": 0.3,
        "diagnosis": "not moving",
        "failure_tags": ["energy_too_high"],
    }
    summary = {"final_episodic_return": -100, "total_timesteps": 50000, "training_time_s": 30.0}

    result_code, result_overrides = revise(
        "run fast",
        "old code",
        judgment,
        summary,
        client=client,
        model="claude-opus-4-6",
        env=get_env_spec("HalfCheetah-v5"),
    )

    assert result_code == code
    assert isinstance(result_overrides, dict)
    call_kwargs = client.messages.create.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "not moving" in user_msg
    assert "energy_too_high" in user_msg
