"""Shared test fixtures."""

from __future__ import annotations

import pytest

import p2p.inference.llm_client


@pytest.fixture(autouse=True)
def _reset_llm_client():
    """Reset the cached Anthropic client singleton between tests."""
    p2p.inference.llm_client._client = None
    yield
    p2p.inference.llm_client._client = None


@pytest.fixture(autouse=True)
def _block_real_llm_calls(monkeypatch: pytest.MonkeyPatch):
    """Block real Gemini/OpenAI API calls that bypass the mocked Anthropic client.

    ``create_message()`` routes by model prefix: ``gemini*`` → ``_call_gemini()``,
    ``gpt-*``/``o1*`` → ``_call_openai()``.  These paths ignore the ``client``
    parameter, so a mocked Anthropic client offers no protection.  When
    ``LLM_MODEL`` is set to a non-Anthropic model in the environment
    (e.g. ``gemini-2.0-flash``), tests that rely on the default ``model``
    parameter would make real API calls.

    This fixture replaces both functions with stubs that raise immediately,
    turning a silent (and potentially rate-limited) real call into a loud
    failure.
    """

    def _blocked_gemini(**kwargs):
        raise RuntimeError(
            "Test attempted a real Gemini API call. "
            "Pass an explicit model=<name> or mock create_message() directly."
        )

    def _blocked_openai(**kwargs):
        raise RuntimeError(
            "Test attempted a real OpenAI API call. "
            "Pass an explicit model=<name> or mock create_message() directly."
        )

    monkeypatch.setattr(p2p.inference.llm_client, "_call_gemini", _blocked_gemini)
    monkeypatch.setattr(p2p.inference.llm_client, "_call_openai", _blocked_openai)
