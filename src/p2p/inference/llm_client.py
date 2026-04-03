"""Multi-provider LLM client with retry-on-rate-limit and event logging.

Supports Anthropic (Claude), OpenAI (GPT/o-series), and Google Gemini.
Provider is auto-detected from the model name passed to ``create_message()``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx

from p2p.settings import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 128000
DEFAULT_TIMEOUT = httpx.Timeout(3600.0, connect=10.0)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic | None:
    """Return a lazily-initialized, reusable Anthropic client.

    Uses ``settings.ANTHROPIC_API_KEY`` when set.
    Returns ``None`` if the key is missing — non-Anthropic providers
    (OpenAI, Gemini) do not need this client.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        if not ANTHROPIC_API_KEY:
            return None
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


_MAX_RETRIES = 50
_INITIAL_WAIT = 30  # seconds
_MAX_WAIT = 300  # cap backoff at 5 minutes


class LLMRateLimitError(Exception):
    """Raised when all retries are exhausted due to rate limiting / overload."""


def _retry_with_backoff(
    call_fn: Callable[[], Any],
    is_retryable: Callable[[Exception], bool],
    exhaust_exc: BaseException,
    *,
    is_rate_limit: Callable[[Exception], bool] | None = None,
) -> Any:
    """Call *call_fn* with exponential backoff on retryable errors.

    Parameters
    ----------
    call_fn:
        Zero-arg callable that performs the API request.
    is_retryable:
        ``(exc) -> bool`` — return True if *exc* should be retried.
    exhaust_exc:
        Exception instance to raise when all retries fail.
    is_rate_limit:
        Optional ``(exc) -> bool``.  When provided and the last exception
        matches, ``LLMRateLimitError`` is raised instead of *exhaust_exc*
        so callers can distinguish rate-limit exhaustion from other errors.
    """
    wait = _INITIAL_WAIT
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return call_fn()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
        if attempt == _MAX_RETRIES:
            if is_rate_limit and last_exc and is_rate_limit(last_exc):
                raise LLMRateLimitError(str(last_exc)) from last_exc
            raise last_exc if last_exc else exhaust_exc
        logger.warning(
            "API error %s (attempt %d/%d), waiting %ds...",
            type(last_exc).__name__,
            attempt,
            _MAX_RETRIES,
            wait,
        )
        time.sleep(wait)
        wait = min(wait * 2, _MAX_WAIT)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Unified response wrapper
# ---------------------------------------------------------------------------


@dataclass
class _ContentBlock:
    """Provider-agnostic content block (text, thinking, or tool_use)."""

    type: str  # "text", "thinking", "tool_use"
    text: str = ""
    thinking: str = ""
    id: str = ""  # tool_use
    name: str = ""  # tool_use
    input: dict | None = None  # tool_use only
    signature: str = ""  # Anthropic: must echo back on thinking blocks in multi-turn
    thought_signature: str = ""  # Gemini 3.x: must echo back on function_call parts


@dataclass
class _Usage:
    """Token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # OpenAI: hidden reasoning token count


@dataclass
class _LLMResponse:
    """Provider-agnostic LLM response.

    Duck-type compatible with ``anthropic.types.Message``: callers access
    ``response.content``, ``response.stop_reason``, and
    ``response.usage.input_tokens / output_tokens``.
    """

    content: list[_ContentBlock]
    stop_reason: str  # "end_turn" or "tool_use"
    usage: _Usage


# ---------------------------------------------------------------------------
# Message normalization (handles _ContentBlock in tool-loop messages)
# ---------------------------------------------------------------------------


def _safe_json_loads(s: str | None) -> dict:
    """Parse JSON string to dict, returning empty dict on failure."""
    if not s:
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _normalize_content(content: Any) -> Any:
    """Convert ``_ContentBlock`` objects in message content to plain dicts.

    When the tool-use loop appends ``response.content`` (a list of
    ``_ContentBlock``) directly into messages, subsequent API calls need
    these converted back to dicts.
    """
    if not isinstance(content, list):
        return content
    result = []
    for item in content:
        if isinstance(item, _ContentBlock):
            if item.type == "text":
                result.append({"type": "text", "text": item.text})
            elif item.type == "thinking":
                d = {"type": "thinking", "thinking": item.thinking}
                if item.signature:
                    d["signature"] = item.signature
                result.append(d)
            elif item.type == "tool_use":
                d = {
                    "type": "tool_use",
                    "id": item.id,
                    "name": item.name,
                    "input": item.input or {},
                }
                if item.thought_signature:
                    d["thought_signature"] = item.thought_signature
                result.append(d)
        else:
            result.append(item)
    return result


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages by converting any ``_ContentBlock`` objects to dicts."""
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            normalized = _normalize_content(content)
            result.append({**msg, "content": normalized})
        else:
            result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Tool schema translators
# ---------------------------------------------------------------------------


def _translate_tools_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schemas to OpenAI Responses API format.

    Responses API uses flat internal tagging::

        {"type": "function", "name": "foo", "parameters": {...}}
    """
    result = []
    for t in tools:
        result.append(
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
                "strict": False,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _extract_user_text(messages: list[dict]) -> str:
    """Extract text from the last user message (handles multimodal content)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = (
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            return " ".join(parts)
    return ""


def _extract_tool_results(messages: list[dict]) -> list[dict]:
    """Extract tool_result blocks from the last user message (tool loop context)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            results = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.append(
                        {
                            "tool_use_id": b.get("tool_use_id", ""),
                            "content": b.get("content", ""),
                        }
                    )
            return results
    return []


def extract_response_text(response: object) -> str:
    """Safely extract text from all content blocks of an LLM response.

    Skips ``thinking`` blocks -- only returns the final answer text.
    """
    parts = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "thinking":
            continue
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def extract_thinking_text(response: object) -> str:
    """Extract thinking text from an extended-thinking response."""
    parts = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "thinking":
            parts.append(getattr(block, "thinking", ""))
    return "\n".join(parts)


def _serialize_content_blocks(response: object) -> dict:
    """Serialize all response content blocks into structured data."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "thinking":
            thinking_parts.append(getattr(block, "thinking", ""))
        elif getattr(block, "type", None) == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input or {},
                }
            )
        elif hasattr(block, "text") and isinstance(block.text, str):
            text_parts.append(block.text)
    result: dict[str, Any] = {
        "response": "\n".join(text_parts),
        "tool_calls": tool_calls,
    }
    if thinking_parts:
        result["thinking"] = "\n".join(thinking_parts)
    return result


_VALID_EFFORTS = {"max", "xhigh", "high", "medium", "low", "minimal"}

# OpenAI supports: "low", "medium", "high", "xhigh" (GPT-5.4 and GPT-5.3-Codex)
_OPENAI_EFFORTS = {"xhigh", "high", "medium", "low"}

# Gemini 3.x thinking_level enum values (LOW, MEDIUM, HIGH, MINIMAL)
_GEMINI_LEVEL_MAP = {
    "max": "HIGH",
    "xhigh": "HIGH",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "minimal": "MINIMAL",
}


# ---------------------------------------------------------------------------
# Provider: Anthropic
# ---------------------------------------------------------------------------


def _call_anthropic(client: anthropic.Anthropic, **kwargs: Any) -> _LLMResponse:
    """Call Anthropic API with retry logic, return unified response."""
    # Normalize messages to convert _ContentBlock objects to dicts
    if "messages" in kwargs:
        kwargs["messages"] = _normalize_messages(kwargs["messages"])

    kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError)):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529:
            return True
        return False

    def _is_rate_limit(exc: Exception) -> bool:
        return isinstance(exc, anthropic.RateLimitError) or (
            isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529
        )

    response = _retry_with_backoff(
        lambda: client.messages.create(**kwargs),
        is_retryable=_is_retryable,
        exhaust_exc=RuntimeError("Anthropic retry exhausted"),
        is_rate_limit=_is_rate_limit,
    )

    # Wrap native response into _LLMResponse
    content: list[_ContentBlock] = []
    for block in response.content:
        if getattr(block, "type", None) == "thinking":
            content.append(
                _ContentBlock(
                    type="thinking",
                    thinking=getattr(block, "thinking", ""),
                    signature=getattr(block, "signature", ""),
                )
            )
        elif getattr(block, "type", None) == "text":
            content.append(_ContentBlock(type="text", text=block.text))
        elif getattr(block, "type", None) == "tool_use":
            content.append(
                _ContentBlock(type="tool_use", id=block.id, name=block.name, input=block.input)
            )

    return _LLMResponse(
        content=content,
        stop_reason=response.stop_reason,
        usage=_Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Provider: OpenAI
# ---------------------------------------------------------------------------

_openai_client: Any = None


def _call_openai(**kwargs: Any) -> _LLMResponse:
    """Call OpenAI Responses API, return unified response.

    Uses the Responses API (not Chat Completions) to access reasoning
    summaries and better tool-use support for GPT-5 reasoning models.
    """
    import openai

    from p2p.settings import OPENAI_API_KEY

    global _openai_client  # noqa: PLW0603
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or set the environment variable."
        )
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    model = kwargs.get("model", "")
    system = kwargs.get("system", "")
    messages_in = kwargs.get("messages", [])
    tools_in = kwargs.get("tools")
    effort = kwargs.get("_thinking_effort", "")

    # Build Responses API input items
    input_items: list[dict[str, Any]] = []

    for msg in _normalize_messages(messages_in):
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, list):
                # Check if these are tool_result blocks
                tool_results = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tr["tool_use_id"],
                                "output": tr.get("content", ""),
                            }
                        )
                else:
                    # Regular content blocks — extract text
                    text_parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                    input_items.append(
                        {
                            "role": "user",
                            "content": "\n".join(text_parts) if text_parts else str(content),
                        }
                    )
            else:
                input_items.append({"role": "user", "content": content})

        elif role == "assistant":
            if isinstance(content, list):
                # Add text as assistant message
                text_parts = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            # Function calls are separate items in Responses API
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": b["id"],
                                    "name": b["name"],
                                    "arguments": json.dumps(b.get("input") or {}),
                                }
                            )
                if text_parts:
                    input_items.append({"role": "assistant", "content": "\n".join(text_parts)})
            else:
                input_items.append({"role": "assistant", "content": content})

    # Build API kwargs
    api_kwargs: dict[str, Any] = {"model": model, "input": input_items}
    if system:
        api_kwargs["instructions"] = system
    if tools_in:
        api_kwargs["tools"] = _translate_tools_openai(tools_in)

    # Reasoning effort + summary
    # "max" is Anthropic-only; map to OpenAI's highest tier ("xhigh")
    # so global THINKING_EFFORT=max works across providers.
    if effort:
        oai_effort = "xhigh" if effort == "max" else effort
        if oai_effort in _OPENAI_EFFORTS:
            api_kwargs["reasoning"] = {"effort": oai_effort, "summary": "detailed"}
        else:
            logger.warning("Unsupported effort for OpenAI: %s, skipping", oai_effort)

    def _call() -> Any:
        global _openai_client  # noqa: PLW0603
        return _openai_client.responses.create(**api_kwargs)

    def _is_retryable(exc: Exception) -> bool:
        global _openai_client  # noqa: PLW0603
        if isinstance(exc, (openai.RateLimitError, openai.InternalServerError)):
            return True
        if isinstance(exc, RuntimeError) and "client has been closed" in str(exc):
            logger.warning("OpenAI client was closed, recreating...")
            _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            return True
        return False

    response = _retry_with_backoff(
        _call,
        is_retryable=_is_retryable,
        exhaust_exc=RuntimeError("OpenAI retry exhausted"),
        is_rate_limit=lambda exc: isinstance(exc, openai.RateLimitError),
    )

    # Convert Responses API output to _LLMResponse
    content_blocks: list[_ContentBlock] = []
    has_function_call = False
    reasoning_tokens = 0

    for item in response.output:
        item_type = getattr(item, "type", "")

        if item_type == "reasoning":
            # Reasoning item — extract summary text
            summaries = getattr(item, "summary", []) or []
            thinking_text = "\n".join(s.text for s in summaries if hasattr(s, "text"))
            if thinking_text:
                content_blocks.append(_ContentBlock(type="thinking", thinking=thinking_text))

        elif item_type == "message":
            # Output message — extract text content
            for part in getattr(item, "content", []):
                if getattr(part, "type", "") == "output_text":
                    content_blocks.append(_ContentBlock(type="text", text=part.text))

        elif item_type == "function_call":
            # Function/tool call
            has_function_call = True
            content_blocks.append(
                _ContentBlock(
                    type="tool_use",
                    id=getattr(item, "call_id", ""),
                    name=getattr(item, "name", ""),
                    input=_safe_json_loads(getattr(item, "arguments", "")),
                )
            )

    # Map stop reason
    stop = "tool_use" if has_function_call else "end_turn"

    # Extract usage
    usage = getattr(response, "usage", None)
    if usage:
        reasoning_tokens = (
            getattr(
                getattr(usage, "output_tokens_details", None),
                "reasoning_tokens",
                0,
            )
            or 0
        )

    return _LLMResponse(
        content=content_blocks,
        stop_reason=stop,
        usage=_Usage(
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            reasoning_tokens=reasoning_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Provider: Google Gemini
# ---------------------------------------------------------------------------

_gemini_client: Any = None


def _build_tool_name_map(messages: list[dict]) -> dict[str, str]:
    """Build a mapping from tool_use_id to tool name from assistant messages.

    Gemini's ``Part.from_function_response`` requires the function name, but
    ``dispatch_tool_calls()`` only stores ``tool_use_id`` in tool_result blocks.
    We recover the name by scanning preceding assistant messages.
    """
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                id_to_name[b["id"]] = b["name"]
    return id_to_name


_GEMINI_UNSUPPORTED_SCHEMA_KEYS = {"oneOf", "anyOf", "allOf", "$ref"}


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """Strip JSON Schema keywords unsupported by Gemini FunctionDeclaration.

    For ``oneOf``/``anyOf``, picks the first variant.  Recurses into nested
    properties and items.
    """
    out: dict[str, Any] = {}
    for key, val in schema.items():
        if key in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            # For oneOf/anyOf, use the first alternative
            if key in ("oneOf", "anyOf") and isinstance(val, list) and val:
                out.update(_sanitize_schema_for_gemini(val[0]))
            continue
        if key == "properties" and isinstance(val, dict):
            out[key] = {k: _sanitize_schema_for_gemini(v) for k, v in val.items()}
        elif key == "items" and isinstance(val, dict):
            out[key] = _sanitize_schema_for_gemini(val)
        else:
            out[key] = val
    return out


def _call_gemini(**kwargs: Any) -> _LLMResponse:
    """Call Google Gemini API, return unified response."""
    from google import genai
    from google.genai import types

    from p2p.settings import GEMINI_API_KEY

    global _gemini_client  # noqa: PLW0603
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file or set the environment variable."
        )
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    model = kwargs.get("model", "")
    system = kwargs.get("system", "")
    messages_in = kwargs.get("messages", [])
    tools_in = kwargs.get("tools")
    effort = kwargs.get("_thinking_effort", "")

    normalized_msgs = _normalize_messages(messages_in)
    tool_name_map = _build_tool_name_map(normalized_msgs)

    # Build Gemini contents
    gemini_contents: list[Any] = []
    for msg in normalized_msgs:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, list):
                tool_results = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    parts = []
                    for tr in tool_results:
                        # Recover tool name from the preceding assistant message
                        tool_use_id = tr.get("tool_use_id", "")
                        func_name = tool_name_map.get(tool_use_id, "tool")
                        # Parse content back to dict
                        result_content = tr.get("content", "")
                        try:
                            result_data = (
                                json.loads(result_content)
                                if isinstance(result_content, str)
                                else result_content
                            )
                        except (json.JSONDecodeError, TypeError):
                            result_data = {"result": result_content}
                        if not isinstance(result_data, dict):
                            result_data = {"result": str(result_data)}
                        parts.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response=result_data,
                            )
                        )
                    gemini_contents.append(types.Content(role="user", parts=parts))
                else:
                    text_parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                    gemini_contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="\n".join(text_parts))],
                        )
                    )
            else:
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        elif role == "assistant":
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text" and b.get("text"):
                            parts.append(types.Part.from_text(text=b["text"]))
                        elif b.get("type") == "tool_use":
                            fc_part = types.Part.from_function_call(
                                name=b["name"],
                                args=b.get("input", {}),
                            )
                            # Gemini 3.x: echo thought_signature on function_call parts
                            sig = b.get("thought_signature", "")
                            if sig:
                                fc_part.thought_signature = sig
                            parts.append(fc_part)
                if parts:
                    gemini_contents.append(types.Content(role="model", parts=parts))
            else:
                gemini_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

    # Build tools
    gemini_tools = None
    if tools_in:
        declarations = []
        for t in tools_in:
            schema = t.get("input_schema", {})
            declarations.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=_sanitize_schema_for_gemini(schema) if schema else None,
                )
            )
        gemini_tools = [types.Tool(function_declarations=declarations)]

    # Thinking config — Gemini 3.x uses thinking_level enum
    # include_thoughts=True is required to get thinking text in the response
    # Gemini 3.1 Pro does not support MINIMAL — fall back to LOW
    thinking_config = None
    if effort and effort in _GEMINI_LEVEL_MAP:
        level = effort
        if level == "minimal" and "pro" in model.lower():
            logger.warning("Gemini Pro does not support MINIMAL thinking — falling back to LOW")
            level = "low"
        thinking_config = types.ThinkingConfig(
            thinking_level=_GEMINI_LEVEL_MAP[level],
            include_thoughts=True,
        )

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        tools=gemini_tools,
        thinking_config=thinking_config,
    )

    _GEMINI_RETRYABLE_TAGS = ("ResourceExhausted", "ServiceUnavailable", "429", "503")

    def _is_gemini_transient(exc: Exception) -> bool:
        exc_name = type(exc).__name__
        exc_str = str(exc)
        return any(tag in exc_name or tag in exc_str for tag in _GEMINI_RETRYABLE_TAGS)

    def _call() -> Any:
        global _gemini_client  # noqa: PLW0603
        return _gemini_client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=config,
        )

    def _is_retryable(exc: Exception) -> bool:
        global _gemini_client  # noqa: PLW0603
        if isinstance(exc, RuntimeError) and "client has been closed" in str(exc):
            logger.warning("Gemini client was closed, recreating...")
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            return True
        return _is_gemini_transient(exc)

    def _is_rate_limit(exc: Exception) -> bool:
        return _is_gemini_transient(exc)

    response = _retry_with_backoff(
        _call,
        is_retryable=_is_retryable,
        exhaust_exc=RuntimeError("Gemini retry exhausted"),
        is_rate_limit=_is_rate_limit,
    )

    # Convert response
    content_blocks: list[_ContentBlock] = []
    has_function_call = False

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "thought", False):
                    content_blocks.append(_ContentBlock(type="thinking", thinking=part.text or ""))
                elif hasattr(part, "function_call") and part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    content_blocks.append(
                        _ContentBlock(
                            type="tool_use",
                            id=f"gemini_{fc.name}_{id(fc)}",
                            name=fc.name,
                            input=dict(fc.args) if fc.args else {},
                            thought_signature=getattr(part, "thought_signature", "") or "",
                        )
                    )
                elif hasattr(part, "text") and part.text:
                    content_blocks.append(_ContentBlock(type="text", text=part.text))

    stop = "tool_use" if has_function_call else "end_turn"

    usage_meta = getattr(response, "usage_metadata", None)
    return _LLMResponse(
        content=content_blocks,
        stop_reason=stop,
        usage=_Usage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            reasoning_tokens=getattr(usage_meta, "thoughts_token_count", 0) or 0,
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_message(client: anthropic.Anthropic | None, **kwargs: Any) -> _LLMResponse:
    """Call an LLM with exponential backoff on transient errors.

    The provider is auto-detected from the ``model`` kwarg:

    - ``gpt-*``, ``o1*``, ``o3*``, ``o4*`` -- OpenAI
    - ``gemini*`` -- Google Gemini
    - Everything else -- Anthropic (default)

    All keyword arguments follow the Anthropic convention (``model``,
    ``system``, ``messages``, ``tools``, etc.). Non-Anthropic providers
    translate these internally.

    Adaptive thinking
    -----------------
    Set ``THINKING_EFFORT`` env-var (default ``"max"``) to auto-inject
    extended thinking. For Anthropic this means ``thinking={type: adaptive}``
    + ``output_config={effort: <level>}``. For OpenAI it maps to
    ``reasoning={effort: <level>}``. For Gemini it sets a thinking budget.

    Returns a ``_LLMResponse`` that is duck-type compatible with the
    previous ``anthropic.types.Message``: callers access
    ``response.content``, ``response.stop_reason``, and
    ``response.usage.input_tokens / output_tokens``.
    """
    from p2p.event_log import emit as _emit
    from p2p.settings import THINKING_EFFORT

    # Pop custom kwargs before forwarding to any provider
    agent_name = kwargs.pop("agent_name", None)
    effort = kwargs.pop("thinking_effort", None) or THINKING_EFFORT
    model = kwargs.get("model", "")

    start = time.monotonic()

    # Route to the appropriate provider
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        kwargs["_thinking_effort"] = effort
        response = _call_openai(**kwargs)

    elif model.startswith("gemini"):
        kwargs["_thinking_effort"] = effort
        response = _call_gemini(**kwargs)

    else:
        # Anthropic (default)
        if client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                f"Cannot use model {model!r}. "
                "Add it to your .env file or set the environment variable."
            )
        if effort and effort in _VALID_EFFORTS and "thinking" not in kwargs:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs.setdefault("output_config", {"effort": effort})
        if "thinking" in kwargs:
            # API constraint: temperature must not be set with thinking
            kwargs.pop("temperature", None)
        response = _call_anthropic(client, **kwargs)

    # Auto-log the LLM call
    duration_ms = int((time.monotonic() - start) * 1000)
    content_data = _serialize_content_blocks(response)
    messages = kwargs.get("messages", [])
    tool_results = _extract_tool_results(messages)

    log_data: dict[str, Any] = {
        "model": kwargs.get("model", "unknown"),
        "system_prompt": kwargs.get("system", ""),
        "user_prompt": _extract_user_text(messages),
        "response": content_data["response"],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
    }
    if effort:
        log_data["thinking_effort"] = effort
    if response.usage.reasoning_tokens:
        log_data["reasoning_tokens"] = response.usage.reasoning_tokens
    if content_data.get("thinking"):
        log_data["thinking"] = content_data["thinking"]
    if content_data["tool_calls"]:
        log_data["tool_calls"] = content_data["tool_calls"]
    if tool_results:
        log_data["tool_results_input"] = tool_results
    if agent_name:
        log_data["agent"] = agent_name

    _emit("llm.call", data=log_data, duration_ms=duration_ms)

    return response
