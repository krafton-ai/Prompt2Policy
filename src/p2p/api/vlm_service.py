"""VLM proxy services — Ollama and vLLM health checks and chat forwarding."""

from __future__ import annotations

from typing import Any

import requests

from p2p.api.schemas import VlmChatRequest, VlmChatResponse, VlmStatusResponse
from p2p.settings import OLLAMA_URL


def vlm_chat(payload: VlmChatRequest) -> VlmChatResponse:
    """Forward a chat request to the local Ollama VLM server."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload.model_dump(exclude_none=True),
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return VlmChatResponse(
            model=data.get("model", payload.model),
            message=data.get("message", {}),
            done=data.get("done", True),
        )
    except requests.ConnectionError as e:
        msg = f"Ollama not reachable at {OLLAMA_URL}: {e}"
        raise ConnectionError(msg) from e


def vlm_status() -> VlmStatusResponse:
    """Check whether local VLM servers (Ollama and/or vLLM) are available."""
    from p2p.inference.vllm_server import vllm_health_check
    from p2p.settings import VLLM_HOST, VLLM_PORT

    providers: list[str] = []
    errors: list[str] = []

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names = [m.get("name", "") for m in models]
        if names:
            providers.append(f"Ollama: {', '.join(names)}")
    except Exception as e:
        errors.append(f"Ollama: {e}")

    # Check vLLM
    if vllm_health_check(VLLM_PORT):
        providers.append(f"vLLM: {VLLM_HOST}:{VLLM_PORT}")

    available = len(providers) > 0
    model_str = "; ".join(providers) if providers else ""
    error_str = "; ".join(errors) if errors and not available else None

    return VlmStatusResponse(available=available, model=model_str, error=error_str)


def vlm_tags() -> dict[str, Any]:
    """Proxy the Ollama /api/tags endpoint."""
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    resp.raise_for_status()
    return resp.json()
