"""Centralized environment configuration.

All environment variables are read here after a single ``load_dotenv()`` call.
Other modules should ``from p2p.settings import …`` instead of calling
``os.environ.get()`` directly.

**Import order matters**: this module MUST be imported before gymnasium / mujoco
so that ``MUJOCO_GL`` is set in the environment before those libraries read it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # idempotent — safe to call multiple times

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM model names
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gemini-3.1-pro-preview")
LLM_MODEL_LIGHT: str = os.environ.get("LLM_MODEL_LIGHT", "gemini-3-flash-preview")

# Extended thinking effort level ("max", "xhigh", "high", "medium", "low", or "" to disable).
# create_message() auto-injects thinking={type: adaptive} + output_config={effort: <level>}.
# "max" is Opus 4.6 only; "xhigh" is OpenAI GPT-5.x only.
THINKING_EFFORT: str = os.environ.get("THINKING_EFFORT", "max")

# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Ollama / VLM dispatch
# ---------------------------------------------------------------------------
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL: str = os.environ.get("VLM_MODEL", "vllm-Qwen/Qwen3.5-27B")
# VLM_BASE_URL defaults to OLLAMA_URL for backwards compat
VLM_BASE_URL: str = os.environ.get("VLM_BASE_URL", OLLAMA_URL)

# Refined initial frame: pad video + send first frame JPEG in Turn 1 for Gemini.
# Fixes center-of-interval sampling missing frame 0 and improves calibration.
VLM_REFINED_INITIAL_FRAME: bool = os.environ.get("VLM_REFINED_INITIAL_FRAME", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Criteria diagnosis: Turn 2 evaluates each visual criterion individually
# before giving a holistic score (Method D style).
VLM_CRITERIA_DIAGNOSIS: bool = os.environ.get("VLM_CRITERIA_DIAGNOSIS", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Motion trail dual: send two videos in Turn 2 — a standard recording and a
# motion-trail version — so the VLM can compare both (Method F style).
VLM_MOTION_TRAIL_DUAL: bool = os.environ.get("VLM_MOTION_TRAIL_DUAL", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# vLLM server
# ---------------------------------------------------------------------------
VLLM_HOST: str = os.environ.get("VLLM_HOST", "0.0.0.0")
VLLM_PORT: int = int(os.environ.get("VLLM_PORT", "8100"))
VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B")

# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------
RUNS_DIR: Path = Path("runs")

_BM_CASE_RE = re.compile(r"^(bm_(?:\d{8}_\d{6}_)?[a-f0-9]+)_case(\d+)$")


def resolve_session_subpath(session_id: str) -> str:
    """Return the **relative** subpath for a session_id.

    Use this when composing paths against a custom ``runs_dir`` that may
    differ from the global ``RUNS_DIR``::

        target_runs_dir / resolve_session_subpath(session_id)

    Benchmark cases use a nested layout::

        bm_abc12345_case0          →  bm_abc12345/case0
        bm_00010101_000000_abc12345_case0  →  bm_00010101_000000_abc12345/case0

    Regular sessions return the session_id unchanged.

    Raises ``ValueError`` if the session_id contains path traversal
    components (``..``, ``/``, ``\\``).
    """
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise ValueError(f"Invalid session_id (path traversal): {session_id!r}")
    m = _BM_CASE_RE.match(session_id)
    if m:
        return f"{m.group(1)}/case{m.group(2)}"
    return session_id


def resolve_session_dir(session_id: str) -> Path:
    """Map a flat session_id to its absolute directory path under ``RUNS_DIR``.

    Convenience wrapper around ``resolve_session_subpath`` for the common
    case where the global ``RUNS_DIR`` is the base.  When working with a
    custom ``runs_dir`` (e.g. from the run orchestrator), use
    ``runs_dir / resolve_session_subpath(session_id)`` instead.
    """
    return RUNS_DIR / resolve_session_subpath(session_id)


# ---------------------------------------------------------------------------
# Human labeling server
# ---------------------------------------------------------------------------
LABELING_SERVER_URL: str = os.environ.get("LABELING_SERVER_URL", "")
LABELING_ANNOTATOR: str = os.environ.get("LABELING_ANNOTATOR", "")

# ---------------------------------------------------------------------------
# MuJoCo — MUJOCO_GL is read by mujoco at import time from os.environ.
# load_dotenv() above ensures .env values are available before that happens,
# as long as this module is imported before gymnasium/mujoco.
# ---------------------------------------------------------------------------
