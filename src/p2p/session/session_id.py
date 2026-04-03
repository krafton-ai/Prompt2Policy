"""Session ID generation with local timezone timestamp."""

from __future__ import annotations

import uuid
from datetime import datetime


def generate_session_id(suffix: str | None = None) -> str:
    """Generate a session ID with local timezone timestamp.

    Examples:
        generate_session_id()            -> "session_20260305_143052_a1b2c3d4"
        generate_session_id("cfg1_seed_42") -> "session_20260305_143052_cfg1_seed_42"
    """
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    tail = suffix or uuid.uuid4().hex[:8]
    return f"session_{ts}_{tail}"
