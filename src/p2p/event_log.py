"""Event-based logging for session observability.

Appends structured JSONL events to ``events.jsonl`` inside each session
directory.  Uses contextvars so downstream code (LLM calls, VLM calls)
can emit events without explicit parameter passing.

Usage in loop.py::

    from p2p.event_log import EventLogger, set_event_logger, reset_event_logger, emit, span

    events = EventLogger(session.path)
    token = set_event_logger(events)
    try:
        emit("session.started", data={"prompt": prompt})
        with span("train", iteration=1):
            run_training(...)
    finally:
        reset_event_logger(token)

Any module can call ``emit()`` / ``span()`` — they are no-ops when no
logger is active.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p2p.contracts import EventDetailRecord

_current_logger: contextvars.ContextVar[EventLogger | None] = contextvars.ContextVar(
    "event_logger", default=None
)
_current_iteration: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_iteration", default=None
)


def get_event_logger() -> EventLogger | None:
    return _current_logger.get()


def set_event_logger(logger: EventLogger | None) -> contextvars.Token:
    return _current_logger.set(logger)


def reset_event_logger(token: contextvars.Token) -> None:
    _current_logger.reset(token)


def set_current_iteration(iteration: int | None) -> contextvars.Token:
    return _current_iteration.set(iteration)


def reset_current_iteration(token: contextvars.Token) -> None:
    _current_iteration.reset(token)


# ---------------------------------------------------------------------------
# Convenience functions (no-op when no logger active)
# ---------------------------------------------------------------------------


def emit(
    event: str,
    *,
    iteration: int | None = None,
    data: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> None:
    logger = _current_logger.get()
    if logger:
        if iteration is None:
            iteration = _current_iteration.get()
        logger.emit(event, iteration=iteration, data=data, duration_ms=duration_ms)


@contextmanager
def span(
    event: str,
    *,
    iteration: int | None = None,
    data: dict[str, Any] | None = None,
):
    logger = _current_logger.get()
    if not logger:
        yield {}
        return

    if iteration is None:
        iteration = _current_iteration.get()

    with logger.span(event, iteration=iteration, data=data) as result:
        yield result


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only JSONL event log for a session."""

    def __init__(self, session_dir: Path) -> None:
        self.path = session_dir / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        iteration: int | None = None,
        data: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq

        entry = {
            "seq": seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "iteration": iteration,
            "data": data or {},
            "duration_ms": duration_ms,
        }
        line = json.dumps(entry, default=str) + "\n"

        with self._lock:
            with self.path.open("a") as f:
                f.write(line)

        return seq

    @contextmanager
    def span(
        self,
        event: str,
        *,
        iteration: int | None = None,
        data: dict[str, Any] | None = None,
    ):
        self.emit(f"{event}.start", iteration=iteration, data=data)
        start = time.monotonic()
        result: dict[str, Any] = {}
        try:
            yield result
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.emit(
                f"{event}.error",
                iteration=iteration,
                data={"error": str(exc), **result},
                duration_ms=duration_ms,
            )
            raise
        else:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.emit(
                f"{event}.end",
                iteration=iteration,
                data=result,
                duration_ms=duration_ms,
            )


# ---------------------------------------------------------------------------
# Reader (for API layer)
# ---------------------------------------------------------------------------

_TRUNCATE_KEYS = ("system_prompt", "user_prompt", "response", "full_prompt", "full_response")
_TRUNCATE_LEN = 200


def read_events(session_dir: Path, *, truncate: bool = True) -> list[dict]:
    """Read events.jsonl, optionally truncating large text fields."""
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return []

    events = []
    for line in events_path.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if truncate:
            data = entry.get("data", {})
            has_full = False
            for key in _TRUNCATE_KEYS:
                if key in data and isinstance(data[key], str) and len(data[key]) > _TRUNCATE_LEN:
                    has_full = True
                    data[key] = data[key][:_TRUNCATE_LEN] + "..."
            entry["has_full_content"] = has_full
        else:
            entry["has_full_content"] = False

        events.append(entry)

    return events


def read_event_by_seq(session_dir: Path, seq: int) -> EventDetailRecord | None:
    """Read a single event by sequence number (full content)."""
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return None

    for line in events_path.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("seq") == seq:
            entry["has_full_content"] = False
            return entry

    return None
