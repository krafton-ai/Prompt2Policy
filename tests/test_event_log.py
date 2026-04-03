"""Tests for event_log module — thread safety, span, format validation."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path

import pytest

from p2p.event_log import (
    EventLogger,
    emit,
    get_event_logger,
    read_event_by_seq,
    read_events,
    reset_current_iteration,
    reset_event_logger,
    set_current_iteration,
    set_event_logger,
    span,
)


@pytest.fixture()
def logger(tmp_path: Path) -> EventLogger:
    return EventLogger(tmp_path)


@pytest.fixture()
def _activate_logger(logger: EventLogger) -> Generator[None]:
    """Set module-level event logger and clean up both contextvars after."""
    token_logger = set_event_logger(logger)
    token_iter = set_current_iteration(None)
    try:
        yield
    finally:
        reset_current_iteration(token_iter)
        reset_event_logger(token_logger)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return parsed entries."""
    text = path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# EventLogger.emit — basic behavior
# ---------------------------------------------------------------------------


def test_emit_returns_monotonic_seq(logger: EventLogger) -> None:
    s1 = logger.emit("a")
    s2 = logger.emit("b")
    s3 = logger.emit("c")
    assert s1 < s2 < s3


def test_emit_writes_valid_jsonl(logger: EventLogger) -> None:
    logger.emit("test.event", iteration=1, data={"key": "value"}, duration_ms=42)

    entries = _read_jsonl(logger.path)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["seq"] == 1
    assert entry["event"] == "test.event"
    assert entry["iteration"] == 1
    assert entry["data"] == {"key": "value"}
    assert entry["duration_ms"] == 42
    assert "timestamp" in entry


def test_emit_defaults_data_to_empty_dict(logger: EventLogger) -> None:
    logger.emit("x")
    entry = _read_jsonl(logger.path)[0]
    assert entry["data"] == {}
    assert entry["iteration"] is None
    assert entry["duration_ms"] is None


# ---------------------------------------------------------------------------
# Thread safety — concurrent emit
# ---------------------------------------------------------------------------


def _run_concurrent(
    logger: EventLogger,
    n_threads: int,
    n_per_thread: int,
    worker_fn: Callable[[EventLogger, int, int], None],
) -> list[dict]:
    """Launch concurrent workers and return all parsed JSONL entries."""
    barrier = threading.Barrier(n_threads)

    def _target(tid: int) -> None:
        barrier.wait()
        worker_fn(logger, tid, n_per_thread)

    threads = [threading.Thread(target=_target, args=(tid,)) for tid in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return _read_jsonl(logger.path)


def test_concurrent_emit_seq_unique_and_gapless(logger: EventLogger) -> None:
    """Multiple threads emitting concurrently must produce unique, gapless seqs.

    Note: file-order may differ from seq-order because JSON serialization
    happens outside the lock.  This is acceptable for a local observability
    log — consumers can sort by seq or timestamp when ordering matters.
    """

    def worker(lg: EventLogger, _tid: int, n: int) -> None:
        for _ in range(n):
            lg.emit("concurrent")

    entries = _run_concurrent(logger, n_threads=10, n_per_thread=100, worker_fn=worker)
    assert len(entries) == 1000

    seqs = [e["seq"] for e in entries]
    assert sorted(seqs) == list(range(1, 1001))


def test_concurrent_emit_file_lines_are_complete_json(logger: EventLogger) -> None:
    """No partial/interleaved lines under concurrent writes."""

    def worker(lg: EventLogger, tid: int, n: int) -> None:
        for i in range(n):
            lg.emit("t", data={"tid": tid, "i": i})

    entries = _run_concurrent(logger, n_threads=8, n_per_thread=50, worker_fn=worker)
    for entry in entries:
        assert "seq" in entry


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------


def test_span_emits_start_and_end(logger: EventLogger) -> None:
    with logger.span("train", iteration=1, data={"lr": 0.01}):
        pass

    entries = _read_jsonl(logger.path)
    assert len(entries) == 2

    start, end = entries
    assert start["event"] == "train.start"
    assert start["iteration"] == 1
    assert start["data"] == {"lr": 0.01}

    assert end["event"] == "train.end"
    assert end["iteration"] == 1
    assert isinstance(end["duration_ms"], int)
    assert end["duration_ms"] >= 0


def test_span_records_duration(logger: EventLogger) -> None:
    with logger.span("slow"):
        time.sleep(0.05)

    end_entry = _read_jsonl(logger.path)[-1]
    assert end_entry["duration_ms"] >= 20  # generous margin for CI


def test_span_on_exception_emits_error_and_reraises(logger: EventLogger) -> None:
    with pytest.raises(ValueError, match="boom"):
        with logger.span("fail", iteration=2) as result:
            result["partial"] = "data"
            raise ValueError("boom")

    entries = _read_jsonl(logger.path)
    assert len(entries) == 2

    error = entries[1]
    assert error["event"] == "fail.error"
    assert error["iteration"] == 2
    assert error["data"]["error"] == "boom"
    assert error["data"]["partial"] == "data"
    assert isinstance(error["duration_ms"], int)


def test_span_yields_mutable_result_dict(logger: EventLogger) -> None:
    with logger.span("op") as result:
        result["output"] = 42

    end_entry = _read_jsonl(logger.path)[-1]
    assert end_entry["data"]["output"] == 42


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def test_emit_noop_when_no_logger() -> None:
    """emit() should silently do nothing when no logger is active."""
    token = set_event_logger(None)
    try:
        emit("should.not.crash")  # no error
    finally:
        reset_event_logger(token)


def test_span_noop_when_no_logger() -> None:
    """span() yields empty dict and does nothing when no logger is active."""
    token = set_event_logger(None)
    try:
        with span("noop") as result:
            result["x"] = 1
        assert result == {"x": 1}
    finally:
        reset_event_logger(token)


def test_get_event_logger_returns_none_by_default() -> None:
    token = set_event_logger(None)
    try:
        assert get_event_logger() is None
    finally:
        reset_event_logger(token)


def test_get_event_logger_returns_active_logger(logger: EventLogger) -> None:
    token = set_event_logger(logger)
    try:
        assert get_event_logger() is logger
    finally:
        reset_event_logger(token)


@pytest.mark.usefixtures("_activate_logger")
def test_module_emit_delegates_to_logger(logger: EventLogger) -> None:
    emit("hello", iteration=5, data={"a": 1})

    entry = _read_jsonl(logger.path)[0]
    assert entry["event"] == "hello"
    assert entry["iteration"] == 5


@pytest.mark.usefixtures("_activate_logger")
def test_module_span_delegates_to_logger(logger: EventLogger) -> None:
    with span("work", iteration=3) as result:
        result["done"] = True

    entries = _read_jsonl(logger.path)
    assert len(entries) == 2
    assert entries[0]["event"] == "work.start"
    assert entries[1]["event"] == "work.end"


@pytest.mark.usefixtures("_activate_logger")
def test_set_current_iteration_used_by_emit(logger: EventLogger) -> None:
    set_current_iteration(7)
    emit("auto.iter")

    entry = _read_jsonl(logger.path)[0]
    assert entry["iteration"] == 7


@pytest.mark.usefixtures("_activate_logger")
def test_explicit_iteration_overrides_current(logger: EventLogger) -> None:
    set_current_iteration(7)
    emit("override", iteration=99)

    entry = _read_jsonl(logger.path)[0]
    assert entry["iteration"] == 99


# ---------------------------------------------------------------------------
# read_events / read_event_by_seq
# ---------------------------------------------------------------------------


def test_read_events_empty_dir(tmp_path: Path) -> None:
    assert read_events(tmp_path) == []


def test_read_events_returns_all_entries(logger: EventLogger) -> None:
    logger.emit("a")
    logger.emit("b")

    events = read_events(logger.path.parent)
    assert len(events) == 2
    assert events[0]["event"] == "a"
    assert events[1]["event"] == "b"


def test_read_events_truncates_long_fields(logger: EventLogger) -> None:
    long_text = "x" * 500
    logger.emit("llm", data={"system_prompt": long_text, "short": "ok"})

    events = read_events(logger.path.parent, truncate=True)
    assert len(events) == 1
    assert events[0]["data"]["system_prompt"].endswith("...")
    assert len(events[0]["data"]["system_prompt"]) < len(long_text)
    assert events[0]["data"]["short"] == "ok"
    assert events[0]["has_full_content"] is True


def test_read_events_no_truncate(logger: EventLogger) -> None:
    long_text = "x" * 500
    logger.emit("llm", data={"system_prompt": long_text})

    events = read_events(logger.path.parent, truncate=False)
    assert events[0]["data"]["system_prompt"] == long_text
    assert events[0]["has_full_content"] is False


def test_read_events_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"seq":1,"event":"ok","data":{}}\nNOT_JSON\n{"seq":2,"event":"ok2","data":{}}\n'
    )

    events = read_events(tmp_path, truncate=False)
    assert len(events) == 2


def test_read_event_by_seq_found(logger: EventLogger) -> None:
    logger.emit("first")
    logger.emit("second")

    event = read_event_by_seq(logger.path.parent, 2)
    assert event is not None
    assert event["event"] == "second"
    assert event["has_full_content"] is False


def test_read_event_by_seq_not_found(logger: EventLogger) -> None:
    logger.emit("only")
    assert read_event_by_seq(logger.path.parent, 999) is None


def test_read_event_by_seq_missing_file(tmp_path: Path) -> None:
    assert read_event_by_seq(tmp_path, 1) is None


# ---------------------------------------------------------------------------
# events.jsonl format validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"seq", "timestamp", "event", "iteration", "data", "duration_ms"}


def test_jsonl_entry_has_all_required_keys(logger: EventLogger) -> None:
    logger.emit("check", iteration=1, data={"k": "v"}, duration_ms=10)

    entry = _read_jsonl(logger.path)[0]
    assert set(entry.keys()) == _REQUIRED_KEYS


def test_timestamp_is_utc_iso(logger: EventLogger) -> None:
    logger.emit("ts")
    entry = _read_jsonl(logger.path)[0]
    ts = datetime.fromisoformat(entry["timestamp"])
    assert ts.tzinfo is not None  # timezone-aware
