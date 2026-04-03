"""Tests for status.json concurrent write protection (issue #218)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from p2p.session.iteration_record import (
    SessionRecord,
    status_lock,
    write_status,
)

_TERMINAL_STATUSES = (
    "completed",
    "error",
    "cancelled",
    "passed",
    "max_iterations",
    "rate_limited",
    "auth_error",
    "invalid_code",
    "failed",
)


@pytest.fixture()
def session(tmp_path: Path) -> SessionRecord:
    """Create a minimal session directory with 'running' status."""
    session_dir = tmp_path / "session_test"
    session_dir.mkdir()
    sr = SessionRecord(session_dir)
    sr.set_status("running")
    return sr


class TestStatusLock:
    def test_lock_serializes_writes(self, tmp_path: Path):
        """Two threads acquiring status_lock should not overlap."""
        order: list[str] = []

        def writer(name: str, delay: float) -> None:
            with status_lock(tmp_path):
                order.append(f"{name}_start")
                time.sleep(delay)
                order.append(f"{name}_end")

        t1 = threading.Thread(target=writer, args=("a", 0.05))
        t2 = threading.Thread(target=writer, args=("b", 0.01))
        t1.start()
        time.sleep(0.01)  # give t1 time to acquire
        t2.start()
        t1.join()
        t2.join()

        # a should complete before b starts
        assert order == ["a_start", "a_end", "b_start", "b_end"]

    def test_lock_file_created(self, tmp_path: Path):
        with status_lock(tmp_path):
            assert (tmp_path / "status.lock").exists()


class TestWriteStatus:
    def test_write_status_creates_file(self, tmp_path: Path):
        write_status(tmp_path, "running")
        data = json.loads((tmp_path / "status.json").read_text())
        assert data["status"] == "running"
        assert "updated_at" in data

    def test_write_status_with_error(self, tmp_path: Path):
        write_status(tmp_path, "error", error="something broke")
        data = json.loads((tmp_path / "status.json").read_text())
        assert data["status"] == "error"
        assert data["error"] == "something broke"


class TestSetStatusIf:
    def test_set_status_if_matching(self, session: SessionRecord):
        result = session.set_status_if("error", only_if=("running",), error="crash")
        assert result is True
        data = session.read_status()
        assert data is not None
        assert data["status"] == "error"
        assert data["error"] == "crash"

    def test_set_status_if_not_matching(self, session: SessionRecord):
        session.set_status("completed")
        result = session.set_status_if("error", only_if=("running",), error="crash")
        assert result is False
        data = session.read_status()
        assert data is not None
        assert data["status"] == "completed"

    def test_set_status_if_no_existing_status(self, tmp_path: Path):
        sr = SessionRecord(tmp_path / "session_empty")
        sr.ensure_dir()
        # No status.json exists — current_status is None, not in only_if
        result = sr.set_status_if("error", only_if=("running",))
        assert result is False

    def test_set_status_if_cancels_running(self, session: SessionRecord):
        result = session.set_status_if("cancelled", only_if=("running", "pending"))
        assert result is True
        assert session.read_status()["status"] == "cancelled"  # type: ignore[index]

    def test_set_status_if_does_not_overwrite_terminal(self, session: SessionRecord):
        """Terminal statuses should not be overwritten by non-terminal ones."""
        for terminal in _TERMINAL_STATUSES:
            session.set_status(terminal)
            result = session.set_status_if("running", only_if=("running",))
            assert result is False
            assert session.read_status()["status"] == terminal  # type: ignore[index]


class TestSetStatusIfConcurrent:
    def test_only_one_writer_wins(self, session: SessionRecord):
        """When two threads race to set_status_if, only one should succeed."""
        results: dict[str, bool] = {}
        barrier = threading.Barrier(2)

        def try_set(name: str, status: str, error: str) -> None:
            barrier.wait()
            results[name] = session.set_status_if(
                status,
                only_if=("running",),
                error=error,
            )

        t1 = threading.Thread(target=try_set, args=("watchdog", "error", "exit code 1"))
        t2 = threading.Thread(target=try_set, args=("stop", "cancelled", ""))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed
        assert sum(results.values()) == 1

        final = session.read_status()
        assert final is not None
        assert final["status"] in ("error", "cancelled")


class TestTouchHeartbeat:
    def test_touch_heartbeat_updates_timestamp(self, session: SessionRecord):
        original = session.read_status()
        assert original is not None
        original_ts = original["updated_at"]

        time.sleep(0.01)
        session.touch_heartbeat()

        updated = session.read_status()
        assert updated is not None
        assert updated["status"] == "running"
        assert updated["updated_at"] != original_ts

    def test_touch_heartbeat_no_status_file(self, tmp_path: Path):
        sr = SessionRecord(tmp_path / "session_no_status")
        sr.ensure_dir()
        # Should not raise
        sr.touch_heartbeat()

    def test_touch_heartbeat_does_not_lose_error_field(self, tmp_path: Path):
        sr = SessionRecord(tmp_path / "session_err")
        sr.ensure_dir()
        sr.set_status("error", error="something broke")

        sr.touch_heartbeat()

        data = sr.read_status()
        assert data is not None
        assert data["status"] == "error"
        assert data["error"] == "something broke"
