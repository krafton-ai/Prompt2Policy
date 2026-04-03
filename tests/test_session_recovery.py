"""Tests for session recovery after backend restart."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from p2p.api.process_manager import recover_stale_sessions
from p2p.utils.process_safety import is_pid_alive


@pytest.fixture()
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("p2p.api.process_manager.RUNS_DIR", tmp_path)
    return tmp_path


def _make_session(runs_dir: Path, session_id: str, status: str, pid: int | None = None) -> Path:
    """Create a minimal session directory with status.json and optional pid file."""
    session_dir = runs_dir / session_id
    session_dir.mkdir()
    status_data = {"status": status, "updated_at": "2025-01-01T00:00:00Z"}
    (session_dir / "status.json").write_text(json.dumps(status_data))
    if pid is not None:
        (session_dir / "pid").write_text(str(pid))
    return session_dir


class TestIsPidAlive:
    def test_own_process_is_alive(self) -> None:
        assert is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self) -> None:
        assert is_pid_alive(99999999) is False


class TestRecoverStaleSessions:
    def test_no_runs_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setattr("p2p.api.process_manager.RUNS_DIR", nonexistent)
        assert recover_stale_sessions() == {}

    def test_completed_session_ignored(self, runs_dir: Path) -> None:
        _make_session(runs_dir, "session_abc", "completed", pid=12345)
        actions = recover_stale_sessions()
        assert actions == {}

    def test_running_session_no_pid_marked_error(self, runs_dir: Path) -> None:
        _make_session(runs_dir, "session_nopid", "running")
        actions = recover_stale_sessions()
        assert actions["session_nopid"] == "marked_error_no_pid"
        status = json.loads((runs_dir / "session_nopid" / "status.json").read_text())
        assert status["status"] == "error"

    def test_running_session_dead_pid_marked_error(self, runs_dir: Path) -> None:
        _make_session(runs_dir, "session_dead", "running", pid=99999999)
        actions = recover_stale_sessions()
        assert actions["session_dead"] == "marked_error_dead"
        status = json.loads((runs_dir / "session_dead" / "status.json").read_text())
        assert status["status"] == "error"

    def test_running_session_alive_pid_reattached(self, runs_dir: Path) -> None:
        # Use our own PID (guaranteed alive)
        _make_session(runs_dir, "session_alive", "running", pid=os.getpid())
        with (
            patch("p2p.api.process_manager.verify_pid_ownership", return_value=True),
            patch("p2p.api.process_manager.threading") as mock_threading,
        ):
            actions = recover_stale_sessions()
        assert actions["session_alive"] == "reattached"
        # Verify a watchdog thread was started
        mock_threading.Thread.assert_called_once()

    def test_alive_pid_but_not_p2p_marked_error(self, runs_dir: Path) -> None:
        # PID is alive but belongs to a different process (PID reuse)
        _make_session(runs_dir, "session_reused", "running", pid=os.getpid())
        with patch("p2p.api.process_manager.verify_pid_ownership", return_value=False):
            actions = recover_stale_sessions()
        assert actions["session_reused"] == "marked_error_dead"
        status = json.loads((runs_dir / "session_reused" / "status.json").read_text())
        assert status["status"] == "error"

    def test_running_session_with_subprocess_log(self, runs_dir: Path) -> None:
        session_dir = _make_session(runs_dir, "session_log", "running", pid=99999999)
        (session_dir / "subprocess.log").write_text("line1\nline2\nTraceback: some error\n")
        actions = recover_stale_sessions()
        assert actions["session_log"] == "marked_error_dead"
        status = json.loads((session_dir / "status.json").read_text())
        assert "Traceback: some error" in status.get("error", "")

    def test_non_session_dirs_ignored(self, runs_dir: Path) -> None:
        # Non-session directories should be skipped
        (runs_dir / "benchmark_123").mkdir()
        (runs_dir / "run_456").mkdir()
        actions = recover_stale_sessions()
        assert actions == {}

    def test_corrupt_pid_file(self, runs_dir: Path) -> None:
        session_dir = _make_session(runs_dir, "session_corrupt", "running")
        (session_dir / "pid").write_text("not_a_number")
        actions = recover_stale_sessions()
        assert actions["session_corrupt"] == "marked_error_bad_pid"
