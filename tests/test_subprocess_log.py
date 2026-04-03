"""Tests for subprocess log capture and crash detection in services.start_session."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from p2p.api.process_manager import start_session
from p2p.config import LoopConfig, TrainConfig
from p2p.session.iteration_record import SessionRecord


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _poll_status(session: SessionRecord, target: str, timeout: float = 2.0) -> dict:
    """Poll session status until it matches *target* or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = session.read_status()
        except Exception:  # noqa: BLE001
            data = None
        if data and data.get("status") == target:
            return data
        time.sleep(0.05)
    # Return latest status even if it didn't match
    try:
        return session.read_status() or {}
    except Exception:  # noqa: BLE001
        return {}


class _FakeProc:
    """Simulates a subprocess.Popen that exits immediately with a given code."""

    def __init__(self, returncode: int):
        self.returncode: int | None = None
        self._final_code = returncode
        self.pid = 12345

    def wait(self) -> int:
        self.returncode = self._final_code
        return self._final_code


def _make_fake_popen(returncode: int):
    """Return a Popen side_effect that writes to the log then returns a FakeProc."""

    def _side_effect(cmd, *, env=None, stdout=None, stderr=None, start_new_session=False):
        # Simulate subprocess writing to the log file
        if stdout is not None and hasattr(stdout, "write"):
            stdout.write("Traceback (most recent call last):\n")
            stdout.write("ModuleNotFoundError: No module named 'p2p.run_record'\n")
            stdout.flush()
        return _FakeProc(returncode)

    return _side_effect


class TestSubprocessLogCapture:
    """Verify that subprocess output is captured and crashes are detected."""

    def test_subprocess_log_file_created(self, runs_dir: Path) -> None:
        """start_session should create subprocess.log in the session directory."""
        with patch("p2p.api.process_manager.subprocess.Popen", side_effect=_make_fake_popen(0)):
            lc = LoopConfig(train=TrainConfig(total_timesteps=1000, seed=42))
            session_id = start_session("test", lc, runs_dir=runs_dir)

        log_path = runs_dir / session_id / "subprocess.log"
        assert log_path.exists(), "subprocess.log should be created"

    def test_subprocess_crash_marks_session_error(self, runs_dir: Path) -> None:
        """When subprocess exits non-zero, session status should become 'error'."""
        with patch("p2p.api.process_manager.subprocess.Popen", side_effect=_make_fake_popen(1)):
            lc = LoopConfig(train=TrainConfig(total_timesteps=1000, seed=42))
            session_id = start_session("test", lc, runs_dir=runs_dir)

        session = SessionRecord(runs_dir / session_id)
        status_data = _poll_status(session, "error")
        assert status_data["status"] == "error"
        assert "exit" in status_data.get("error", "").lower()

    def test_subprocess_crash_captures_log_tail(self, runs_dir: Path) -> None:
        """Error message should include the last lines of subprocess output."""
        with patch("p2p.api.process_manager.subprocess.Popen", side_effect=_make_fake_popen(1)):
            lc = LoopConfig(train=TrainConfig(total_timesteps=1000, seed=42))
            session_id = start_session("test", lc, runs_dir=runs_dir)

        session = SessionRecord(runs_dir / session_id)
        status_data = _poll_status(session, "error")
        assert "ModuleNotFoundError" in status_data.get("error", "")

    def test_subprocess_success_keeps_status(self, runs_dir: Path) -> None:
        """When subprocess exits 0, the watchdog should NOT override status."""
        with patch("p2p.api.process_manager.subprocess.Popen", side_effect=_make_fake_popen(0)):
            lc = LoopConfig(train=TrainConfig(total_timesteps=1000, seed=42))
            session_id = start_session("test", lc, runs_dir=runs_dir)

        # Give watchdog time to finish (it should NOT change status)
        time.sleep(0.3)

        session = SessionRecord(runs_dir / session_id)
        status_data = session.read_status()
        assert status_data["status"] == "running"
