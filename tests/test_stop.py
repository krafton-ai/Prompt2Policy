"""Tests for stop/cancel functionality."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from p2p.api import process_manager
from p2p.session.iteration_record import IterationRecord

# ---------------------------------------------------------------------------
# IterationRecord.derive_status — cancelled takes priority
# ---------------------------------------------------------------------------


class TestDeriveStatusCancelled:
    @pytest.fixture()
    def iteration_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "20260303_120000_abcd1234"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"env_id": "HalfCheetah-v5"}))
        return d

    def test_cancelled_overrides_running(self, iteration_dir: Path):
        metrics = iteration_dir / "metrics"
        metrics.mkdir()
        (metrics / "scalars.jsonl").write_text(json.dumps({"global_step": 100}) + "\n")
        (iteration_dir / "status.json").write_text(json.dumps({"status": "cancelled"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "cancelled"

    def test_cancelled_overrides_completed(self, iteration_dir: Path):
        (iteration_dir / "summary.json").write_text(json.dumps({"final_episodic_return": 0}))
        (iteration_dir / "status.json").write_text(json.dumps({"status": "cancelled"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "cancelled"

    def test_no_cancelled_file_keeps_original_status(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "pending"

    def test_legacy_cancelled_json_still_works(self, iteration_dir: Path):
        """Backward compat: cancelled.json fallback when no status.json."""
        metrics = iteration_dir / "metrics"
        metrics.mkdir()
        (metrics / "scalars.jsonl").write_text(json.dumps({"global_step": 100}) + "\n")
        (iteration_dir / "cancelled.json").write_text(json.dumps({"cancelled": True}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "cancelled"


# ---------------------------------------------------------------------------
# process_manager.stop_session
# ---------------------------------------------------------------------------


class TestStopSession:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        self._orig_runs = process_manager.RUNS_DIR
        process_manager.RUNS_DIR = tmp_path
        process_manager._active_procs.clear()
        yield
        process_manager.RUNS_DIR = self._orig_runs
        process_manager._active_procs.clear()

    def test_stop_unknown_session_returns_false(self):
        assert process_manager.stop_session("nonexistent") is False

    def test_stop_already_finished_returns_false(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 0  # already finished
        process_manager._active_procs["sess_1"] = proc
        assert process_manager.stop_session("sess_1") is False
        assert "sess_1" not in process_manager._active_procs

    @patch("os.killpg")
    @patch("os.getpgid", return_value=99999)
    def test_stop_running_session_terminates(self, _mock_getpgid, _mock_killpg, tmp_path: Path):
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 99999
        proc.poll.return_value = None  # still running
        proc.wait.return_value = None
        process_manager._active_procs["sess_2"] = proc

        # Create session dir with loop_history.json and status.json
        session_dir = tmp_path / "sess_2"
        session_dir.mkdir()
        history = {"session_id": "sess_2", "status": "running", "iterations": []}
        (session_dir / "loop_history.json").write_text(json.dumps(history))
        (session_dir / "status.json").write_text(json.dumps({"status": "running"}))

        assert process_manager.stop_session("sess_2") is True
        assert "sess_2" not in process_manager._active_procs

        # Verify status updated in loop_history
        updated = json.loads((session_dir / "loop_history.json").read_text())
        assert updated["status"] == "cancelled"

    @patch("os.killpg")
    @patch("os.getpgid", return_value=99998)
    def test_stop_session_kills_on_timeout(self, _mock_getpgid, _mock_killpg, tmp_path: Path):
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 99998
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
        process_manager._active_procs["sess_3"] = proc

        session_dir = tmp_path / "sess_3"
        session_dir.mkdir()
        (session_dir / "loop_history.json").write_text(
            json.dumps({"session_id": "sess_3", "status": "running"})
        )

        assert process_manager.stop_session("sess_3") is True
