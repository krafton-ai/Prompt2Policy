"""Tests for process isolation: PID verification and PGID ownership checks.

Covers the defensive guards added in issue #380 to prevent cancelling
one job from killing unrelated processes via recycled PIDs.
"""

from __future__ import annotations

import os
import signal
import subprocess
from unittest.mock import MagicMock, patch

from p2p.api import process_manager as pm
from p2p.utils import process_safety as ps

# ---------------------------------------------------------------------------
# verify_pid_ownership
# ---------------------------------------------------------------------------


class TestVerifyPidOwnership:
    def test_rejects_non_p2p_process(self):
        """Our pytest process lacks 'p2p' in cmdline — always rejected."""
        pid = os.getpid()
        assert ps.verify_pid_ownership(pid, expected_cmdline="pytest") is False
        assert ps.verify_pid_ownership(pid, expected_cmdline=None) is False

    def test_nonexistent_pid_returns_false(self):
        assert ps.verify_pid_ownership(999999999, expected_cmdline="anything") is False

    def test_requires_both_p2p_and_expected_cmdline(self):
        """Spawn a subprocess with both 'p2p' and a marker in cmdline."""
        import sys
        import time

        marker = "session_test_12345_marker"
        proc = subprocess.Popen(
            [sys.executable, "-c", f"import p2p.settings, time; time.sleep(60)  # {marker}"],
            start_new_session=True,
        )
        try:
            time.sleep(0.1)
            # Has both "p2p" and the marker → True
            assert ps.verify_pid_ownership(proc.pid, expected_cmdline=marker) is True
            # Has "p2p" but not "nonexistent" → False
            assert ps.verify_pid_ownership(proc.pid, expected_cmdline="nonexistent") is False
            # Has "p2p", no expected_cmdline → True (generic check)
            assert ps.verify_pid_ownership(proc.pid) is True
        finally:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# verify_pgid_ownership
# ---------------------------------------------------------------------------


class TestVerifyPgidOwnership:
    def test_start_new_session_process_has_pgid_eq_pid(self):
        """Subprocess with start_new_session=True should have PGID == PID."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            start_new_session=True,
        )
        try:
            pgid = ps.verify_pgid_ownership(proc.pid)
            assert pgid is not None
            assert pgid == proc.pid
        finally:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def test_child_process_has_pgid_ne_pid(self):
        """Child without start_new_session inherits parent PGID != child PID."""
        proc = subprocess.Popen(
            ["sleep", "60"],
        )
        try:
            pgid = ps.verify_pgid_ownership(proc.pid)
            assert pgid is None
        finally:
            proc.kill()
            proc.wait()

    def test_dead_process_returns_none(self):
        assert ps.verify_pgid_ownership(999999999) is None


# ---------------------------------------------------------------------------
# safe_killpg
# ---------------------------------------------------------------------------


class TestSafeKillpg:
    @patch.object(ps, "verify_pid_ownership", return_value=False)
    @patch("os.killpg")
    def test_returns_false_when_pid_not_owned(self, mock_killpg, mock_verify):
        assert ps.safe_killpg(12345, expected_cmdline="sess_1") is False
        mock_killpg.assert_not_called()

    @patch.object(ps, "verify_pid_ownership", return_value=True)
    @patch.object(ps, "verify_pgid_ownership", return_value=None)
    @patch("os.kill")
    def test_falls_back_to_single_pid_on_pgid_mismatch(self, mock_kill, mock_pgid, mock_verify):
        assert ps.safe_killpg(12345) is True
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @patch.object(ps, "verify_pid_ownership", return_value=True)
    @patch.object(ps, "verify_pgid_ownership", return_value=12345)
    @patch("os.killpg")
    @patch.object(ps, "is_pid_alive", return_value=False)
    def test_kills_pgid_when_verified(self, mock_alive, mock_killpg, mock_pgid, mock_verify):
        assert ps.safe_killpg(12345) is True
        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)

    @patch.object(ps, "verify_pid_ownership", return_value=True)
    @patch.object(ps, "verify_pgid_ownership", return_value=12345)
    @patch("os.killpg")
    @patch.object(ps, "is_pid_alive", return_value=True)
    @patch("p2p.utils.process_safety.time.sleep")
    def test_escalates_to_sigkill_after_timeout(
        self, mock_sleep, mock_alive, mock_killpg, mock_pgid, _
    ):
        ps.safe_killpg(12345)
        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(12345, signal.SIGTERM)
        mock_killpg.assert_any_call(12345, signal.SIGKILL)

    @patch.object(ps, "verify_pid_ownership", return_value=True)
    @patch.object(ps, "verify_pgid_ownership", return_value=12345)
    @patch("os.killpg", side_effect=PermissionError("Operation not permitted"))
    def test_returns_false_on_sigterm_permission_error(self, mock_killpg, mock_pgid, mock_verify):
        assert ps.safe_killpg(12345) is False


# ---------------------------------------------------------------------------
# _kill_pid_tree delegates to safe_killpg
# ---------------------------------------------------------------------------


class TestKillPidTree:
    @patch("p2p.api.process_manager.safe_killpg", return_value=False)
    @patch.object(pm, "get_descendant_pids", return_value=[])
    @patch.object(pm, "force_kill_pids")
    def test_skips_children_when_verification_fails(self, mock_force, mock_desc, mock_safe):
        pm._kill_pid_tree(12345, session_id="session_abc")
        mock_safe.assert_called_once_with(12345, expected_cmdline="session_abc")
        mock_force.assert_not_called()

    @patch("p2p.api.process_manager.safe_killpg", return_value=True)
    @patch.object(pm, "get_descendant_pids", return_value=[111, 222])
    @patch.object(pm, "force_kill_pids")
    def test_kills_children_when_verification_passes(self, mock_force, mock_desc, mock_safe):
        pm._kill_pid_tree(12345, session_id="session_abc")
        mock_safe.assert_called_once_with(12345, expected_cmdline="session_abc")
        mock_force.assert_called_once_with([111, 222])


# ---------------------------------------------------------------------------
# kill_run_process_standalone delegates to safe_killpg
# ---------------------------------------------------------------------------


class TestKillRunProcessStandalone:
    def _make_local_run(self, pid: int = 12345, session_id: str = "sess_1") -> dict:
        return {
            "run_id": "run_1",
            "spec": {"parameters": {"session_id": session_id}},
            "state": "running",
            "pid": pid,
            "node_id": "local",
        }

    @patch("p2p.scheduler.job_scheduler.safe_killpg", return_value=False)
    @patch("p2p.scheduler.job_scheduler.force_kill_pids")
    @patch("p2p.scheduler.job_scheduler.get_descendant_pids", return_value=[])
    def test_delegates_to_safe_killpg(self, mock_desc, mock_force, mock_safe):
        from p2p.scheduler.job_scheduler import kill_run_process_standalone

        run = self._make_local_run()
        kill_run_process_standalone(run)
        mock_safe.assert_called_once_with(12345, expected_cmdline="sess_1")

    @patch("p2p.scheduler.job_scheduler.safe_killpg", return_value=False)
    @patch("p2p.scheduler.job_scheduler.force_kill_pids")
    @patch("p2p.scheduler.job_scheduler.get_descendant_pids", return_value=[])
    def test_empty_session_id_passes_none(self, mock_desc, mock_force, mock_safe):
        from p2p.scheduler.job_scheduler import kill_run_process_standalone

        run = self._make_local_run(session_id="")
        kill_run_process_standalone(run)
        mock_safe.assert_called_once_with(12345, expected_cmdline=None)

    def test_no_pid_is_noop(self):
        from p2p.scheduler.job_scheduler import kill_run_process_standalone

        run = {"run_id": "r1", "spec": {"parameters": {}}, "state": "running", "node_id": "local"}
        kill_run_process_standalone(run)


# ---------------------------------------------------------------------------
# cancel_job scheduler PID verification
# ---------------------------------------------------------------------------


class TestCancelJobSchedulerVerification:
    @patch("p2p.scheduler.job_queries.write_job_manifest")
    @patch("p2p.scheduler.job_queries.read_job_manifest")
    @patch("p2p.scheduler.job_queries.verify_pid_ownership", return_value=True)
    def test_scheduler_alive_and_verified_just_writes_cancelled(
        self, mock_verify, mock_read, mock_write
    ):
        from p2p.scheduler.job_queries import cancel_job

        manifest = {
            "job_id": "job_1",
            "status": "running",
            "scheduler_pid": 12345,
            "runs": [],
        }
        mock_read.return_value = manifest

        cancel_job("job_1")
        assert manifest["status"] == "cancelled"
        mock_write.assert_called_once()

    @patch("p2p.scheduler.job_queries.write_job_manifest")
    @patch("p2p.scheduler.job_queries.read_job_manifest")
    @patch("p2p.scheduler.job_queries.verify_pid_ownership", return_value=False)
    def test_scheduler_pid_recycled_uses_direct_kill(self, mock_verify, mock_read, mock_write):
        """When scheduler PID is recycled, should fall through to direct kill path."""
        from p2p.scheduler.job_queries import cancel_job

        manifest = {
            "job_id": "job_1",
            "status": "running",
            "scheduler_pid": 12345,
            "runs": [
                {"run_id": "r1", "state": "pending"},
            ],
        }
        mock_read.return_value = manifest

        cancel_job("job_1")
        assert manifest["runs"][0]["state"] == "cancelled"


# ---------------------------------------------------------------------------
# LocalBackend.cancel with PGID verification
# ---------------------------------------------------------------------------


class TestLocalBackendCancel:
    @patch("p2p.scheduler.backend.safe_killpg", return_value=True)
    def test_delegates_to_safe_killpg(self, mock_safe):
        from p2p.scheduler.backend import LocalBackend

        backend = LocalBackend()
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        with backend._lock:
            backend._procs["run_1"] = proc
            backend._statuses["run_1"] = {"run_id": "run_1", "state": "running"}

        result = backend.cancel("run_1")
        mock_safe.assert_called_once_with(12345, expected_cmdline="run_1")
        assert result is True

    def test_cancel_unknown_run_returns_false(self):
        from p2p.scheduler.backend import LocalBackend

        backend = LocalBackend()
        assert backend.cancel("nonexistent") is False
