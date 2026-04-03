"""Tests for API-server-level CPU core allocation tracking for sessions."""

from __future__ import annotations

import pytest

from p2p.training.cpu_manager import CPUManager


@pytest.fixture()
def _patch_cpu_manager(monkeypatch: pytest.MonkeyPatch):
    """Provide a small CPUManager (16 cores, 2 reserved) for services."""
    mgr = CPUManager(total_cores=16, reserved=2)
    monkeypatch.setattr("p2p.training.cpu_manager._cpu_manager", mgr)
    return mgr


class TestReleaseSessionCores:
    def test_release_frees_cores(self, _patch_cpu_manager):
        import p2p.api.process_manager as svc

        mgr = _patch_cpu_manager
        alloc_id = "session_test1"
        mgr.allocate(alloc_id, 4)
        with svc._session_cpu_allocs_lock:
            svc._session_cpu_allocs["test1"] = alloc_id

        assert mgr.available_count() == 10  # 14 usable - 4 allocated

        svc._release_session_cores("test1")

        assert mgr.available_count() == 14  # all usable cores free
        assert "test1" not in svc._session_cpu_allocs

    def test_release_idempotent(self, _patch_cpu_manager):
        import p2p.api.process_manager as svc

        mgr = _patch_cpu_manager
        alloc_id = "session_test2"
        mgr.allocate(alloc_id, 4)
        with svc._session_cpu_allocs_lock:
            svc._session_cpu_allocs["test2"] = alloc_id

        svc._release_session_cores("test2")
        svc._release_session_cores("test2")  # second call is no-op

        assert mgr.available_count() == 14

    def test_release_unknown_session_is_noop(self):
        import p2p.api.process_manager as svc

        # Should not raise
        svc._release_session_cores("nonexistent_session")


class TestRecoverReAllocates:
    def test_recover_reallocates_cores_for_alive_session(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.process_manager as svc
        from p2p.training.cpu_manager import CPUManager

        mgr = CPUManager(total_cores=16, reserved=2)
        monkeypatch.setattr("p2p.training.cpu_manager._cpu_manager", mgr)

        monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(svc, "verify_pid_ownership", lambda pid, **kw: True)
        # Stub _reattach_session to avoid spawning background watchdog threads
        monkeypatch.setattr(svc, "_reattach_session", lambda session, pid: None)

        # Create a session dir that looks like it's running
        session_dir = tmp_path / "session_alive1"
        session_dir.mkdir()
        import json

        (session_dir / "status.json").write_text(json.dumps({"status": "running"}))
        (session_dir / "pid").write_text("99999")
        (session_dir / "cpu_alloc_count").write_text("6")

        assert mgr.available_count() == 14

        actions = svc.recover_stale_sessions()

        assert actions["session_alive1"] == "reattached"
        # 6 cores should now be allocated
        assert mgr.available_count() == 8
        assert "session_alive1" in svc._session_cpu_allocs

        # Clean up tracking state
        svc._release_session_cores("session_alive1")

    def test_recover_does_not_allocate_for_dead_session(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.process_manager as svc
        from p2p.training.cpu_manager import CPUManager

        mgr = CPUManager(total_cores=16, reserved=2)
        monkeypatch.setattr("p2p.training.cpu_manager._cpu_manager", mgr)

        monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(svc, "verify_pid_ownership", lambda pid, **kw: False)

        session_dir = tmp_path / "session_dead1"
        session_dir.mkdir()
        import json

        (session_dir / "status.json").write_text(json.dumps({"status": "running"}))
        (session_dir / "pid").write_text("99998")
        (session_dir / "cpu_alloc_count").write_text("6")

        actions = svc.recover_stale_sessions()

        assert actions["session_dead1"] == "marked_error_dead"
        # No cores should be allocated
        assert mgr.available_count() == 14
        assert "session_dead1" not in svc._session_cpu_allocs
