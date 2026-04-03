"""Tests for stale session detection (heartbeat + is_stale)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from p2p.session.iteration_record import SessionRecord, write_status


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "session_abc123"
    d.mkdir()
    return d


class TestWriteStatusTimestamp:
    def test_write_status_includes_updated_at(self, session_dir: Path):
        write_status(session_dir, "running")
        data = json.loads((session_dir / "status.json").read_text())
        assert "updated_at" in data
        # Should be a valid ISO 8601 timestamp
        ts = datetime.fromisoformat(data["updated_at"])
        assert ts.tzinfo is not None  # timezone-aware

    def test_write_status_preserves_error(self, session_dir: Path):
        write_status(session_dir, "error", error="boom")
        data = json.loads((session_dir / "status.json").read_text())
        assert data["status"] == "error"
        assert data["error"] == "boom"
        assert "updated_at" in data


class TestTouchHeartbeat:
    def test_touch_heartbeat_updates_timestamp(self, session_dir: Path):
        sr = SessionRecord(session_dir)
        sr.set_status("running")

        old_data = json.loads((session_dir / "status.json").read_text())
        old_ts = old_data["updated_at"]

        sr.touch_heartbeat()

        new_data = json.loads((session_dir / "status.json").read_text())
        assert new_data["status"] == "running"  # status unchanged
        assert new_data["updated_at"] >= old_ts  # timestamp updated

    def test_touch_heartbeat_noop_without_status_file(self, session_dir: Path):
        sr = SessionRecord(session_dir)
        # No status.json exists — should not raise
        sr.touch_heartbeat()
        assert not (session_dir / "status.json").exists()


class TestIsStale:
    def test_non_running_status_is_not_stale(self):
        from p2p.api.process_manager import is_stale

        now = datetime.now(timezone.utc).isoformat()
        assert is_stale({"status": "completed", "updated_at": now}) is False
        assert is_stale({"status": "error", "updated_at": now}) is False

    def test_running_with_fresh_timestamp_is_not_stale(self):
        from p2p.api.process_manager import is_stale

        now = datetime.now(timezone.utc).isoformat()
        assert is_stale({"status": "running", "updated_at": now}) is False

    def test_running_with_old_timestamp_is_stale(self):
        from p2p.api.process_manager import is_stale

        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        assert is_stale({"status": "running", "updated_at": old}) is True

    def test_running_without_updated_at_is_not_stale(self):
        from p2p.api.process_manager import is_stale

        assert is_stale({"status": "running"}) is False

    def test_none_status_data_is_not_stale(self):
        from p2p.api.process_manager import is_stale

        assert is_stale(None) is False


class TestIsStalePidCheck:
    """PID-based process alive check in stale detection."""

    def _old_status(self) -> dict:
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        return {"status": "running", "updated_at": old}

    def test_pid_alive_and_p2p_process_not_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.process_manager as pm

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(pm, "RUNS_DIR", tmp_path)
        session_dir = tmp_path / "sess1"
        session_dir.mkdir()
        (session_dir / "pid").write_text("12345")

        monkeypatch.setattr(pm, "verify_pid_ownership", lambda pid, **kw: True)

        assert pm.is_stale(self._old_status(), "sess1") is False

    def test_pid_alive_but_not_p2p_process_is_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.process_manager as pm

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(pm, "RUNS_DIR", tmp_path)
        session_dir = tmp_path / "sess2"
        session_dir.mkdir()
        (session_dir / "pid").write_text("12345")

        monkeypatch.setattr(pm, "verify_pid_ownership", lambda pid, **kw: False)

        assert pm.is_stale(self._old_status(), "sess2") is True

    def test_pid_dead_is_stale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import p2p.api.process_manager as pm

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(pm, "RUNS_DIR", tmp_path)
        session_dir = tmp_path / "sess3"
        session_dir.mkdir()
        (session_dir / "pid").write_text("12345")

        monkeypatch.setattr(pm, "verify_pid_ownership", lambda pid, **kw: False)

        assert pm.is_stale(self._old_status(), "sess3") is True

    def test_no_pid_file_is_stale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import p2p.api.process_manager as pm

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(pm, "RUNS_DIR", tmp_path)
        session_dir = tmp_path / "sess4"
        session_dir.mkdir()
        # No pid file written

        assert pm.is_stale(self._old_status(), "sess4") is True

    def test_fresh_timestamp_skips_pid_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.process_manager as pm

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(pm, "RUNS_DIR", tmp_path)
        now = datetime.now(timezone.utc).isoformat()
        status = {"status": "running", "updated_at": now}

        # PID check should never be reached — no session dir or pid file needed

        assert pm.is_stale(status, "sess5") is False


class TestGetSessionIsStale:
    def test_get_session_returns_is_stale_for_stale_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.services as svc

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)

        session_dir = tmp_path / "session_stale"
        session_dir.mkdir()

        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        (session_dir / "status.json").write_text(
            json.dumps({"status": "running", "updated_at": old_ts})
        )

        result = svc.get_session("session_stale")
        assert result is not None
        assert result.is_stale is True

    def test_get_session_returns_not_stale_for_fresh_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.services as svc

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
        monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)

        session_dir = tmp_path / "session_fresh"
        session_dir.mkdir()

        now = datetime.now(timezone.utc).isoformat()
        (session_dir / "status.json").write_text(
            json.dumps({"status": "running", "updated_at": now})
        )

        result = svc.get_session("session_fresh")
        assert result is not None
        assert result.is_stale is False
