"""Tests for E2E session routing through the unified scheduler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from p2p.config import LoopConfig, TrainConfig
from p2p.scheduler import node_store
from p2p.scheduler.controllers import SessionController
from p2p.scheduler.job_queries import cancel_job, find_job_for_session
from p2p.scheduler.manifest_io import read_job_manifest, set_jobs_dir, write_job_manifest


@pytest.fixture(autouse=True)
def _clean_state(tmp_path):
    """Use temp directories for all state."""
    node_store.set_store_path(tmp_path / "nodes.json")
    set_jobs_dir(tmp_path / "jobs")
    yield


@pytest.fixture()
def _mock_spawn():
    """Prevent actually spawning subprocesses."""
    with patch("p2p.scheduler.controllers._spawn_job_scheduler") as mock:
        mock.return_value = None
        yield mock


def _lc(**overrides) -> LoopConfig:
    return LoopConfig(train=TrainConfig(total_timesteps=1000), **overrides)


class TestSessionControllerSessionId:
    """SessionController.run() with explicit session_id."""

    def test_session_id_propagated_to_run_spec(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="walk forward",
            loop_config=_lc(),
            backend="local",
            session_id="session_20260313_120000_abc",
        )

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None
        spec = manifest["runs"][0]["spec"]
        assert spec["run_id"] == "session_20260313_120000_abc"
        assert spec["parameters"]["session_id"] == "session_20260313_120000_abc"

    def test_auto_generated_id_when_none(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="run",
            loop_config=_lc(),
            backend="local",
        )

        manifest = read_job_manifest(job["job_id"])
        spec = manifest["runs"][0]["spec"]
        # Should be a run_xxx auto-generated ID
        assert spec["run_id"].startswith("run_")

    def test_scheduler_proc_stored(self, _mock_spawn) -> None:
        ctrl = SessionController()
        ctrl.run(prompt="test", loop_config=_lc(), backend="local")

        # _spawn_job_scheduler is mocked to return None
        assert ctrl.scheduler_proc is None

    def test_scheduler_proc_set_on_real_spawn(self) -> None:
        """When spawning is not mocked, scheduler_proc holds the Popen."""
        ctrl = SessionController()
        with patch("p2p.scheduler.controllers._spawn_job_scheduler") as mock:
            import subprocess

            fake_proc = subprocess.Popen(["true"])
            mock.return_value = fake_proc
            ctrl.run(prompt="test", loop_config=_lc(), backend="local")

        assert ctrl.scheduler_proc is fake_proc
        fake_proc.wait()


class TestFindJobForSession:
    """find_job_for_session() lookup."""

    def test_finds_running_session(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            backend="local",
            session_id="session_target",
        )

        # Manually set run state to running (normally done by scheduler)
        manifest = read_job_manifest(job["job_id"])
        manifest["runs"][0]["state"] = "running"
        write_job_manifest(manifest)

        found = find_job_for_session("session_target")
        assert found == job["job_id"]

    def test_returns_none_for_unknown_session(self, _mock_spawn) -> None:
        ctrl = SessionController()
        ctrl.run(prompt="test", loop_config=_lc(), backend="local", session_id="session_other")

        found = find_job_for_session("session_unknown")
        assert found is None

    def test_ignores_completed_jobs(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            backend="local",
            session_id="session_done",
        )

        # Mark job as completed
        manifest = read_job_manifest(job["job_id"])
        manifest["status"] = "completed"
        manifest["runs"][0]["state"] = "completed"
        write_job_manifest(manifest)

        found = find_job_for_session("session_done")
        assert found is None


class TestStartSessionRouting:
    """POST /api/sessions always routes through SessionController."""

    def test_creates_manifest_with_session_id(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="walk forward",
            loop_config=_lc(),
            backend="local",
            session_id="session_20260313_test",
        )

        assert job["job_type"] == "session"
        assert job["status"] == "running"

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None
        assert manifest["backend"] == "local"
        spec = manifest["runs"][0]["spec"]
        assert spec["parameters"]["session_id"] == "session_20260313_test"
        assert spec["parameters"]["prompt"] == "walk forward"


class TestStopSessionRouting:
    """POST /api/sessions/{id}/stop: scheduler-first, process_manager fallback."""

    def test_scheduler_stop_cancels_job(self, _mock_spawn) -> None:
        """Stopping a scheduler-managed session cancels the job."""
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            backend="local",
            session_id="session_stop_target",
        )

        # Set run to running state
        manifest = read_job_manifest(job["job_id"])
        manifest["runs"][0]["state"] = "running"
        write_job_manifest(manifest)

        # Find and cancel
        job_id = find_job_for_session("session_stop_target")
        assert job_id is not None
        cancel_job(job_id)

        # Verify manifest is cancelled
        manifest = read_job_manifest(job_id)
        assert manifest["status"] == "cancelled"

    def test_stop_falls_back_for_legacy_session(self, _mock_spawn) -> None:
        """Sessions not in scheduler fall back to process_manager."""
        # No scheduler jobs exist
        found = find_job_for_session("session_legacy")
        assert found is None
