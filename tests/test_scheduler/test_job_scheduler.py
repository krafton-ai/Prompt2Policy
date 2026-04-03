"""Tests for _NodeCoreAllocator, Backend-delegated ops, and staged execution."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from p2p.scheduler.backend import LocalBackend
from p2p.scheduler.job_scheduler import (
    _check_liveness,
    _init_session,
    _NodeCoreAllocator,
    _NodeGPUAllocator,
    _run_staged,
    _SchedulerState,
    _submit_run,
    kill_run_process_standalone,
)
from p2p.scheduler.manifest_io import set_jobs_dir, write_job_manifest
from p2p.scheduler.types import RunRecord, RunSpec, RunState

_FAKE_SSH_NODE = {"node_id": "ssh1", "host": "h", "user": "u", "port": 22, "max_cores": 8}


def _make_run(
    *,
    run_id: str = "run_1",
    state: RunState = "running",
    pid: int | None = 42,
    node_id: str = "local",
    session_id: str = "sess_1",
) -> RunRecord:
    spec: RunSpec = {
        "run_id": run_id,
        "entry_point": "p2p.session.run_session",
        "parameters": {"session_id": session_id, "prompt": "test"},
        "cpu_cores": 2,
    }
    run: RunRecord = {
        "run_id": run_id,
        "spec": spec,
        "state": state,
        "node_id": node_id,
        "remote_dir": "",
        "synced": False,
    }
    if pid is not None:
        run["pid"] = pid
    return run


# ---------------------------------------------------------------------------
# _NodeCoreAllocator
# ---------------------------------------------------------------------------


class TestNodeCoreAllocator:
    def test_allocate_returns_requested_cores(self) -> None:
        alloc = _NodeCoreAllocator()
        cores = alloc.allocate("n1", max_cores=8, num_cores=4)

        assert cores == [0, 1, 2, 3]
        assert alloc.available("n1") == 4

    def test_allocate_returns_none_when_insufficient(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=2)

        result = alloc.allocate("n1", max_cores=2, num_cores=5)

        assert result is None
        assert alloc.available("n1") == 2  # pool unchanged

    def test_release_returns_cores_sorted(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=8)
        alloc.allocate("n1", max_cores=8, num_cores=8)
        assert alloc.available("n1") == 0

        alloc.release("n1", [7, 3, 1])

        assert alloc.available("n1") == 3
        # Verify released cores come back in sorted order via allocate
        cores = alloc.allocate("n1", max_cores=8, num_cores=3)
        assert cores == [1, 3, 7]

    def test_reserve_removes_specific_cores(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=8)

        alloc.reserve("n1", [2, 5])

        assert alloc.available("n1") == 6
        # Verify reserved cores are gone: allocate all remaining, check 2 and 5 absent
        remaining = alloc.allocate("n1", max_cores=8, num_cores=6)
        assert remaining is not None
        assert 2 not in remaining
        assert 5 not in remaining

    def test_allocate_then_release_roundtrip(self) -> None:
        alloc = _NodeCoreAllocator()
        cores = alloc.allocate("n1", max_cores=4, num_cores=4)
        assert cores is not None
        assert alloc.available("n1") == 0

        alloc.release("n1", cores)

        assert alloc.available("n1") == 4
        # Can allocate again
        cores2 = alloc.allocate("n1", max_cores=4, num_cores=4)
        assert cores2 == [0, 1, 2, 3]

    def test_ensure_node_with_zero_cores(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=0)

        assert alloc.available("n1") == 0
        # Reserve on empty pool is a no-op
        alloc.reserve("n1", [0, 1])
        assert alloc.available("n1") == 0

    def test_ensure_node_idempotent(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=4)
        alloc.allocate("n1", max_cores=4, num_cores=2)

        # Second ensure_node should NOT reset the pool
        alloc.ensure_node("n1", max_cores=4)
        assert alloc.available("n1") == 2

    def test_reserve_skips_cores_not_in_pool(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.ensure_node("n1", max_cores=4)  # pool = [0, 1, 2, 3]

        alloc.reserve("n1", [1, 99])  # 99 is not in pool

        assert alloc.available("n1") == 3
        remaining = alloc.allocate("n1", max_cores=4, num_cores=3)
        assert remaining is not None
        assert 1 not in remaining

    def test_multiple_nodes_independent(self) -> None:
        alloc = _NodeCoreAllocator()
        alloc.allocate("n1", max_cores=4, num_cores=2)
        alloc.allocate("n2", max_cores=8, num_cores=3)

        assert alloc.available("n1") == 2
        assert alloc.available("n2") == 5

    def test_available_unknown_node_returns_zero(self) -> None:
        alloc = _NodeCoreAllocator()
        assert alloc.available("unknown") == 0


# ---------------------------------------------------------------------------
# _check_liveness — Backend-delegated liveness checking
# ---------------------------------------------------------------------------


class TestCheckLiveness:
    def test_running_state_keeps_running(self) -> None:
        run = _make_run(state="running")
        backend = MagicMock()
        backend.status.return_value = {"run_id": "run_1", "state": "running"}

        _check_liveness(run, backend)

        assert run["state"] == "running"
        assert "completed_at" not in run

    def test_completed_state_transitions(self) -> None:
        run = _make_run(state="running")
        backend = MagicMock()
        backend.status.return_value = {
            "run_id": "run_1",
            "state": "completed",
            "completed_at": "2026-01-01T00:00:00",
        }

        _check_liveness(run, backend)

        assert run["state"] == "completed"
        assert run["completed_at"] == "2026-01-01T00:00:00"

    def test_error_state_with_message(self) -> None:
        run = _make_run(state="running")
        backend = MagicMock()
        backend.status.return_value = {
            "run_id": "run_1",
            "state": "error",
            "error": "Remote process exited without status",
        }

        _check_liveness(run, backend)

        assert run["state"] == "error"
        assert "without status" in run["error"]


# ---------------------------------------------------------------------------
# _submit_run — Backend-delegated submission
# ---------------------------------------------------------------------------


class TestSubmitRun:
    def test_successful_submit_updates_run_record(self) -> None:
        run = _make_run(state="pending", pid=None)
        backend = MagicMock()
        backend.submit.return_value = {
            "run_id": "run_1",
            "state": "running",
            "pid": 12345,
            "node_id": "local",
            "started_at": "2026-01-01T00:00:00",
        }

        with patch("p2p.scheduler.job_scheduler._init_session"):
            _submit_run(run, backend, allocated_cores=[0, 1])

        assert run["state"] == "running"
        assert run["pid"] == 12345
        assert run["node_id"] == "local"
        assert run["allocated_cores"] == [0, 1]

    def test_failed_submit_sets_error(self) -> None:
        run = _make_run(state="pending", pid=None)
        backend = MagicMock()
        backend.submit.return_value = {
            "run_id": "run_1",
            "state": "error",
            "error": "Failed to sync code",
        }

        with patch("p2p.scheduler.job_scheduler._init_session"):
            _submit_run(run, backend)

        assert run["state"] == "error"
        assert "sync code" in run["error"]

    def test_ssh_submit_captures_remote_dir(self) -> None:
        run = _make_run(state="pending", pid=None)
        backend = MagicMock()
        backend.submit.return_value = {
            "run_id": "run_1",
            "state": "running",
            "pid": 999,
            "node_id": "ssh1",
            "remote_dir": "/tmp/p2p-abc123",
            "started_at": "2026-01-01T00:00:00",
        }

        with patch("p2p.scheduler.job_scheduler._init_session"):
            _submit_run(run, backend)

        assert run["remote_dir"] == "/tmp/p2p-abc123"
        assert run["node_id"] == "ssh1"


# ---------------------------------------------------------------------------
# _init_session — pre-submit session initialization
# ---------------------------------------------------------------------------


class TestInitSession:
    def test_creates_status_and_config(self, tmp_path, monkeypatch) -> None:
        from p2p.config import LoopConfig, TrainConfig

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)

        lc = LoopConfig(train=TrainConfig(total_timesteps=1000))
        run = _make_run(state="pending", pid=None, session_id="sess_init")
        run["spec"]["parameters"]["loop_config"] = lc.to_json()

        _init_session(run)

        session_dir = tmp_path / "sess_init"
        assert session_dir.exists()

        status = json.loads((session_dir / "status.json").read_text())
        assert status["status"] == "running"

        config = json.loads((session_dir / "session_config.json").read_text())
        assert config["prompt"] == "test"
        assert "train" in config

    def test_benchmark_case_creates_nested_dir(self, tmp_path, monkeypatch) -> None:
        """Benchmark cases use nested layout: bm_xxx/case0/."""
        from p2p.config import LoopConfig, TrainConfig

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)

        lc = LoopConfig(train=TrainConfig(total_timesteps=500))
        run: RunRecord = {
            "run_id": "bm_abc12345_case3",
            "spec": {
                "run_id": "bm_abc12345_case3",
                "entry_point": "p2p.session.run_session",
                "parameters": {
                    "session_id": "bm_abc12345_case3",
                    "prompt": "walk forward",
                    "loop_config": lc.to_json(),
                },
                "cpu_cores": 2,
                "tags": {
                    "job_type": "benchmark",
                    "benchmark_id": "bm_abc12345",
                    "case_index": "3",
                    "env_id": "Ant-v5",
                },
            },
            "state": "pending",
            "node_id": "",
            "remote_dir": "",
            "synced": False,
        }

        _init_session(run)

        # Nested directory is a real directory (not symlink)
        nested_dir = tmp_path / "bm_abc12345" / "case3"
        assert nested_dir.is_dir()

        # No flat symlink exists
        symlink_path = tmp_path / "bm_abc12345_case3"
        assert not symlink_path.exists()

        # Status and config written in the nested directory
        status = json.loads((nested_dir / "status.json").read_text())
        assert status["status"] == "running"

        config = json.loads((nested_dir / "session_config.json").read_text())
        assert config["prompt"] == "walk forward"

    def test_benchmark_case_timestamped_creates_nested_dir(self, tmp_path, monkeypatch) -> None:
        """Timestamped benchmark IDs (bm_YYYYMMDD_HHMMSS_hex) use nested layout."""
        from p2p.config import LoopConfig, TrainConfig

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)

        lc = LoopConfig(train=TrainConfig(total_timesteps=500))
        bm_id = "bm_00010101_000000_abc12345"
        run: RunRecord = {
            "run_id": f"{bm_id}_case2",
            "spec": {
                "run_id": f"{bm_id}_case2",
                "entry_point": "p2p.session.run_session",
                "parameters": {
                    "session_id": f"{bm_id}_case2",
                    "prompt": "run fast",
                    "loop_config": lc.to_json(),
                },
                "cpu_cores": 2,
                "tags": {
                    "job_type": "benchmark",
                    "benchmark_id": bm_id,
                    "case_index": "2",
                    "env_id": "HalfCheetah-v5",
                },
            },
            "state": "pending",
            "node_id": "",
            "remote_dir": "",
            "synced": False,
        }

        _init_session(run)

        nested_dir = tmp_path / bm_id / "case2"
        assert nested_dir.is_dir()
        assert not (tmp_path / f"{bm_id}_case2").exists()

        status = json.loads((nested_dir / "status.json").read_text())
        assert status["status"] == "running"

    def test_benchmark_case_idempotent(self, tmp_path, monkeypatch) -> None:
        """Calling _init_session twice does not raise."""
        from p2p.config import LoopConfig, TrainConfig

        monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)

        lc = LoopConfig(train=TrainConfig(total_timesteps=500))
        run: RunRecord = {
            "run_id": "bm_aaa00000_case0",
            "spec": {
                "run_id": "bm_aaa00000_case0",
                "entry_point": "p2p.session.run_session",
                "parameters": {
                    "session_id": "bm_aaa00000_case0",
                    "prompt": "test",
                    "loop_config": lc.to_json(),
                },
                "cpu_cores": 2,
                "tags": {
                    "job_type": "benchmark",
                    "benchmark_id": "bm_aaa00000",
                    "case_index": "0",
                    "env_id": "HalfCheetah-v5",
                },
            },
            "state": "pending",
            "node_id": "",
            "remote_dir": "",
            "synced": False,
        }

        _init_session(run)
        _init_session(run)  # second call should not raise

        assert (tmp_path / "bm_aaa00000" / "case0").is_dir()


# ---------------------------------------------------------------------------
# kill_run_process_standalone — SIGKILL escalation
# ---------------------------------------------------------------------------


class TestKillRunProcessEscalation:
    """Verify kill_run_process_standalone delegates to safe_killpg.

    The SIGTERM/SIGKILL escalation logic is now in safe_killpg() and
    tested in tests/test_process_isolation.py::TestSafeKillpg.
    """

    def test_local_run_delegates_to_safe_killpg(self) -> None:
        run = _make_run(pid=100, node_id="local")
        with patch("p2p.scheduler.job_scheduler.safe_killpg", return_value=True) as mock_safe:
            kill_run_process_standalone(run)
        session_id = run["spec"]["parameters"].get("session_id", "")
        mock_safe.assert_called_once_with(100, expected_cmdline=session_id or None)

    def test_no_pid_is_noop(self) -> None:
        run = _make_run(pid=None, node_id="local")
        # Should return without error — no kill attempted
        kill_run_process_standalone(run)


# ---------------------------------------------------------------------------
# _run_staged
# ---------------------------------------------------------------------------


def _make_staged_manifest(tmp_path, *, gate_threshold=0.7, start_from_stage=1):
    """Create a minimal staged benchmark manifest with 2 stages."""
    set_jobs_dir(tmp_path / "jobs")

    runs = [
        {
            "run_id": f"bm_case{i}",
            "spec": {
                "run_id": f"bm_case{i}",
                "entry_point": "p2p.session.run_session",
                "parameters": {"session_id": f"bm_case{i}", "prompt": "test"},
                "cpu_cores": 1,
                "tags": {"case_index": str(i), "stage": str(1 if i < 2 else 2)},
            },
            "state": "pending",
            "node_id": "",
            "remote_dir": "",
            "synced": False,
            "session_group": f"case_{i}",
        }
        for i in range(4)
    ]

    manifest = {
        "job_id": "job_staged",
        "job_type": "benchmark",
        "status": "running",
        "created_at": "2026-01-01T00:00:00",
        "backend": "local",
        "config": {
            "mode": "staged",
            "max_parallel": 10,
            "pass_threshold": 0.7,
            "start_from_stage": start_from_stage,
            "gate_threshold": gate_threshold,
        },
        "runs": runs,
        "metadata": {
            "benchmark_id": "bm_test",
            "mode": "staged",
            "current_stage": 0,
            "total_stages": 2,
            "stages": [
                {
                    "stage": 1,
                    "name": "Batch 1",
                    "gate_threshold": gate_threshold,
                    "max_parallel": 10,
                    "case_indices": [0, 1],
                    "status": "pending",
                    "gate_result": None,
                },
                {
                    "stage": 2,
                    "name": "Batch 2",
                    "gate_threshold": 0.0,
                    "max_parallel": 10,
                    "case_indices": [2, 3],
                    "status": "pending",
                    "gate_result": None,
                },
            ],
            "test_cases": [
                {"index": i, "env_id": f"Env-{i}", "instruction": f"test {i}"} for i in range(4)
            ],
        },
    }
    write_job_manifest(manifest)
    return manifest


class TestRunStaged:
    def test_staged_gate_passed_runs_both_stages(self, tmp_path) -> None:
        manifest = _make_staged_manifest(tmp_path, gate_threshold=0.5)
        for r in manifest["runs"]:
            r["state"] = "completed"
            r["synced"] = True

        allocator = _NodeCoreAllocator()
        local_be = LocalBackend()
        state = _SchedulerState(manifest, allocator, _NodeGPUAllocator(), local_be, {})

        def _mock_info(sid):
            return {"best_score": 0.8, "status": "passed"}

        with (
            patch(
                "p2p.benchmark.benchmark_helpers.lightweight_session_info",
                side_effect=_mock_info,
            ),
            patch(
                "p2p.scheduler.job_scheduler.read_job_manifest",
                return_value=manifest,
            ),
            patch("p2p.scheduler.job_scheduler.write_job_manifest"),
        ):
            result = _run_staged(state, "job_staged")

        assert result is True
        stages = manifest["metadata"]["stages"]
        assert stages[0]["status"] == "gate_passed"
        assert stages[0]["gate_result"]["passed"] is True
        assert stages[1]["status"] == "completed"  # last stage, gt=0.0

    def test_staged_gate_failed_skips_remaining(self, tmp_path) -> None:
        manifest = _make_staged_manifest(tmp_path, gate_threshold=0.9)
        # Only stage 1 runs are completed; stage 2 stays pending
        for r in manifest["runs"]:
            if r["spec"]["tags"].get("stage") == "1":
                r["state"] = "completed"
                r["synced"] = True

        allocator = _NodeCoreAllocator()
        local_be = LocalBackend()
        state = _SchedulerState(manifest, allocator, _NodeGPUAllocator(), local_be, {})

        def _mock_info(sid):
            return {"best_score": 0.3, "status": "failed"}

        with (
            patch(
                "p2p.benchmark.benchmark_helpers.lightweight_session_info",
                side_effect=_mock_info,
            ),
            patch(
                "p2p.scheduler.job_scheduler.read_job_manifest",
                return_value=manifest,
            ),
            patch("p2p.scheduler.job_scheduler.write_job_manifest"),
        ):
            result = _run_staged(state, "job_staged")

        assert result is True
        stages = manifest["metadata"]["stages"]
        assert stages[0]["status"] == "gate_failed"
        assert stages[1]["status"] == "skipped"
        # Stage 2 runs should be cancelled
        stage2_runs = [r for r in manifest["runs"] if r["spec"]["tags"].get("stage") == "2"]
        for r in stage2_runs:
            assert r["state"] == "cancelled"

    def test_staged_start_from_stage_skips_earlier(self, tmp_path) -> None:
        manifest = _make_staged_manifest(tmp_path, gate_threshold=0.0, start_from_stage=2)
        for r in manifest["runs"]:
            r["state"] = "completed"
            r["synced"] = True

        allocator = _NodeCoreAllocator()
        local_be = LocalBackend()
        state = _SchedulerState(manifest, allocator, _NodeGPUAllocator(), local_be, {})

        with (
            patch(
                "p2p.scheduler.job_scheduler.read_job_manifest",
                return_value=manifest,
            ),
            patch("p2p.scheduler.job_scheduler.write_job_manifest"),
        ):
            result = _run_staged(state, "job_staged")

        assert result is True
        stages = manifest["metadata"]["stages"]
        assert stages[0]["status"] == "skipped"
        assert stages[1]["status"] == "completed"
