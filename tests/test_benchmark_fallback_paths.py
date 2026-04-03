"""Integration tests for legacy/pointer benchmark fallback paths.

Covers the dual-path dispatch in benchmark_service.py where:
- Legacy benchmarks (``benchmark_`` prefix) use standalone manifests with scheduler PID
- Pointer benchmarks (``bm_`` prefix) delegate to scheduler job manifests

Closes #236
"""

from __future__ import annotations

import json

import pytest

from p2p.api.benchmark_service import (
    _get_job_id,
    _is_scheduler_alive,
    _loop_config_from_manifest,
    get_benchmark,
    get_benchmark_config,
    list_benchmarks,
    stop_benchmark,
)
from p2p.api.entity_lifecycle import _is_benchmark_dir
from p2p.config import LoopConfig, TrainConfig

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """Override RUNS_DIR to a temp directory for all tests."""
    monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
    monkeypatch.setattr("p2p.api.benchmark_service.RUNS_DIR", tmp_path)
    monkeypatch.setattr("p2p.api.entity_lifecycle.RUNS_DIR", tmp_path)
    monkeypatch.setattr("p2p.benchmark.benchmark_helpers.RUNS_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def stub_legacy_deps(monkeypatch):
    """Stub session info and scheduler liveness for legacy benchmark tests."""
    monkeypatch.setattr(
        "p2p.api.benchmark_service._lightweight_session_info",
        lambda sid: {
            "status": "completed",
            "best_score": 0.8,
            "is_stale": False,
            "iterations_completed": 3,
        },
    )
    monkeypatch.setattr("p2p.api.benchmark_service._is_scheduler_alive", lambda m: False)


def _write_manifest(runs_dir, benchmark_id: str, manifest: dict) -> None:
    """Write a benchmark.json manifest under runs_dir/benchmark_id."""
    d = runs_dir / benchmark_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark.json").write_text(json.dumps(manifest, indent=2))


def _make_legacy_manifest(
    benchmark_id: str = "benchmark_20260101_test",
    *,
    status: str = "completed",
    scheduler_pid: int | None = None,
    with_loop_config: bool = False,
) -> dict:
    """Build a legacy-format manifest (benchmark_ prefix, flat fields)."""
    m: dict = {
        "benchmark_id": benchmark_id,
        "created_at": "2026-01-01T00:00:00",
        "status": status,
        "total_timesteps": 1_000_000,
        "seed": 1,
        "max_iterations": 5,
        "pass_threshold": 0.7,
        "test_cases": [
            {
                "index": 0,
                "env_id": "HalfCheetah-v5",
                "instruction": "run forward",
                "category": "locomotion",
                "difficulty": "easy",
                "session_id": "s0",
            },
        ],
    }
    if scheduler_pid is not None:
        m["scheduler_pid"] = scheduler_pid
    if with_loop_config:
        lc = LoopConfig(train=TrainConfig(total_timesteps=1_000_000, seed=1))
        m["loop_config"] = lc.to_json()
    return m


def _make_pointer_manifest(
    benchmark_id: str = "bm_20260101_test",
    job_id: str = "job_abc123",
) -> dict:
    """Build a pointer manifest (bm_ prefix, delegates to job)."""
    return {
        "type": "pointer",
        "job_id": job_id,
        "benchmark_id": benchmark_id,
        "created_at": "2026-01-01T00:00:00",
    }


def _make_job_benchmark_data(benchmark_id: str, **overrides) -> dict:
    """Build mock data matching get_job_benchmark return shape."""
    base = {
        "benchmark_id": benchmark_id,
        "created_at": "2026-01-01T00:00:00",
        "status": "completed",
        "total_cases": 2,
        "completed_cases": 2,
        "passed_cases": 1,
        "success_rate": 0.5,
        "average_score": 0.6,
        "cumulative_score": 1.2,
        "mode": "flat",
        "current_stage": 0,
        "total_stages": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _is_benchmark_dir
# ---------------------------------------------------------------------------


class TestIsBenchmarkDir:
    def test_matches_benchmark_prefix(self, runs_dir):
        _write_manifest(runs_dir, "benchmark_test1", {"benchmark_id": "benchmark_test1"})
        assert _is_benchmark_dir(runs_dir / "benchmark_test1") is True

    def test_matches_bm_prefix(self, runs_dir):
        _write_manifest(runs_dir, "bm_test1", {"type": "pointer", "benchmark_id": "bm_test1"})
        assert _is_benchmark_dir(runs_dir / "bm_test1") is True

    def test_rejects_session_dir(self, runs_dir):
        d = runs_dir / "session_test1"
        d.mkdir()
        (d / "benchmark.json").write_text("{}")
        assert _is_benchmark_dir(d) is False

    def test_rejects_missing_manifest(self, runs_dir):
        d = runs_dir / "benchmark_no_manifest"
        d.mkdir()
        assert _is_benchmark_dir(d) is False


# ---------------------------------------------------------------------------
# _get_job_id
# ---------------------------------------------------------------------------


class TestGetJobId:
    def test_returns_job_id_for_pointer(self):
        manifest = {"type": "pointer", "job_id": "job_123"}
        assert _get_job_id(manifest) == "job_123"

    def test_returns_none_for_legacy(self):
        manifest = {"benchmark_id": "benchmark_test", "status": "running"}
        assert _get_job_id(manifest) is None

    def test_returns_none_for_non_pointer_type(self):
        manifest = {"type": "standalone", "job_id": "job_123"}
        assert _get_job_id(manifest) is None

    def test_returns_none_for_empty_manifest(self):
        assert _get_job_id({}) is None


# ---------------------------------------------------------------------------
# _loop_config_from_manifest
# ---------------------------------------------------------------------------


class TestLoopConfigFromManifest:
    def test_extracts_loop_config(self):
        lc = LoopConfig(train=TrainConfig(total_timesteps=500_000, seed=42))
        manifest = {"loop_config": lc.to_json()}
        result = _loop_config_from_manifest(manifest)
        assert result is not None
        assert result.train.total_timesteps == 500_000

    def test_returns_none_for_legacy_manifest(self):
        manifest = {"total_timesteps": 500_000, "seed": 42}
        assert _loop_config_from_manifest(manifest) is None


# ---------------------------------------------------------------------------
# _is_scheduler_alive
# ---------------------------------------------------------------------------


class TestIsSchedulerAlive:
    def test_legacy_manifest_alive_pid(self, monkeypatch):
        """Legacy manifest with a live PID -> True."""
        monkeypatch.setattr(
            "p2p.api.benchmark_service.verify_pid_ownership", lambda pid, **kw: True
        )
        manifest = {"scheduler_pid": 12345}
        assert _is_scheduler_alive(manifest) is True

    def test_legacy_manifest_dead_pid(self, monkeypatch):
        """Legacy manifest with a dead PID -> False."""
        monkeypatch.setattr(
            "p2p.api.benchmark_service.verify_pid_ownership", lambda pid, **kw: False
        )
        manifest = {"scheduler_pid": 99999}
        assert _is_scheduler_alive(manifest) is False

    def test_legacy_manifest_no_pid(self):
        """Legacy manifest with no scheduler_pid -> False."""
        manifest = {"benchmark_id": "benchmark_test"}
        assert _is_scheduler_alive(manifest) is False

    def test_pointer_manifest_delegates_to_job_queries(self, monkeypatch):
        """Pointer manifest delegates to job_queries.is_scheduler_alive."""
        captured = {}
        monkeypatch.setattr(
            "p2p.scheduler.job_queries.is_scheduler_alive",
            lambda job_id: captured.update({"job_id": job_id}) or True,
        )
        manifest = {"type": "pointer", "job_id": "job_xyz"}
        assert _is_scheduler_alive(manifest) is True
        assert captured["job_id"] == "job_xyz"


# ---------------------------------------------------------------------------
# list_benchmarks
# ---------------------------------------------------------------------------


class TestListBenchmarks:
    def test_lists_legacy_benchmark(self, runs_dir, stub_legacy_deps):
        """Legacy benchmark appears in listing via manifest-based aggregation."""
        manifest = _make_legacy_manifest()
        _write_manifest(runs_dir, manifest["benchmark_id"], manifest)

        result = list_benchmarks()
        assert len(result) == 1
        assert result[0].benchmark_id == manifest["benchmark_id"]
        assert result[0].status == "completed"

    def test_lists_pointer_benchmark(self, runs_dir, monkeypatch):
        """Pointer benchmark delegates to get_job_benchmark for listing."""
        bm_id = "bm_20260101_test"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_abc")
        _write_manifest(runs_dir, bm_id, pointer)

        mock_data = _make_job_benchmark_data(
            bm_id,
            total_cases=5,
            completed_cases=5,
            passed_cases=3,
            success_rate=0.6,
            average_score=0.65,
            cumulative_score=3.25,
        )
        monkeypatch.setattr(
            "p2p.scheduler.benchmark_aggregation.get_job_benchmark",
            lambda job_id: mock_data,
        )

        result = list_benchmarks()
        assert len(result) == 1
        assert result[0].benchmark_id == bm_id
        assert result[0].total_cases == 5

    def test_lists_both_legacy_and_pointer(self, runs_dir, stub_legacy_deps, monkeypatch):
        """Mixed runs_dir with legacy + pointer benchmarks lists both."""
        legacy = _make_legacy_manifest()
        _write_manifest(runs_dir, legacy["benchmark_id"], legacy)

        bm_id = "bm_20260101_mixed"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_mix")
        _write_manifest(runs_dir, bm_id, pointer)

        monkeypatch.setattr(
            "p2p.scheduler.benchmark_aggregation.get_job_benchmark",
            lambda job_id: _make_job_benchmark_data(bm_id, status="running"),
        )

        result = list_benchmarks()
        assert len(result) == 2
        ids = {r.benchmark_id for r in result}
        assert legacy["benchmark_id"] in ids
        assert bm_id in ids

    def test_skips_pointer_when_job_manifest_missing(self, runs_dir, monkeypatch):
        """Pointer benchmark is skipped if job manifest is unavailable."""
        pointer = _make_pointer_manifest()
        _write_manifest(runs_dir, pointer["benchmark_id"], pointer)

        monkeypatch.setattr(
            "p2p.scheduler.benchmark_aggregation.get_job_benchmark",
            lambda job_id: None,
        )

        result = list_benchmarks()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_benchmark_config
# ---------------------------------------------------------------------------


class TestGetBenchmarkConfig:
    def test_pointer_delegates_to_job_manifest(self, runs_dir, monkeypatch):
        """Pointer benchmark reads config from job manifest."""
        bm_id = "bm_config_test"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_cfg")
        _write_manifest(runs_dir, bm_id, pointer)

        job_config = {"total_timesteps": 2_000_000, "max_iterations": 10}
        monkeypatch.setattr(
            "p2p.scheduler.manifest_io.read_job_manifest",
            lambda job_id: {"config": job_config},
        )

        result = get_benchmark_config(bm_id)
        assert result == job_config

    def test_pointer_returns_none_when_job_missing(self, runs_dir, monkeypatch):
        """Pointer benchmark returns None if job manifest is gone."""
        bm_id = "bm_config_gone"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_gone")
        _write_manifest(runs_dir, bm_id, pointer)

        monkeypatch.setattr(
            "p2p.scheduler.manifest_io.read_job_manifest",
            lambda job_id: None,
        )

        assert get_benchmark_config(bm_id) is None

    def test_legacy_with_loop_config(self, runs_dir):
        """Legacy manifest with loop_config extracts structured config."""
        manifest = _make_legacy_manifest(with_loop_config=True)
        _write_manifest(runs_dir, manifest["benchmark_id"], manifest)

        result = get_benchmark_config(manifest["benchmark_id"])
        assert result is not None
        assert result["total_timesteps"] == 1_000_000
        assert result["seed"] == 1

    def test_legacy_without_loop_config(self, runs_dir):
        """Legacy manifest without loop_config returns empty dict + manifest fields."""
        manifest = _make_legacy_manifest()
        manifest["mode"] = "flat"
        manifest["csv_file"] = "test.csv"
        _write_manifest(runs_dir, manifest["benchmark_id"], manifest)

        result = get_benchmark_config(manifest["benchmark_id"])
        assert result is not None
        assert result.get("mode") == "flat"
        assert result.get("csv_file") == "test.csv"
        # No LoopConfig fields
        assert "total_timesteps" not in result

    def test_nonexistent_benchmark_returns_none(self, runs_dir):
        assert get_benchmark_config("nonexistent_id") is None


# ---------------------------------------------------------------------------
# get_benchmark
# ---------------------------------------------------------------------------


class TestGetBenchmark:
    def test_pointer_delegates_to_job_aggregation(self, runs_dir, monkeypatch):
        """Pointer benchmark delegates to get_job_benchmark for detail."""
        bm_id = "bm_detail_test"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_det")
        _write_manifest(runs_dir, bm_id, pointer)

        mock_data = _make_job_benchmark_data(
            bm_id,
            pass_threshold=0.7,
            by_category={},
            by_difficulty={},
            by_env={},
            test_cases=[],
            stages=[],
            start_from_stage=1,
            max_iterations=5,
        )
        monkeypatch.setattr(
            "p2p.scheduler.benchmark_aggregation.get_job_benchmark",
            lambda job_id: mock_data,
        )

        result = get_benchmark(bm_id)
        assert result is not None
        assert result.benchmark_id == bm_id
        assert result.total_cases == 2

    def test_pointer_returns_none_when_job_missing(self, runs_dir, monkeypatch):
        """Pointer benchmark returns None if job manifest is gone."""
        bm_id = "bm_detail_gone"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_gone")
        _write_manifest(runs_dir, bm_id, pointer)

        monkeypatch.setattr(
            "p2p.scheduler.benchmark_aggregation.get_job_benchmark",
            lambda job_id: None,
        )

        assert get_benchmark(bm_id) is None

    def test_legacy_benchmark_builds_detail(self, runs_dir, stub_legacy_deps):
        """Legacy benchmark builds detail from manifest + session info."""
        manifest = _make_legacy_manifest(with_loop_config=True)
        _write_manifest(runs_dir, manifest["benchmark_id"], manifest)

        result = get_benchmark(manifest["benchmark_id"])
        assert result is not None
        assert result.benchmark_id == manifest["benchmark_id"]
        assert result.total_cases == 1
        assert result.completed_cases == 1
        assert result.test_cases[0].best_score == 0.8

    def test_nonexistent_returns_none(self, runs_dir):
        assert get_benchmark("nonexistent_id") is None


# ---------------------------------------------------------------------------
# stop_benchmark
# ---------------------------------------------------------------------------


class TestStopBenchmark:
    def test_pointer_delegates_to_cancel_job(self, runs_dir, monkeypatch):
        """Pointer benchmark cancels via cancel_job and updates pointer status."""
        bm_id = "bm_stop_test"
        pointer = _make_pointer_manifest(benchmark_id=bm_id, job_id="job_stop")
        _write_manifest(runs_dir, bm_id, pointer)

        cancelled_jobs = []
        monkeypatch.setattr(
            "p2p.scheduler.job_queries.cancel_job",
            lambda job_id: cancelled_jobs.append(job_id),
        )

        stopped, count = stop_benchmark(bm_id)
        assert stopped is True
        assert count == 0
        assert cancelled_jobs == ["job_stop"]

        # Pointer manifest should be updated to cancelled
        updated = json.loads((runs_dir / bm_id / "benchmark.json").read_text())
        assert updated["status"] == "cancelled"

    def test_legacy_kills_scheduler_and_stops_sessions(self, runs_dir, monkeypatch):
        """Legacy benchmark kills scheduler PID and stops individual sessions."""
        manifest = _make_legacy_manifest(scheduler_pid=12345, status="running")
        _write_manifest(runs_dir, manifest["benchmark_id"], manifest)

        killed_pids = []
        stopped_sessions = []

        monkeypatch.setattr(
            "p2p.api.benchmark_service._kill_scheduler",
            lambda m: killed_pids.append(m.get("scheduler_pid")),
        )

        def mock_stop_session(sid):
            stopped_sessions.append(sid)
            return True

        monkeypatch.setattr("p2p.api.benchmark_service.stop_session", mock_stop_session)

        stopped, count = stop_benchmark(manifest["benchmark_id"])
        assert stopped is True
        assert count == 1
        assert killed_pids == [12345]
        assert stopped_sessions == ["s0"]

        # Manifest should be updated
        updated = json.loads((runs_dir / manifest["benchmark_id"] / "benchmark.json").read_text())
        assert updated["status"] == "cancelled"
        assert "scheduler_pid" not in updated

    def test_nonexistent_returns_false(self, runs_dir):
        stopped, count = stop_benchmark("nonexistent_id")
        assert stopped is False
        assert count == 0
