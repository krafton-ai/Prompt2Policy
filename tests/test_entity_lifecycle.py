"""Characterization tests for entity_lifecycle module.

Records current behavior as a safety net for future refactoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2p.api.entity_lifecycle import (
    delete_benchmark,
    delete_session,
    hard_delete_entity,
    inject_metadata,
    is_entity_deleted,
    list_trash,
    read_entity_metadata,
    restore_benchmark,
    restore_entity,
    restore_session,
    soft_delete_entity,
    update_benchmark_metadata,
    update_entity_metadata,
    update_session_metadata,
)
from p2p.scheduler import manifest_io
from p2p.scheduler.manifest_io import set_jobs_dir


@pytest.fixture()
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch RUNS_DIR to a temporary directory."""
    monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
    monkeypatch.setattr("p2p.api.entity_lifecycle.RUNS_DIR", tmp_path)
    original_jobs_dir = manifest_io._JOBS_DIR
    set_jobs_dir(tmp_path / "scheduler" / "jobs")
    yield tmp_path
    set_jobs_dir(original_jobs_dir)


@pytest.fixture()
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch _jobs_dir to a temporary directory."""
    jdir = tmp_path / "jobs"
    jdir.mkdir()
    monkeypatch.setattr("p2p.api.entity_lifecycle._jobs_dir", lambda: jdir)
    return jdir


def _write_meta(entity_dir: Path, meta: dict) -> None:
    entity_dir.mkdir(parents=True, exist_ok=True)
    (entity_dir / "metadata.json").write_text(json.dumps(meta))


def _write_status(entity_dir: Path, status: str) -> None:
    entity_dir.mkdir(parents=True, exist_ok=True)
    (entity_dir / "status.json").write_text(json.dumps({"status": status}))


def _read_meta(entity_dir: Path) -> dict:
    return json.loads((entity_dir / "metadata.json").read_text())


# ---------------------------------------------------------------------------
# read_entity_metadata
# ---------------------------------------------------------------------------


class TestReadEntityMetadata:
    def test_returns_metadata_when_exists(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "test", "starred": True})
        result = read_entity_metadata(tmp_path)
        assert result == {"alias": "test", "starred": True}

    def test_returns_empty_dict_when_no_file(self, tmp_path: Path) -> None:
        assert read_entity_metadata(tmp_path) == {}

    def test_returns_empty_dict_when_empty_file(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "metadata.json").write_text("")
        assert read_entity_metadata(tmp_path) == {}

    def test_returns_empty_dict_when_invalid_json(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "metadata.json").write_text("{bad json")
        assert read_entity_metadata(tmp_path) == {}


# ---------------------------------------------------------------------------
# inject_metadata
# ---------------------------------------------------------------------------


class TestInjectMetadata:
    def test_inject_into_dict_with_metadata(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "my-run", "starred": True, "tags": ["a"]})
        model: dict = {}
        inject_metadata(model, tmp_path)
        assert model["alias"] == "my-run"
        assert model["starred"] is True
        assert model["tags"] == ["a"]

    def test_inject_into_dict_without_metadata(self, tmp_path: Path) -> None:
        model: dict = {}
        inject_metadata(model, tmp_path)
        assert model["alias"] == ""
        assert model["starred"] is False
        assert model["tags"] == []

    def test_inject_into_object(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "obj-alias", "starred": True, "tags": ["x"]})

        class FakeModel:
            pass

        model = FakeModel()
        inject_metadata(model, tmp_path)
        assert model.alias == "obj-alias"  # type: ignore[attr-defined]
        assert model.starred is True  # type: ignore[attr-defined]
        assert model.tags == ["x"]  # type: ignore[attr-defined]

    def test_inject_into_object_without_metadata(self, tmp_path: Path) -> None:
        class FakeModel:
            pass

        model = FakeModel()
        inject_metadata(model, tmp_path)
        assert model.alias == ""  # type: ignore[attr-defined]
        assert model.starred is False  # type: ignore[attr-defined]
        assert model.tags == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# is_entity_deleted
# ---------------------------------------------------------------------------


class TestIsEntityDeleted:
    def test_returns_true_when_deleted(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"deleted_at": "2024-01-01T00:00:00+00:00"})
        assert is_entity_deleted(tmp_path) is True

    def test_returns_false_when_not_deleted(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "ok"})
        assert is_entity_deleted(tmp_path) is False

    def test_returns_false_when_no_metadata(self, tmp_path: Path) -> None:
        assert is_entity_deleted(tmp_path) is False


# ---------------------------------------------------------------------------
# update_entity_metadata
# ---------------------------------------------------------------------------


class TestUpdateEntityMetadata:
    def test_sets_alias(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {})
        result = update_entity_metadata(tmp_path, alias="new-alias")
        assert result["alias"] == "new-alias"
        assert _read_meta(tmp_path)["alias"] == "new-alias"

    def test_sets_starred(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {})
        result = update_entity_metadata(tmp_path, starred=True)
        assert result["starred"] is True

    def test_sets_tags(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {})
        result = update_entity_metadata(tmp_path, tags=["a", "b"])
        assert result["tags"] == ["a", "b"]

    def test_merges_with_existing(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "old", "starred": False})
        result = update_entity_metadata(tmp_path, starred=True)
        assert result["alias"] == "old"
        assert result["starred"] is True

    def test_none_values_are_ignored(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "keep"})
        result = update_entity_metadata(tmp_path, alias=None)
        assert result["alias"] == "keep"

    def test_creates_metadata_file_if_missing(self, tmp_path: Path) -> None:
        result = update_entity_metadata(tmp_path, alias="created")
        assert result["alias"] == "created"
        assert (tmp_path / "metadata.json").exists()


# ---------------------------------------------------------------------------
# soft_delete_entity / restore_entity
# ---------------------------------------------------------------------------


class TestSoftDeleteEntity:
    def test_sets_deleted_at(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "to-delete"})
        result = soft_delete_entity(tmp_path)
        assert result is True
        meta = _read_meta(tmp_path)
        assert "deleted_at" in meta
        assert meta["alias"] == "to-delete"

    def test_overwrites_deleted_at_on_repeat(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"deleted_at": "2024-01-01T00:00:00+00:00"})
        soft_delete_entity(tmp_path)
        meta = _read_meta(tmp_path)
        assert meta["deleted_at"] != "2024-01-01T00:00:00+00:00"


class TestRestoreEntity:
    def test_removes_deleted_at(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"deleted_at": "2024-01-01", "alias": "restored"})
        result = restore_entity(tmp_path)
        assert result is True
        meta = _read_meta(tmp_path)
        assert "deleted_at" not in meta
        assert meta["alias"] == "restored"

    def test_no_op_if_not_deleted(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"alias": "ok"})
        restore_entity(tmp_path)
        meta = _read_meta(tmp_path)
        assert "deleted_at" not in meta
        assert meta["alias"] == "ok"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestUpdateSessionMetadata:
    def test_updates_metadata(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_001"
        _write_meta(session_dir, {})
        result = update_session_metadata("session_001", alias="sess-alias")
        assert result["alias"] == "sess-alias"

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Session"):
            update_session_metadata("nonexistent", alias="x")


class TestDeleteSession:
    def test_soft_deletes(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_002"
        _write_meta(session_dir, {})
        _write_status(session_dir, "completed")
        delete_session("session_002")
        assert "deleted_at" in _read_meta(session_dir)

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Session"):
            delete_session("nonexistent")

    def test_raises_when_running(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_003"
        _write_meta(session_dir, {})
        _write_status(session_dir, "running")
        with pytest.raises(ValueError, match="running"):
            delete_session("session_003")

    def test_raises_when_pending(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_004"
        _write_meta(session_dir, {})
        _write_status(session_dir, "pending")
        with pytest.raises(ValueError, match="running"):
            delete_session("session_004")

    def test_deletes_when_no_status_json(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_006"
        _write_meta(session_dir, {})
        delete_session("session_006")
        assert "deleted_at" in _read_meta(session_dir)


class TestRestoreSession:
    def test_restores(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_005"
        _write_meta(session_dir, {"deleted_at": "2024-01-01"})
        restore_session("session_005")
        assert "deleted_at" not in _read_meta(session_dir)

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Session"):
            restore_session("nonexistent")


# ---------------------------------------------------------------------------
# Benchmark lifecycle
# ---------------------------------------------------------------------------


class TestUpdateBenchmarkMetadata:
    def test_updates_metadata(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_001"
        _write_meta(bench_dir, {})
        (bench_dir / "benchmark.json").write_text(json.dumps({"status": "done"}))
        result = update_benchmark_metadata("benchmark_001", alias="bench-alias")
        assert result["alias"] == "bench-alias"

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Benchmark"):
            update_benchmark_metadata("nonexistent", alias="x")


class TestDeleteBenchmark:
    def test_soft_deletes(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_002"
        _write_meta(bench_dir, {})
        _write_status(bench_dir, "completed")
        (bench_dir / "benchmark.json").write_text(json.dumps({"status": "done"}))
        delete_benchmark("benchmark_002")
        assert "deleted_at" in _read_meta(bench_dir)

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Benchmark"):
            delete_benchmark("nonexistent")

    def test_raises_when_running(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_003"
        _write_meta(bench_dir, {})
        _write_status(bench_dir, "running")
        with pytest.raises(ValueError, match="running"):
            delete_benchmark("benchmark_003")

    def test_deletes_when_no_status_json(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_005"
        _write_meta(bench_dir, {})
        delete_benchmark("benchmark_005")
        assert "deleted_at" in _read_meta(bench_dir)


class TestRestoreBenchmark:
    def test_restores(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_004"
        _write_meta(bench_dir, {"deleted_at": "2024-01-01"})
        (bench_dir / "benchmark.json").write_text(json.dumps({"status": "done"}))
        restore_benchmark("benchmark_004")
        assert "deleted_at" not in _read_meta(bench_dir)

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Benchmark"):
            restore_benchmark("nonexistent")


# ---------------------------------------------------------------------------
# hard_delete_entity
# ---------------------------------------------------------------------------


class TestHardDeleteEntity:
    def test_permanently_deletes(self, runs_dir: Path) -> None:
        entity_dir = runs_dir / "session_del"
        _write_meta(entity_dir, {"deleted_at": "2024-01-01"})
        hard_delete_entity("session_del")
        assert not entity_dir.exists()

    def test_raises_when_not_found(self, runs_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Entity"):
            hard_delete_entity("nonexistent")

    def test_raises_when_not_in_trash(self, runs_dir: Path) -> None:
        entity_dir = runs_dir / "session_active"
        _write_meta(entity_dir, {"alias": "active"})
        with pytest.raises(ValueError, match="trash"):
            hard_delete_entity("session_active")

    def test_raises_when_running(self, runs_dir: Path) -> None:
        entity_dir = runs_dir / "session_running"
        _write_meta(entity_dir, {"deleted_at": "2024-01-01"})
        _write_status(entity_dir, "running")
        with pytest.raises(ValueError, match="running"):
            hard_delete_entity("session_running")

    def test_job_hard_delete_cascades_benchmark(self, runs_dir: Path, jobs_dir: Path) -> None:
        """Hard-deleting a job also removes its linked benchmark pointer dir."""
        # Create job directory in jobs_dir with manifest referencing a benchmark
        job_dir = jobs_dir / "job_abc123"
        job_dir.mkdir()
        _write_meta(job_dir, {"deleted_at": "2024-01-01"})
        manifest = {"metadata": {"benchmark_id": "bm_linked"}}
        (job_dir / "manifest.json").write_text(json.dumps(manifest))

        # Create the benchmark pointer directory in runs_dir
        bm_dir = runs_dir / "bm_linked"
        bm_dir.mkdir()
        (bm_dir / "benchmark.json").write_text(json.dumps({"type": "pointer"}))

        hard_delete_entity("job_abc123")

        assert not job_dir.exists()
        assert not bm_dir.exists()

    def test_job_hard_delete_without_benchmark(self, runs_dir: Path, jobs_dir: Path) -> None:
        """Hard-deleting a job works even if no benchmark pointer exists."""
        job_dir = jobs_dir / "job_no_bm"
        job_dir.mkdir()
        _write_meta(job_dir, {"deleted_at": "2024-01-01"})
        (job_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        hard_delete_entity("job_no_bm")

        assert not job_dir.exists()

    def test_job_hard_delete_missing_benchmark_dir(self, runs_dir: Path, jobs_dir: Path) -> None:
        """Hard-deleting a job succeeds even if linked benchmark dir is already gone."""
        job_dir = jobs_dir / "job_gone_bm"
        job_dir.mkdir()
        _write_meta(job_dir, {"deleted_at": "2024-01-01"})
        manifest = {"metadata": {"benchmark_id": "bm_already_gone"}}
        (job_dir / "manifest.json").write_text(json.dumps(manifest))

        hard_delete_entity("job_gone_bm")

        assert not job_dir.exists()

    def test_job_hard_delete_skips_running_benchmark(self, runs_dir: Path, jobs_dir: Path) -> None:
        """Benchmark dir is NOT removed if it is still running."""
        job_dir = jobs_dir / "job_running_bm"
        job_dir.mkdir()
        _write_meta(job_dir, {"deleted_at": "2024-01-01"})
        manifest = {"metadata": {"benchmark_id": "bm_running"}}
        (job_dir / "manifest.json").write_text(json.dumps(manifest))

        bm_dir = runs_dir / "bm_running"
        bm_dir.mkdir()
        (bm_dir / "benchmark.json").write_text(json.dumps({"type": "pointer"}))
        _write_status(bm_dir, "running")

        hard_delete_entity("job_running_bm")

        assert not job_dir.exists()
        assert bm_dir.exists()  # benchmark still running, not deleted


# ---------------------------------------------------------------------------
# list_trash
# ---------------------------------------------------------------------------


class TestListTrash:
    def test_returns_empty_when_no_runs_dir(
        self, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("p2p.api.entity_lifecycle.RUNS_DIR", runs_dir / "nonexistent")
        assert list_trash() == []

    def test_lists_deleted_session(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_trash1"
        _write_meta(session_dir, {"deleted_at": "2024-06-01T00:00:00+00:00", "alias": "trashed"})
        _write_status(session_dir, "completed")
        (session_dir / "loop_history.json").write_text(json.dumps({"prompt": "test prompt"}))

        items = list_trash()
        assert len(items) == 1
        assert items[0]["entity_id"] == "session_trash1"
        assert items[0]["entity_type"] == "session"
        assert items[0]["alias"] == "trashed"
        assert items[0]["prompt"] == "test prompt"
        assert items[0]["status"] == "completed"

    def test_lists_deleted_benchmark(self, runs_dir: Path) -> None:
        bench_dir = runs_dir / "benchmark_trash1"
        _write_meta(bench_dir, {"deleted_at": "2024-06-01T00:00:00+00:00"})
        (bench_dir / "benchmark.json").write_text(
            json.dumps({"status": "done", "created_at": "2024-04-01"})
        )

        items = list_trash()
        assert len(items) == 1
        assert items[0]["entity_type"] == "benchmark"
        assert items[0]["status"] == "done"

    def test_excludes_non_deleted_entities(self, runs_dir: Path) -> None:
        session_dir = runs_dir / "session_active"
        _write_meta(session_dir, {"alias": "active"})
        assert list_trash() == []

    def test_sorted_by_deleted_at_descending(self, runs_dir: Path) -> None:
        s1 = runs_dir / "session_old"
        _write_meta(s1, {"deleted_at": "2024-01-01T00:00:00+00:00"})
        (s1 / "loop_history.json").write_text("{}")

        s2 = runs_dir / "session_new"
        _write_meta(s2, {"deleted_at": "2024-06-01T00:00:00+00:00"})
        (s2 / "loop_history.json").write_text("{}")

        items = list_trash()
        assert len(items) == 2
        assert items[0]["entity_id"] == "session_new"
        assert items[1]["entity_id"] == "session_old"

    def test_skips_unknown_entity_type(self, runs_dir: Path) -> None:
        unknown_dir = runs_dir / "unknown_entity"
        _write_meta(unknown_dir, {"deleted_at": "2024-01-01T00:00:00+00:00"})
        assert list_trash() == []

    def test_skips_files_in_runs_dir(self, runs_dir: Path) -> None:
        (runs_dir / "some_file.txt").write_text("not a dir")
        assert list_trash() == []

    def test_session_detected_by_loop_history(self, runs_dir: Path) -> None:
        """Dir without session_ prefix detected as session via loop_history.json."""
        d = runs_dir / "custom_name"
        _write_meta(d, {"deleted_at": "2024-01-01T00:00:00+00:00"})
        (d / "loop_history.json").write_text(json.dumps({"prompt": "p", "status": "done"}))

        items = list_trash()
        assert len(items) == 1
        assert items[0]["entity_type"] == "session"
        assert items[0]["prompt"] == "p"
