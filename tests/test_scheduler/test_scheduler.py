"""Tests for manifest I/O and job scheduler types."""

import pytest

from p2p.scheduler.manifest_io import (
    list_job_ids,
    read_job_manifest,
    set_jobs_dir,
    write_job_manifest,
)
from p2p.scheduler.types import JobManifest, RunRecord, now_iso


@pytest.fixture(autouse=True)
def _tmp_jobs_dir(tmp_path):
    set_jobs_dir(tmp_path / "jobs")


def _make_manifest(job_id: str = "job_test1") -> JobManifest:
    run: RunRecord = {
        "run_id": "run_abc",
        "spec": {
            "run_id": "run_abc",
            "entry_point": "p2p.session.run_session",
            "parameters": {"session_id": "run_abc", "prompt": "test"},
            "cpu_cores": 2,
        },
        "state": "pending",
        "node_id": "",
        "remote_dir": "",
        "synced": False,
    }
    return {
        "job_id": job_id,
        "job_type": "session",
        "status": "running",
        "created_at": now_iso(),
        "backend": "local",
        "config": {"prompt": "test"},
        "runs": [run],
    }


class TestManifestIO:
    def test_write_and_read(self) -> None:
        manifest = _make_manifest()
        write_job_manifest(manifest)

        loaded = read_job_manifest("job_test1")
        assert loaded is not None
        assert loaded["job_id"] == "job_test1"
        assert len(loaded["runs"]) == 1

    def test_read_nonexistent(self) -> None:
        assert read_job_manifest("nonexistent") is None

    def test_list_job_ids(self) -> None:
        write_job_manifest(_make_manifest("job_a"))
        write_job_manifest(_make_manifest("job_b"))

        ids = list_job_ids()
        assert set(ids) == {"job_a", "job_b"}

    def test_list_empty(self) -> None:
        assert list_job_ids() == []

    def test_atomic_write(self, tmp_path) -> None:
        """Verify no .tmp files are left behind."""
        set_jobs_dir(tmp_path / "jobs2")
        manifest = _make_manifest("job_atomic")
        write_job_manifest(manifest)

        job_dir = tmp_path / "jobs2" / "job_atomic"
        files = list(job_dir.iterdir())
        assert all(f.suffix != ".tmp" for f in files)

    def test_overwrite_manifest(self) -> None:
        manifest = _make_manifest()
        write_job_manifest(manifest)

        manifest["status"] = "completed"
        write_job_manifest(manifest)

        loaded = read_job_manifest("job_test1")
        assert loaded is not None
        assert loaded["status"] == "completed"
