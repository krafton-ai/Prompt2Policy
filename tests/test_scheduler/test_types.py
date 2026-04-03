"""Tests for scheduler type definitions."""

from p2p.scheduler.types import Job, JobManifest, NodeConfig, RunRecord, RunSpec, RunStatus


def test_run_spec_creation() -> None:
    spec: RunSpec = {
        "run_id": "run_abc",
        "entry_point": "p2p.session.run_session",
        "parameters": {"prompt": "test", "seed": 1},
        "cpu_cores": 2,
    }
    assert spec["run_id"] == "run_abc"
    assert spec["entry_point"] == "p2p.session.run_session"
    assert spec["cpu_cores"] == 2


def test_run_spec_with_tags() -> None:
    spec: RunSpec = {
        "run_id": "run_abc",
        "entry_point": "p2p.session.run_session",
        "parameters": {},
        "cpu_cores": 1,
        "tags": {"job_type": "benchmark", "stage": "1"},
    }
    assert spec["tags"]["job_type"] == "benchmark"


def test_run_status_minimal() -> None:
    status: RunStatus = {"run_id": "run_1", "state": "running"}
    assert status["state"] == "running"


def test_run_status_full() -> None:
    status: RunStatus = {
        "run_id": "run_1",
        "state": "completed",
        "pid": 12345,
        "exit_code": 0,
        "node_id": "node-1",
        "started_at": "2025-01-01T00:00:00Z",
        "completed_at": "2025-01-01T01:00:00Z",
    }
    assert status["exit_code"] == 0
    assert status["node_id"] == "node-1"


def test_node_config() -> None:
    config: NodeConfig = {
        "node_id": "gpu-1",
        "host": "10.0.0.1",
        "user": "researcher",
        "port": 22,
        "base_dir": "/home/researcher/p2p",
        "max_cores": 60,
    }
    assert config["max_cores"] == 60


def test_job() -> None:
    job: Job = {
        "job_id": "job_abc",
        "job_type": "benchmark",
        "run_ids": ["run_1", "run_2"],
        "status": "running",
        "created_at": "2025-01-01T00:00:00Z",
    }
    assert len(job["run_ids"]) == 2
    assert job["job_type"] == "benchmark"


def test_run_record() -> None:
    record: RunRecord = {
        "run_id": "run_1",
        "spec": {
            "run_id": "run_1",
            "entry_point": "p2p.session.run_session",
            "parameters": {},
            "cpu_cores": 2,
        },
        "state": "pending",
        "node_id": "",
        "remote_dir": "",
        "synced": False,
    }
    assert record["state"] == "pending"
    assert record["synced"] is False


def test_job_manifest() -> None:
    manifest: JobManifest = {
        "job_id": "job_abc",
        "job_type": "session",
        "status": "running",
        "created_at": "2025-01-01T00:00:00Z",
        "backend": "local",
        "config": {"prompt": "test"},
        "runs": [
            {
                "run_id": "run_1",
                "spec": {
                    "run_id": "run_1",
                    "entry_point": "p2p.session.run_session",
                    "parameters": {},
                    "cpu_cores": 2,
                },
                "state": "pending",
                "node_id": "",
                "remote_dir": "",
                "synced": False,
            }
        ],
    }
    assert manifest["backend"] == "local"
    assert len(manifest["runs"]) == 1
