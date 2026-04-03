"""Tests for scheduler API routes."""

import pytest
from fastapi.testclient import TestClient

from p2p.scheduler import node_store
from p2p.scheduler.manifest_io import set_jobs_dir


@pytest.fixture(autouse=True)
def _clean_state(tmp_path):
    node_store.set_store_path(tmp_path / "nodes.json")
    set_jobs_dir(tmp_path / "jobs")
    yield


@pytest.fixture()
def client():
    from p2p.api.app import create_app

    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Node endpoints
# ---------------------------------------------------------------------------


class TestNodeRoutes:
    def test_list_nodes_has_localhost_seed(self, client) -> None:
        resp = client.get("/api/scheduler/nodes")
        assert resp.status_code == 200
        nodes = resp.json()
        assert len(nodes) >= 1
        assert any(n["node_id"] == "localhost" for n in nodes)

    def test_add_node(self, client) -> None:
        resp = client.post(
            "/api/scheduler/nodes",
            json={
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "user",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["node_id"] == "n1"
        assert data["port"] == 22  # default

    def test_add_duplicate_node(self, client) -> None:
        payload = {"node_id": "n1", "host": "10.0.0.1", "user": "user"}
        client.post("/api/scheduler/nodes", json=payload)
        resp = client.post("/api/scheduler/nodes", json=payload)
        assert resp.status_code == 409

    def test_list_after_add(self, client) -> None:
        client.post(
            "/api/scheduler/nodes",
            json={
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "u",
            },
        )
        resp = client.get("/api/scheduler/nodes")
        assert any(n["node_id"] == "n1" for n in resp.json())

    def test_update_node(self, client) -> None:
        client.post(
            "/api/scheduler/nodes",
            json={
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "u",
            },
        )
        resp = client.put("/api/scheduler/nodes/n1", json={"host": "10.0.0.2"})
        assert resp.status_code == 200
        assert resp.json()["host"] == "10.0.0.2"

    def test_update_nonexistent(self, client) -> None:
        resp = client.put("/api/scheduler/nodes/nope", json={"host": "x"})
        assert resp.status_code == 404

    def test_update_empty_body(self, client) -> None:
        client.post(
            "/api/scheduler/nodes",
            json={
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "u",
            },
        )
        resp = client.put("/api/scheduler/nodes/n1", json={})
        assert resp.status_code == 400

    def test_remove_node(self, client) -> None:
        client.post(
            "/api/scheduler/nodes",
            json={
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "u",
            },
        )
        resp = client.delete("/api/scheduler/nodes/n1")
        assert resp.status_code == 200

        resp = client.get("/api/scheduler/nodes")
        assert all(n["node_id"] != "n1" for n in resp.json())

    def test_remove_nonexistent(self, client) -> None:
        resp = client.delete("/api/scheduler/nodes/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------


class TestJobRoutes:
    def test_list_jobs_empty(self, client) -> None:
        resp = client.get("/api/scheduler/jobs")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_get_nonexistent_job(self, client) -> None:
        resp = client.get("/api/scheduler/jobs/nonexistent")
        assert resp.status_code == 404

    def test_cancel_nonexistent_job(self, client) -> None:
        resp = client.post("/api/scheduler/jobs/nonexistent/cancel")
        assert resp.status_code == 404

    def test_get_benchmark_nonexistent(self, client) -> None:
        resp = client.get("/api/scheduler/jobs/nonexistent/benchmark")
        assert resp.status_code == 404

    def test_get_benchmark_from_manifest(self, client, tmp_path) -> None:
        """Benchmark endpoint returns aggregated data from job manifest."""
        from p2p.scheduler.manifest_io import write_job_manifest
        from p2p.scheduler.types import JobManifest, RunRecord, RunSpec

        job_id = "job_test_bm"
        bm_id = "bm_test123"
        run_spec = RunSpec(
            run_id=f"{bm_id}_case0_s1",
            entry_point="p2p.session.run_session",
            parameters={"session_id": f"{bm_id}_case0_s1", "env_id": "Ant-v5"},
            cpu_cores=2,
            tags={
                "job_type": "benchmark",
                "benchmark_id": bm_id,
                "case_index": "0",
                "env_id": "Ant-v5",
                "config_id": "cfg_test",
                "seed": "1",
            },
        )
        manifest: JobManifest = {
            "job_id": job_id,
            "job_type": "benchmark",
            "status": "running",
            "created_at": "2026-01-01T00:00:00+00:00",
            "backend": "local",
            "config": {"mode": "flat", "pass_threshold": 0.7, "max_iterations": 5},
            "runs": [
                RunRecord(
                    run_id=run_spec["run_id"],
                    spec=run_spec,
                    state="pending",
                    node_id="",
                    remote_dir="",
                    synced=False,
                ),
            ],
            "metadata": {
                "benchmark_id": bm_id,
                "mode": "flat",
                "total_cases": 1,
                "total_runs": 1,
                "gate_threshold": 0.7,
                "test_cases": [
                    {
                        "index": 0,
                        "env_id": "Ant-v5",
                        "instruction": "Walk forward",
                        "category": "locomotion",
                        "difficulty": "easy",
                    },
                ],
            },
        }
        write_job_manifest(manifest)

        resp = client.get(f"/api/scheduler/jobs/{job_id}/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark_id"] == bm_id
        assert data["total_cases"] == 1
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["env_id"] == "Ant-v5"
        assert data["test_cases"][0]["category"] == "locomotion"
        assert data["mode"] == "flat"


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


class TestRunRoutes:
    def test_list_runs_empty(self, client) -> None:
        resp = client.get("/api/scheduler/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_nonexistent_run(self, client) -> None:
        resp = client.get("/api/scheduler/runs/nonexistent")
        assert resp.status_code == 404
