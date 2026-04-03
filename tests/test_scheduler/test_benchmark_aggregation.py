"""Tests for benchmark_aggregation module."""

from unittest.mock import patch

from p2p.scheduler.benchmark_aggregation import get_job_benchmark
from p2p.scheduler.manifest_io import set_jobs_dir, write_job_manifest


class TestGetJobBenchmarkSessionId:
    """Verify get_job_benchmark uses the best run's session_id, not runs[0]."""

    def test_session_id_matches_best_run(self, tmp_path) -> None:
        """When the best-scoring run is not runs[0], session_id should
        still point to the best run."""
        set_jobs_dir(tmp_path / "jobs")

        manifest = {
            "job_id": "bm-1",
            "job_type": "benchmark",
            "backend": "local",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00",
            "metadata": {
                "benchmark_id": "bm-1",
                "test_cases": [
                    {"index": 0, "env_id": "HalfCheetah-v5", "instruction": "run"},
                ],
            },
            "config": {"mode": "flat", "pass_threshold": 0.7, "max_iterations": 5},
            "runs": [
                {
                    "run_id": "run-low",
                    "state": "completed",
                    "node_id": "n1",
                    "spec": {"tags": {"case_index": "0"}},
                },
                {
                    "run_id": "run-high",
                    "state": "completed",
                    "node_id": "n2",
                    "spec": {"tags": {"case_index": "0"}},
                },
            ],
        }
        write_job_manifest(manifest)

        def _mock_info(session_id: str) -> dict | None:
            if session_id == "run-low":
                return {
                    "status": "passed",
                    "best_score": 0.3,
                    "iterations_completed": 2,
                    "is_stale": False,
                    "iteration_scores": [0.1, 0.3],
                }
            if session_id == "run-high":
                return {
                    "status": "passed",
                    "best_score": 0.9,
                    "iterations_completed": 3,
                    "is_stale": False,
                    "iteration_scores": [0.2, 0.5, 0.9],
                }
            return None

        with patch(
            "p2p.scheduler.benchmark_aggregation.lightweight_session_info",
            side_effect=_mock_info,
        ):
            result = get_job_benchmark("bm-1")

        assert result is not None
        cases = result["test_cases"]
        assert len(cases) == 1
        # The best run is "run-high" (score 0.9), not "run-low" (runs[0])
        assert cases[0]["session_id"] == "run-high"
        assert cases[0]["best_score"] == 0.9
