"""Tests for job controllers."""

from unittest.mock import patch

import pytest

from p2p.config import LoopConfig, TrainConfig
from p2p.scheduler import node_store
from p2p.scheduler.controllers import (
    BenchmarkController,
    SessionController,
)
from p2p.scheduler.job_queries import get_job, list_jobs
from p2p.scheduler.manifest_io import read_job_manifest, set_jobs_dir
from p2p.scheduler.ssh_utils import resolve_node


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
    """Build a minimal LoopConfig for controller tests."""
    return LoopConfig(train=TrainConfig(total_timesteps=1000), **overrides)


class TestSessionController:
    def test_run_creates_manifest(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(prompt="test prompt", loop_config=_lc(), backend="local")

        assert job["job_type"] == "session"
        assert job["status"] == "running"
        assert len(job["run_ids"]) == 1

        # Verify manifest was written
        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None
        assert manifest["backend"] == "local"
        assert len(manifest["runs"]) == 1
        assert manifest["runs"][0]["state"] == "pending"

    def test_job_is_queryable(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(prompt="test", loop_config=_lc(), backend="local")

        fetched = get_job(job["job_id"])
        assert fetched is not None
        assert fetched["job_id"] == job["job_id"]

    def test_jobs_appear_in_list(self, _mock_spawn) -> None:
        ctrl = SessionController()
        ctrl.run(prompt="test1", loop_config=_lc(), backend="local")
        ctrl.run(prompt="test2", loop_config=_lc(), backend="local")

        jobs = list_jobs()
        assert len(jobs) == 2

    def test_spawn_is_called(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(prompt="test", loop_config=_lc(), backend="local")
        _mock_spawn.assert_called_once_with(job["job_id"])

    def test_multi_config_multi_seed(self, _mock_spawn) -> None:
        ctrl = SessionController()
        configs = [
            {"config_id": "c1", "params": {"lr": 1e-3}},
            {"config_id": "c2", "params": {"lr": 3e-4}},
        ]
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            configs=configs,
            seeds=[1, 2, 3],
            backend="ssh",
        )

        # Bundled: 1 RunSpec with all configs/seeds embedded in LoopConfig
        assert len(job["run_ids"]) == 1
        assert job["metadata"]["total_runs"] == 1

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None
        assert len(manifest["runs"]) == 1

        # Verify LoopConfig contains configs and seeds
        import json

        spec = manifest["runs"][0]["spec"]
        lc_data = json.loads(spec["parameters"]["loop_config"])
        assert len(lc_data["configs"]) == 2
        assert lc_data["seeds"] == [1, 2, 3]

        # CPU should reflect full matrix: 2 configs × 3 seeds × per-run cores
        assert spec["cpu_cores"] == 6 * max(2, _lc().train.num_envs)

    def test_single_config_single_seed(self, _mock_spawn) -> None:
        ctrl = SessionController()
        lc = LoopConfig(train=TrainConfig(total_timesteps=1000, seed=42))
        job = ctrl.run(prompt="test", loop_config=lc, backend="local")

        assert len(job["run_ids"]) == 1

    def test_state_counts_in_metadata(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            configs=[{"config_id": "c1"}, {"config_id": "c2"}],
            seeds=[1, 2],
            backend="ssh",
        )

        # Bundled: 1 run containing all configs/seeds
        meta = job["metadata"]
        assert meta["state_counts"] == {"pending": 1}
        assert meta["total_runs"] == 1

    def test_node_allocation_unassigned(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(prompt="test", loop_config=_lc(), seeds=[1, 2, 3], backend="ssh")

        # Bundled: 1 run, pending with empty node_id -> "unassigned"
        alloc = job["metadata"]["node_allocation"]
        assert "unassigned" in alloc
        assert alloc["unassigned"]["total"] == 1
        assert alloc["unassigned"]["pending"] == 1

    def test_explicit_session_id_in_run_spec(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            backend="local",
            session_id="session_20260313_custom_id",
        )

        manifest = read_job_manifest(job["job_id"])
        spec = manifest["runs"][0]["spec"]
        assert spec["run_id"] == "session_20260313_custom_id"
        assert spec["parameters"]["session_id"] == "session_20260313_custom_id"

    def test_session_affinity_detected(self, _mock_spawn) -> None:
        ctrl = SessionController()
        job = ctrl.run(
            prompt="test",
            loop_config=_lc(),
            configs=[{"config_id": "c1"}, {"config_id": "c2"}],
            seeds=[1],
            backend="ssh",
        )

        # Bundled: 1 run with a session_group → affinity is True
        assert job["metadata"]["session_affinity"] is True
        # No node assigned yet
        assert job["metadata"]["affinity_node"] is None


class TestBenchmarkController:
    def test_run_creates_manifest(self, _mock_spawn) -> None:
        ctrl = BenchmarkController()
        test_cases = [
            {"env_id": "HalfCheetah-v5", "instruction": "Run fast"},
            {"env_id": "Ant-v5", "instruction": "Walk forward"},
        ]

        job = ctrl.run(
            test_cases=test_cases,
            loop_config=_lc(),
            mode="flat",
            backend="local",
        )

        assert job["job_type"] == "benchmark"
        assert job["metadata"]["mode"] == "flat"
        assert job["metadata"]["total_cases"] == 2

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None
        assert len(manifest["runs"]) == 2
        assert all(r["state"] == "pending" for r in manifest["runs"])

    def test_spawn_is_called(self, _mock_spawn) -> None:
        ctrl = BenchmarkController()
        test_cases = [{"env_id": "HalfCheetah-v5", "instruction": "test"}]
        job = ctrl.run(test_cases=test_cases, loop_config=_lc(), mode="flat", backend="local")
        _mock_spawn.assert_called_once_with(job["job_id"])

    def test_multi_seed_per_test_case(self, _mock_spawn) -> None:
        ctrl = BenchmarkController()
        test_cases = [
            {"env_id": "HalfCheetah-v5", "instruction": "Run"},
            {"env_id": "Ant-v5", "instruction": "Walk"},
        ]

        job = ctrl.run(
            test_cases=test_cases,
            loop_config=_lc(),
            seeds=[1, 2],
            mode="flat",
            backend="ssh",
        )

        # Bundled: 1 run per test case = 2 runs total
        assert len(job["run_ids"]) == 2
        assert job["metadata"]["total_runs"] == 2

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None

        # Each test case should have its own session_group
        groups = {r["session_group"] for r in manifest["runs"]}
        assert len(groups) == 2  # 2 test cases = 2 groups

        # Verify seeds are embedded in LoopConfig
        import json

        for run in manifest["runs"]:
            lc_data = json.loads(run["spec"]["parameters"]["loop_config"])
            assert lc_data["seeds"] == [1, 2]

    def test_single_seed_per_test_case(self, _mock_spawn) -> None:
        ctrl = BenchmarkController()
        test_cases = [
            {"env_id": "HalfCheetah-v5", "instruction": "Run"},
        ]

        job = ctrl.run(
            test_cases=test_cases,
            loop_config=_lc(),
            mode="flat",
            backend="local",
        )

        # 1 test case × 1 seed = 1 run
        assert len(job["run_ids"]) == 1

    def test_staged_mode_creates_stages_and_tags(self, _mock_spawn, tmp_path) -> None:
        import json

        from p2p.settings import RUNS_DIR

        ctrl = BenchmarkController()
        test_cases = [
            {"env_id": "HalfCheetah-v5", "instruction": "Run", "difficulty": "easy"},
            {"env_id": "Ant-v5", "instruction": "Walk", "difficulty": "medium"},
            {"env_id": "Hopper-v5", "instruction": "Jump", "difficulty": "hard"},
        ]

        job = ctrl.run(
            test_cases=test_cases,
            loop_config=_lc(),
            mode="staged",
            num_stages=2,
            gate_threshold=0.7,
            backend="ssh",
        )

        manifest = read_job_manifest(job["job_id"])
        assert manifest is not None

        # Metadata should contain stages
        metadata = manifest["metadata"]
        assert metadata["mode"] == "staged"
        assert metadata["total_stages"] > 0
        assert len(metadata["stages"]) > 0

        # Each run should have a stage tag
        for run in manifest["runs"]:
            assert "stage" in run["spec"]["tags"]

        # All case indices across stages should cover all test cases
        all_indices = []
        for s in metadata["stages"]:
            all_indices.extend(s["case_indices"])
        assert sorted(all_indices) == list(range(3))

        # Pointer file should exist
        bm_id = metadata["benchmark_id"]
        pointer_path = RUNS_DIR / bm_id / "benchmark.json"
        assert pointer_path.exists()
        pointer = json.loads(pointer_path.read_text())
        assert pointer["type"] == "pointer"
        assert pointer["job_id"] == job["job_id"]
        assert pointer["benchmark_id"] == bm_id

    def test_benchmark_no_affinity(self, _mock_spawn) -> None:
        ctrl = BenchmarkController()
        test_cases = [
            {"env_id": "HalfCheetah-v5", "instruction": "Run"},
            {"env_id": "Ant-v5", "instruction": "Walk"},
        ]

        job = ctrl.run(
            test_cases=test_cases,
            loop_config=_lc(),
            mode="flat",
            backend="ssh",
        )

        # Multiple case groups → no session affinity
        assert job["metadata"]["session_affinity"] is False
        assert job["metadata"]["affinity_node"] is None


def _node(node_id: str, host: str = "10.0.0.1", max_cores: int = 8) -> dict:
    return {
        "node_id": node_id,
        "host": host,
        "user": "u",
        "port": 22,
        "base_dir": "/tmp/p2p",
        "max_cores": max_cores,
    }


class TestResolveNodeSkipNodes:
    """Tests for resolve_node skip_nodes parameter."""

    @pytest.fixture(autouse=True)
    def _mock_load_nodes(self):
        """Isolate load_nodes so tests don't see real production nodes."""
        nodes: list[dict] = []

        def _add(n: dict) -> None:
            nodes.append(n)

        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=nodes):
            self._nodes = nodes
            self._add = _add
            yield

    def _setup_nodes(self):
        """Register three test nodes."""
        self._add(_node("n1", "10.0.0.1", max_cores=8))
        self._add(_node("n2", "10.0.0.2", max_cores=8))
        self._add(_node("n3", "10.0.0.3", max_cores=4))

    def test_skip_nodes_excludes_from_auto_assign(self):
        self._setup_nodes()
        used: dict[str, int] = {}
        # Without skip_nodes, n1 or n2 (both have 8 free) would be picked
        node = resolve_node(None, used, 1, skip_nodes={"n1", "n2"})
        assert node is not None
        assert node["node_id"] == "n3"

    def test_skip_nodes_all_skipped_returns_none(self):
        self._setup_nodes()
        used: dict[str, int] = {}
        node = resolve_node(None, used, 1, skip_nodes={"n1", "n2", "n3"})
        assert node is None

    def test_skip_nodes_ignored_for_explicit_hint(self):
        """Affinity hints bypass skip_nodes."""
        self._setup_nodes()
        used: dict[str, int] = {}
        node = resolve_node("n1", used, 1, skip_nodes={"n1"})
        assert node is not None
        assert node["node_id"] == "n1"

    def test_skip_nodes_none_has_no_effect(self):
        self._setup_nodes()
        used: dict[str, int] = {}
        node_a = resolve_node(None, used, 1)
        node_b = resolve_node(None, used, 1, skip_nodes=None)
        assert node_a is not None
        assert node_b is not None
        assert node_a["node_id"] == node_b["node_id"]


class TestFailedNodeCascade:
    """Tests that failed nodes are skipped via resolve_node skip_nodes."""

    def test_failed_node_skipped_next_run_goes_elsewhere(self):
        """When a node fails, resolve_node with skip_nodes excludes it."""
        from p2p.scheduler.ssh_utils import resolve_node

        test_nodes = [
            _node("good", "10.0.0.1", max_cores=8),
            _node("bad", "10.0.0.2", max_cores=16),
        ]

        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=test_nodes):
            # First call: "bad" has most free cores, gets selected
            node1 = resolve_node(None, {}, 1)
            assert node1 is not None
            assert node1["node_id"] == "bad"

            # Second call: "bad" in skip_nodes, should go to "good"
            node2 = resolve_node(None, {}, 1, skip_nodes={"bad"})
            assert node2 is not None
            assert node2["node_id"] == "good"

    def test_all_nodes_failed_returns_none(self):
        """When all nodes are in skip_nodes, resolve_node returns None."""
        from p2p.scheduler.ssh_utils import resolve_node

        test_nodes = [_node("n1", "10.0.0.1", max_cores=8)]

        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=test_nodes):
            node = resolve_node(None, {}, 1, skip_nodes={"n1"})

        assert node is None
