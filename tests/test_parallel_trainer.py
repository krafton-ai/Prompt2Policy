"""Tests for parallel_trainer — subprocess coordination and result aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from p2p.training.parallel_trainer import _aggregate, run_parallel_trainings

# ---------------------------------------------------------------------------
# _aggregate (pure function)
# ---------------------------------------------------------------------------


def _make_configs(n: int = 2) -> list[dict]:
    return [{"config_id": f"cfg_{i}", "label": f"config {i}", "params": {}} for i in range(n)]


def test_aggregate_single_config_single_seed():
    configs = _make_configs(1)
    seeds = [42]
    completed = {"cfg_0_seed_42": {"final_episodic_return": 100.0}}

    result = _aggregate(configs, seeds, completed)

    assert result["best_config_id"] == "cfg_0"
    assert result["best_run_id"] == "cfg_0_seed_42"
    assert result["configs"]["cfg_0"]["mean_final_return"] == 100.0
    assert result["configs"]["cfg_0"]["std_final_return"] == 0.0


def test_aggregate_multiple_configs_picks_best():
    configs = _make_configs(2)
    seeds = [1, 2]
    completed = {
        "cfg_0_seed_1": {"final_episodic_return": 100.0},
        "cfg_0_seed_2": {"final_episodic_return": 200.0},
        "cfg_1_seed_1": {"final_episodic_return": 500.0},
        "cfg_1_seed_2": {"final_episodic_return": 600.0},
    }

    result = _aggregate(configs, seeds, completed)

    assert result["best_config_id"] == "cfg_1"
    # Best individual run in best config
    assert result["best_run_id"] == "cfg_1_seed_2"
    assert result["configs"]["cfg_1"]["mean_final_return"] == 550.0


def test_aggregate_missing_run_defaults_to_zero():
    configs = _make_configs(1)
    seeds = [1, 2]
    completed = {"cfg_0_seed_1": {"final_episodic_return": 300.0}}
    # cfg_0_seed_2 is missing

    result = _aggregate(configs, seeds, completed)

    assert result["configs"]["cfg_0"]["mean_final_return"] == 150.0  # (300+0)/2


def test_aggregate_empty_completed():
    configs = _make_configs(1)
    seeds = [1]
    completed = {}

    result = _aggregate(configs, seeds, completed)

    assert result["configs"]["cfg_0"]["mean_final_return"] == 0.0


def test_aggregate_none_return_treated_as_zero():
    configs = _make_configs(1)
    seeds = [1]
    completed = {"cfg_0_seed_1": {"final_episodic_return": None}}

    result = _aggregate(configs, seeds, completed)

    assert result["configs"]["cfg_0"]["mean_final_return"] == 0.0


# ---------------------------------------------------------------------------
# run_parallel_trainings (subprocess mocked)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_cpu_manager():
    """Provide a CPUManager with 8 cores (2 reserved)."""
    from p2p.training.cpu_manager import CPUManager

    mgr = CPUManager(total_cores=8, reserved=0)
    with patch("p2p.training.parallel_trainer.get_cpu_manager", return_value=mgr):
        yield mgr


def _write_reward_fn(tmp_path: Path) -> Path:
    reward_path = tmp_path / "reward_fn_src.py"
    reward_path.write_text("def compute_reward(obs): return 1.0\n")
    return reward_path


def _fake_popen_factory(iteration_dir: Path, return_code: int = 0):
    """Return a Popen constructor that writes a summary.json on poll()."""

    def _fake_popen(cmd, **kwargs):
        # Extract iteration_id from --iteration-id argument
        iteration_id = None
        for i, arg in enumerate(cmd):
            if arg == "--iteration-id" and i + 1 < len(cmd):
                iteration_id = cmd[i + 1]
                break

        mock_proc = MagicMock()
        mock_proc.poll.return_value = return_code
        mock_proc.returncode = return_code

        # Write summary.json so IterationRecord.read_summary() works
        if iteration_id:
            run_dir = iteration_dir / iteration_id
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = {"final_episodic_return": 500.0}
            (run_dir / "summary.json").write_text(json.dumps(summary))

        return mock_proc

    return _fake_popen


def test_run_parallel_trainings_launches_all_runs(tmp_path, _mock_cpu_manager):
    configs = _make_configs(2)
    seeds = [1, 2]
    reward_path = _write_reward_fn(tmp_path)
    iteration_dir = tmp_path / "iter_1"

    fake_popen = _fake_popen_factory(iteration_dir)

    with patch(
        "p2p.training.parallel_trainer.subprocess.Popen", side_effect=fake_popen
    ) as mock_popen:
        result = run_parallel_trainings(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_path,
            base_config_dict={"num_envs": 1},
            iteration_dir=iteration_dir,
            max_parallel=4,
        )

    # 2 configs x 2 seeds = 4 launches
    assert mock_popen.call_count == 4
    assert result["best_config_id"] in ("cfg_0", "cfg_1")
    assert "configs" in result


def test_run_parallel_trainings_writes_config_json(tmp_path, _mock_cpu_manager):
    configs = [{"config_id": "fast", "label": "fast", "params": {"learning_rate": 1e-3}}]
    seeds = [42]
    reward_path = _write_reward_fn(tmp_path)
    iteration_dir = tmp_path / "iter_1"

    fake_popen = _fake_popen_factory(iteration_dir)

    with patch("p2p.training.parallel_trainer.subprocess.Popen", side_effect=fake_popen):
        run_parallel_trainings(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_path,
            base_config_dict={"num_envs": 1, "total_timesteps": 100},
            iteration_dir=iteration_dir,
        )

    config_path = iteration_dir / "fast_seed_42" / "config.json"
    assert config_path.exists()
    config_data = json.loads(config_path.read_text())
    assert config_data["learning_rate"] == 1e-3
    assert config_data["seed"] == 42


def test_run_parallel_trainings_saves_aggregation_json(tmp_path, _mock_cpu_manager):
    configs = _make_configs(1)
    seeds = [1]
    reward_path = _write_reward_fn(tmp_path)
    iteration_dir = tmp_path / "iter_1"

    fake_popen = _fake_popen_factory(iteration_dir)

    with patch("p2p.training.parallel_trainer.subprocess.Popen", side_effect=fake_popen):
        run_parallel_trainings(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_path,
            base_config_dict={},
            iteration_dir=iteration_dir,
        )

    agg_path = iteration_dir / "aggregation.json"
    assert agg_path.exists()
    agg = json.loads(agg_path.read_text())
    assert "best_config_id" in agg
    assert "configs" in agg


def test_run_parallel_trainings_copies_reward_fn(tmp_path, _mock_cpu_manager):
    configs = _make_configs(1)
    seeds = [1]
    reward_path = _write_reward_fn(tmp_path)
    iteration_dir = tmp_path / "iter_1"

    fake_popen = _fake_popen_factory(iteration_dir)

    with patch("p2p.training.parallel_trainer.subprocess.Popen", side_effect=fake_popen):
        run_parallel_trainings(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_path,
            base_config_dict={},
            iteration_dir=iteration_dir,
        )

    dest = iteration_dir / "reward_fn.py"
    assert dest.exists()
    assert "compute_reward" in dest.read_text()
