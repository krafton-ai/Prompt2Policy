"""Tests for the Trainer protocol and LocalTrainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from p2p.training.trainer import LocalTrainer, Trainer


def test_local_trainer_satisfies_protocol():
    """LocalTrainer must be a structural subtype of Trainer."""
    trainer = LocalTrainer()
    assert isinstance(trainer, Trainer)


def test_local_trainer_delegates_to_run_parallel_trainings(tmp_path: Path):
    """LocalTrainer.train() should forward to run_parallel_trainings with bound params."""
    configs = [{"config_id": "default", "label": "default", "params": {}}]
    seeds = [1]
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("def reward_fn(obs, action, next_obs, info): return 1.0, {}")
    iteration_dir = tmp_path / "iter_1"

    expected_agg = {
        "best_config_id": "default",
        "best_run_id": "default_seed_1",
        "configs": {},
    }

    trainer = LocalTrainer(cores_per_run=4, max_parallel=2, cores_pool=[0, 1, 2, 3])

    with patch(
        "p2p.training.parallel_trainer.run_parallel_trainings", return_value=expected_agg
    ) as mock:
        result = trainer.train(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_path,
            base_config_dict={"num_envs": 1},
            iteration_dir=iteration_dir,
            env_id="HalfCheetah-v5",
        )

    assert result == expected_agg
    mock.assert_called_once_with(
        configs=configs,
        seeds=seeds,
        reward_fn_path=reward_path,
        base_config_dict={"num_envs": 1},
        iteration_dir=iteration_dir,
        env_id="HalfCheetah-v5",
        cores_per_run=4,
        max_parallel=2,
        cores_pool=[0, 1, 2, 3],
        heartbeat_fn=None,
        no_cpu_affinity=False,
        gpu_pool=None,
    )


def test_local_trainer_passes_heartbeat_fn(tmp_path: Path):
    """LocalTrainer should pass heartbeat_fn to run_parallel_trainings."""
    heartbeat = MagicMock()
    trainer = LocalTrainer(heartbeat_fn=heartbeat)

    with patch("p2p.training.parallel_trainer.run_parallel_trainings", return_value={}) as mock:
        trainer.train(
            configs=[{"config_id": "a", "label": "a", "params": {}}],
            seeds=[1],
            reward_fn_path=tmp_path / "r.py",
            base_config_dict={},
            iteration_dir=tmp_path,
            env_id="HalfCheetah-v5",
        )

    assert mock.call_args.kwargs["heartbeat_fn"] is heartbeat


def test_custom_trainer_satisfies_protocol():
    """A custom class implementing train() should satisfy the Trainer protocol."""

    class FakeTrainer:
        def train(self, configs, seeds, reward_fn_path, base_config_dict, iteration_dir, env_id):
            return {
                "best_config_id": "fake",
                "best_run_id": "fake_seed_1",
                "configs": {},
            }

    assert isinstance(FakeTrainer(), Trainer)
