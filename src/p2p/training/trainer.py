"""Trainer abstraction for pluggable execution backends (#220).

Defines the ``Trainer`` protocol and ``LocalTrainer`` implementation.
The loop delegates training execution to a Trainer, decoupling the
"what to train" from "how to execute it".

Design sketch::

    loop.py
      → run_iteration(trainer=trainer)
          → trainer.train(configs, seeds, ...) → IterationAggregation
                ├── LocalTrainer   (local subprocess via parallel_trainer)
                └── (future) ScheduledTrainer(Scheduler(SSHBackend()))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from p2p.contracts import IterationAggregation, RunConfigEntry


@runtime_checkable
class Trainer(Protocol):
    """Protocol for training execution backends.

    Implementors handle the details of spawning, monitoring, and collecting
    results from training runs.  The loop and iteration_runner only interact
    with this interface.
    """

    def train(
        self,
        configs: list[RunConfigEntry],
        seeds: list[int],
        reward_fn_path: Path,
        base_config_dict: dict,
        iteration_dir: Path,
        env_id: str,
    ) -> IterationAggregation: ...


class LocalTrainer:
    """Execute training runs as local subprocesses.

    Wraps ``parallel_trainer.run_parallel_trainings()`` with resource
    parameters bound at construction time.

    Parameters
    ----------
    cores_per_run:
        CPU cores per training run. 0 = auto.
    max_parallel:
        Maximum concurrent training runs. 0 = auto.
    cores_pool:
        Pre-allocated CPU core IDs for benchmark mode.
    heartbeat_fn:
        Optional callback invoked during polling.
    """

    def __init__(
        self,
        cores_per_run: int = 0,
        max_parallel: int = 0,
        cores_pool: list[int] | None = None,
        heartbeat_fn: Callable[[], None] | None = None,
        no_cpu_affinity: bool = False,
        gpu_pool: list[int] | None = None,
    ) -> None:
        self._cores_per_run = cores_per_run
        self._max_parallel = max_parallel
        self._cores_pool = cores_pool
        self._heartbeat_fn = heartbeat_fn
        self._no_cpu_affinity = no_cpu_affinity
        self._gpu_pool = gpu_pool

    def train(
        self,
        configs: list[RunConfigEntry],
        seeds: list[int],
        reward_fn_path: Path,
        base_config_dict: dict,
        iteration_dir: Path,
        env_id: str,
    ) -> IterationAggregation:
        from p2p.training.parallel_trainer import run_parallel_trainings

        return run_parallel_trainings(
            configs=configs,
            seeds=seeds,
            reward_fn_path=reward_fn_path,
            base_config_dict=base_config_dict,
            iteration_dir=iteration_dir,
            env_id=env_id,
            cores_per_run=self._cores_per_run,
            max_parallel=self._max_parallel,
            cores_pool=self._cores_pool,
            heartbeat_fn=self._heartbeat_fn,
            no_cpu_affinity=self._no_cpu_affinity,
            gpu_pool=self._gpu_pool,
        )
