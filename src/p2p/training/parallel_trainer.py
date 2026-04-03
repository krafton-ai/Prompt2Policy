"""Parallel training coordinator for multi-config × seeds within a loop iteration.

Spawns one subprocess per (config, seed) pair via ``python -m p2p.executor``,
manages CPU core allocation, polls for completion, and aggregates results.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from p2p.contracts import round_metric
from p2p.session.iteration_record import IterationRecord
from p2p.training.cpu_manager import get_cpu_manager
from p2p.utils.subprocess_utils import python_cmd
from p2p.utils.utils import read_log_tail

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from p2p.contracts import IterationAggregation, RunAggregationEntry, RunConfigEntry


def run_parallel_trainings(
    configs: list[RunConfigEntry],
    seeds: list[int],
    reward_fn_path: Path,
    base_config_dict: dict,
    iteration_dir: Path,
    env_id: str = "HalfCheetah-v5",
    cores_per_run: int = 0,
    max_parallel: int = 0,
    heartbeat_fn: Callable[[], None] | None = None,
    cores_pool: list[int] | None = None,
    no_cpu_affinity: bool = False,
    gpu_pool: list[int] | None = None,
) -> IterationAggregation:
    """Run all config x seed trainings in parallel and return aggregation.

    Each training runs as a subprocess via ``python -m p2p.executor``.

    Parameters
    ----------
    configs:
        List of hyperparameter configurations to compare.
    seeds:
        List of seed values to repeat for each config.
    reward_fn_path:
        Path to the shared reward function file.
    base_config_dict:
        Base TrainConfig as a dict (seed/iteration_id will be overridden per run).
    iteration_dir:
        Directory for this loop iteration (e.g. ``session_xxx/iter_1/``).
    env_id:
        MuJoCo environment ID.
    cores_per_run:
        CPU cores per run. 0 = auto (defaults to 2).
    max_parallel:
        Max concurrent runs. 0 = auto (fill available cores).
    cores_pool:
        Pre-allocated CPU cores from the central CPUManager. When provided,
        cores are split into ``cores_per_run``-sized chunks instead of
        allocating from a local CPUManager. This prevents cross-session
        core contention in benchmark mode.
    gpu_pool:
        Pre-allocated GPU device IDs. When provided, each subprocess gets
        ``CUDA_VISIBLE_DEVICES`` set to a single GPU in round-robin order.

    Returns
    -------
    IterationAggregation with best_config_id, best_run_id, and per-config stats.
    """
    iteration_dir.mkdir(parents=True, exist_ok=True)

    # Copy reward file into iteration dir if not already there
    reward_dest = iteration_dir / "reward_fn.py"
    if reward_fn_path != reward_dest:
        reward_dest.write_text(reward_fn_path.read_text())

    cores_per = cores_per_run or 2

    # Mode selection for CPU allocation
    if no_cpu_affinity:
        # No CPU pinning: skip CPUManager, rely on max_parallel only
        _chunks = None
        cpu_mgr = None
        effective_max = max_parallel if max_parallel > 0 else len(configs) * len(seeds)
    elif cores_pool is not None:
        _chunks = [cores_pool[i : i + cores_per] for i in range(0, len(cores_pool), cores_per)]
        _available_chunks: list[list[int]] = list(_chunks)  # pool of free chunks
        effective_max = len(_chunks)
        cpu_mgr = None  # not used — cores are pre-assigned
    else:
        _chunks = None
        cpu_mgr = get_cpu_manager()
        auto_p = max(1, cpu_mgr.available_count() // cores_per) if cores_per > 0 else 1
        effective_max = max_parallel or min(auto_p, len(configs) * len(seeds))

    # Build schedule
    schedule: list[dict] = []
    for cfg in configs:
        config_id = cfg["config_id"]

        for seed in seeds:
            run_id = f"{config_id}_seed_{seed}"
            run_config = dict(base_config_dict)
            run_config.update(cfg.get("params", {}))
            run_config["seed"] = seed
            run_config["iteration_id"] = run_id

            # Write config JSON for this run
            run_dir = iteration_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

            schedule.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "run_id": run_id,
                }
            )

    # Launch and manage subprocesses
    base_env = {**os.environ}
    # run_id -> (proc, alloc_id, log_file, cores_used)
    active: dict[str, tuple[subprocess.Popen, str, object, list[int] | None]] = {}
    pending = list(schedule)
    completed: dict[str, dict] = {}  # run_id -> summary dict
    failed: dict[str, str] = {}  # run_id -> error description
    total = len(schedule)
    gpu_launch_counter = 0  # round-robin counter for GPU assignment

    logger.info(
        "Starting: %d configs x %d seeds = %d runs, max_parallel=%d, cores_per=%d",
        len(configs),
        len(seeds),
        total,
        effective_max,
        cores_per,
    )

    try:
        while pending or active:
            # Launch new runs if resources allow
            def _can_alloc() -> bool:
                if no_cpu_affinity:
                    return True
                if _chunks is not None:
                    return len(_available_chunks) > 0
                return cpu_mgr is not None and cpu_mgr.can_fit(cores_per)

            while pending and len(active) < effective_max and _can_alloc():
                entry = pending.pop(0)
                alloc_id = f"parallel_{entry['run_id']}"

                if no_cpu_affinity:
                    cores = None
                elif _chunks is not None:
                    cores = _available_chunks.pop(0)
                else:
                    cores = cpu_mgr.allocate(alloc_id, cores_per)
                    if cores is None:
                        pending.insert(0, entry)
                        break

                config_path = iteration_dir / entry["run_id"] / "config.json"
                cmd = [
                    *python_cmd(unbuffered=True),
                    "-m",
                    "p2p.executor",
                    "--reward-fn",
                    str(reward_dest),
                    "--config",
                    str(config_path),
                    "--runs-dir",
                    str(iteration_dir),
                    "--iteration-id",
                    entry["run_id"],
                    "--env-id",
                    env_id,
                ]

                if cores:
                    if shutil.which("taskset"):
                        core_list = ",".join(str(c) for c in cores)
                        cmd = ["taskset", "-c", core_list, *cmd]
                    else:
                        logger.warning(
                            "taskset not found, skipping CPU pinning for %s",
                            entry["run_id"],
                        )

                if gpu_pool:
                    run_env = dict(base_env)
                    gpu_id = gpu_pool[gpu_launch_counter % len(gpu_pool)]
                    run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    gpu_launch_counter += 1
                else:
                    run_env = base_env

                log_path = iteration_dir / entry["run_id"] / "subprocess.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(log_path, "w", buffering=1)  # noqa: SIM115
                try:
                    proc = subprocess.Popen(
                        cmd,
                        env=run_env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log_file.close()
                    if cpu_mgr is not None:
                        cpu_mgr.release(alloc_id)
                    if _chunks is not None and cores is not None:
                        _available_chunks.append(cores)
                    raise

                active[entry["run_id"]] = (proc, alloc_id, log_file, cores)
                gpu_info = run_env.get("CUDA_VISIBLE_DEVICES", "inherit")
                logger.info(
                    "Launched %s (cores=%s, gpu=%s) [%d/%d done, %d active, %d pending]",
                    entry["run_id"],
                    cores,
                    gpu_info,
                    len(completed),
                    total,
                    len(active),
                    len(pending),
                )

            # Poll for completions
            for run_id in list(active):
                proc, alloc_id, log_file, cores_used = active[run_id]
                if proc.poll() is not None:
                    log_file.close()
                    if cpu_mgr is not None:
                        cpu_mgr.release(alloc_id)
                    if _chunks is not None and cores_used is not None:
                        _available_chunks.append(cores_used)
                    del active[run_id]

                    if proc.returncode != 0:
                        log_path = iteration_dir / run_id / "subprocess.log"
                        err_tail = read_log_tail(log_path, n=10)
                        failed[run_id] = f"exit code {proc.returncode}\n{err_tail}"
                        logger.error("FAILED %s (exit %d)", run_id, proc.returncode)
                    else:
                        rec = IterationRecord(iteration_dir / run_id)
                        summary = rec.read_summary() or {}
                        completed[run_id] = summary
                        logger.info("Finished %s [%d/%d done]", run_id, len(completed), total)

            # Early abort: all launched runs failed, none succeeded, more pending
            if failed and not completed and not active and pending:
                first_err = next(iter(failed.values()))
                raise RuntimeError(
                    f"All {len(failed)} training runs failed, aborting "
                    f"{len(pending)} remaining. First error:\n{first_err}"
                )

            if active:
                if heartbeat_fn:
                    heartbeat_fn()
                time.sleep(2)

        # All runs finished — check if everything failed
        if not completed:
            first_err = next(iter(failed.values())) if failed else "unknown"
            raise RuntimeError(
                f"All {len(failed)} training runs failed. First error:\n{first_err}"
            )

        if failed:
            logger.warning("%d/%d runs failed", len(failed), total)

    finally:
        # Clean up any remaining active processes
        for run_id, (proc, alloc_id, log_file, _cores) in active.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            log_file.close()
            if cpu_mgr is not None:
                cpu_mgr.release(alloc_id)

    # Aggregate results per config
    aggregation = _aggregate(configs, seeds, completed)

    # Save aggregation.json in iteration_dir
    (iteration_dir / "aggregation.json").write_text(json.dumps(aggregation, indent=2))

    return aggregation


def _aggregate(
    configs: list[RunConfigEntry],
    seeds: list[int],
    completed: dict[str, dict],
) -> IterationAggregation:
    """Compute per-config aggregation and select best config."""
    config_stats: dict[str, RunAggregationEntry] = {}
    best_config_id = ""
    best_mean_return = float("-inf")

    for cfg in configs:
        config_id = cfg["config_id"]
        per_seed: list[dict[str, float]] = []

        for seed in seeds:
            run_id = f"{config_id}_seed_{seed}"
            summary = completed.get(run_id, {})
            final_return = round_metric(float(summary.get("final_episodic_return", 0) or 0))
            per_seed.append(
                {
                    "seed": float(seed),
                    "best_score": 0.0,  # filled after judging
                    "final_return": final_return,
                }
            )

        returns = [p["final_return"] for p in per_seed]
        mean_ret = round_metric(float(np.mean(returns))) if returns else 0.0

        config_stats[config_id] = {
            "mean_best_score": 0.0,  # filled after judging
            "std_best_score": 0.0,
            "mean_final_return": mean_ret,
            "std_final_return": round_metric(float(np.std(returns))) if returns else 0.0,
            "per_seed": per_seed,
        }

        if mean_ret > best_mean_return:
            best_mean_return = mean_ret
            best_config_id = config_id

    # Find best individual run (highest return in best config)
    best_run_id = ""
    best_return = float("-inf")
    for seed in seeds:
        run_id = f"{best_config_id}_seed_{seed}"
        summary = completed.get(run_id, {})
        ret = float(summary.get("final_episodic_return", 0) or 0)
        if ret > best_return:
            best_return = ret
            best_run_id = run_id

    return {
        "best_config_id": best_config_id,
        "best_run_id": best_run_id,
        "configs": config_stats,
    }
