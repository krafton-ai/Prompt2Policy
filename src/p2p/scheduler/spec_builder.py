"""Build RunSpec from high-level job configurations.

Each function converts a job-type-specific config into one or more RunSpecs.
Adding a new job type means adding a new builder function here — no changes
to the scheduler or backends.
"""

from __future__ import annotations

import dataclasses
import uuid

from p2p.config import LoopConfig, TrainConfig
from p2p.scheduler.types import RunSpec
from p2p.settings import ANTHROPIC_API_KEY

DEFAULT_CONFIG: dict = {"config_id": "default", "label": "default", "params": {}}


def _run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def _base_env() -> dict[str, str]:
    from p2p.settings import VLM_REFINED_INITIAL_FRAME

    env: dict[str, str] = {}
    if ANTHROPIC_API_KEY:
        env["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    env["VLM_REFINED_INITIAL_FRAME"] = "true" if VLM_REFINED_INITIAL_FRAME else "false"
    return env


def _cpu_budget(
    configs: list[dict] | None,
    seeds: list[int] | None,
    fallback_seed: int,
    cores_per_run: int | None,
    num_envs: int,
) -> int:
    """Compute total CPU cores: n_configs × n_seeds × per-run cores."""
    n_configs = len(configs) if configs else 1
    n_seeds = len(seeds) if seeds else 1
    per_run = cores_per_run or max(2, num_envs)
    return n_configs * n_seeds * per_run


def _gpu_budget(engine: str) -> int:
    """Compute GPU count: 1 for IsaacLab sessions, 0 for MuJoCo."""
    return 1 if engine == "isaaclab" else 0


def session_to_spec(
    *,
    prompt: str = "",
    loop_config: LoopConfig | None = None,
    session_id: str | None = None,
    configs: list[dict] | None = None,
    seeds: list[int] | None = None,
) -> RunSpec:
    """Convert session config to a single RunSpec.

    When *configs* or *seeds* are provided they are embedded into
    ``LoopConfig`` so that a single ``run_loop()`` call handles the
    full multi-config × multi-seed matrix.
    """
    if loop_config is None:
        loop_config = LoopConfig()

    # Embed configs/seeds into LoopConfig so run_loop handles the matrix.
    # skip_vram_scaling: the remote run_session will scale on its own GPU.
    if configs is not None or seeds is not None:
        replace_kw: dict = {"skip_vram_scaling": True}
        if configs is not None:
            replace_kw["configs"] = configs
        if seeds is not None:
            replace_kw["seeds"] = seeds
        loop_config = dataclasses.replace(loop_config, **replace_kw)

    skip_cpu = loop_config.no_cpu_affinity

    if skip_cpu:
        cpu = 0
    else:
        cpu = _cpu_budget(
            configs,
            seeds,
            loop_config.train.seed,
            loop_config.cores_per_run,
            loop_config.train.num_envs,
        )

    run_id = session_id or _run_id()
    params: dict = {
        "session_id": run_id,
        "prompt": prompt,
        "loop_config": loop_config.to_json(),
    }
    tags: dict[str, str] = {"job_type": "session"}
    if skip_cpu:
        tags["no_cpu_affinity"] = "true"

    return RunSpec(
        run_id=run_id,
        entry_point="p2p.session.run_session",
        parameters=params,
        env=_base_env(),
        cpu_cores=cpu,
        gpu_count=_gpu_budget(loop_config.train.engine),
        tags=tags,
    )


def benchmark_case_to_spec(
    *,
    benchmark_id: str,
    case_index: int,
    env_id: str,
    instruction: str,
    base_loop_config: LoopConfig,
    configs: list[dict] | None = None,
    seeds: list[int] | None = None,
) -> RunSpec:
    """Convert a single benchmark test case into a RunSpec.

    Rebuilds TrainConfig per-case with the correct env_id so each
    environment gets its own Zoo preset HPs.  All *configs* and *seeds*
    are embedded in the LoopConfig so a single ``run_loop()`` call
    handles the full matrix.
    """
    base_train = base_loop_config.train
    # Fields that from_preset must receive to apply Zoo-tuned HPs
    # per environment.  Non-HP fields (trajectory_stride, num_evals, …)
    # are forwarded so they are not silently reset to defaults.
    preset_kwargs = {
        "env_id": env_id,
        "seed": base_train.seed,
        "total_timesteps": base_train.total_timesteps,
        "checkpoint_interval": base_train.checkpoint_interval,
        "num_envs": base_train.num_envs,
        "side_info": base_train.side_info,
        "engine": base_train.engine,
        "trajectory_stride": base_train.trajectory_stride,
        "num_evals": base_train.num_evals,
    }
    case_train = (
        TrainConfig.from_preset(**preset_kwargs)
        if base_loop_config.use_zoo_preset
        else dataclasses.replace(base_train, env_id=env_id, seed=base_train.seed)
    )

    replace_kw: dict = {"train": case_train, "skip_vram_scaling": True}
    if configs is not None:
        replace_kw["configs"] = configs
    if seeds is not None:
        replace_kw["seeds"] = seeds
    case_config = dataclasses.replace(base_loop_config, **replace_kw)

    run_id = f"{benchmark_id}_case{case_index}"

    skip_cpu = base_loop_config.no_cpu_affinity

    if skip_cpu:
        cpu = 0
    else:
        cpu = _cpu_budget(
            configs,
            seeds,
            base_train.seed,
            base_loop_config.cores_per_run,
            case_train.num_envs,
        )

    params: dict = {
        "session_id": run_id,
        "prompt": instruction,
        "loop_config": case_config.to_json(),
    }

    tags: dict[str, str] = {
        "job_type": "benchmark",
        "benchmark_id": benchmark_id,
        "case_index": str(case_index),
        "env_id": env_id,
    }
    if skip_cpu:
        tags["no_cpu_affinity"] = "true"

    return RunSpec(
        run_id=run_id,
        entry_point="p2p.session.run_session",
        parameters=params,
        env=_base_env(),
        cpu_cores=cpu,
        gpu_count=_gpu_budget(case_train.engine),
        tags=tags,
    )
