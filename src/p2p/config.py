"""TrainConfig, LoopConfig dataclasses and hyperparameter bounds definition."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from p2p.settings import LLM_MODEL, VLM_MODEL

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from p2p.contracts import RunConfigEntry

_MAX_BATCH_SIZE: int = 1_000_000
"""Upper bound for batch_size (num_envs * num_steps).

Prevents GPU OOM when HP tuning increases num_steps for envs with large
num_envs (e.g. IsaacLab envs with 4096 envs).  If batch_size would exceed
this, num_steps is clamped down in __post_init__.
"""

_ISAACLAB_GPU_RESERVE_MB: int = 10_000
"""VRAM reserved for Phase 2 rendering worker (~5.5 GB) + Phase 1 eval
subprocess (~4 GB) + OS/CUDA overhead.
"""

_ISAACLAB_BASE_VRAM_MB: int = 1500
"""Base VRAM per IsaacLab training process (Isaac Sim overhead, independent of num_envs)."""

_ISAACLAB_VRAM_PER_ENV: float = 0.75
"""Approximate VRAM in MB per parallel environment instance.

Derived from empirical measurements at 4096 envs:
  Shadow Hand (obs=157): ~4500 MB → 0.73 MB/env
  G1 (obs=69):           ~3500 MB → 0.49 MB/env
  Anymal (obs=48):       ~2500 MB → 0.24 MB/env
Using 0.75 as a conservative upper bound.
"""

_ISAACLAB_VRAM_SAFETY: float = 1.2
"""Safety multiplier for VRAM estimates to account for bootstrap peak memory."""

_MAX_SCALED_MINIBATCHES: int = 32
"""Upper bound for num_minibatches after sqrt-scaling in from_preset().

Prevents excessive gradient steps when scaling large-base environments
(e.g. Ant num_minibatches=16) to many envs. The base case (num_envs=1) is
never capped — this only applies to the sqrt-scaled value.
"""

TARGET_EPISODE_DURATION_S: float = 5.0
"""Target real-time episode duration in seconds.

max_episode_steps is computed as ``int(TARGET_EPISODE_DURATION_S / dt)``
when ``dt`` is known for the environment (via ``EnvSpec.dt``).
Environments without a known ``dt`` fall back to the TrainConfig default (300).
"""

DEFAULT_JUDGMENT_SELECT: str = "last"
"""Single source of truth for the judgment_select default.

"best" picks the checkpoint with the highest intent_score;
"last" picks the final (chronologically last) checkpoint.
"""


@dataclass
class TrainConfig:
    """Training configuration for PPO (SB3)."""

    # Iteration
    iteration_id: str = ""
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "auto"

    # Environment
    env_id: str = "HalfCheetah-v5"
    num_envs: int = 1
    max_episode_steps: int = 300
    side_info: bool = False
    engine: str = "mujoco"
    terminate_when_unhealthy: bool = False

    # PPO hyperparameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Network
    net_arch: list[int] = field(default_factory=lambda: [256, 256])

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True
    reward_clip: float = 10.0
    obs_clip: float = 10.0

    # Logging & checkpoints
    eval_interval: int = 0  # 0 = auto (10%, 50%, 100%)
    checkpoint_interval: int = 100_000
    num_evals: int = 4  # number of eval checkpoints (videos + trajectories)
    reward_term_avg_window: int = 25  # last-N rollouts for per-term averages
    judgment_select: str = DEFAULT_JUDGMENT_SELECT
    num_eval_rounds: int = 3  # eval rounds per checkpoint (total episodes = num_envs * rounds)
    num_eval_envs: int = 0  # parallel eval workers (0 = use num_envs)
    parallel_eval: bool = True  # True = parallel VecEnv eval, False = sequential (main branch)
    trajectory_precision: int = 4  # decimal places for trajectory floats
    trajectory_stride: int = 1  # save every Nth step (1 = all steps)
    # Derived (computed at runtime)
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Coerce net_arch: int → [int, int], ensure list[int]
        if isinstance(self.net_arch, int):
            self.net_arch = [self.net_arch, self.net_arch]
        # Cap num_steps to keep batch_size within GPU memory budget
        if self.num_envs * self.num_steps > _MAX_BATCH_SIZE and self.num_envs > 0:
            clamped = max(1, _MAX_BATCH_SIZE // self.num_envs)
            logger.warning(
                "num_steps clamped %d -> %d (batch_size cap: %d, num_envs: %d)",
                self.num_steps,
                clamped,
                _MAX_BATCH_SIZE,
                self.num_envs,
            )
            self.num_steps = clamped
        self.batch_size = self.num_envs * self.num_steps
        if self.num_minibatches > 0:
            self.minibatch_size = self.batch_size // self.num_minibatches
        if self.batch_size > 0:
            self.num_iterations = self.total_timesteps // self.batch_size
        # Sequential eval is slow — cap episodes to avoid long waits
        if not self.parallel_eval and self.num_eval_rounds > 10:
            self.num_eval_rounds = 10

    # Keys that the revise agent is allowed to tune
    _TUNABLE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "learning_rate",
            "ent_coef",
            "vf_coef",
            "clip_coef",
            "max_grad_norm",
            "gae_lambda",
            "gamma",
            "num_steps",
            "update_epochs",
            "target_kl",
            "total_timesteps",
            "net_arch",
            "normalize_obs",
            "normalize_reward",
            "reward_clip",
            "obs_clip",
            "max_episode_steps",
        }
    )

    def apply_updates(self, changes: dict) -> TrainConfig:
        """Create new TrainConfig with HP changes applied. Only tunable params allowed.

        Unsafe params (env_id, seed, device, num_envs) are
        silently ignored. Numeric values are clamped to HP_BOUNDS.
        Derived fields are recomputed via __post_init__.
        """
        d = json.loads(self.to_json())
        # Remove derived fields so __post_init__ recomputes them
        for key in ("batch_size", "minibatch_size", "num_iterations"):
            d.pop(key, None)
        for k, v in changes.items():
            if k in self._TUNABLE_KEYS:
                # Clamp numeric values to HP_BOUNDS
                if k in HP_BOUNDS and isinstance(v, (int, float)):
                    lo, hi = HP_BOUNDS[k]
                    if lo is not None:
                        v = max(lo, v)
                    if hi is not None:
                        v = min(hi, v)
                    v = type(v)(v)
                d[k] = v
        return TrainConfig(**d)

    @classmethod
    def from_preset(cls, **overrides: object) -> TrainConfig:
        """Create TrainConfig with Zoo-tuned HPs for the given env_id.

        Falls back to dataclass defaults if no preset exists.
        *overrides* must include ``env_id``; other fields
        (e.g. total_timesteps, seed) are applied after the preset.

        When ``num_envs`` differs from the Zoo-tuned ``_zoo_n_envs``,
        applies sqrt-scaling to balance GPU efficiency and learning:
          - minibatch_size grows by sqrt(env_ratio)
          - learning_rate grows by sqrt(minibatch_growth)
        Example (HalfCheetah, n_envs 1→16):
          minibatch 64→256, lr 2.06e-5→4.1e-5, num_minibatches 8→32
        """
        import math

        from p2p.training.hp_presets import get_preset

        env_id = str(overrides.get("env_id", "HalfCheetah-v5"))
        preset = dict(get_preset(env_id) or {})

        # --- Sqrt-scaling for num_envs mismatch ---
        zoo_n_envs = preset.pop("_zoo_n_envs", 1)
        actual_n_envs = int(overrides.get("num_envs", zoo_n_envs))
        zoo_num_steps = preset.get("num_steps", 1024)
        zoo_num_minibatches = preset.get("num_minibatches", 4)
        zoo_lr = preset.get("learning_rate", 3e-4)

        env_ratio = actual_n_envs / zoo_n_envs
        if env_ratio > 1 and "num_minibatches" not in overrides:
            zoo_minibatch = max(1, (zoo_n_envs * zoo_num_steps) // zoo_num_minibatches)
            # Minibatch grows by sqrt(env_ratio) — balance GPU utilization vs learning
            mb_scale = math.sqrt(env_ratio)
            new_minibatch = int(zoo_minibatch * mb_scale)
            new_batch = actual_n_envs * zoo_num_steps
            preset["num_minibatches"] = min(
                _MAX_SCALED_MINIBATCHES,
                max(1, new_batch // new_minibatch),
            )
            # lr sqrt-scaled for the minibatch growth
            if "learning_rate" not in overrides:
                preset["learning_rate"] = zoo_lr * math.sqrt(mb_scale)

        kwargs: dict = {"env_id": env_id, **preset, **overrides}
        # Remove derived/internal fields in case they leak in
        for key in ("batch_size", "minibatch_size", "num_iterations", "_zoo_n_envs"):
            kwargs.pop(key, None)
        # Auto-derive engine from env_id if not explicitly set
        if kwargs.get("engine", "mujoco") == "mujoco" and env_id.startswith("Isaac-"):
            from p2p.training.env_spec import ENV_REGISTRY

            spec = ENV_REGISTRY.get(env_id)
            if spec:
                kwargs["engine"] = spec.engine
        # Auto-compute max_episode_steps from dt when not set by caller or preset
        if "max_episode_steps" not in kwargs:
            from p2p.training.env_spec import max_steps_for_duration

            dt_steps = max_steps_for_duration(env_id, TARGET_EPISODE_DURATION_S)
            if dt_steps is not None:
                kwargs["max_episode_steps"] = dt_steps
            else:
                logger.debug(
                    "max_episode_steps auto-compute skipped for %s (dt unknown); "
                    "using TrainConfig default (%d)",
                    env_id,
                    300,
                )
        return cls(**kwargs)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> TrainConfig:
        d = json.loads(s)
        for key in _DERIVED_KEYS | _LEGACY_KEYS:
            d.pop(key, None)
        return cls(**d)


# ---------------------------------------------------------------------------
# Canonical HP bounds & boolean keys (single source of truth for all modules)
# ---------------------------------------------------------------------------

HP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "learning_rate": (1e-6, 3e-3),
    "ent_coef": (0.0001, 0.1),
    "vf_coef": (0.1, 2.0),
    "clip_coef": (0.1, 0.4),
    "max_grad_norm": (0.1, 5.0),
    "gae_lambda": (0.8, 1.0),
    "gamma": (0.95, 0.999),
    "num_steps": (256, 8192),
    "update_epochs": (3, 20),
    "num_minibatches": (2, 64),
    "target_kl": (0.001, 0.05),
    "total_timesteps": (500_000, 50_000_000),
    "reward_clip": (1.0, 50.0),
    "obs_clip": (1.0, 50.0),
}

BOOL_HP_KEYS: frozenset[str] = frozenset({"normalize_obs", "normalize_reward"})


@dataclass
class LoopConfig:
    """Complete configuration for a loop session.

    Bundles all loop session parameters into a single object.
    Built from CLI args, API requests, or tests.
    """

    # RL
    train: TrainConfig = field(default_factory=TrainConfig)
    configs: list[RunConfigEntry] | None = None
    seeds: list[int] | None = None

    # Loop control
    max_iterations: int = 5
    pass_threshold: float = 0.7

    # LLM / VLM
    model: str = LLM_MODEL
    vlm_model: str = VLM_MODEL
    thinking_effort: str = ""
    refined_initial_frame: bool = True

    # Execution
    runs_dir: Path = Path("runs")
    cores_per_run: int = 0
    max_parallel: int = 0
    cores_pool: list[int] | None = None
    gpu_pool: list[int] | None = None
    no_cpu_affinity: bool = False

    # Motion overlays (VLM video preprocessing)
    criteria_diagnosis: bool = True
    motion_trail_dual: bool = True

    # Flags
    hp_tuning: bool = True
    use_code_judge: bool = False
    review_reward: bool = True
    review_judge: bool = True
    judgment_select: str = DEFAULT_JUDGMENT_SELECT
    use_zoo_preset: bool = True

    # Intent elicitation
    elaborated_intent: str = ""

    # When True, skip VRAM-based num_envs scaling (used during spec
    # building on the API server where the local GPU differs from the
    # target node — the remote run_session will scale on its own GPU).
    skip_vram_scaling: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runs_dir, Path):
            self.runs_dir = Path(self.runs_dir)
        if isinstance(self.train, dict):
            self.train = _train_from_dict(self.train)
        if self.train.engine == "isaaclab" and not self.skip_vram_scaling:
            _scale_isaaclab_num_envs(self)

    def to_json(self) -> str:
        d = asdict(self)
        d["runs_dir"] = str(d["runs_dir"])
        d.pop("skip_vram_scaling", None)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> LoopConfig:
        d = json.loads(s)
        d["train"] = _train_from_dict(d["train"])
        return cls(**d)


def _estimate_isaaclab_vram(num_envs: int) -> int:
    """Estimated VRAM in MB for one IsaacLab training process."""
    raw = _ISAACLAB_BASE_VRAM_MB + _ISAACLAB_VRAM_PER_ENV * num_envs
    return int(raw * _ISAACLAB_VRAM_SAFETY)


def _scale_isaaclab_num_envs(lc: LoopConfig) -> None:
    """Scale down num_envs for IsaacLab envs to fit GPU VRAM budget.

    Queries the GPU total memory, reserves space for Phase 2 worker and
    Phase 1 eval, then computes the per-process budget.  If the current
    num_envs would exceed that budget, halves num_envs (and doubles
    num_steps to preserve batch_size) until it fits.

    Idempotent: already-scaled values fit the budget and trigger early return.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        device_str = lc.train.device or "auto"
        if device_str != "auto" and not device_str.startswith("cuda"):
            return
        # NOTE: current_device() returns 0 if CUDA context is not yet initialized,
        # which is typically fine — CUDA_VISIBLE_DEVICES remaps physical GPUs if set.
        if device_str == "auto":
            dev_idx = torch.cuda.current_device()
        elif ":" in device_str:
            dev_idx = int(device_str.split(":")[-1])
        else:
            dev_idx = 0
        gpu_total_mb = torch.cuda.get_device_properties(dev_idx).total_memory // (1024 * 1024)
    except Exception:
        logger.debug("VRAM scaling skipped: could not query GPU", exc_info=True)
        return

    num_configs = len(lc.configs) if lc.configs else 1
    num_seeds = len(lc.seeds) if lc.seeds else 1
    num_parallel = num_configs * num_seeds

    # With multi-GPU (gpu_pool), each GPU only runs a fraction of parallel
    # processes. Compute per-GPU occupancy for accurate VRAM budgeting.
    if lc.gpu_pool and len(lc.gpu_pool) > 1:
        sessions_per_gpu = max(1, math.ceil(num_parallel / len(lc.gpu_pool)))
    else:
        sessions_per_gpu = num_parallel

    budget_per_process = (gpu_total_mb - _ISAACLAB_GPU_RESERVE_MB) // max(sessions_per_gpu, 1)
    if budget_per_process <= 0:
        # GPU smaller than reserved headroom; proceed with original num_envs
        logger.error(
            "GPU VRAM (%d MB) too small for reserve (%d MB). OOM is likely.",
            gpu_total_mb,
            _ISAACLAB_GPU_RESERVE_MB,
        )
        return

    estimated_vram = _estimate_isaaclab_vram(lc.train.num_envs)
    if estimated_vram <= budget_per_process:
        return

    original_num_envs = lc.train.num_envs
    original_num_steps = lc.train.num_steps
    num_envs = original_num_envs
    num_steps = original_num_steps

    max_num_steps = int(HP_BOUNDS["num_steps"][1])
    while _estimate_isaaclab_vram(num_envs) > budget_per_process and num_envs > 64:
        num_envs //= 2
        num_steps = min(num_steps * 2, max_num_steps)

    final_est = _estimate_isaaclab_vram(num_envs)
    if final_est > budget_per_process:
        logger.error(
            "VRAM budget still exceeded after scaling to num_envs=%d "
            "(estimated %d MB > budget %d MB). OOM may occur.",
            num_envs,
            final_est,
            budget_per_process,
        )

    if num_envs == original_num_envs:
        return

    logger.warning(
        "IsaacLab VRAM scaling: num_envs %d -> %d, num_steps %d -> %d "
        "(GPU: %d MB, budget/process: %d MB, %d parallel)",
        original_num_envs,
        num_envs,
        original_num_steps,
        num_steps,
        gpu_total_mb,
        budget_per_process,
        num_parallel,
    )
    lc.train.num_envs = num_envs
    lc.train.num_steps = num_steps
    lc.train.__post_init__()


def _train_from_dict(d: dict) -> TrainConfig:
    """Build TrainConfig from a plain dict, stripping derived and legacy fields."""
    clean = {k: v for k, v in d.items() if k not in _DERIVED_KEYS and k not in _LEGACY_KEYS}
    return TrainConfig(**clean)


_DERIVED_KEYS = frozenset({"batch_size", "minibatch_size", "num_iterations"})

# Stripped keys — no longer part of TrainConfig but may appear in saved configs.
_LEGACY_KEYS = frozenset(
    {
        "algorithm",  # removed: SAC support dropped (#254)
        "sac",  # removed: SACConfig dropped (#254)
    }
)


# ---------------------------------------------------------------------------
# Factory function — single construction point for LoopConfig (#303)
# ---------------------------------------------------------------------------


def loop_config_from_params(
    *,
    # TrainConfig fields
    total_timesteps: int = 1_000_000,
    seed: int = 1,
    num_envs: int = 1,
    env_id: str = "HalfCheetah-v5",
    side_info: bool = False,
    engine: str = "mujoco",
    checkpoint_interval: int | None = None,  # None = auto
    num_evals: int = 4,
    trajectory_stride: int = 1,
    device: str = "auto",
    max_episode_steps: int | None = None,
    terminate_when_unhealthy: bool = False,
    # LoopConfig fields
    configs: list[RunConfigEntry] | None = None,
    seeds: list[int] | None = None,
    max_iterations: int = 5,
    pass_threshold: float = 0.7,
    model: str = "",  # "" = use LLM_MODEL default
    vlm_model: str | None = None,  # None = use VLM_MODEL default
    thinking_effort: str = "",
    refined_initial_frame: bool = True,
    runs_dir: str | Path = Path("runs"),
    cores_per_run: int = 0,
    max_parallel: int = 0,
    cores_pool: list[int] | None = None,
    no_cpu_affinity: bool = False,
    criteria_diagnosis: bool = True,
    motion_trail_dual: bool = True,
    hp_tuning: bool = True,
    use_code_judge: bool = False,
    review_reward: bool = True,
    review_judge: bool = True,
    judgment_select: str = DEFAULT_JUDGMENT_SELECT,
    use_zoo_preset: bool = True,
    elaborated_intent: str = "",
    **_extra: object,  # ignore unknown keys (forward compatibility)
) -> LoopConfig:
    """Build a LoopConfig from flat parameters.

    Centralizes checkpoint_interval auto-computation, model/vlm_model
    default resolution, and zoo-preset dispatch so that callers don't
    duplicate this logic.

    Default resolution:
    - ``checkpoint_interval=None`` → ``max(100_000, total_timesteps // 5)``
    - ``model=""`` → ``LLM_MODEL`` (empty string = not specified)
    - ``vlm_model=None`` → ``VLM_MODEL``; ``vlm_model=""`` is kept as-is
      (empty string = VLM explicitly disabled)
    """
    if checkpoint_interval is None:
        checkpoint_interval = max(100_000, total_timesteps // 5)

    # Auto-derive engine from env_id when not explicitly set
    if engine == "mujoco" and env_id.startswith("Isaac-"):
        from p2p.training.env_spec import ENV_REGISTRY

        spec = ENV_REGISTRY.get(env_id)
        if spec:
            engine = spec.engine

    train_kwargs: dict = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "num_envs": num_envs,
        "env_id": env_id,
        "side_info": side_info,
        "engine": engine,
        "checkpoint_interval": checkpoint_interval,
        "num_evals": num_evals,
        "trajectory_stride": trajectory_stride,
        "device": device,
        "terminate_when_unhealthy": terminate_when_unhealthy,
    }
    if max_episode_steps is not None:
        train_kwargs["max_episode_steps"] = max_episode_steps

    train_config = (
        TrainConfig.from_preset(**train_kwargs) if use_zoo_preset else TrainConfig(**train_kwargs)
    )

    return LoopConfig(
        train=train_config,
        configs=configs,
        seeds=seeds,
        max_iterations=max_iterations,
        pass_threshold=pass_threshold,
        model=model or LLM_MODEL,
        vlm_model=vlm_model if vlm_model is not None else VLM_MODEL,
        thinking_effort=thinking_effort,
        refined_initial_frame=refined_initial_frame,
        runs_dir=runs_dir,
        cores_per_run=cores_per_run,
        max_parallel=max_parallel,
        cores_pool=cores_pool,
        no_cpu_affinity=no_cpu_affinity,
        criteria_diagnosis=criteria_diagnosis,
        motion_trail_dual=motion_trail_dual,
        hp_tuning=hp_tuning,
        use_code_judge=use_code_judge,
        review_reward=review_reward,
        review_judge=review_judge,
        judgment_select=judgment_select,
        use_zoo_preset=use_zoo_preset,
        elaborated_intent=elaborated_intent,
    )
