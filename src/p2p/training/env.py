"""Gymnasium environment wrappers and factory functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from p2p.training.isaaclab_backend import OBJ_POS_ATTRS as _OBJ_POS_ATTRS
from p2p.training.isaaclab_backend import OBJ_ROT_ATTRS as _OBJ_ROT_ATTRS

logger = logging.getLogger(__name__)


class CustomRewardWrapper(gym.Wrapper):
    """Replace environment reward with a user-defined reward function.

    The reward_fn receives (obs, action, next_obs, info) and returns
    (total_reward, terms_dict). terms_dict is stored in info["reward_terms"]
    for per-component logging.

    When ``side_info=True``, ``info["mj_data"]`` and ``info["mj_model"]``
    are injected before calling reward_fn. If the reward function accepts
    6 positional args (obs, action, next_obs, info, mj_data, mj_model),
    they are passed as extra positional args. Otherwise they are available
    via info["mj_data"] and info["mj_model"].
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable[..., tuple[float, dict[str, float]]],
        *,
        side_info: bool = False,
        engine: str = "mujoco",
    ):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.side_info = side_info
        self._last_obs: np.ndarray | None = None
        # Resolve backend once at init, not per-step
        if side_info:
            from p2p.training.simulator import get_simulator

            self._backend = get_simulator(engine)
        else:
            self._backend = None
        # Detect whether reward_fn expects mj_data/mj_model as extra args
        self._uses_extra_args = self._detect_extra_args(reward_fn)
        self._episode_start = True

    @staticmethod
    def _detect_extra_args(fn: Callable) -> bool:
        """Return True if fn accepts > 4 positional parameters (including optional)."""
        import inspect

        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return False
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) > 4

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        # Reset stateful reward_fn's internal state at episode boundary
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        self._episode_start = True
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs_before = self._get_obs()
        obs, _reward, terminated, truncated, info = self.env.step(action)
        mj_data = None
        mj_model = None
        side_keys: list[str] = []
        if self._backend is not None:
            backend = self._backend
            if backend.has_physics_state(self.unwrapped):
                side = backend.extract_side_info(self.unwrapped)
                info.update(side)
                side_keys = list(side.keys())
                # Backward compat: extract mj_data/mj_model for 6-arg path
                mj_data = side.get("mj_data")
                mj_model = side.get("mj_model")
        info["_episode_start"] = self._episode_start
        self._episode_start = False
        if self._uses_extra_args:
            total_reward, terms = self.reward_fn(obs_before, action, obs, info, mj_data, mj_model)
        else:
            total_reward, terms = self.reward_fn(obs_before, action, obs, info)
        # Remove unpicklable engine-specific objects before info is sent over
        # SubprocVecEnv pipes (pickle would crash or OOM).
        for _k in side_keys:
            info.pop(_k, None)
        info["reward_terms"] = terms
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs)
        return obs, float(total_reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Snapshot current observation before step."""
        base = self.unwrapped
        if hasattr(base, "_get_obs"):
            try:
                return base._get_obs()
            except Exception:
                logger.debug("_get_obs() failed, using last observation", exc_info=True)
        # Fallback: use cached observation from last reset/step
        if self._last_obs is not None:
            return self._last_obs
        raise RuntimeError("No observation available — call reset() first")


# ---------------------------------------------------------------------------
# IsaacLab VecEnv reward wrapper
# ---------------------------------------------------------------------------


class _BoolBatch(torch.Tensor):
    """Bool tensor subclass where ``if batch:`` calls ``.any()``.

    LLM-generated reward functions often write ``if info.get("_episode_start"):``.
    That pattern is valid for scalar bools (MuJoCo per-env wrappers) but raises
    ``RuntimeError`` on multi-element tensors.  This subclass makes the scalar
    pattern work transparently while remaining a real ``torch.Tensor`` — so
    ``torch.zeros_like()``, boolean indexing, and all other torch ops work.
    """

    @staticmethod
    def _make(t: torch.Tensor) -> "_BoolBatch":
        """Create a _BoolBatch from an existing bool tensor (no copy)."""
        return t.as_subclass(_BoolBatch)

    def __bool__(self) -> bool:
        return bool(self.as_subclass(torch.Tensor).any())


class IsaacLabRewardVecWrapper:
    """SB3-compatible VecEnv wrapper that overrides reward with an LLM reward_fn.

    Sits between ``Sb3VecEnvWrapper`` and ``VecNormalize``.  Calls the
    reward function ONCE per step with batched tensors ``(num_envs, ...)``.
    The LLM writes vectorized code using ``[:, idx]`` slicing.

    Implements the minimal SB3 ``VecEnv`` interface via delegation so that
    ``VecNormalize`` and PPO work transparently.
    """

    _POS_ATTRS = _OBJ_POS_ATTRS
    _ROT_ATTRS = _OBJ_ROT_ATTRS

    def __init__(
        self,
        venv: Any,
        reward_fn: Callable | None = None,
        reward_code: str = "",
        raw_env: Any = None,
    ) -> None:
        self.venv = venv
        self._torch = torch
        # Expose VecEnv attributes that SB3 / VecNormalize expect
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.metadata = getattr(venv, "metadata", {})

        # Resolve the raw IsaacLab env and cache the robot articulation
        # reference once (not per-step) for efficient robot_data access.
        self._robot: Any = None
        self._unwrapped_env: Any = None
        # Cache env_origins for coordinate frame conversion.
        # Some IsaacLab envs (InHandManipulation, AutoMate) store positions
        # in local frame (env_origins subtracted).  We convert back to world
        # frame so everything in batch_info is consistent with body_pos_w.
        self._env_origins: Any = None
        if raw_env is not None:
            uw = raw_env.unwrapped if hasattr(raw_env, "unwrapped") else raw_env
            self._unwrapped_env = uw
            scene = getattr(uw, "scene", None)
            if scene is not None:
                try:
                    self._robot = scene["robot"]
                except (KeyError, TypeError):
                    arts = list(getattr(scene, "articulations", {}).values())
                    self._robot = arts[0] if arts else None
                self._env_origins = getattr(scene, "env_origins", None)

        # Cache non-robot scene articulations/objects for reward_fn access.
        self._scene_articulations: dict[str, Any] = {}
        if raw_env is not None:
            scene = getattr(self._unwrapped_env, "scene", None)
            if scene is not None:
                for key, art in getattr(scene, "articulations", {}).items():
                    if key == "robot":
                        continue
                    self._scene_articulations[key] = art
                for key, obj in getattr(scene, "rigid_objects", {}).items():
                    self._scene_articulations[key] = obj

        # Cache device from robot data (avoids hardcoded "cuda:0")
        if self._robot is not None:
            self._device = self._robot.data.root_pos_w.device
        else:
            self._device = torch.device("cuda:0")

        # Build ONE reward function (batched — called once for all envs)
        self._reward_fn: Callable = self._build_reward_fn(reward_fn, reward_code)

        # State tracking
        self._last_obs: np.ndarray | None = None
        self._last_actions = np.zeros((self.num_envs, *self.action_space.shape), dtype=np.float32)
        self._episode_starts = np.ones(self.num_envs, dtype=bool)
        self._custom_ep_rew_buf = np.zeros(self.num_envs, dtype=np.float64)
        self._reward_fn_err_count = 0

    # --- reward_fn construction ----------------------------------------

    @staticmethod
    def _build_reward_fn(
        reward_fn: Callable | None,
        reward_code: str,
    ) -> Callable:
        """Build a single reward function (called once per step for all envs)."""
        if reward_code:
            from p2p.training.reward_loader import (
                LegacyRewardWrapper,
                _sanitize_escape_sequences,
                _strip_numpy_imports,
            )
            from p2p.training.simulator import get_simulator

            sanitized = _sanitize_escape_sequences(_strip_numpy_imports(reward_code))
            compiled = compile(sanitized, "<reward_fn>", "exec")
            sim = get_simulator("isaaclab")
            ns: dict[str, Any] = {"np": np, "numpy": np}
            sim.inject_reward_namespace(ns)
            exec(compiled, ns)  # noqa: S102
            fn = ns.get("reward_fn")
            if fn is None:
                raise ValueError("Reward code does not define 'reward_fn'")
            return LegacyRewardWrapper(fn, source=reward_code, engine="isaaclab")
        if reward_fn is not None:
            return reward_fn
        raise ValueError("Either reward_fn or reward_code must be provided")

    # --- SB3 VecEnv interface (delegation) ----------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate anything not defined here to the inner venv."""
        return getattr(self.venv, name)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._last_obs = obs.copy()
        # Initialize actions to zero so step_wait doesn't crash before step_async
        self._last_actions = np.zeros((self.num_envs, *self.action_space.shape), dtype=np.float32)
        self._episode_starts[:] = True
        self._custom_ep_rew_buf[:] = 0.0
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = actions.copy()  # SB3 reuses action buffer
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs, _rewards, dones, infos = self.venv.step_wait()

        obs_before = self._last_obs if self._last_obs is not None else obs
        torch = self._torch
        device = self._device

        # Convert numpy arrays to CUDA tensors — the reward_fn uses torch ops.
        obs_t = torch.as_tensor(obs_before, device=device)
        act_t = torch.as_tensor(self._last_actions, device=device)
        next_obs_t = torch.as_tensor(obs, device=device)

        # Call reward_fn ONCE for all envs (batched tensors).
        robot_data = self._robot.data if self._robot is not None else None
        batch_info: dict[str, Any] = {
            "robot_data": robot_data,
            "_episode_start": _BoolBatch._make(
                torch.as_tensor(self._episode_starts, dtype=torch.bool, device=device)
            ),
        }
        # Convert position attrs from local to world frame (see _env_origins).
        uw = self._unwrapped_env
        if uw is not None:
            origins = self._env_origins
            for attr in (*self._POS_ATTRS, *self._ROT_ATTRS):
                val = getattr(uw, attr, None)
                if val is None:
                    continue
                if attr in self._POS_ATTRS and origins is not None:
                    if val.shape[0] != origins.shape[0] or origins.shape[-1] != 3:
                        logger.warning("env_origins shape mismatch for %s; skipping", attr)
                        continue
                    val = val + origins
                batch_info[attr] = val

        # Scene articulations (non-robot) — e.g. cabinet, manipulated objects
        if self._scene_articulations:
            batch_info["scene"] = {k: art.data for k, art in self._scene_articulations.items()}

        try:
            r, terms = self._reward_fn(obs_t, act_t, next_obs_t, batch_info)
            if isinstance(r, torch.Tensor):
                custom_rewards = r.detach().cpu().numpy().astype(np.float32).flatten()
            else:
                custom_rewards = np.asarray(r, dtype=np.float32).flatten()
            # Convert term tensors to per-step mean floats for logging
            reward_terms: dict[str, float] = {}
            for k, v in terms.items():
                if isinstance(v, torch.Tensor):
                    reward_terms[k] = float(v.detach().cpu().mean())
                else:
                    reward_terms[k] = float(np.mean(v))
            self._reward_fn_err_count = 0
        except Exception:
            self._reward_fn_err_count += 1
            if self._reward_fn_err_count <= 5:
                logger.warning(
                    "reward_fn failed (batched, error %d)",
                    self._reward_fn_err_count,
                    exc_info=True,
                )
            elif self._reward_fn_err_count == 6:
                logger.error(
                    "reward_fn has failed %d consecutive times — training on "
                    "ZERO reward. The reward function likely has a bug.",
                    self._reward_fn_err_count,
                )
            custom_rewards = np.zeros(self.num_envs, dtype=np.float32)
            reward_terms = {}

        # Populate per-env infos with reward_terms (same for all envs in batch)
        for i in range(self.num_envs):
            infos[i]["reward_terms"] = reward_terms

        # Track custom episode rewards; handle episode boundaries
        self._custom_ep_rew_buf += custom_rewards
        done_indices = np.where(dones)[0]
        for i in done_indices:
            ep = infos[i].get("episode")
            if ep is not None:
                ep["r"] = float(self._custom_ep_rew_buf[i])
        self._custom_ep_rew_buf[done_indices] = 0.0

        # Episode start tracking (no stateful reset — batched rewards use
        # _episode_start flag inside the reward_fn for per-env state management)
        self._episode_starts = dones.copy()
        self._last_obs = obs.copy()
        self._last_actions = None  # consumed

        return obs, custom_rewards, dones, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        self.venv.close()

    def seed(self, seed: int | None = None) -> list[int | None]:
        if hasattr(self.venv, "seed"):
            return self.venv.seed(seed)
        return [seed] * self.num_envs

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        if hasattr(self.venv, "env_is_wrapped"):
            return self.venv.env_is_wrapped(wrapper_class, indices)
        return [False] * self.num_envs


# ---------------------------------------------------------------------------
# Shared env construction — single source of truth for train & eval
# ---------------------------------------------------------------------------


def _disable_debug_vis(cfg: object) -> None:
    """Recursively disable ``debug_vis`` on all nested config objects.

    IsaacLab task configs (velocity commands, sensors, etc.) often ship with
    ``debug_vis=True`` which draws arrows/markers that waste GPU cycles during
    training and clutter eval videos.
    """
    for attr in list(vars(cfg)):
        if attr.startswith("_"):
            continue
        val = getattr(cfg, attr, None)
        if attr == "debug_vis" and isinstance(val, bool):
            setattr(cfg, attr, False)
        elif hasattr(val, "__dict__") and not callable(val):
            _disable_debug_vis(val)


def _disable_early_termination(env_cfg: object) -> None:
    """Disable early termination so eval episodes run full length.

    Preserves ``time_out`` so episodes still end at ``max_episode_steps``.

    Handles both environment architectures:

    * **Manager-based** envs have an ``env_cfg.terminations`` object whose
      named terms are set to ``None``.
    * **Direct RL** envs hard-code termination in ``_get_dones()`` using
      config scalars (``termination_height``, ``fall_dist``).  We push
      these to extreme values (``-inf`` / ``inf``) so the checks never
      fire.  ``fall_penalty``
      is zeroed to avoid reward distortion from the (now-unreachable)
      penalty branch.
    """
    # --- Direct RL: locomotion (Ant, Humanoid, HumanoidAMP) ---
    if hasattr(env_cfg, "termination_height"):
        logger.info(
            "Disabled Direct RL locomotion termination (termination_height %s -> -inf)",
            env_cfg.termination_height,
        )
        env_cfg.termination_height = -float("inf")

    # --- Direct RL: dexterous manipulation (Shadow Hand, Allegro) ---
    if hasattr(env_cfg, "fall_dist"):
        logger.info(
            "Disabled Direct RL dexterous termination (fall_dist %s -> inf)",
            env_cfg.fall_dist,
        )
        env_cfg.fall_dist = float("inf")

    if hasattr(env_cfg, "fall_penalty"):
        logger.info(
            "Zeroed fall_penalty (%s -> 0.0) to avoid reward distortion",
            env_cfg.fall_penalty,
        )
        env_cfg.fall_penalty = 0.0

    # --- Direct RL: HumanoidAMP early_termination flag ---
    if hasattr(env_cfg, "early_termination"):
        logger.info(
            "Disabled Direct RL early_termination flag (%s -> False)",
            env_cfg.early_termination,
        )
        env_cfg.early_termination = False

    # --- Manager-based envs: named termination terms ---
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return
    for name in list(vars(terms)):
        if name.startswith("_") or name == "time_out":
            continue
        term = getattr(terms, name, None)
        if term is not None and hasattr(term, "func"):
            setattr(terms, name, None)
            logger.debug("Disabled termination term '%s'", name)


def _disable_direct_rl_termination(env: gym.Env) -> None:
    """Monkey-patch ``_get_dones`` on Direct RL envs to suppress early termination.

    Some Direct RL envs (e.g. ANYmal-C) hard-code termination conditions in
    ``_get_dones()`` using contact-force checks with no config scalar to
    override.  This function patches the method on the env *instance* so that
    the ``died`` tensor is always zeros, preserving only ``time_out``.

    Must be called **after** ``gym.make()`` — it operates on the live env, not
    the config.
    """
    import types

    import torch

    unwrapped = env.unwrapped

    try:
        from isaaclab.envs import DirectRLEnv
    except ImportError:
        return

    if not isinstance(unwrapped, DirectRLEnv):
        return

    if getattr(unwrapped, "_sar_termination_disabled", False):
        return
    unwrapped._sar_termination_disabled = True

    original_get_dones = unwrapped._get_dones

    def _patched_get_dones(self: DirectRLEnv) -> tuple:
        _died, time_out = original_get_dones()
        return torch.zeros_like(_died), time_out

    unwrapped._get_dones = types.MethodType(_patched_get_dones, unwrapped)
    logger.info(
        "Monkey-patched _get_dones on %s to disable early termination",
        type(unwrapped).__name__,
    )


_isaaclab_bootstrapped = False


def _bootstrap_isaaclab(*, enable_cameras: bool = False) -> None:
    """Initialize Isaac Sim runtime and register IsaacLab envs with Gymnasium.

    Must be called once per process before ``gym.make()`` on IsaacLab envs.
    Subsequent calls are no-ops.

    Args:
        enable_cameras: Enable offscreen rendering for video capture.
            Required for ``render_mode="rgb_array"`` (eval videos).
            Uses ``AppLauncher`` which loads the correct rendering kit file.
    """
    global _isaaclab_bootstrapped  # noqa: PLW0603
    if _isaaclab_bootstrapped:
        return

    import os

    if enable_cameras:
        os.environ["ENABLE_CAMERAS"] = "1"

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=enable_cameras)  # noqa: F841

    # Fix Nucleus asset root for pip installs: the "cloud" root (used by
    # IsaacLab assets) is None by default.  Copy from the "default" root
    # which Isaac Sim sets to the S3 CDN URL.
    import carb

    settings = carb.settings.get_settings()
    cloud_root = settings.get("/persistent/isaac/asset_root/cloud")
    if cloud_root is None:
        default_root = settings.get("/persistent/isaac/asset_root/default")
        if default_root:
            settings.set("/persistent/isaac/asset_root/cloud", default_root)
            logger.info("Set Nucleus cloud root to: %s", default_root)

    import isaaclab_tasks  # noqa: F401 — triggers gym.register() calls

    import p2p.isaaclab_envs  # noqa: F401 — registers custom SAR envs

    _isaaclab_bootstrapped = True
    logger.info("IsaacLab runtime bootstrapped, envs registered with Gymnasium")


def _make_env(
    env_id: str,
    reward_fn: Callable,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
    *,
    side_info: bool = False,
    terminate_when_unhealthy: bool | None = None,
    engine: str = "mujoco",
) -> gym.Env:
    """Create base env + CustomRewardWrapper. Used by both train and eval.

    For IsaacLab single-env eval, wraps with CustomRewardWrapper just like
    MuJoCo (num_envs=1, so ``robot_data`` tensors are ``(1, ...)`` and the
    reward_fn's ``[0]`` indexing works correctly).
    """
    if engine == "isaaclab":
        # Enable cameras when rendering is needed (eval videos)
        _bootstrap_isaaclab(enable_cameras=render_mode == "rgb_array")
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1)
        if render_mode == "rgb_array":
            env_cfg.sim.render_interval = env_cfg.decimation

        _disable_debug_vis(env_cfg)
        if terminate_when_unhealthy is False:
            _disable_early_termination(env_cfg)

        env = gym.make(env_id, cfg=env_cfg, render_mode=render_mode)

        if terminate_when_unhealthy is False:
            _disable_direct_rl_termination(env)

        # Apply custom reward wrapper (same as MuJoCo). With num_envs=1 the
        # robot_data tensors are (1, ...) so [0] indexing works correctly.
        return CustomRewardWrapper(env, reward_fn, side_info=side_info, engine=engine)

    kwargs: dict[str, Any] = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    if terminate_when_unhealthy is not None:
        kwargs["terminate_when_unhealthy"] = terminate_when_unhealthy
    try:
        env = gym.make(env_id, **kwargs)
    except TypeError as e:
        if "terminate_when_unhealthy" not in str(e):
            raise
        kwargs.pop("terminate_when_unhealthy", None)
        env = gym.make(env_id, **kwargs)
    return CustomRewardWrapper(env, reward_fn, side_info=side_info, engine=engine)


# ---------------------------------------------------------------------------
# SB3-style env factories (train)
# ---------------------------------------------------------------------------


def make_env_sb3(
    env_id: str,
    reward_fn: Callable,
    seed: int,
    max_episode_steps: int | None = None,
    *,
    side_info: bool = False,
    terminate_when_unhealthy: bool = False,
    engine: str = "mujoco",
) -> Callable[[], gym.Env]:
    """SB3 env factory with a pre-built reward function (stateless rewards)."""

    def thunk() -> gym.Env:
        env = _make_env(
            env_id,
            reward_fn,
            max_episode_steps,
            side_info=side_info,
            terminate_when_unhealthy=terminate_when_unhealthy,
            engine=engine,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_sb3_from_code(
    env_id: str,
    reward_code: str,
    seed: int,
    max_episode_steps: int | None = None,
    *,
    side_info: bool = False,
    terminate_when_unhealthy: bool = False,
    engine: str = "mujoco",
) -> Callable[[], gym.Env]:
    """SB3 env factory that loads reward from source code.

    Each call to thunk() executes the code fresh, producing an independent
    reward function instance. This is essential for stateful closures
    (e.g. back_tumble.py) running in SubprocVecEnv where each subprocess
    needs its own state.
    """
    from p2p.training.reward_loader import _sanitize_escape_sequences, _strip_numpy_imports

    sanitized_code = _sanitize_escape_sequences(_strip_numpy_imports(reward_code))

    def thunk() -> gym.Env:
        from p2p.training.reward_loader import LegacyRewardWrapper
        from p2p.training.simulator import get_simulator

        ns: dict[str, Any] = {"np": np, "numpy": np}
        get_simulator(engine).inject_reward_namespace(ns)
        exec(sanitized_code, ns)  # noqa: S102
        fn = ns.get("reward_fn")
        if fn is None:
            raise ValueError("Reward code does not define 'reward_fn'")
        # Wrap in LegacyRewardWrapper so .reset() can re-create the closure
        # from source, clearing stateful variables between episodes.
        fn = LegacyRewardWrapper(fn, source=reward_code, engine=engine)
        env = _make_env(
            env_id,
            fn,
            max_episode_steps,
            side_info=side_info,
            terminate_when_unhealthy=terminate_when_unhealthy,
            engine=engine,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Eval env factory
# ---------------------------------------------------------------------------


def make_eval_env(
    env_id: str,
    reward_fn: Callable,
    max_episode_steps: int | None = None,
    *,
    side_info: bool = False,
    engine: str = "mujoco",
) -> gym.Env:
    """Create a single eval env. Same wrapper stack as train + render support.

    Disables ``terminate_when_unhealthy`` (Humanoid, Ant, Hopper, Walker2d)
    so eval episodes always run to ``max_episode_steps`` and produce
    full-length videos regardless of policy quality.
    """
    return _make_env(
        env_id,
        reward_fn,
        max_episode_steps,
        render_mode="rgb_array",
        side_info=side_info,
        terminate_when_unhealthy=False,
        engine=engine,
    )


def make_eval_vec_env(
    env_id: str,
    reward_fn: Callable,
    num_envs: int,
    seed: int,
    max_episode_steps: int | None = None,
    *,
    side_info: bool = False,
    reward_code: str = "",
    engine: str = "mujoco",
) -> Any:
    """Create a SubprocVecEnv for parallel eval (no rendering).

    Each env gets unique seed: seed, seed+1, ..., seed+num_envs-1.
    VecNormalize wrapping is handled by the caller.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if reward_code:
        env_fns = [
            make_env_sb3_from_code(
                env_id,
                reward_code,
                seed + i,
                max_episode_steps,
                side_info=side_info,
                engine=engine,
            )
            for i in range(num_envs)
        ]
    else:
        env_fns = [
            make_env_sb3(
                env_id,
                reward_fn,
                seed + i,
                max_episode_steps,
                side_info=side_info,
                engine=engine,
            )
            for i in range(num_envs)
        ]
    return SubprocVecEnv(env_fns)
