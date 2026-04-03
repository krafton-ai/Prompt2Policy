"""SB3-based training: PPO.

Produces the following artifact structure:
  - metrics/scalars.jsonl  (training + eval entries)
  - videos/eval_{step}_{p10|median|p90}.mp4  (parallel eval)
  - videos/eval_{step}_ep{N}.mp4  (sequential fallback)
  - trajectory_{step}_{p10|median|p90}.npz
  - checkpoints/final.zip
"""

# ruff: noqa: I001 — p2p.settings must be imported before gymnasium/mujoco
from __future__ import annotations

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports

import json
import logging
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from p2p.config import TrainConfig
from p2p.contracts import EvalResult, EvalScalar, TrainSummary, round_metric
from p2p.training.env import (
    make_env_sb3,
    make_env_sb3_from_code,
    make_eval_env,
    make_eval_vec_env,
)

logger = logging.getLogger(__name__)

try:
    from p2p.training.ppo_diagnostic import PPODiagnostic as PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
except ImportError as e:
    raise ImportError(
        "SB3 backend requires stable-baselines3. Install with: pip install stable-baselines3"
    ) from e


# IsaacLab eval uses fewer parallel envs than training to limit GPU memory.
_MAX_ISAACLAB_EVAL_ENVS = 16

# ---------------------------------------------------------------------------
# Trajectory entry builder (single source of truth for field set)
# ---------------------------------------------------------------------------


def build_trajectory_entry(
    *,
    step: int,
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    reward: float,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    dt: float,
    env_id: str,
    env_unwrapped: Any,
    engine: str = "mujoco",
) -> dict[str, Any]:
    """Build a single trajectory entry dict.

    This is the canonical implementation — every trajectory producer
    (evaluation, sample collection, etc.) should call this to guarantee
    field parity.
    """
    terms = info.get("reward_terms", {})
    entry: dict[str, Any] = {
        "step": step,
        "timestamp": float(step * dt),
        "obs": np.asarray(obs).tolist(),
        "action": np.asarray(action).tolist(),
        "next_obs": np.asarray(next_obs).tolist(),
        "reward": float(reward),
        "reward_terms": {k: float(v) for k, v in terms.items()},
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    from p2p.training.simulator import get_simulator

    backend = get_simulator(engine)
    if backend.has_physics_state(env_unwrapped):
        entry.update(backend.build_trajectory_fields(env_unwrapped, env_id, action, info))
    return entry


# ---------------------------------------------------------------------------
# Evaluation helper (mirrors ppo.py::run_evaluation output format)
# ---------------------------------------------------------------------------


def _round_floats(obj: Any, precision: int) -> Any:
    """Recursively round floats in nested dicts/lists."""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, precision) for v in obj]
    return obj


def _run_single_eval_episode(
    env_id: str,
    reward_fn: Callable,
    model: Any,
    seed: int,
    max_episode_steps: int | None,
    vec_normalize: Any | None,
    *,
    side_info: bool = False,
    trajectory_stride: int = 5,
    trajectory_precision: int = 4,
    engine: str = "mujoco",
    existing_env: Any | None = None,
) -> tuple[float, int, dict[str, float], list[np.ndarray], list[dict], int]:
    """Run one deterministic eval episode.

    Args:
        existing_env: For IsaacLab, pass the training VecEnv to avoid
            creating a second scene (Isaac Sim single-scene limitation).
            If None, creates a new eval env (MuJoCo path).

    Returns (reward, steps, term_means, frames, trajectory, render_fps).
    """
    if existing_env is not None:
        # IsaacLab: use the training VecEnv directly
        venv = existing_env
        # Unwrap to get the raw IsaacLab env for rendering
        raw = venv
        while hasattr(raw, "venv"):
            raw = raw.venv
        env = raw.env if hasattr(raw, "env") else raw
        dt = getattr(env.unwrapped, "step_dt", 0.02)
        render_fps = max(1, int(1 / dt))
        owns_env = False
    else:
        env = make_eval_env(
            env_id,
            reward_fn,
            max_episode_steps,
            side_info=side_info,
            engine=engine,
        )
        venv = None
        dt = getattr(env.unwrapped, "step_dt", None) or getattr(env.unwrapped, "dt", 1 / 30)
        render_fps = max(1, int(env.metadata.get("render_fps", 1 / dt)))
        owns_env = True

    def _to_numpy(x: Any) -> np.ndarray:
        """Convert obs (numpy or dict of torch tensors) to flat numpy."""
        if isinstance(x, dict):
            # IsaacLab: obs is {"policy": tensor(1, obs_dim)}
            v = x.get("policy", next(iter(x.values())))
            if hasattr(v, "detach"):
                return v[0].detach().cpu().numpy()
            return np.asarray(v).flatten()
        return np.asarray(x).flatten()

    def _normalize_obs(o: np.ndarray) -> np.ndarray:
        if vec_normalize is None or not getattr(vec_normalize, "norm_obs", False):
            return o
        return vec_normalize.normalize_obs(o.reshape(1, -1)).reshape(-1)

    frames: list[np.ndarray] = []
    seg_masks: list[np.ndarray] = []
    trajectory: list[dict] = []

    # Segmentation renderer for MuJoCo (used by motion overlay flow arrows)
    _seg_renderer = None
    if engine == "mujoco" and venv is None:
        try:
            import mujoco as _mj

            _seg_renderer = _mj.Renderer(
                env.unwrapped.model,
                height=env.unwrapped.mujoco_renderer.height,
                width=env.unwrapped.mujoco_renderer.width,
            )
        except Exception:
            pass  # segmentation unavailable — flow arrows will run without mask

    # Reset: VecEnv (IsaacLab) vs single env (MuJoCo)
    if venv is not None:
        obs_raw = venv.reset()
        obs = np.asarray(obs_raw).flatten()
        # IsaacLab Direct RL envs randomize episode_length_buf on full
        # reset to stagger training resets, causing spurious time_out
        # with num_envs=1.  Zero it so eval runs its full length.
        if hasattr(env.unwrapped, "episode_length_buf"):
            env.unwrapped.episode_length_buf[:] = 0
    else:
        obs_raw, _ = env.reset(seed=seed)
        obs = _to_numpy(obs_raw)

    total_reward = 0.0
    term_accum: dict[str, float] = defaultdict(float)
    steps = 0
    ep_max_steps = max_episode_steps or 1000

    while steps < ep_max_steps:
        # Render frame (catch warmup errors for IsaacLab)
        try:
            frame = env.render()
            if frame is not None and hasattr(frame, "size") and frame.size > 0:
                frames.append(frame)
                # Capture segmentation mask alongside RGB
                if _seg_renderer is not None:
                    try:
                        _seg_renderer.update_scene(env.unwrapped.data)
                        _seg_renderer.enable_segmentation_rendering()
                        seg = _seg_renderer.render()
                        # body_id > 0 = robot, 0 = world/floor
                        seg_masks.append((seg[..., 0] > 0).astype(np.uint8))
                    except Exception:
                        seg_masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
        except (TypeError, RuntimeError):
            pass  # IsaacLab render warmup — skip frame

        action, _ = model.predict(_normalize_obs(obs), deterministic=True)
        action_np = np.asarray(action).flatten()

        # Step: VecEnv (IsaacLab) vs single env (MuJoCo)
        if venv is not None:
            obs_raw, rewards, dones, infos = venv.step(action_np.reshape(1, -1))
            obs_next = np.asarray(obs_raw).flatten()
            reward_val = float(rewards[0])
            info = infos[0] if isinstance(infos, list) else infos
            terminated = bool(dones[0])
            truncated = False
            # Check episode info from Sb3VecEnvWrapper
            ep_info = info.get("episode") if isinstance(info, dict) else None
            if ep_info is not None:
                total_reward = float(ep_info["r"])
        else:
            action_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_obs_raw, reward_val, terminated, truncated, info = env.step(action_clipped)
            obs_next = _to_numpy(next_obs_raw)
            if hasattr(reward_val, "item"):
                reward_val = reward_val.item()
            if hasattr(terminated, "item"):
                terminated = bool(terminated.item())
            if hasattr(truncated, "item"):
                truncated = bool(truncated.item())
            ep_info = None

        if ep_info is None:
            total_reward += reward_val

        terms = info.get("reward_terms", {}) if isinstance(info, dict) else {}
        for k, v in terms.items():
            term_accum[k] += float(v) if not hasattr(v, "item") else v.item()

        if steps % trajectory_stride == 0 or terminated or truncated:
            traj_entry = build_trajectory_entry(
                step=steps,
                obs=obs,
                action=action_np,
                next_obs=obs_next,
                reward=reward_val,
                info=info if isinstance(info, dict) else {},
                terminated=terminated,
                truncated=truncated,
                dt=dt,
                env_id=env_id,
                env_unwrapped=env.unwrapped,
                engine=engine,
            )
            trajectory.append(_round_floats(traj_entry, trajectory_precision))

        steps += 1
        obs = obs_next
        if terminated or truncated:
            break

    if _seg_renderer is not None:
        _seg_renderer.close()
    if owns_env:
        env.close()

    term_means = {k: float(v / max(steps, 1)) for k, v in term_accum.items()}
    return total_reward, steps, term_means, frames, trajectory, render_fps, seg_masks


def _save_video(frames: list[np.ndarray], path: Path, render_fps: int = 30) -> None:
    """Write frames to an mp4 file with real-time playback speed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = min(render_fps, 30)
    n_original = len(frames)
    n_resampled = max(1, round(n_original * output_fps / render_fps))
    indices = np.linspace(0, n_original - 1, n_resampled).round().astype(int)
    writer = imageio.get_writer(
        str(path), fps=output_fps, output_params=["-movflags", "+faststart"]
    )
    for i in indices:
        writer.append_data(frames[i])
    writer.close()


def _save_seg_masks(seg_masks: list[np.ndarray], video_path: Path) -> None:
    """Save segmentation masks as compressed .npz alongside the video.

    Masks are binary (H, W) uint8 arrays where 1 = robot body.
    Saved only when non-empty; the motion overlay pipeline checks for the
    file and falls back to mask-free mode if absent.
    """
    if not seg_masks:
        return
    out = video_path.with_suffix(".seg.npz")
    try:
        np.savez_compressed(str(out), masks=np.stack(seg_masks))
    except Exception:
        pass  # non-critical — overlay will work without masks


def _save_trajectory(trajectory: list[dict], path: Path) -> None:
    """Write trajectory steps to a compressed NPZ file.

    Each field is stored as a columnar numpy array for ~7x smaller files
    compared to the legacy JSONL format.  ``load_trajectory`` reconstructs
    the original ``list[dict]`` transparently.
    """
    if not trajectory:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    sample = trajectory[0]
    arrays: dict[str, np.ndarray] = {}
    bool_fields: list[str] = []
    int_fields: list[str] = []

    for key in sample:
        if key == "reward_terms":
            term_names = sorted(sample["reward_terms"].keys())
            if term_names:
                arrays["_reward_term_names"] = np.array(term_names)
                arrays["reward_terms"] = np.array(
                    [
                        [step.get("reward_terms", {}).get(t, 0.0) for t in term_names]
                        for step in trajectory
                    ],
                    dtype=np.float32,
                )
            continue

        values = [step.get(key) for step in trajectory]
        first = values[0]

        if isinstance(first, bool):
            arrays[key] = np.array(values, dtype=np.uint8)
            bool_fields.append(key)
        elif isinstance(first, int):
            arrays[key] = np.array(values, dtype=np.int64)
            int_fields.append(key)
        elif isinstance(first, (float, np.floating)):
            arrays[key] = np.array(values, dtype=np.float32)
        elif isinstance(first, list):
            arrays[key] = np.array(values, dtype=np.float32)

    if bool_fields:
        arrays["_bool_fields"] = np.array(bool_fields)
    if int_fields:
        arrays["_int_fields"] = np.array(int_fields)

    np.savez_compressed(str(path), **arrays)


def _percentile_index(sorted_data: list, percentile: float) -> int:
    """Return the index in sorted_data closest to the given percentile."""
    idx = int(len(sorted_data) * percentile / 100)
    return min(idx, len(sorted_data) - 1)


def _sync_vec_normalize(
    train_vn: VecNormalize, eval_venv: Any, config: TrainConfig
) -> VecNormalize:
    """Wrap eval SubprocVecEnv with VecNormalize, copying stats from training."""
    eval_vn = VecNormalize(
        eval_venv,
        norm_obs=config.normalize_obs,
        norm_reward=False,
        clip_obs=config.obs_clip,
        clip_reward=config.reward_clip,
        gamma=config.gamma,
    )
    eval_vn.obs_rms = train_vn.obs_rms.copy()
    eval_vn.ret_rms = train_vn.ret_rms.copy()
    eval_vn.training = False
    eval_vn.norm_reward = False
    return eval_vn


def _run_parallel_eval_episodes(
    env_id: str,
    reward_fn: Callable,
    model: Any,
    num_envs: int,
    num_episodes: int,
    seed: int,
    max_episode_steps: int | None,
    vec_normalize: Any | None,
    config: TrainConfig,
    *,
    side_info: bool = False,
    reward_code: str = "",
    engine: str = "mujoco",
    eval_venv: Any | None = None,
    max_steps: int = 0,
) -> list[tuple[float, int]]:
    """Phase 1: run episodes in parallel, return [(return, seed), ...].

    Args:
        eval_venv: Pre-created VecEnv (IsaacLab path). When provided, this
            VecEnv is reused for all rounds instead of creating a new
            SubprocVecEnv per round.  The caller owns the env and is
            responsible for closing it.
        max_steps: Maximum number of steps per round.  When > 0, the loop
            breaks after this many steps even if some episodes have not
            terminated.  Episodes that did not complete use accumulated
            reward as their return (needed for IsaacLab episodes that can
            run 1000+ steps).
    """
    from math import ceil

    results: list[tuple[float, int]] = []
    num_rounds = ceil(num_episodes / num_envs)
    owns_venv = eval_venv is None

    for round_idx in range(num_rounds):
        round_base_seed = seed + round_idx * num_envs
        envs_this_round = min(num_envs, num_episodes - len(results))

        if owns_venv:
            round_venv = make_eval_vec_env(
                env_id,
                reward_fn,
                envs_this_round,
                round_base_seed,
                max_episode_steps,
                side_info=side_info,
                reward_code=reward_code,
                engine=engine,
            )
        else:
            round_venv = eval_venv

        if vec_normalize is not None and isinstance(vec_normalize, VecNormalize):
            round_venv = _sync_vec_normalize(vec_normalize, round_venv, config)

        try:
            done_mask = [False] * envs_this_round
            episode_returns: list[float | None] = [None] * envs_this_round
            accum_rewards = np.zeros(envs_this_round, dtype=np.float64)

            if hasattr(round_venv, "seed"):
                round_venv.seed(round_base_seed)
            obs = round_venv.reset()

            step_count = 0
            while not all(done_mask):
                if max_steps > 0 and step_count >= max_steps:
                    break
                actions, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = round_venv.step(actions)
                accum_rewards += np.asarray(rewards[:envs_this_round], dtype=np.float64)
                step_count += 1

                for i in range(envs_this_round):
                    if dones[i] and not done_mask[i]:
                        ep_info = infos[i].get("episode")
                        if ep_info is not None:
                            episode_returns[i] = float(ep_info["r"])
                        else:
                            episode_returns[i] = float(accum_rewards[i])
                        done_mask[i] = True
                        accum_rewards[i] = 0.0

            for i in range(envs_this_round):
                ep_seed = round_base_seed + i
                # Use episode return if completed, else truncated accumulated reward
                if episode_returns[i] is not None:
                    ret = episode_returns[i]
                else:
                    ret = float(accum_rewards[i])
                results.append((ret, ep_seed))
        finally:
            if owns_venv:
                round_venv.close()

    return results


def run_evaluation_sb3(
    env_id: str,
    reward_fn: Callable,
    model: Any,
    video_path: Path,
    seed: int = 0,
    max_episode_steps: int | None = None,
    vec_normalize: Any | None = None,
    *,
    side_info: bool = False,
    num_eval_rounds: int = 1,
    num_eval_envs: int = 0,
    parallel_eval: bool = True,
    config: TrainConfig | None = None,
    reward_code: str = "",
    trajectory_stride: int = 5,
    trajectory_precision: int = 4,
    engine: str = "mujoco",
) -> EvalResult:
    """Run deterministic eval episodes and save representative videos.

    Two modes controlled by ``parallel_eval``:
      Parallel (parallel_eval=True):
        Phase 1: Fast stats via SubprocVecEnv (no rendering)
        Phase 2: Re-run 3 representative episodes (P10, median, P90) for
                 video + trajectory capture
        Output: videos/eval_{step}_{p10|median|p90}.mp4

      Sequential (parallel_eval=False):
        Run episodes one by one, save per-episode videos
        Output: videos/eval_{step}_ep{N}.mp4
    """
    # Freeze VecNormalize during eval
    _prev_training = None
    _prev_norm_reward = None
    if vec_normalize is not None:
        _prev_training = vec_normalize.training
        _prev_norm_reward = vec_normalize.norm_reward
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    step_label = video_path.stem.split("_")[-1]
    videos_dir = video_path.parent
    iter_dir = videos_dir.parent
    videos_dir.mkdir(parents=True, exist_ok=True)

    effective_eval_envs = (
        num_eval_envs if num_eval_envs > 0 else (config.num_envs if config else 1)
    )
    use_parallel = parallel_eval and config is not None

    try:
        if use_parallel:
            # ── Phase 1: parallel stats collection ─────────────────
            num_episodes = effective_eval_envs * num_eval_rounds
            episode_data = _run_parallel_eval_episodes(
                env_id,
                reward_fn,
                model,
                effective_eval_envs,
                num_episodes,
                seed,
                max_episode_steps,
                vec_normalize,
                config,
                side_info=side_info,
                reward_code=reward_code,
                engine=engine,
            )
            all_returns = [r for r, _s in episode_data]

            # Sort by return for percentile selection
            sorted_data = sorted(episode_data, key=lambda x: x[0])

            # ── Phase 2: capture P10, median, P90 ─────────────────
            capture_targets = {
                "p10": _percentile_index(sorted_data, 10),
                "median": _percentile_index(sorted_data, 50),
                "p90": _percentile_index(sorted_data, 90),
            }

            # Use median episode for EvalResult metadata
            median_steps = 0
            median_terms: dict[str, float] = {}

            for label, idx in capture_targets.items():
                _ret, ep_seed = sorted_data[idx]
                _reward, steps, terms, frames, traj, render_fps, seg = _run_single_eval_episode(
                    env_id,
                    reward_fn,
                    model,
                    ep_seed,
                    max_episode_steps,
                    vec_normalize,
                    side_info=side_info,
                    trajectory_stride=trajectory_stride,
                    trajectory_precision=trajectory_precision,
                    engine=engine,
                )
                video_path = videos_dir / f"eval_{step_label}_{label}.mp4"
                _save_video(frames, video_path, render_fps)
                _save_seg_masks(seg, video_path)
                _save_trajectory(traj, iter_dir / f"trajectory_{step_label}_{label}.npz")

                if label == "median":
                    median_steps = steps
                    median_terms = terms

            capture_steps = median_steps
            capture_terms = median_terms

        else:
            # ── Sequential fallback ───────────────────────────────
            best_reward = float("-inf")
            best_steps = 0
            best_terms: dict[str, float] = {}
            all_returns = []

            seq_num_episodes = 10  # fixed count for sequential fallback
            consecutive_failures = 0
            last_error: Exception | None = None
            for ep_idx in range(seq_num_episodes):
                ep_seed = seed + ep_idx
                try:
                    reward, steps, terms, frames, traj, render_fps, seg = _run_single_eval_episode(
                        env_id,
                        reward_fn,
                        model,
                        ep_seed,
                        max_episode_steps,
                        vec_normalize,
                        side_info=side_info,
                        trajectory_stride=trajectory_stride,
                        trajectory_precision=trajectory_precision,
                        engine=engine,
                    )
                except Exception as e:
                    consecutive_failures += 1
                    last_error = e
                    logger.warning(
                        "Eval episode %d/%d failed (seed=%d, env=%s): %s",
                        ep_idx + 1,
                        seq_num_episodes,
                        ep_seed,
                        env_id,
                        e,
                    )
                    if consecutive_failures >= 3:
                        raise RuntimeError(
                            f"3 consecutive eval episodes failed (env={env_id}, "
                            f"last seed={ep_seed})"
                        ) from e
                    continue
                else:
                    consecutive_failures = 0
                    all_returns.append(reward)
                    ep_video_path = videos_dir / f"eval_{step_label}_ep{ep_idx}.mp4"
                    _save_video(frames, ep_video_path, render_fps)
                    _save_seg_masks(seg, ep_video_path)
                    ep_traj_path = iter_dir / f"trajectory_{step_label}_ep{ep_idx}.npz"
                    _save_trajectory(traj, ep_traj_path)
                    if reward > best_reward:
                        best_reward = reward
                        best_steps = steps
                        best_terms = terms

            if not all_returns:
                raise RuntimeError(
                    f"All {seq_num_episodes} eval episodes failed for {env_id} — "
                    f"no results to save. Last error: {last_error}"
                )
            capture_steps = best_steps
            capture_terms = best_terms

    finally:
        if vec_normalize is not None:
            vec_normalize.training = _prev_training
            vec_normalize.norm_reward = _prev_norm_reward

    result: EvalResult = {
        "total_reward": round_metric(float(max(all_returns))),
        "episode_length": capture_steps,
        "reward_terms": capture_terms,
    }

    if len(all_returns) > 1:
        returns_arr = np.array(all_returns)
        result["num_eval_rounds"] = num_eval_rounds
        result["mean_return"] = round_metric(float(np.mean(returns_arr)))
        result["std_return"] = round_metric(float(np.std(returns_arr)))
        result["min_return"] = round_metric(float(np.min(returns_arr)))
        result["max_return"] = round_metric(float(np.max(returns_arr)))
        result["median_return"] = round_metric(float(np.median(returns_arr)))
        result["p10_return"] = round_metric(float(np.percentile(returns_arr, 10)))
        result["p90_return"] = round_metric(float(np.percentile(returns_arr, 90)))
        result["all_returns"] = [round_metric(float(r)) for r in all_returns]

    return result


# ---------------------------------------------------------------------------
# TensorBoard tag grouping (mirrors dashboard sections)
# ---------------------------------------------------------------------------

_TB_TAG_MAP: dict[str, str] = {
    # Episodic Return
    "episodic_return": "return/episodic_return",
    # Episode Stats
    "episodic_return_std": "episode_stats/return_std",
    "episodic_return_min": "episode_stats/return_min",
    "episodic_return_max": "episode_stats/return_max",
    "episode_length": "episode_stats/episode_length",
    # Loss
    "policy_loss": "loss/clip_loss",
    "value_loss": "loss/value_loss",
    "entropy": "loss/exploration_loss",
    # PPO Diagnostic
    "clip_fraction": "ppo_diagnostic/clip_fraction",
    "approx_kl": "ppo_diagnostic/approx_kl",
    "kl_mean_term": "ppo_diagnostic/kl_mean_term",
    "kl_var_term": "ppo_diagnostic/kl_var_term",
    "mean_shift_normalized": "ppo_diagnostic/mean_shift_normalized",
    # Policy & Gradient Health
    "policy_std": "health/policy_std",
    "grad_norm": "health/grad_norm",
    # General
    "learning_rate": "general/learning_rate",
    "sps": "general/sps",
    # Throughput
    "rollout_time": "throughput/rollout_time",
    "train_time": "throughput/train_time",
    "elapsed_time": "throughput/elapsed_time",
}


def _tb_tag(key: str) -> str:
    """Map flat scalars key to grouped TensorBoard tag."""
    if key in _TB_TAG_MAP:
        return _TB_TAG_MAP[key]
    if key.startswith("reward_term_"):
        return f"reward_terms/{key.removeprefix('reward_term_')}"
    return f"other/{key}"


# ---------------------------------------------------------------------------
# SB3 Callback (streams metrics in scalars.jsonl format)
# ---------------------------------------------------------------------------


class P2PCallback(BaseCallback):
    """SB3 callback: metrics logging, video capture, diagnostics."""

    def __init__(
        self,
        config: TrainConfig,
        reward_fn: Callable,
        env_id: str,
        iteration_dir: Path,
        total_timesteps: int,
        algorithm: str,
        session_heartbeat_fn: Callable[[], None] | None = None,
        reward_code: str = "",
    ):
        super().__init__(verbose=0)
        self.config = config
        self.reward_fn = reward_fn
        self.reward_code = reward_code
        self.env_id = env_id
        self.iteration_dir = iteration_dir
        self.total_timesteps = total_timesteps
        self.algorithm = algorithm
        self._session_heartbeat_fn = session_heartbeat_fn
        self._last_heartbeat_t: float = 0.0

        # Metrics
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._update_count = 0

        # Per-env accumulators for reward terms
        self._env_reward_terms: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Rolling buffer of completed episodes (last 10)
        self._episode_reward_terms: list[dict[str, float]] = []

        # Timing: rollout (data collection) vs train (gradient update)
        self._rollout_start_t: float = 0.0
        self._prev_rollout_end_t: float = 0.0
        self._prev_step: int = 0  # for per-rollout SPS
        self._recent_sps: list[float] = []  # rolling window
        self._eps_at_rollout_start: int = 0

        # Scalars file handle
        self._metrics_dir = iteration_dir / "metrics"
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._scalars_path = self._metrics_dir / "scalars.jsonl"
        self._scalars_f = open(self._scalars_path, "a")  # noqa: SIM115

        # Active eval subprocess (for cleanup on termination)
        self._eval_proc: subprocess.Popen[str] | None = None

        # TensorBoard writer (same metrics, TB format)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer: SummaryWriter | None = SummaryWriter(
                log_dir=str(iteration_dir / "tb_logs")
            )
        except ImportError:
            self._tb_writer = None

        # Video schedule
        self._video_dir = iteration_dir / "videos"
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._video_thresholds = sorted(
            max(1, int(i / config.num_evals * total_timesteps))
            for i in range(1, config.num_evals + 1)
        )
        self._next_video_idx = 0

        # Progress tracking
        self._t_start = time.time()
        self._progress_interval = max(1, total_timesteps // 20)
        self._last_progress_step = 0

    # --- Per-step ---------------------------------------------------------

    def _on_step(self) -> bool:
        step = self.num_timesteps
        infos = self.locals.get("infos", [])

        # Per-env accumulation: reward terms
        for env_idx, info in enumerate(infos):
            rt = info.get("reward_terms")
            if rt:
                for k, v in rt.items():
                    self._env_reward_terms[env_idx][k].append(float(v))

            # Episode completions
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)

                # Finalize per-term means for this episode
                term_means: dict[str, float] = {}
                for k, vals in self._env_reward_terms[env_idx].items():
                    term_means[k] = float(np.mean(vals)) if vals else 0.0
                self._episode_reward_terms.append(term_means)

                # Reset accumulator for this env
                self._env_reward_terms[env_idx] = defaultdict(list)

        # Session heartbeat (every 60s) — prevents dashboard stale detection
        now = time.time()
        if self._session_heartbeat_fn and now - self._last_heartbeat_t >= 60:
            self._last_heartbeat_t = now
            self._session_heartbeat_fn()

        # Progress log
        if step - self._last_progress_step >= self._progress_interval:
            self._last_progress_step = step
            pct = step / self.total_timesteps * 100
            elapsed = time.time() - self._t_start
            sps = step / max(elapsed, 1e-6)
            eta = (self.total_timesteps - step) / max(sps, 1e-6)
            ret_str = f"{self.episode_returns[-1]:.1f}" if self.episode_returns else "N/A"
            avg_str = "N/A"
            if len(self.episode_returns) >= 2:
                avg_str = f"{np.mean(self.episode_returns[-10:]):.1f}"
            logger.info(
                "%5.1f%% | step %d/%d | ret=%s avg10=%s eps=%d | %.0f stp/s | ETA %.0fs",
                pct,
                step,
                self.total_timesteps,
                ret_str,
                avg_str,
                len(self.episode_returns),
                sps,
                eta,
            )

        # Video capture
        while (
            self._next_video_idx < len(self._video_thresholds)
            and step >= self._video_thresholds[self._next_video_idx]
        ):
            target_step = self._video_thresholds[self._next_video_idx]
            self._next_video_idx += 1
            try:
                video_path = self._video_dir / f"eval_{target_step}.mp4"

                if self.config.engine == "isaaclab":
                    eval_result = self._run_isaaclab_eval(video_path, target_step)
                else:
                    eval_result = run_evaluation_sb3(
                        self.env_id,
                        self.reward_fn,
                        self.model,
                        video_path,
                        seed=self.config.seed,
                        max_episode_steps=self.config.max_episode_steps,
                        vec_normalize=self.model.get_vec_normalize_env(),
                        side_info=self.config.side_info,
                        num_eval_rounds=self.config.num_eval_rounds,
                        num_eval_envs=self.config.num_eval_envs,
                        parallel_eval=self.config.parallel_eval,
                        config=self.config,
                        reward_code=self.reward_code,
                        trajectory_stride=self.config.trajectory_stride,
                        trajectory_precision=self.config.trajectory_precision,
                        engine=self.config.engine,
                    )
                # Write eval entry (use target_step to match video filename)
                eval_entry: EvalScalar = {
                    "global_step": target_step,
                    "type": "eval",
                    **eval_result,
                }
                self._scalars_f.write(json.dumps(_round_floats(eval_entry, 4)) + "\n")
                self._scalars_f.flush()
                pct_label = target_step / self.total_timesteps
                if "median_return" in eval_result:
                    logger.info(
                        "Eval @ %.0f%%: median=%.1f mean=%.1f±%.1f best=%.1f (%d rounds)",
                        pct_label * 100,
                        eval_result["median_return"],
                        eval_result["mean_return"],
                        eval_result["std_return"],
                        eval_result["total_reward"],
                        eval_result["num_eval_rounds"],
                    )
                elif "mean_return" in eval_result:
                    logger.info(
                        "Eval @ %.0f%%: best=%.1f mean=%.1f±%.1f (%d rounds)",
                        pct_label * 100,
                        eval_result["total_reward"],
                        eval_result["mean_return"],
                        eval_result["std_return"],
                        eval_result["num_eval_rounds"],
                    )
                else:
                    logger.info(
                        "Eval @ %.0f%%: reward=%.1f",
                        pct_label * 100,
                        eval_result["total_reward"],
                    )
            except Exception as e:
                logger.warning("Eval failed at step %d: %s", target_step, e)

        return True

    def _run_isaaclab_eval(self, video_path: Path, step: int) -> EvalResult:
        """Run eval for IsaacLab in a separate subprocess.

        Isaac Sim only supports one scene per process and enabling cameras
        adds 7-12x overhead. We spawn a subprocess that loads the checkpoint,
        runs eval with video capture, and writes results — same approach as
        IsaacLab's own play.py (separate process from training).

        Output: same files as MuJoCo parallel eval
        (eval_{step}_p10.mp4, eval_{step}_median.mp4, eval_{step}_p90.mp4).
        """
        import json as _json

        step_label = video_path.stem.split("_")[-1]
        iter_dir = video_path.parent.parent

        # Save torch-only checkpoint for subprocess. SB3's model.save() uses
        # cloudpickle which embeds numpy module references. Isaac Sim's
        # rendering kit bundles numpy 1.26 but training runs with venv numpy
        # 2.x — the cross-version cloudpickle is not loadable. Saving just
        # the policy state dict + metadata as pure torch/JSON avoids this.
        ckpt_dir = iter_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"eval_ckpt_{step_label}.pt"

        import torch as _torch

        policy = self.model.policy
        ckpt_data = {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": self.model.observation_space.shape[0],
            "act_dim": self.model.action_space.shape[0],
            "act_low": self.model.action_space.low.tolist(),
            "act_high": self.model.action_space.high.tolist(),
            "net_arch": self.model.policy_kwargs.get("net_arch", [64, 64]),
        }

        # Save VecNormalize running stats so the eval subprocess can
        # reconstruct the same observation normalization.
        _vn = self.model.get_vec_normalize_env()
        if _vn is not None:
            ckpt_data["obs_rms_mean"] = _vn.obs_rms.mean.tolist()
            ckpt_data["obs_rms_var"] = _vn.obs_rms.var.tolist()
            ckpt_data["obs_rms_count"] = float(_vn.obs_rms.count)
            ckpt_data["normalize_obs"] = True
            ckpt_data["obs_clip"] = float(_vn.clip_obs)
            ckpt_data["gamma"] = float(_vn.gamma)

        _torch.save(ckpt_data, str(ckpt_path))

        result_path = iter_dir / f"eval_result_{step_label}.json"
        max_steps = str(self.config.max_episode_steps or 300)
        common_args = [
            "--env-id",
            self.config.env_id,
            "--checkpoint",
            str(ckpt_path),
            "--output-dir",
            str(iter_dir),
            "--step-label",
            step_label,
            "--max-steps",
            max_steps,
            "--seed",
            str(self.config.seed),
            "--stride",
            str(self.config.trajectory_stride),
            "--precision",
            str(self.config.trajectory_precision),
            "--result-file",
            str(result_path),
            "--device",
            "cuda:0",  # IsaacLab requires CUDA
        ]

        # Save reward code for eval subprocess to apply custom reward
        if self.reward_code:
            reward_code_path = ckpt_dir / f"reward_code_{step_label}.py"
            reward_code_path.write_text(self.reward_code)
            common_args.extend(["--reward-code", str(reward_code_path)])

        # Phase 1: fast stats (headless, num_envs=N, no rendering).
        # Cap eval envs for IsaacLab — training may use 4096 envs but eval
        # only needs enough for statistical coverage (~64 × 3 rounds = 192 eps).
        # Each eval subprocess creates a full Isaac Sim scene on GPU.
        phase1_envs = self.config.num_eval_envs or min(
            self.config.num_envs, _MAX_ISAACLAB_EVAL_ENVS
        )
        logger.info(
            "IsaacLab eval Phase 1: %d envs × %d rounds (headless)...",
            phase1_envs,
            self.config.num_eval_rounds,
        )
        from p2p.utils.subprocess_utils import python_cmd

        cmd_p1 = [
            *python_cmd(),
            "-m",
            "p2p.evaluator_isaaclab",
            "--phase",
            "1",
            "--num-envs",
            str(phase1_envs),
            "--num-rounds",
            str(self.config.num_eval_rounds),
            *common_args,
        ]

        # Acquire GPU eval lock to serialize with Phase 2 worker rendering.
        # Both Phase 1 (headless eval, ~3GB) and Phase 2 (camera render, ~5GB)
        # need GPU alongside training (~14GB).  On a 24GB GPU, running both
        # concurrently causes OOM.  The lock ensures only one runs at a time.
        gpu_lock_path = iter_dir.parent / ".phase2_queue" / ".gpu_eval.lock"
        if gpu_lock_path.parent.exists():
            import filelock

            gpu_lock = filelock.FileLock(gpu_lock_path, timeout=600)
        else:
            gpu_lock = None

        try:
            if gpu_lock:
                gpu_lock.acquire()
            rc, stderr = self._run_eval_subprocess(cmd_p1, timeout=600)
        except filelock.Timeout:
            logger.warning("GPU eval lock timeout — Phase 2 worker may be stuck")
            rc, stderr = -1, "GPU eval lock timeout"
        finally:
            if gpu_lock and gpu_lock.is_locked:
                gpu_lock.release()

        if rc != 0:
            logger.warning("Phase 1 failed (rc=%d): %s", rc, (stderr or "")[-300:])
            # Skip Phase 2 if Phase 1 failed and no result file
            if not result_path.exists():
                return {"total_reward": 0.0, "episode_length": 0, "reward_terms": {}}

        # Phase 2 (video rendering): queue a render request for the persistent
        # Phase 2 worker launched by iteration_runner.  The worker bootstraps
        # Isaac Sim once and reuses one env for all requests.
        queue_dir = iter_dir.parent / ".phase2_queue"
        if queue_dir.exists():
            import time as _req_time

            request = {
                "run_id": iter_dir.name,
                "run_dir": str(iter_dir),
                "checkpoint": str(ckpt_path),
                "step_label": step_label,
                "result_file": str(result_path),
                "seed": self.config.seed,
            }
            # Atomic write: temp file + rename to avoid partial reads
            ts = int(_req_time.monotonic() * 1000)
            tmp_path = queue_dir / f".tmp_{ts}.json"
            final_path = queue_dir / f"request_{ts}_{iter_dir.name}_{step_label}.json"
            tmp_path.write_text(_json.dumps(request, indent=2))
            tmp_path.rename(final_path)
            logger.info("Phase 2 render request queued: %s", final_path.name)

        if result_path.exists():
            with open(result_path) as f:
                result = _json.load(f)
            logger.info("IsaacLab eval: %s", result.get("mean_return", "?"))
            return result

        logger.warning("IsaacLab eval: no result file at %s", result_path)
        return {"total_reward": 0.0, "episode_length": 0, "reward_terms": {}}

    def _run_eval_subprocess(self, cmd: list[str], *, timeout: int = 600) -> tuple[int, str]:
        """Run an eval subprocess with proper cleanup on termination.

        Uses ``Popen`` + ``start_new_session=True`` so the child gets its
        own process group. If the training process receives SIGTERM
        (``SystemExit``) or ``KeyboardInterrupt`` while waiting, the
        child's entire process group is killed before re-raising.

        Returns (returncode, stderr_text).
        """
        self._eval_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, stderr = self._eval_proc.communicate(timeout=timeout)
            return self._eval_proc.returncode, stderr or ""
        except subprocess.TimeoutExpired:
            self._kill_eval_proc()
            return -1, "eval subprocess timed out"
        except (SystemExit, KeyboardInterrupt):
            self._kill_eval_proc()
            raise
        finally:
            self._eval_proc = None

    def _kill_eval_proc(self) -> None:
        """Kill the active eval subprocess and its entire process group."""
        import os
        import signal

        proc = self._eval_proc
        if proc is None or proc.poll() is not None:
            return
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pass
        # Kill the entire process group (includes xvfb-run, uv, Isaac Kit)
        if pgid is not None and pgid != os.getpgid(os.getpid()):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
        else:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Eval subprocess PID %d did not exit after SIGKILL", proc.pid)

    # --- Per rollout start (timing) ----------------------------------------

    def _on_rollout_start(self) -> None:
        self._rollout_start_t = time.time()
        self._eps_at_rollout_start = len(self.episode_returns)

    # --- Per rollout end (diagnostics) ------------------------------------

    def _on_rollout_end(self) -> None:
        now = time.time()

        # Timing: rollout = data collection, train = previous gradient update
        rollout_time = now - self._rollout_start_t if self._rollout_start_t > 0 else 0.0
        train_time = (
            (self._rollout_start_t - self._prev_rollout_end_t)
            if self._prev_rollout_end_t > 0
            else 0.0
        )
        self._prev_rollout_end_t = now

        if self.model.logger is None:
            return

        name_map = getattr(self.model.logger, "name_to_value", {})

        step = self.num_timesteps
        key_map = {
            "train/policy_gradient_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/entropy_loss": "entropy",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/explained_variance": "explained_variance",
            "train/kl_mean_term": "kl_mean_term",
            "train/kl_var_term": "kl_var_term",
            "train/mean_shift_normalized": "mean_shift_normalized",
        }

        log_entry: dict[str, Any] = {
            "global_step": step,
            "iteration": self._update_count,
        }
        has_data = False
        for sb3_key, our_key in key_map.items():
            if sb3_key in name_map:
                log_entry[our_key] = float(name_map[sb3_key])
                has_data = True

        # Policy std (continuous action spaces)
        if "train/std" in name_map:
            log_entry["policy_std"] = float(name_map["train/std"])

        # Episode statistics
        if self.episode_returns:
            recent = self.episode_returns[-10:]
            log_entry["episodic_return"] = float(np.mean(recent))
            log_entry["episodic_return_std"] = float(np.std(recent))
            log_entry["episodic_return_min"] = float(np.min(recent))
            log_entry["episodic_return_max"] = float(np.max(recent))
        if self.episode_lengths:
            log_entry["episode_length"] = float(np.mean(self.episode_lengths[-10:]))

        # Episodes completed in this rollout
        log_entry["episodes_per_rollout"] = len(self.episode_returns) - self._eps_at_rollout_start

        # Gradient norm (from previous train() call's last minibatch)
        try:
            total_norm_sq = 0.0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            if total_norm_sq > 0:
                log_entry["grad_norm"] = round(total_norm_sq**0.5, 6)
        except Exception:
            logger.debug("grad_norm computation failed", exc_info=True)

        # Instantaneous SPS: steps collected this rollout / wall-clock for this cycle
        cycle_time = rollout_time + train_time
        steps_this_rollout = step - self._prev_step
        if cycle_time > 0 and steps_this_rollout > 0:
            instant_sps = steps_this_rollout / cycle_time
            self._recent_sps.append(instant_sps)
            if len(self._recent_sps) > 20:
                self._recent_sps.pop(0)
        self._prev_step = step
        log_entry["sps"] = int(np.mean(self._recent_sps)) if self._recent_sps else 0

        log_entry["learning_rate"] = self.model.learning_rate

        # Cumulative wall-clock time
        log_entry["elapsed_time"] = round(time.time() - self._t_start, 2)

        # Throughput timing
        log_entry["rollout_time"] = round(rollout_time, 4)
        if train_time > 0:
            log_entry["train_time"] = round(train_time, 4)
        log_entry["device"] = str(self.model.device)

        # Per-term reward means (last 10 episodes)
        if self._episode_reward_terms:
            recent = self._episode_reward_terms[-10:]
            all_terms: set[str] = set()
            for ep_terms in recent:
                all_terms.update(ep_terms.keys())
            for term_name in sorted(all_terms):
                values = [ep.get(term_name, 0.0) for ep in recent]
                log_entry[f"reward_term_{term_name}"] = round(float(np.mean(values)), 6)

        # Write when SB3 logger has train/* metrics OR we have episode stats
        if has_data or self.episode_returns:
            self._scalars_f.write(json.dumps(log_entry) + "\n")
            self._scalars_f.flush()

            # Mirror to TensorBoard (grouped by prefix)
            if self._tb_writer is not None:
                for k, v in log_entry.items():
                    if isinstance(v, (int, float)):
                        tag = _tb_tag(k)
                        self._tb_writer.add_scalar(tag, v, global_step=step)

        self._update_count += 1

    # --- Finalize ---------------------------------------------------------

    def finalize(self) -> None:
        try:
            self._scalars_f.close()
        except Exception:
            logger.warning("Failed to close scalars file", exc_info=True)
        if self._tb_writer is not None:
            self._tb_writer.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train(
    config: TrainConfig,
    reward_fn: Callable,
    iteration_dir: Path,
    reward_code: str = "",
    session_heartbeat_fn: Callable[[], None] | None = None,
) -> TrainSummary:
    """Run SB3 PPO training. Returns summary metrics dict."""
    return _train_ppo(config, reward_fn, iteration_dir, reward_code, session_heartbeat_fn)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def _train_ppo(
    config: TrainConfig,
    reward_fn: Callable,
    iteration_dir: Path,
    reward_code: str,
    session_heartbeat_fn: Callable[[], None] | None = None,
) -> TrainSummary:
    _set_seed(config)
    tc = config
    total_timesteps = tc.total_timesteps

    # Environment
    envs = _make_vec_env(config, reward_fn, reward_code, tc.num_envs)

    # Model
    batch_size = tc.num_envs * tc.num_steps // tc.num_minibatches
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=tc.learning_rate,
        n_steps=tc.num_steps,
        batch_size=batch_size,
        n_epochs=tc.update_epochs,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_coef,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        max_grad_norm=tc.max_grad_norm,
        target_kl=tc.target_kl,
        policy_kwargs=dict(net_arch=tc.net_arch),
        seed=config.seed,
        device=config.device,
        verbose=0,
    )

    callback = P2PCallback(
        config=config,
        reward_fn=reward_fn,
        env_id=config.env_id,
        iteration_dir=iteration_dir,
        total_timesteps=total_timesteps,
        algorithm="ppo",
        session_heartbeat_fn=session_heartbeat_fn,
        reward_code=reward_code,
    )

    logger.info("Training %d timesteps, %d envs", total_timesteps, tc.num_envs)

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        callback.finalize()
        envs.close()
    elapsed = time.time() - t0

    # Checkpoint
    ckpt_path = iteration_dir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_path / "final"))
    if isinstance(envs, VecNormalize):
        envs.save(str(ckpt_path / "vecnormalize.pkl"))

    return _build_summary(callback, total_timesteps, elapsed, "ppo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(config: TrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_reward_signature(fn: Callable) -> None:
    """Raise ValueError if reward_fn has fewer than 4 positional parameters.

    Called before spawning SubprocVecEnv workers so that a bad signature
    is caught early instead of crashing in every subprocess.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return

    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    n = len(params)
    if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
        return

    if n < 4:
        raise ValueError(
            f"reward_fn(obs, action, next_obs, info) requires 4 positional parameters but got {n}."
        )


def _make_isaaclab_vec_env(
    config: TrainConfig,
    num_envs: int,
    *,
    render: bool = False,
    enable_cameras: bool = False,
    reward_fn: Callable | None = None,
    reward_code: str = "",
) -> VecNormalize:
    """Create an IsaacLab VecEnv via Sb3VecEnvWrapper.

    Args:
        render: If True, create env with render_mode="rgb_array".
        enable_cameras: If True, bootstrap Isaac Sim with camera support
            (required for any rendering, even if render=False now).
            Set True at training start so eval can render later.
        reward_fn: LLM-generated reward function to override built-in reward.
        reward_code: Source code string for stateful reward closures.
    """
    from p2p.training.env import (
        IsaacLabRewardVecWrapper,
        _bootstrap_isaaclab,
        _disable_debug_vis,
        _disable_direct_rl_termination,
        _disable_early_termination,
    )

    _bootstrap_isaaclab(enable_cameras=enable_cameras or render)
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(config.env_id, device="cuda:0", num_envs=num_envs)
    if render:
        env_cfg.sim.render_interval = env_cfg.decimation
    _disable_debug_vis(env_cfg)
    _disable_early_termination(env_cfg)
    import gymnasium as gym

    raw_env = gym.make(
        config.env_id,
        cfg=env_cfg,
        render_mode="rgb_array" if render else None,
    )
    _disable_direct_rl_termination(raw_env)
    venv = Sb3VecEnvWrapper(raw_env)

    # Insert custom reward wrapper between Sb3VecEnvWrapper and VecNormalize
    if reward_fn is not None or reward_code:
        venv = IsaacLabRewardVecWrapper(
            venv,
            reward_fn=reward_fn,
            reward_code=reward_code,
            raw_env=raw_env,
        )

    if config.normalize_obs or config.normalize_reward:
        venv = VecNormalize(
            venv,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=config.obs_clip,
            clip_reward=config.reward_clip,
            gamma=config.gamma,
        )
    return venv


def _make_vec_env(
    config: TrainConfig,
    reward_fn: Callable,
    reward_code: str,
    num_envs: int,
) -> SubprocVecEnv | VecNormalize:
    """Create SubprocVecEnv, optionally wrapped with VecNormalize.

    Uses code-based factory when reward_code is available (supports stateful
    closures like back_tumble). Falls back to callable-based factory otherwise.

    For IsaacLab envs, uses the official ``Sb3VecEnvWrapper`` which converts
    GPU tensors to numpy and adapts the already-vectorized env to SB3's VecEnv
    interface (no SubprocVecEnv needed).
    """
    # --- IsaacLab path: use Sb3VecEnvWrapper (already vectorized on GPU) ---
    if config.engine == "isaaclab":
        return _make_isaaclab_vec_env(
            config,
            num_envs,
            render=False,
            enable_cameras=False,
            reward_fn=reward_fn,
            reward_code=reward_code,
        )

    side_info = config.side_info

    # --- MuJoCo path: SubprocVecEnv with per-worker env factories ---
    if reward_code:
        # Sanitize before exec — strip numpy imports that cause
        # UnboundLocalError inside closures, fix escape sequences.
        from p2p.training.reward_loader import _sanitize_escape_sequences, _strip_numpy_imports

        reward_code = _sanitize_escape_sequences(_strip_numpy_imports(reward_code))

        # Pre-flight: exec reward code once and validate the signature
        # before spawning subprocess workers.
        from p2p.training.simulator import get_simulator

        _ns: dict[str, Any] = {"np": np, "numpy": np}
        get_simulator(config.engine).inject_reward_namespace(_ns)
        exec(reward_code, _ns)  # noqa: S102
        _fn = _ns.get("reward_fn")
        if _fn is not None:
            _validate_reward_signature(_fn)
        env_fns = [
            make_env_sb3_from_code(
                config.env_id,
                reward_code,
                config.seed + i,
                config.max_episode_steps,
                side_info=side_info,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                engine=config.engine,
            )
            for i in range(num_envs)
        ]
    else:
        _validate_reward_signature(reward_fn)
        env_fns = [
            make_env_sb3(
                config.env_id,
                reward_fn,
                config.seed + i,
                config.max_episode_steps,
                side_info=side_info,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                engine=config.engine,
            )
            for i in range(num_envs)
        ]
    venv = SubprocVecEnv(env_fns)

    if config.normalize_obs or config.normalize_reward:
        venv = VecNormalize(
            venv,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=config.obs_clip,
            clip_reward=config.reward_clip,
            gamma=config.gamma,
        )
    return venv


def _build_summary(
    callback: P2PCallback,
    total_timesteps: int,
    elapsed: float,
    algorithm: str,
) -> TrainSummary:
    returns = callback.episode_returns
    return {
        "total_timesteps": total_timesteps,
        "training_time_s": elapsed,
        "final_episodic_return": (round_metric(float(np.mean(returns[-10:]))) if returns else 0.0),
        "total_episodes": len(returns),
        "algorithm": algorithm,
    }
