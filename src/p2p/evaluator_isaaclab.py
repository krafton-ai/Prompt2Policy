"""IsaacLab eval subprocess -- calls shared eval functions from sb3_trainer.

Three modes:

Phase 1 (--phase 1): Fast stats via num_envs=N, headless, no rendering.
  Called per-run during training.  Writes eval_result_{step}.json.

Phase 2 (--phase 2): Single-run video rendering.
  Legacy mode for manual use.  Reads Phase 1 result, re-runs P10/median/P90
  with num_envs=1 + xvfb-run rendering.

Batch Phase 2 (--batch-manifest <json>): Consolidated video rendering.
  Launched ONCE by iteration_runner after all training completes.
  Bootstraps Isaac Sim once, creates one env, then loops over all
  config×seed runs swapping only the checkpoint + VecNormalize stats.
  Saves ~(N-1) × 30s of Isaac Sim bootstrap overhead.

Output (same as MuJoCo parallel eval):
  - videos/eval_{step}_p10.mp4, eval_{step}_median.mp4, eval_{step}_p90.mp4
  - trajectory_{step}_p10.npz, trajectory_{step}_median.npz, ...
  - eval_result_{step}.json

Usage:
    # Phase 1 (headless)
    uv run python -m p2p.evaluator_isaaclab --phase 1 --num-envs 10 ...
    # Phase 2 single-run (xvfb-run for rendering)
    xvfb-run -a uv run python -m p2p.evaluator_isaaclab --phase 2 --num-envs 1 ...
    # Batch Phase 2 (xvfb-run for rendering)
    xvfb-run -a uv run python -m p2p.evaluator_isaaclab --batch-manifest manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _disable_debug_vis(cfg: object) -> None:
    """Import from env.py -- deferred to avoid importing before AppLauncher."""
    from p2p.training.env import _disable_debug_vis as _impl

    _impl(cfg)


def _disable_early_termination(env_cfg: object) -> None:
    """Import from env.py -- deferred to avoid importing before AppLauncher."""
    from p2p.training.env import _disable_early_termination as _impl

    _impl(env_cfg)


def _disable_direct_rl_termination(env: object) -> None:
    """Import from env.py -- deferred to avoid importing before AppLauncher."""
    from p2p.training.env import _disable_direct_rl_termination as _impl

    _impl(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="IsaacLab eval subprocess")
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument(
        "--batch-manifest",
        default="",
        help="JSON manifest for consolidated Phase 2",
    )
    parser.add_argument(
        "--worker-queue",
        default="",
        help="Queue directory for persistent Phase 2 worker",
    )
    parser.add_argument("--env-id", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--step-label", default="")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--result-file", default="")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--reward-code", default="", help="Path to reward_fn source file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.worker_queue:
        _main_worker(args)
    elif args.batch_manifest:
        _main_batch(args)
    elif args.phase in (1, 2):
        required = ["env_id", "checkpoint", "output_dir", "step_label", "result_file"]
        missing = [f for f in required if not getattr(args, f)]
        if missing:
            parser.error(
                f"--phase requires: {', '.join('--' + f.replace('_', '-') for f in missing)}"
            )
        _main_single(args)
    else:
        parser.error("Must specify --phase 1|2 or --batch-manifest")


def _main_single(args: argparse.Namespace) -> None:
    """Phase 1 or single-run Phase 2 (legacy)."""
    # --- Bootstrap (must happen before any IsaacLab/gymnasium imports) ---
    if args.phase == 2:
        os.environ["ENABLE_CAMERAS"] = "1"
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=False, enable_cameras=True)
    else:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True)

    _fix_asset_root()
    _register_custom_envs()

    # --- Imports after SimulationApp ---
    import torch
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from stable_baselines3 import PPO

    from p2p.contracts import round_metric
    from p2p.training.sb3_trainer import _run_parallel_eval_episodes

    device = args.device
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    net_arch = ckpt.get("net_arch", [64, 64])

    # --- Create env ---
    env_cfg = parse_env_cfg(args.env_id, device=device, num_envs=args.num_envs)
    _disable_debug_vis(env_cfg)
    _disable_early_termination(env_cfg)
    _fix_eval_yaw(env_cfg)

    if args.phase == 2:
        _configure_phase2_camera(env_cfg, env_id=args.env_id)

    import gymnasium as gym

    raw_env = gym.make(
        args.env_id,
        cfg=env_cfg,
        render_mode="rgb_array" if args.phase == 2 else None,
    )
    _disable_direct_rl_termination(raw_env)

    if args.phase == 2:
        _verify_camera_asset(raw_env)

    venv = Sb3VecEnvWrapper(raw_env)

    # Apply custom reward wrapper if reward code is provided
    if args.reward_code:
        reward_code_text = Path(args.reward_code).read_text()
        from p2p.training.env import IsaacLabRewardVecWrapper

        venv = IsaacLabRewardVecWrapper(
            venv,
            reward_code=reward_code_text,
            raw_env=raw_env,
        )
        logger.info("Custom reward wrapper applied from %s", args.reward_code)

    # Restore VecNormalize if the training checkpoint includes running stats.
    if ckpt.get("normalize_obs"):
        venv = _apply_vec_normalize(venv, ckpt)

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=net_arch),
        device=device,
    )
    model.policy.load_state_dict(ckpt["policy_state_dict"])
    model.policy.eval()

    try:
        if args.phase == 1:
            _run_phase1(
                args,
                venv,
                model,
                _run_parallel_eval_episodes,
                round_metric,
            )
        else:
            _run_phase2(args, venv, model, output_dir, videos_dir)
    finally:
        venv.close()
        launcher.app.close()


def _main_batch(args: argparse.Namespace) -> None:
    """Consolidated Phase 2: one bootstrap, one env, all config×seed runs."""
    manifest_path = Path(args.batch_manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    env_id = manifest["env_id"]
    runs = manifest["runs"]
    num_rounds = manifest.get("num_rounds", 3)
    max_steps = manifest.get("max_steps", 300)
    stride = manifest.get("stride", 5)
    precision = manifest.get("precision", 4)
    device = manifest.get("device", "cuda:0")

    if not runs:
        logger.warning("Batch manifest has no runs, exiting")
        return

    logger.info(
        "Batch Phase 2: %d runs for %s (bootstrap once, one env)",
        len(runs),
        env_id,
    )

    # --- Bootstrap Isaac Sim ONCE ---
    os.environ["ENABLE_CAMERAS"] = "1"
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=False, enable_cameras=True)
    _fix_asset_root()
    _register_custom_envs()

    # --- Imports after SimulationApp ---
    import numpy as np
    import torch
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    # --- Create env ONCE ---
    env_cfg = parse_env_cfg(env_id, device=device, num_envs=1)
    _disable_debug_vis(env_cfg)
    _disable_early_termination(env_cfg)
    _fix_eval_yaw(env_cfg)
    _configure_phase2_camera(env_cfg, env_id=env_id)

    import gymnasium as gym

    raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")
    _disable_direct_rl_termination(raw_env)
    _verify_camera_asset(raw_env)

    base_venv = Sb3VecEnvWrapper(raw_env)

    # Apply custom reward wrapper (same reward code for all runs within iteration)
    reward_code_path = runs[0].get("reward_code", "")
    if reward_code_path and Path(reward_code_path).exists():
        reward_code_text = Path(reward_code_path).read_text()
        from p2p.training.env import IsaacLabRewardVecWrapper

        base_venv = IsaacLabRewardVecWrapper(
            base_venv,
            reward_code=reward_code_text,
            raw_env=raw_env,
        )
        logger.info("Custom reward wrapper applied from %s", reward_code_path)

    # Track current state for hot-swapping
    current_net_arch: list | None = None
    model: PPO | None = None
    vec_norm_venv: VecNormalize | None = None

    try:
        for i, run_entry in enumerate(runs):
            run_id = run_entry["run_id"]
            run_dir = Path(run_entry["run_dir"])
            ckpt_path = run_entry["checkpoint"]
            step_label = run_entry["step_label"]
            result_file = Path(run_entry["result_file"])
            seed = run_entry.get("seed", 0)

            logger.info(
                "--- Run %d/%d: %s (step %s) ---",
                i + 1,
                len(runs),
                run_id,
                step_label,
            )

            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            net_arch = ckpt.get("net_arch", [64, 64])

            # Determine the effective venv (with or without VecNormalize)
            if ckpt.get("normalize_obs"):
                if vec_norm_venv is None:
                    vec_norm_venv = _apply_vec_normalize(base_venv, ckpt)
                else:
                    # Update stats in-place for subsequent runs
                    vec_norm_venv.training = False
                    vec_norm_venv.obs_rms.mean = np.array(ckpt["obs_rms_mean"])
                    vec_norm_venv.obs_rms.var = np.array(ckpt["obs_rms_var"])
                    vec_norm_venv.obs_rms.count = ckpt["obs_rms_count"]
                    vec_norm_venv.clip_obs = ckpt.get("obs_clip", 10.0)
                effective_venv = vec_norm_venv
            else:
                effective_venv = base_venv

            # Create or re-create PPO model if net_arch changed
            if net_arch != current_net_arch:
                model = PPO(
                    "MlpPolicy",
                    effective_venv,
                    policy_kwargs=dict(net_arch=net_arch),
                    device=device,
                )
                current_net_arch = net_arch
                logger.info("Created PPO model with net_arch=%s", net_arch)

            # Load policy weights
            model.policy.load_state_dict(ckpt["policy_state_dict"])
            model.policy.eval()

            # Run Phase 2 for this run
            output_dir = run_dir
            videos_dir = run_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            _run_phase2_for_run(
                env_id=env_id,
                venv=effective_venv,
                model=model,
                step_label=step_label,
                result_file=result_file,
                seed=seed,
                max_steps=max_steps,
                stride=stride,
                precision=precision,
                output_dir=output_dir,
                videos_dir=videos_dir,
                num_rounds=num_rounds,
            )
            _patch_eval_scalars(run_dir, step_label, result_file)
            logger.info("Run %s done", run_id)

    finally:
        # Close the base env (VecNormalize.close() delegates to wrapped env)
        if vec_norm_venv is not None:
            vec_norm_venv.close()
        else:
            base_venv.close()
        launcher.app.close()

    logger.info("Batch Phase 2 complete: %d runs processed", len(runs))


def _main_worker(args: argparse.Namespace) -> None:
    """Persistent Phase 2 worker: one bootstrap, one env, poll for requests.

    Launched by iteration_runner before training starts.  Polls a queue
    directory for render request JSON files written by training callbacks.
    Processes each request (load checkpoint, render 3 episodes, save
    videos/trajectories), then waits for the next.  Exits when a ``stop``
    file appears and no pending requests remain.
    """
    import time as _time

    queue_dir = Path(args.worker_queue)

    # Read worker config (written by iteration_runner before launch)
    config_path = queue_dir / "worker_config.json"
    with open(config_path) as f:
        wconfig = json.load(f)

    env_id = wconfig["env_id"]
    num_rounds = wconfig.get("num_rounds", 3)
    max_steps = wconfig.get("max_steps", 300)
    stride = wconfig.get("stride", 5)
    precision = wconfig.get("precision", 4)
    device = wconfig.get("device", "cuda:0")
    reward_code_path = wconfig.get("reward_code", "")

    logger.info("Phase 2 worker starting for %s (queue: %s)", env_id, queue_dir)

    # --- Bootstrap Isaac Sim ONCE ---
    # Serialize initialization across all workers on this node.
    # Isaac Sim cannot handle concurrent AppLauncher() calls even on
    # different GPUs — the kernel init is a machine-level singleton.
    # After init completes, workers run concurrently on their own GPUs.
    import filelock

    init_lock = filelock.FileLock("/tmp/p2p_isaaclab_init.lock", timeout=300)
    os.environ["ENABLE_CAMERAS"] = "1"

    logger.info("Waiting for Isaac Sim init lock...")
    with init_lock:
        logger.info("Acquired init lock, bootstrapping Isaac Sim + env")
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=False, enable_cameras=True)
        _fix_asset_root()
        _register_custom_envs()

        # --- Imports after SimulationApp ---
        import numpy as np  # noqa: F811
        import torch  # noqa: F811
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from stable_baselines3 import PPO  # noqa: F811
        from stable_baselines3.common.vec_env import VecNormalize

        # Create env inside the lock — concurrent gym.make() causes a USD
        # loading race that silently fails to materialize visual meshes,
        # producing videos with invisible robots (#393).
        env_cfg = parse_env_cfg(env_id, device=device, num_envs=1)
        _disable_debug_vis(env_cfg)
        _disable_early_termination(env_cfg)
        _fix_eval_yaw(env_cfg)
        _configure_phase2_camera(env_cfg, env_id=env_id)

        import gymnasium as gym

        raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")
        _disable_direct_rl_termination(raw_env)
        _verify_camera_asset(raw_env)
        try:
            _verify_render_visible(raw_env, env_id)
        except RenderInvisibleError:
            logger.warning("Invisible robot detected, recreating env...")
            raw_env.close()
            raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")
            _verify_camera_asset(raw_env)
            # Don't verify again — the retry in iteration_runner handles
            # persistent failures.
    logger.info("Isaac Sim init + env creation complete, released lock")

    base_venv = Sb3VecEnvWrapper(raw_env)

    # Apply custom reward wrapper (same for all runs within iteration)
    if reward_code_path and Path(reward_code_path).exists():
        reward_code_text = Path(reward_code_path).read_text()
        from p2p.training.env import IsaacLabRewardVecWrapper

        base_venv = IsaacLabRewardVecWrapper(
            base_venv,
            reward_code=reward_code_text,
            raw_env=raw_env,
        )
        logger.info("Custom reward wrapper applied from %s", reward_code_path)

    # Track state for hot-swapping across requests
    current_net_arch: list | None = None
    model: PPO | None = None
    vec_norm_venv: VecNormalize | None = None

    stop_path = queue_dir / "stop"
    processed = 0

    logger.info("Phase 2 worker ready, polling for requests...")

    try:
        while True:
            # Find pending requests (sorted for deterministic ordering)
            requests = sorted(queue_dir.glob("request_*.json"))

            if not requests:
                if stop_path.exists():
                    break
                _time.sleep(2)
                continue

            for req_path in requests:
                try:
                    with open(req_path) as f:
                        req = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Partially written or disappeared — skip, retry next poll
                    continue

                run_id = req.get("run_id", "unknown")
                run_dir = Path(req["run_dir"])
                ckpt_path = req["checkpoint"]
                step_label = req["step_label"]
                result_file = Path(req["result_file"])
                seed = req.get("seed", 0)

                logger.info("Processing: %s step %s", run_id, step_label)

                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                net_arch = ckpt.get("net_arch", [64, 64])

                # VecNormalize: create once, update stats in-place after
                if ckpt.get("normalize_obs"):
                    if vec_norm_venv is None:
                        vec_norm_venv = _apply_vec_normalize(base_venv, ckpt)
                    else:
                        vec_norm_venv.training = False
                        vec_norm_venv.obs_rms.mean = np.array(ckpt["obs_rms_mean"])
                        vec_norm_venv.obs_rms.var = np.array(ckpt["obs_rms_var"])
                        vec_norm_venv.obs_rms.count = ckpt["obs_rms_count"]
                        vec_norm_venv.clip_obs = ckpt.get("obs_clip", 10.0)
                    effective_venv = vec_norm_venv
                else:
                    effective_venv = base_venv

                # PPO model: re-create only if net_arch changed
                if net_arch != current_net_arch:
                    model = PPO(
                        "MlpPolicy",
                        effective_venv,
                        policy_kwargs=dict(net_arch=net_arch),
                        device=device,
                    )
                    current_net_arch = net_arch
                    logger.info("Created PPO model with net_arch=%s", net_arch)

                model.policy.load_state_dict(ckpt["policy_state_dict"])
                model.policy.eval()

                # Render — acquire GPU lock to avoid OOM with Phase 1 eval
                videos_dir = run_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)

                import filelock

                gpu_lock = queue_dir / ".gpu_eval.lock"
                try:
                    with filelock.FileLock(gpu_lock, timeout=600):
                        _run_phase2_for_run(
                            env_id=env_id,
                            venv=effective_venv,
                            model=model,
                            step_label=step_label,
                            result_file=result_file,
                            seed=seed,
                            max_steps=max_steps,
                            stride=stride,
                            precision=precision,
                            output_dir=run_dir,
                            videos_dir=videos_dir,
                            num_rounds=num_rounds,
                        )
                except filelock.Timeout:
                    logger.warning(
                        "GPU lock timeout for %s step %s, skipping",
                        run_id,
                        step_label,
                    )
                    continue

                # Patch scalars.jsonl with Phase 2 data (episode_length,
                # reward_terms) so the dashboard shows per-term eval charts.
                # MuJoCo writes these synchronously; for IsaacLab we backfill.
                _patch_eval_scalars(run_dir, step_label, result_file)

                # Mark processed (rename request → done)
                done_path = req_path.with_name(req_path.name.replace("request_", "done_"))
                req_path.rename(done_path)
                processed += 1
                logger.info("Done: %s step %s (%d total)", run_id, step_label, processed)

            # After processing batch, check stop
            if stop_path.exists():
                remaining = sorted(queue_dir.glob("request_*.json"))
                if not remaining:
                    break

    finally:
        if vec_norm_venv is not None:
            vec_norm_venv.close()
        else:
            base_venv.close()
        launcher.app.close()

    logger.info("Phase 2 worker exiting (%d requests processed)", processed)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _patch_eval_scalars(run_dir: Path, step_label: str, result_file: Path) -> None:
    """Backfill episode_length and reward_terms into scalars.jsonl.

    In MuJoCo, the eval callback writes these synchronously.  In IsaacLab,
    Phase 1 writes ``reward_terms: {}`` and Phase 2 fills them in
    ``eval_result_{step}.json``.  This function patches scalars.jsonl so
    the dashboard per-term eval chart works identically to MuJoCo.
    """
    scalars_path = run_dir / "metrics" / "scalars.jsonl"
    if not scalars_path.exists() or not result_file.exists():
        return

    try:
        with open(result_file) as f:
            phase2 = json.load(f)
        ep_len = phase2.get("episode_length", 0)
        terms = phase2.get("reward_terms", {})
        if not terms:
            return  # Nothing to patch

        target_step = int(step_label)
        lines = scalars_path.read_text().strip().split("\n")
        patched = False
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if entry.get("type") == "eval" and entry.get("global_step") == target_step:
                entry["episode_length"] = ep_len
                entry["reward_terms"] = terms
                lines[i] = json.dumps(entry)
                patched = True
                break

        if patched:
            scalars_path.write_text("\n".join(lines) + "\n")
            logger.info("Patched scalars eval at step %s with reward_terms", step_label)
    except Exception:
        logger.warning("Failed to patch scalars.jsonl for step %s", step_label, exc_info=True)


def _register_custom_envs() -> None:
    """Register custom SAR environments with Gymnasium.

    Must be called after AppLauncher bootstrap so that isaaclab_tasks and
    p2p.isaaclab_envs can safely import Omniverse/Isaac Sim modules.
    """
    import isaaclab_tasks  # noqa: F401, I001 — triggers standard gym.register() calls
    import p2p.isaaclab_envs  # noqa: F401 — registers custom SAR envs (e.g. pen spinning)


def _fix_asset_root() -> None:
    """Ensure cloud asset root is set (some envs need it for USD loading)."""
    import carb

    settings = carb.settings.get_settings()
    cloud_root = settings.get("/persistent/isaac/asset_root/cloud")
    if cloud_root is None:
        default_root = settings.get("/persistent/isaac/asset_root/default")
        if default_root:
            settings.set("/persistent/isaac/asset_root/cloud", default_root)


def _fix_eval_yaw(env_cfg: object) -> None:
    """Fix eval yaw to 0 so the robot always faces +X (consistent camera angle)."""
    events = getattr(env_cfg, "events", None)
    if events is not None:
        reset_base = getattr(events, "reset_base", None)
        if reset_base is not None and isinstance(reset_base.params, dict):
            pose_range = reset_base.params.get("pose_range", {})
            pose_range["yaw"] = (0.0, 0.0)


def _configure_phase2_camera(env_cfg: object, env_id: str = "") -> None:
    """Configure camera for Phase 2 video rendering."""
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (640, 480)

    # Per-env camera presets — OFFSETS from the tracked asset root
    # (origin_type="asset_root").  Validated via local capture tests.
    # The Ant default (0.5, -3.0, 1.5) is the reference "good" angle.
    eid = env_id.lower()
    if "dexsuite" in eid:
        # Dexsuite (Kuka arm + Allegro hand on table): wide arm view
        env_cfg.viewer.eye = (1.0, -2.0, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    elif "hand-over" in eid:
        # Two-hand handover: uses env_origin (not asset_root) because the
        # articulation is "robot0", not "robot", so asset tracking fails.
        # Side view to show both hands (stacked vertically in the scene).
        env_cfg.viewer.origin_type = "env_origin"
        env_cfg.viewer.eye = (-1.0, 0.25, 0.8)
        env_cfg.viewer.lookat = (0.05, 0.05, 0.5)
    elif "shadow" in eid or "spin-pen" in eid:
        # Dexterous hand: close front-above view of palm and fingers
        env_cfg.viewer.eye = (0.2, -0.6, 0.3)
        env_cfg.viewer.lookat = (0.0, -0.1, 0.0)
    elif "allegro" in eid:
        # Standalone Allegro hand: same close-up as Shadow
        env_cfg.viewer.eye = (0.2, -0.6, 0.3)
        env_cfg.viewer.lookat = (0.0, -0.1, 0.0)
    elif "quadcopter" in eid:
        # Aerial: very close (quadcopter body is tiny)
        env_cfg.viewer.eye = (0.15, -0.4, 0.15)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    elif "cartpole" in eid or "pendulum" in eid:
        # Small robot: close side view
        env_cfg.viewer.eye = (0.5, -2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    elif "franka" in eid or "cabinet" in eid or "drawer" in eid:
        # Tabletop manipulation: front-side view of arm + workspace
        env_cfg.viewer.eye = (1.0, -2.0, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    elif "humanoid" in eid:
        # Tall biped: lowered eye + lookat to keep feet in frame
        env_cfg.viewer.eye = (0.8, -3.0, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    elif "g1" in eid or "h1" in eid or "digit" in eid or "cassie" in eid:
        # Medium bipeds: similar to humanoid
        env_cfg.viewer.eye = (0.8, -3.5, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    elif "galbot" in eid or "agibot" in eid:
        # Tall mobile manipulator: pull back and raise to show full body
        env_cfg.viewer.eye = (1.2, -2.5, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    elif "reach" in eid or "lift" in eid or "stack" in eid or "place" in eid:
        # Manipulation (reach/lift/stack/place): front-side of workspace
        env_cfg.viewer.eye = (1.0, -2.0, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    else:
        # Default: full-body locomotion (Ant, Anymal, quadrupeds, etc.)
        env_cfg.viewer.eye = (0.5, -3.0, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)


def _verify_camera_asset(raw_env: object) -> None:
    """Verify camera tracking asset name, fall back to first articulation."""
    _unwrapped = raw_env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    scene = getattr(_unwrapped, "scene", None)
    if scene is not None:
        arts = list(getattr(scene, "articulations", {}).keys())
        if "robot" not in arts and arts:
            actual = arts[0]
            logger.info("Asset 'robot' not found; tracking '%s' instead", actual)
            vcc = getattr(_unwrapped, "viewport_camera_controller", None)
            if vcc is not None:
                vcc.update_view_to_asset_root(actual)


# Pixel std below this threshold in the center crop indicates the robot
# mesh was not materialized (invisible robot from USD race #393).
# Empirically: ground-only frames have std < 15, frames with a robot > 25.
_RENDER_INVISIBLE_STD_THRESHOLD = 15
_RENDER_VERIFY_WARMUP_STEPS = 10


class RenderInvisibleError(RuntimeError):
    """Raised when the render warmup detects an invisible robot."""


def _verify_render_visible(raw_env: object, env_id: str) -> None:
    """Render warmup frames and verify the robot mesh is visible.

    The USD loading race (#393) can produce envs where physics runs
    correctly but the robot's visual mesh is never materialized —
    yielding videos with only the ground grid visible.  This function
    renders a few warmup frames and checks that the rendered image has
    enough pixel variance to indicate a visible robot.

    Raises ``RenderInvisibleError`` if the robot appears invisible,
    signaling the caller to recreate the env or retry.  Always resets
    the env after verification to ensure a clean state.
    """
    import numpy as np

    try:
        raw_env.reset()
        for _ in range(_RENDER_VERIFY_WARMUP_STEPS):
            action = raw_env.action_space.sample()
            raw_env.step(action)
        frame = raw_env.render()
        if frame is None:
            logger.warning("Render warmup returned None for %s", env_id)
            return
        arr = np.asarray(frame)
        h, w = arr.shape[:2]
        crop = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        std = float(crop.std())
        logger.info(
            "Render warmup for %s: frame %s, center std=%.1f",
            env_id,
            arr.shape,
            std,
        )
        if std < _RENDER_INVISIBLE_STD_THRESHOLD:
            raise RenderInvisibleError(
                f"Low pixel variance (std={std:.1f}) for {env_id} — "
                f"robot mesh likely not materialized"
            )
    except RenderInvisibleError:
        raise
    except Exception as e:
        logger.warning("Render warmup check failed for %s: %s", env_id, e)
    finally:
        # Reset to clean state regardless of outcome, so the env
        # is ready for the actual evaluation loop.
        try:
            raw_env.reset()
        except Exception:
            pass


def _apply_vec_normalize(venv, ckpt: dict):
    """Wrap venv with VecNormalize and restore running stats from checkpoint."""
    import numpy as np
    from stable_baselines3.common.vec_env import VecNormalize

    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=ckpt.get("obs_clip", 10.0),
        gamma=ckpt.get("gamma", 0.99),
    )
    venv.training = False
    venv.obs_rms.mean = np.array(ckpt["obs_rms_mean"])
    venv.obs_rms.var = np.array(ckpt["obs_rms_var"])
    venv.obs_rms.count = ckpt["obs_rms_count"]
    return venv


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def _run_phase1(args, venv, model, run_parallel_fn, round_metric_fn) -> None:
    """Phase 1: collect episode returns via shared parallel eval function."""
    import numpy as np

    from p2p.config import TrainConfig

    logger.info("Phase 1: %d envs x %d rounds...", args.num_envs, args.num_rounds)

    num_episodes = args.num_envs * args.num_rounds
    config = TrainConfig(env_id=args.env_id, num_envs=args.num_envs, seed=args.seed)

    episode_data = run_parallel_fn(
        args.env_id,
        None,  # reward_fn (not used -- reward comes from wrapper or env)
        model,
        args.num_envs,
        num_episodes,
        args.seed,
        args.max_steps,  # max_episode_steps
        None,  # vec_normalize (already applied in venv)
        config,
        engine="isaaclab",
        eval_venv=venv,
        max_steps=args.max_steps,
    )

    all_returns = [r for r, _ in episode_data]
    arr = np.array(all_returns)

    result = {
        "total_reward": round_metric_fn(float(np.max(arr))),
        "episode_length": 0,
        "reward_terms": {},
        "num_eval_rounds": len(all_returns),
        "mean_return": round_metric_fn(float(np.mean(arr))),
        "std_return": round_metric_fn(float(np.std(arr))),
        "min_return": round_metric_fn(float(np.min(arr))),
        "max_return": round_metric_fn(float(np.max(arr))),
        "median_return": round_metric_fn(float(np.median(arr))),
        "p10_return": round_metric_fn(float(np.percentile(arr, 10))),
        "p90_return": round_metric_fn(float(np.percentile(arr, 90))),
        "all_returns": [round_metric_fn(float(r)) for r in all_returns],
    }

    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Phase 1 done: %d eps, mean=%.3f, median=%.3f",
        len(all_returns),
        result["mean_return"],
        result["median_return"],
    )


# ---------------------------------------------------------------------------
# Phase 2 (single-run, legacy)
# ---------------------------------------------------------------------------


def _run_phase2(args, venv, model, output_dir: Path, videos_dir: Path) -> None:
    """Phase 2: render P10/median/P90 via shared single-episode eval function."""
    _run_phase2_for_run(
        env_id=args.env_id,
        venv=venv,
        model=model,
        step_label=args.step_label,
        result_file=Path(args.result_file),
        seed=args.seed,
        max_steps=args.max_steps,
        stride=args.stride,
        precision=args.precision,
        output_dir=output_dir,
        videos_dir=videos_dir,
        num_rounds=args.num_rounds,
    )


# ---------------------------------------------------------------------------
# Phase 2 core (shared by single-run and batch modes)
# ---------------------------------------------------------------------------


def _run_phase2_for_run(
    *,
    env_id: str,
    venv,
    model,
    step_label: str,
    result_file: Path,
    seed: int,
    max_steps: int,
    stride: int,
    precision: int,
    output_dir: Path,
    videos_dir: Path,
    num_rounds: int,
) -> None:
    """Run Phase 2 (video capture) for a single run directory.

    Used by both single-run mode and batch mode.
    """
    from p2p.training.sb3_trainer import (
        _percentile_index,
        _run_single_eval_episode,
        _save_trajectory,
        _save_video,
    )

    # Read Phase 1 result to get returns for percentile selection
    if result_file.exists():
        with open(result_file) as f:
            phase1 = json.load(f)
        all_returns = phase1.get("all_returns", [])
    else:
        logger.warning(
            "No Phase 1 result for %s, running %d episodes directly",
            result_file,
            num_rounds,
        )
        all_returns = []

    # Pick P10/median/P90 indices (same logic as MuJoCo eval)
    if all_returns:
        sorted_returns = sorted(enumerate(all_returns), key=lambda x: x[1])
        capture_targets = {
            "p10": sorted_returns[_percentile_index(sorted_returns, 10)],
            "median": sorted_returns[_percentile_index(sorted_returns, 50)],
            "p90": sorted_returns[_percentile_index(sorted_returns, 90)],
        }
    else:
        capture_targets = {
            "p10": (0, 0.0),
            "median": (1, 0.0),
            "p90": (2, 0.0),
        }

    logger.info("Phase 2: capturing P10/median/P90 (num_envs=1)...")

    median_steps = 0
    median_terms: dict[str, float] = {}

    for label, (orig_idx, orig_return) in capture_targets.items():
        logger.info("  %s (orig_return=%.3f): running with render...", label, orig_return)

        _reward, steps, terms, frames, traj, render_fps, _seg = _run_single_eval_episode(
            env_id,
            None,  # reward_fn (not used -- reward comes from wrapper or env)
            model,
            seed,  # seed (IsaacLab ignores per-reset seed)
            max_steps,
            None,  # vec_normalize (already applied in venv)
            engine="isaaclab",
            existing_env=venv,
            trajectory_stride=stride,
            trajectory_precision=precision,
        )

        if frames:
            vpath = videos_dir / f"eval_{step_label}_{label}.mp4"
            _save_video(frames, vpath, render_fps)
            logger.info("  %s: %s (%d frames)", label, vpath.name, len(frames))

        if traj:
            tpath = output_dir / f"trajectory_{step_label}_{label}.npz"
            _save_trajectory(traj, tpath)

        if label == "median":
            median_steps = steps
            median_terms = terms

        logger.info(
            "  %s: reward=%.2f, steps=%d, frames=%d",
            label,
            _reward,
            steps,
            len(frames),
        )

    # Update result with Phase 2 metadata
    if result_file.exists():
        with open(result_file) as f:
            result = json.load(f)
    else:
        result = {}
    result["episode_length"] = median_steps
    result["reward_terms"] = median_terms

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Phase 2 done for step %s", step_label)


if __name__ == "__main__":
    main()
