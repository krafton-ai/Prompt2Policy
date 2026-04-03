"""CLI: python -m p2p.executor --reward-fn reward.py [--config config.json] [--runs-dir runs/]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports
from p2p.config import TrainConfig
from p2p.session.session_id import generate_session_id
from p2p.training.reward_loader import load_from_file
from p2p.training.runner import run_training

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prompt2Policy Executor — run a single training session",
    )
    parser.add_argument(
        "--reward-fn",
        required=True,
        help="Path to reward function .py file",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TrainConfig JSON file (optional)",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Base directory for run records (default: runs)",
    )
    parser.add_argument(
        "--env-id",
        default="HalfCheetah-v5",
        help="Environment ID (default: HalfCheetah-v5)",
    )
    parser.add_argument(
        "--iteration-id",
        default=None,
        help="Override iteration_id (subdirectory name under runs-dir)",
    )
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            return 1
        config = TrainConfig.from_json(config_path.read_text())
        config.env_id = args.env_id
    else:
        config = TrainConfig(env_id=args.env_id)

    reward_path = Path(args.reward_fn)
    if not reward_path.exists():
        logger.error("Reward file not found: %s", reward_path)
        return 1

    reward = load_from_file(reward_path, engine=config.engine)
    source = reward_path.read_text()

    if not config.iteration_id:
        config.iteration_id = args.iteration_id or "iter_1"
    elif args.iteration_id:
        config.iteration_id = args.iteration_id

    runs_dir = Path(args.runs_dir)
    # If runs_dir is already a session or iteration directory, use it directly.
    # Otherwise create a new session (CLI direct usage).
    if runs_dir.name.startswith(("session_", "iter_")):
        session_dir = runs_dir
    else:
        session_id = generate_session_id()
        session_dir = runs_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    iteration_dir = run_training(config, reward, reward_source=source, runs_dir=session_dir)
    logger.info("Completed: %s", iteration_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
