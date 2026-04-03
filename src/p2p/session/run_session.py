"""CLI entrypoint for running a loop session in a subprocess."""

# ruff: noqa: I001 — p2p.settings must be imported before gymnasium/mujoco
from __future__ import annotations

import p2p.settings  # noqa: F401 — load .env before gymnasium/mujoco imports

import argparse
import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--loop-config",
        required=True,
        help="JSON-serialized LoopConfig",
    )
    args = parser.parse_args()

    from p2p.config import LoopConfig
    from p2p.session.loop import run_loop

    loop_config = LoopConfig.from_json(args.loop_config)

    # Install SIGTERM handler to mark session as error on kill
    from p2p.session.iteration_record import SessionRecord
    from p2p.settings import resolve_session_dir

    session = SessionRecord(resolve_session_dir(args.session_id))

    def _sigterm_handler(signum: int, frame: object) -> None:
        session.set_status("error", error="Process terminated (SIGTERM)")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        result = run_loop(
            args.prompt,
            loop_config,
            session_id=args.session_id,
        )
    except RuntimeError as exc:
        if "MUJOCO_GL" in str(exc):
            current = os.environ.get("MUJOCO_GL", "(not set)")
            raise RuntimeError(
                f"MuJoCo GL backend error (MUJOCO_GL={current}). "
                "Set MUJOCO_GL in your .env file: "
                "'egl' for Linux headless, or remove it entirely for macOS."
            ) from exc
        raise
    logger.info(
        "Session %s finished: status=%s, best=%.2f",
        result["session_id"],
        result["status"],
        result["best_score"],
    )


if __name__ == "__main__":
    main()
