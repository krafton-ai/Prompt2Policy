"""Training orchestration — run_training() creates an IterationRecord."""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from p2p.config import TrainConfig
from p2p.contracts import RewardSpec, RewardTerm
from p2p.session.iteration_record import IterationRecord
from p2p.training.reward_function import RewardFunction

logger = logging.getLogger(__name__)


def _generate_iteration_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:12]}"


def _parse_reward_spec(reward_fn: Callable, source: str | None = None) -> RewardSpec:
    """Extract reward spec from reward function source or docstring.

    Delegates to :func:`p2p.training.reward_loader._parse_docstring` for the actual
    parsing so there is a single implementation of the LaTeX / terms extractor.
    Returns structured ``RewardSpec`` with ``list[RewardTerm]``.
    """
    from p2p.training.reward_loader import _parse_docstring

    latex, terms = _parse_docstring(reward_fn, source or "")
    doc = inspect.getdoc(reward_fn) or ""
    return {"latex": latex, "terms": terms, "description": doc}


def run_training(
    config: TrainConfig,
    reward_fn: Callable,
    reward_source: str | None = None,
    runs_dir: str | Path = "runs",
    session_heartbeat_fn: Callable[[], None] | None = None,
) -> Path:
    """Execute a training run and create a complete IterationRecord.

    Args:
        config: Training configuration.
        reward_fn: Custom reward function.
        reward_source: Source code of the reward function (auto-detected if None).
        runs_dir: Base directory for iteration records.

    Returns:
        Path to the iteration directory.
    """
    runs_dir = Path(runs_dir)

    # Generate iteration_id
    if not config.iteration_id:
        config.iteration_id = _generate_iteration_id()

    rec = IterationRecord(runs_dir / config.iteration_id)

    # Save config + reward artifacts via IterationRecord
    rec.save_config(config)

    if reward_source is None:
        reward_source = inspect.getsource(reward_fn)
    rec.save_reward_source(reward_source)

    if isinstance(reward_fn, RewardFunction):
        # Use structured_terms if available (LegacyRewardWrapper), else convert
        from p2p.training.reward_loader import LegacyRewardWrapper

        if isinstance(reward_fn, LegacyRewardWrapper):
            structured_terms = reward_fn.structured_terms
        else:
            structured_terms: list[RewardTerm] = [
                {"name": k, "description": v} for k, v in reward_fn.terms.items()
            ]
        reward_spec: RewardSpec = {
            "latex": reward_fn.latex,
            "terms": structured_terms,
            "description": reward_fn.description,
        }
    else:
        reward_spec = _parse_reward_spec(reward_fn, source=reward_source)
    rec.save_reward_spec(reward_spec)

    rec.set_status("running")
    try:
        from p2p.training.sb3_trainer import train as sb3_train

        summary = sb3_train(
            config,
            reward_fn,
            rec.path,
            reward_code=reward_source or "",
            session_heartbeat_fn=session_heartbeat_fn,
        )

        rec.save_summary(summary)
        rec.set_status("completed")
    except ImportError as exc:
        msg = f"Missing dependency: {exc}. Install with: uv sync"
        rec.set_status("error", error=msg)
        raise ImportError(msg) from exc
    except Exception as exc:
        rec.set_status("error", error=str(exc))
        raise

    logger.info("IterationRecord saved to: %s", rec.path)
    return rec.path
