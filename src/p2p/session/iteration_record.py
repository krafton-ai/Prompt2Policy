"""Typed accessor for an iteration directory."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Container, Generator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from p2p.config import TrainConfig
    from p2p.contracts import (
        EntityMetadata,
        HumanLabelEntry,
        JudgmentResult,
        RewardSpec,
        SessionAnalysis,
        SessionConfig,
        StatusData,
        StatusLiteral,
        TrainSummary,
    )


def read_json_safe(path: Path) -> dict | None:
    """Read JSON from *path*, tolerating corruption or partial writes."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text else None
    except (json.JSONDecodeError, OSError):
        return None


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via temp-file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = -1
        Path(tmp).replace(path)
    except BaseException:
        if fd != -1:
            with contextlib.suppress(OSError):
                os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def status_lock(directory: str | Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for status.json in *directory*.

    Uses ``fcntl.flock`` on a dedicated lock file so that concurrent
    threads (within the same process) and concurrent processes (e.g.
    the loop subprocess vs. the API server watchdog) are serialized.
    """
    lock_path = Path(directory) / "status.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_status_unlocked(
    directory: str | Path, status: StatusLiteral, error: str | None = None
) -> None:
    """Write status.json — caller MUST already hold ``status_lock``."""
    data: StatusData = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        data["error"] = error
    _atomic_write(Path(directory) / "status.json", json.dumps(data))


def write_status(directory: str | Path, status: StatusLiteral, error: str | None = None) -> None:
    """Write status.json to *directory* (iteration dir or session dir)."""
    with status_lock(directory):
        _write_status_unlocked(directory, status, error=error)


@dataclass
class IterationData:
    """Record of a single loop iteration (moved from loop.py).

    Revision fields: reward_reasoning, hp_reasoning, hp_changes,
    training_dynamics, revise_diagnosis.
    """

    iteration: int
    iteration_dir: str
    reward_code: str
    summary: TrainSummary
    judgment: JudgmentResult
    reward_reasoning: str = ""
    hp_reasoning: str = ""
    hp_changes: dict[str, Any] = field(default_factory=dict)
    training_dynamics: str = ""
    revise_diagnosis: str = ""
    lesson: str = ""
    based_on: int = 0  # iteration number whose reward code was used as base
    is_multi_config: bool = False
    aggregation: dict[str, Any] | None = None


class IterationRecord:
    """Typed accessor for the standard iteration directory layout.

    All path properties are deterministic derivations from ``self.path``.
    Read methods return parsed data or ``None`` when files don't exist.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- path properties ---------------------------------------------------

    @property
    def iteration_id(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def reward_fn_path(self) -> Path:
        return self.path / "reward_fn.py"

    @property
    def reward_spec_path(self) -> Path:
        return self.path / "reward_spec.json"

    @property
    def summary_path(self) -> Path:
        return self.path / "summary.json"

    @property
    def scalars_path(self) -> Path:
        return self.path / "metrics" / "scalars.jsonl"

    @property
    def trajectory_path(self) -> Path:
        """Return the latest trajectory file, preferring .npz over legacy .jsonl."""
        per_step = sorted(self.path.glob("trajectory_*.npz"))
        if per_step:
            return per_step[-1]
        per_step = sorted(self.path.glob("trajectory_*.jsonl"))
        if per_step:
            return per_step[-1]
        # Fallback: prefer legacy .jsonl if it exists, otherwise .npz
        legacy = self.path / "trajectory.jsonl"
        if legacy.exists():
            return legacy
        return self.path / "trajectory.npz"

    @property
    def prompt_path(self) -> Path:
        return self.path / "prompt.txt"

    @property
    def videos_dir(self) -> Path:
        return self.path / "videos"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def judgment_path(self) -> Path:
        return self.path / "judgment.json"

    @property
    def revised_reward_path(self) -> Path:
        return self.path / "reward_fn_revised.py"

    # -- read methods ------------------------------------------------------

    def read_config(self) -> dict[str, Any] | None:
        """Return parsed config.json (serialized ``TrainConfig`` dataclass)."""
        return self._read_json(self.config_path)

    def read_summary(self) -> TrainSummary | None:
        return self._read_json(self.summary_path)  # type: ignore[return-value]

    def read_reward_spec(self) -> RewardSpec:
        default: RewardSpec = {"latex": "", "terms": [], "description": ""}
        return self._read_json(self.reward_spec_path) or default  # type: ignore[return-value]

    def read_reward_source(self) -> str:
        if self.reward_fn_path.exists():
            return self.reward_fn_path.read_text()
        return ""

    def read_judgment(self) -> JudgmentResult | None:
        return self._read_json(self.judgment_path)  # type: ignore[return-value]

    # -- write methods -----------------------------------------------------

    def _ensure_dir(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: TrainConfig) -> None:
        self._ensure_dir()
        self.config_path.write_text(config.to_json())

    def save_reward_source(self, source: str) -> None:
        self._ensure_dir()
        self.reward_fn_path.write_text(source)

    def save_reward_spec(self, spec: RewardSpec) -> None:
        self._ensure_dir()
        self.reward_spec_path.write_text(json.dumps(spec, indent=2))

    def save_summary(self, summary: TrainSummary) -> None:
        self._ensure_dir()
        self.summary_path.write_text(json.dumps(summary, indent=2))

    def save_judgment(self, judgment: JudgmentResult) -> None:
        self._ensure_dir()
        self.judgment_path.write_text(json.dumps(judgment, indent=2))

    # -- human label -------------------------------------------------------

    @property
    def human_label_path(self) -> Path:
        return self.path / "human_label.json"

    def read_human_labels(self) -> dict[str, HumanLabelEntry] | None:
        """Read per-video human labels. Keys are video filenames."""
        return self._read_json(self.human_label_path)  # type: ignore[return-value]

    def save_human_labels(self, data: dict[str, HumanLabelEntry]) -> None:
        """Write per-video human labels atomically."""
        self._ensure_dir()
        _atomic_write(self.human_label_path, json.dumps(data, indent=2))

    def save_revised_reward(self, code: str) -> None:
        self._ensure_dir()
        self.revised_reward_path.write_text(code)

    def set_status(self, status: StatusLiteral, error: str | None = None) -> None:
        self._ensure_dir()
        write_status(self.path, status, error=error)

    # -- derived status ----------------------------------------------------

    def derive_status(self) -> str:
        # Explicit status.json takes priority
        status_data = self._read_json(self.status_path)
        if status_data and "status" in status_data:
            return status_data["status"]
        # Fallback: infer from files (backward compat)
        if (self.path / "cancelled.json").exists():
            return "cancelled"
        if self.summary_path.exists():
            return "completed"
        if self.scalars_path.exists():
            return "running"
        if self.config_path.exists():
            return "pending"
        return "unknown"

    def compute_progress(self) -> float | None:
        config = self.read_config()
        if config is None:
            return None
        total = config.get("total_timesteps", 1)
        if not self.scalars_path.exists():
            return 0.0
        lines = self.scalars_path.read_text().strip().split("\n")
        if not lines or not lines[-1]:
            return 0.0
        last = json.loads(lines[-1])
        return min(1.0, last.get("global_step", 0) / total)

    # -- file listing ------------------------------------------------------

    def video_filenames(self) -> list[str]:
        if not self.videos_dir.exists():
            return []
        return sorted(
            f.name
            for f in self.videos_dir.glob("*.mp4")
            if not f.stem.endswith(("_vlm", "_flow", "_motion"))
        )

    def parse_scalars(self) -> tuple[list[dict], list[dict]]:
        """Parse scalars.jsonl into (training, evaluation) lists."""
        training: list[dict] = []
        evaluation: list[dict] = []
        if not self.scalars_path.exists():
            return training, evaluation
        for line in self.scalars_path.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "eval":
                evaluation.append(entry)
            else:
                training.append(entry)
        return training, evaluation

    # -- validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of issues (empty = valid)."""
        issues: list[str] = []
        if not self.path.exists():
            issues.append(f"Iteration directory does not exist: {self.path}")
            return issues
        if not self.config_path.exists():
            issues.append("Missing config.json")
        if not self.reward_fn_path.exists():
            issues.append("Missing reward_fn.py")
        return issues

    # -- helpers -----------------------------------------------------------

    _read_json = staticmethod(read_json_safe)

    def __repr__(self) -> str:
        return f"IterationRecord({self.path})"


class SessionRecord:
    """Typed accessor for a session directory.

    Handles session-level operations (status, loop history).
    Individual iterations within a session use ``IterationRecord``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def session_id(self) -> str:
        return self.path.name

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def history_path(self) -> Path:
        return self.path / "loop_history.json"

    @property
    def config_path(self) -> Path:
        return self.path / "session_config.json"

    def ensure_dir(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    # -- status ------------------------------------------------------------

    def set_status(self, status: StatusLiteral, error: str | None = None) -> None:
        self.ensure_dir()
        write_status(self.path, status, error=error)

    def set_status_if(
        self,
        status: StatusLiteral,
        *,
        only_if: Container[str],
        error: str | None = None,
    ) -> bool:
        """Atomically set status only when the current status is in *only_if*.

        Holds a file lock across the read-check-write to prevent races
        between the loop subprocess, watchdog thread, and stop-session
        API handler.  Returns True if the status was updated.
        """
        self.ensure_dir()
        with status_lock(self.path):
            current = read_json_safe(self.status_path)
            current_status = current.get("status") if current else None
            if current_status not in only_if:
                return False
            _write_status_unlocked(self.path, status, error=error)
            return True

    def read_status(self) -> StatusData | None:
        return self._read_json(self.status_path)  # type: ignore[return-value]

    def touch_heartbeat(self) -> None:
        """Update only the ``updated_at`` timestamp without changing status."""
        with status_lock(self.path):
            status_data = read_json_safe(self.status_path)
            if status_data is None:
                return
            status_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write(self.status_path, json.dumps(status_data))

    # -- loop history ------------------------------------------------------

    def save_history(self, data: dict) -> None:
        """Save loop history dict (from ``dataclasses.asdict(LoopResult)``)."""
        self.ensure_dir()
        _atomic_write(self.history_path, json.dumps(data, indent=2))

    def read_history(self) -> dict | None:
        return self._read_json(self.history_path)

    # -- session config ----------------------------------------------------

    _SESSION_CONFIG_REQUIRED_KEYS = frozenset({"prompt", "train"})

    def read_session_config(self) -> SessionConfig | None:
        """Read session_config.json, returning None if missing or incomplete."""
        data = self._read_json(self.config_path)
        if data is None:
            return None
        missing = self._SESSION_CONFIG_REQUIRED_KEYS - data.keys()
        if missing:
            logger.warning("%s missing required keys: %s", self.config_path, missing)
            return None
        return data  # type: ignore[return-value]

    # -- metadata ----------------------------------------------------------

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    def read_metadata(self) -> EntityMetadata:
        data = self._read_json(self.metadata_path)
        if data is None:
            return {}  # type: ignore[return-value]
        return data  # type: ignore[return-value]

    def save_metadata(self, data: EntityMetadata) -> None:
        self.ensure_dir()
        _atomic_write(self.metadata_path, json.dumps(data, indent=2))

    def update_metadata(self, **kwargs: Any) -> EntityMetadata:
        """Merge **kwargs into existing metadata and persist."""
        meta = self.read_metadata()
        meta.update(kwargs)  # type: ignore[typeddict-item]
        self.save_metadata(meta)
        return meta

    # -- analysis cache ----------------------------------------------------

    @property
    def analysis_path(self) -> Path:
        return self.path / "analysis.json"

    def save_analysis(self, data: dict) -> None:
        self.ensure_dir()
        self.analysis_path.write_text(json.dumps(data, indent=2))

    _ANALYSIS_REQUIRED_KEYS = frozenset(
        {
            "session_id",
            "analysis_en",
            "key_findings",
            "recommendations",
            "tool_calls_used",
            "model",
            "created_at",
        }
    )

    def read_analysis(self) -> SessionAnalysis | None:
        """Read analysis.json, returning None if missing or incomplete."""
        data = self._read_json(self.analysis_path)
        if data is None:
            return None
        missing = self._ANALYSIS_REQUIRED_KEYS - data.keys()
        if missing:
            logger.warning("%s missing required keys: %s", self.analysis_path, missing)
            return None
        return data  # type: ignore[return-value]

    # -- iteration access --------------------------------------------------

    def iteration_records(self) -> list[IterationRecord]:
        """Return IterationRecords for all iterations in this session, sorted by name."""
        if not self.path.exists():
            return []
        iterations = [
            IterationRecord(d)
            for d in self.path.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]
        return sorted(iterations, key=lambda r: r.iteration_id)

    def get_iteration(self, iteration_id: str) -> IterationRecord | None:
        """Get a specific iteration by ID, or None if not found."""
        candidate = self.path / iteration_id
        if candidate.is_dir() and (candidate / "config.json").exists():
            return IterationRecord(candidate)
        return None

    # -- helpers -----------------------------------------------------------

    _read_json = staticmethod(read_json_safe)

    def __repr__(self) -> str:
        return f"SessionRecord({self.path})"
