"""Data types for the scheduler module.

RunSpec is the atomic execution unit that backends see.
Job is the higher-level grouping that controllers manage.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 12):
    from typing import Any, Literal, NotRequired, TypedDict
else:
    from typing import Any, Literal

    from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Status literals
# ---------------------------------------------------------------------------

RunState = Literal["pending", "running", "completed", "error", "cancelled"]
JobState = RunState  # same values

# Session-level statuses that indicate successful completion.
# Used by Backend implementations and job_scheduler to decide
# whether a dead process ended successfully or in error.
SUCCESS_STATES: frozenset[str] = frozenset(("completed", "passed", "max_iterations"))

# RunState values that indicate a run is finished (no further transitions).
TERMINAL_STATES: frozenset[str] = frozenset(("completed", "error", "cancelled"))

BackendType = Literal["local", "ssh"]
JobType = Literal["session", "benchmark"]


def now_iso() -> str:
    """UTC ISO 8601 timestamp."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RunSpec — the only thing backends see
# ---------------------------------------------------------------------------


class RunSpec(TypedDict):
    """Atomic execution unit. Backends translate this into a command."""

    run_id: str
    entry_point: str  # Python module, e.g. "p2p.session.run_session"
    parameters: dict[str, Any]  # CLI args as dict (backend can rewrite paths)
    env: NotRequired[dict[str, str]]  # extra env vars (ANTHROPIC_API_KEY, etc.)
    cpu_cores: int  # resource requirement
    gpu_count: NotRequired[int]  # GPU resource requirement (0 = no GPU needed)
    tags: NotRequired[dict[str, str]]  # opaque labels for grouping


class RunStatus(TypedDict):
    """Runtime status of a single run."""

    run_id: str
    state: RunState
    pid: NotRequired[int]
    exit_code: NotRequired[int]
    node_id: NotRequired[str]
    remote_dir: NotRequired[str]  # SSH runs: remote working directory
    started_at: NotRequired[str]
    completed_at: NotRequired[str]
    error: NotRequired[str]


# ---------------------------------------------------------------------------
# NodeConfig — SSH node definition
# ---------------------------------------------------------------------------


class NodeConfig(TypedDict):
    """One SSH-accessible compute node."""

    node_id: str
    host: str
    user: str
    port: int  # default 22
    base_dir: NotRequired[str]  # override remote working directory (default: /tmp/p2p-{sha}/)
    max_cores: int  # total CPU cores available for scheduling on this node
    num_gpus: NotRequired[int]  # number of GPUs available (0 = no GPU)
    gpu_memory_mb: NotRequired[int]  # total VRAM per GPU in MB
    sync_mode: NotRequired[str]  # "rsync" (default) or "git" (git pull, preserves venv)
    enabled: NotRequired[bool]  # whether to include in scheduling (default True)


# ---------------------------------------------------------------------------
# Job — higher-level grouping managed by controllers
# ---------------------------------------------------------------------------


class Job(TypedDict):
    """A group of runs managed by a controller."""

    job_id: str
    job_type: JobType
    run_ids: list[str]  # owned RunSpec run_ids
    status: JobState
    created_at: str
    completed_at: NotRequired[str]
    error: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]  # type-specific extra info
    backend: NotRequired[BackendType]
    config: NotRequired[dict[str, Any]]


# ---------------------------------------------------------------------------
# File-based job manifest (replaces in-memory state)
# ---------------------------------------------------------------------------


class RunRecord(TypedDict):
    """Per-run persistent state inside a JobManifest."""

    run_id: str
    spec: RunSpec
    state: RunState
    node_id: str
    remote_dir: str
    pid: NotRequired[int]
    started_at: NotRequired[str]
    completed_at: NotRequired[str]
    synced: bool
    error: NotRequired[str]
    session_group: NotRequired[str]  # node affinity: same group → same node
    allocated_cores: NotRequired[list[int]]  # pinned CPU core IDs
    allocated_gpus: NotRequired[list[int]]  # pinned GPU device IDs


class JobManifest(TypedDict):
    """File-based manifest — the single source of truth for a job."""

    job_id: str
    job_type: JobType
    status: JobState
    created_at: str
    completed_at: NotRequired[str]
    scheduler_pid: NotRequired[int]
    backend: BackendType
    config: dict[str, Any]
    runs: list[RunRecord]
    metadata: NotRequired[dict[str, Any]]
    error: NotRequired[str]
