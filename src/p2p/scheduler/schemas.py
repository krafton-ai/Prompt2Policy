"""Pydantic models for scheduler API request/response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from p2p.config import DEFAULT_JUDGMENT_SELECT
from p2p.scheduler.types import BackendType, JobState, RunState

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class NodeCreateRequest(BaseModel):
    node_id: str
    host: str
    user: str
    port: int = 22
    base_dir: str = ""  # empty = auto (/tmp/p2p-{sha}/)
    max_cores: int = Field(1, ge=1)
    num_gpus: int = 0
    gpu_memory_mb: int = 0
    sync_mode: str = "rsync"  # "rsync" or "git"
    enabled: bool = True


class NodeUpdateRequest(BaseModel):
    host: str | None = None
    user: str | None = None
    port: int | None = None
    base_dir: str | None = None
    max_cores: int | None = Field(None, ge=1)
    num_gpus: int | None = None
    gpu_memory_mb: int | None = None
    sync_mode: str | None = None
    enabled: bool | None = None


class NodeResponse(BaseModel):
    node_id: str
    host: str
    user: str
    port: int
    base_dir: str = ""
    max_cores: int
    num_gpus: int = 0
    gpu_memory_mb: int = 0
    sync_mode: str = "rsync"
    enabled: bool = True
    used_cores: int = 0
    online: bool = False
    active_runs: int = 0


class NodeCheckResponse(BaseModel):
    node_id: str
    online: bool
    uv_available: bool = False
    gpu: str | None = None
    mps_active: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class RunStatusResponse(BaseModel):
    run_id: str
    state: RunState
    node_id: str = ""
    pid: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class SubmitBenchmarkJobRequest(BaseModel):
    """Submit a benchmark job (multiple test cases)."""

    backend: BackendType = "ssh"
    csv_file: str | None = None  # CSV filename under benchmark/; None = default
    pass_threshold: float = 0.7
    total_timesteps: int = 1_000_000
    max_iterations: int = 5
    seed: int = 1
    seeds: list[int] = Field(default_factory=lambda: [1])
    num_configs: int = Field(1, ge=1, le=10)
    num_envs: int = 1
    model: str = ""  # LLM model override (empty = server default)
    vlm_model: str | None = None
    max_parallel: int = 30
    cores_per_run: int = 0
    filter_envs: list[str] = []
    filter_categories: list[str] = []
    filter_difficulties: list[str] = []
    mode: Literal["staged", "flat"] = "staged"
    num_stages: int = 25
    gate_threshold: float = 0.7
    start_from_stage: int = 1
    side_info: bool = False
    trajectory_stride: int = Field(1, ge=1)
    use_zoo_preset: bool = True
    hp_tuning: bool = False
    use_code_judge: bool = False
    review_reward: bool = True
    review_judge: bool = True
    device: str = "auto"  # "auto" | "cpu"
    thinking_effort: str = ""
    refined_initial_frame: bool = True
    criteria_diagnosis: bool = True
    motion_trail_dual: bool = True
    judgment_select: str = DEFAULT_JUDGMENT_SELECT
    allowed_nodes: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: JobState
    run_ids: list[str]
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    metadata: dict | None = None
    backend: str | None = None
    config: dict | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
