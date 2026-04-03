from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from p2p.config import DEFAULT_JUDGMENT_SELECT
from p2p.contracts import StatusLiteral


class IterationSummary(BaseModel):
    """API summary for one iteration. Based on contracts.TrainSummary + contracts.RewardSpec."""

    iteration_id: str
    session_id: str
    env_id: str
    status: StatusLiteral
    created_at: str
    total_timesteps: int
    final_episodic_return: float | None
    reward_latex: str
    reward_description: str
    video_urls: list[str]
    progress: float | None


class IterationDetail(IterationSummary):
    """Full iteration data. Based on contracts.TrainSummary + RewardSpec + JudgmentResult."""

    config: dict
    reward_spec: dict
    reward_source: str
    summary: dict | None
    eval_results: list[dict]
    judgment: dict | None = None  # from judgment.json (per-checkpoint scores)
    training: list[dict] = []  # training scalars (from scalars.jsonl)


class MetricsResponse(BaseModel):
    training: list[dict]
    evaluation: list[dict]


class RunConfigEntrySchema(BaseModel):
    """API mirror of contracts.RunConfigEntry."""

    config_id: str
    label: str = ""
    params: dict = {}


class ElaborateIntentRequest(BaseModel):
    """Request to decompose an intent into behavioral criteria."""

    prompt: str
    env_id: str = "HalfCheetah-v5"
    model: str | None = None


class IntentCriterionSchema(BaseModel):
    title: str
    description: str
    category: str
    default_on: bool


class ElaborateIntentResponse(BaseModel):
    criteria: list[IntentCriterionSchema]


class StartSessionRequest(BaseModel):
    """Session creation request. Corresponds to contracts.SessionConfig."""

    prompt: str = ""
    total_timesteps: int = 1_000_000
    seed: int = 1
    max_iterations: int = 5
    pass_threshold: float = 0.7
    env_id: str = "HalfCheetah-v5"
    num_envs: int = 1
    model: str = ""  # LLM model override (empty = server default)
    vlm_model: str | None = None
    thinking_effort: str = ""
    side_info: bool = False
    # Multi-config × seeds support
    configs: list[RunConfigEntrySchema] = []
    num_configs: int = 0  # auto-generate N perturbed configs (ignored if configs given)
    seeds: list[int] = []
    cores_per_run: int = 0  # 0 = auto
    max_parallel: int = 0  # 0 = auto
    num_evals: int = Field(4, ge=1)  # number of eval checkpoints (videos + trajectories)
    # HP strategy
    use_zoo_preset: bool = True  # use RL Baselines3 Zoo tuned HPs as starting point
    hp_tuning: bool = False  # allow LLM to tune HPs (False = reward-only mode)
    use_code_judge: bool = False
    review_reward: bool = True  # review generated reward code before training
    review_judge: bool = True  # review generated judge code before training
    trajectory_stride: int = Field(1, ge=1)  # save every Nth step (1 = all steps)
    judgment_select: str = DEFAULT_JUDGMENT_SELECT
    elaborated_intent: str = ""  # pre-built elaborated intent (from elicitation UI)
    refined_initial_frame: bool = True  # pad video + first frame JPEG in Turn 1
    criteria_diagnosis: bool = True
    motion_trail_dual: bool = True
    terminate_when_unhealthy: bool = False  # disable early termination for unhealthy states


class StartSessionResponse(BaseModel):
    session_id: str
    status: StatusLiteral


class LoopIterationSummary(BaseModel):
    """Per-iteration summary.

    Based on contracts.JudgmentResult + ReviseResult + IterationAggregation.
    """

    iteration: int
    iteration_dir: str
    intent_score: float | None
    best_checkpoint: str = ""
    checkpoint_scores: dict[str, float] = {}
    checkpoint_diagnoses: dict[str, str] = {}
    # Per-episode rollout scores (keyed by "{step}_ep{N}")
    rollout_scores: dict[str, float] = {}
    rollout_diagnoses: dict[str, str] = {}
    diagnosis: str
    failure_tags: list[str]
    reward_code: str
    reward_diff_summary: str = ""
    final_return: float | None
    video_urls: list[str] = []
    # Multi-config iteration fields
    is_multi_config: bool = False
    aggregation: dict | None = None  # from aggregation.json
    best_config_id: str = ""
    best_run_id: str = ""
    # Video source info (which run's video is displayed)
    video_source_run_id: str = ""
    video_source_return: float | None = None
    # Timing
    elapsed_time_s: float | None = None
    # Revise agent output
    reward_reasoning: str = ""
    hp_reasoning: str = ""
    hp_changes: dict[str, Any] = {}
    training_dynamics: str = ""
    revise_diagnosis: str = ""
    based_on: int = 0
    # Per-judge raw outputs (shown separately in frontend)
    code_diagnosis: str = ""
    code_score: float | None = None
    vlm_diagnosis: str = ""
    vlm_score: float | None = None
    vlm_criteria: str = ""
    criteria_scores: list[dict] = []
    scoring_method: str = ""
    checkpoint_code_diagnoses: dict[str, str] = {}
    checkpoint_vlm_diagnoses: dict[str, str] = {}
    checkpoint_code_scores: dict[str, float] = {}
    checkpoint_vlm_scores: dict[str, float] = {}
    rollout_code_diagnoses: dict[str, str] = {}
    rollout_vlm_diagnoses: dict[str, str] = {}
    rollout_code_scores: dict[str, float] = {}
    rollout_vlm_scores: dict[str, float] = {}
    rollout_synthesis_traces: dict[str, list[dict]] = {}
    rollout_criteria_scores: dict[str, list[dict]] = {}
    # VLM preview video URLs (center-of-interval sampled)
    rollout_vlm_preview_urls: dict[str, str] = {}
    # Motion trail preview URLs (when --motion-trail-dual was enabled)
    rollout_motion_preview_urls: dict[str, str] = {}
    # VLM frames-per-second used for preview sampling
    vlm_fps: int = 0
    # Human label status (from human_label.json)
    human_label: dict | None = None


class SessionDetail(BaseModel):
    """Session overview. Based on contracts.LoopResult + EntityMetadata."""

    session_id: str
    prompt: str
    status: StatusLiteral
    best_iteration: int
    best_score: float
    iterations: list[LoopIterationSummary]
    error: str | None = None
    env_id: str = ""
    created_at: str = ""
    total_timesteps: int = 0
    pass_threshold: float = 0.7
    is_stale: bool = False
    # User metadata
    alias: str = ""
    starred: bool = False
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Iteration sub-runs (multi-config × seeds)
# ---------------------------------------------------------------------------


class IterationRunEntry(BaseModel):
    """API mirror of contracts.IterationRunInfo."""

    config_id: str
    seed: int
    run_id: str
    status: StatusLiteral
    final_return: float | None = None
    intent_score: float | None = None
    video_urls: list[str] = []


class MeanStdArray(BaseModel):
    """API mirror of contracts.MeanStdSeries."""

    mean: list[float]
    std: list[float]


class AggregatedMetricsResponse(BaseModel):
    """API mirror of contracts.AggregatedMetrics."""

    config_id: str
    seeds: list[int]
    available_metrics: list[str]
    global_steps: list[int]
    metrics: dict[str, MeanStdArray]


class RunMetricsResponse(BaseModel):
    """API mirror of contracts.RunMetricsDetail."""

    run_id: str
    config_id: str
    seed: int
    training: list[dict]
    evaluation: list[dict]
    video_urls: list[str]
    # Per-checkpoint judgment data
    checkpoint_scores: dict[str, float] = {}
    checkpoint_diagnoses: dict[str, str] = {}
    checkpoint_code_diagnoses: dict[str, str] = {}
    checkpoint_vlm_diagnoses: dict[str, str] = {}
    checkpoint_code_scores: dict[str, float] = {}
    checkpoint_vlm_scores: dict[str, float] = {}
    rollout_scores: dict[str, float] = {}
    rollout_diagnoses: dict[str, str] = {}
    rollout_code_diagnoses: dict[str, str] = {}
    rollout_vlm_diagnoses: dict[str, str] = {}
    rollout_code_scores: dict[str, float] = {}
    rollout_vlm_scores: dict[str, float] = {}
    rollout_synthesis_traces: dict[str, list[dict]] = {}
    rollout_vlm_preview_urls: dict[str, str] = {}
    vlm_fps: int = 0
    best_checkpoint: str = ""
    intent_score: float | None = None
    diagnosis: str = ""
    config: dict | None = None


class UpdateMetadataRequest(BaseModel):
    """Partial update for contracts.EntityMetadata fields."""

    alias: str | None = None
    starred: bool | None = None
    tags: list[str] | None = None


class UpdateMetadataResponse(BaseModel):
    alias: str
    starred: bool
    tags: list[str]


class TrashItem(BaseModel):
    entity_id: str
    entity_type: str  # "session" | "benchmark"
    alias: str = ""
    deleted_at: str = ""
    created_at: str = ""
    prompt: str = ""
    status: str = ""


class StopResponse(BaseModel):
    stopped: bool
    detail: str


class ResourceAutoResponse(BaseModel):
    cores_per_run: int
    num_envs: int
    max_parallel: int
    total_runs: int
    num_batches: int
    time_score: float
    estimated_processes: int
    usable_cores: int


class ResourceStatusResponse(BaseModel):
    total_cores: int
    reserved_cores: int
    available_cores: int
    active_runs: int
    allocations: list[dict] = []


class RunProcessInfo(BaseModel):
    run_id: str
    pid: int
    cores: list[int]


class CoreProcessInfo(BaseModel):
    session_id: str
    pid: int
    cores: list[int]
    runs: list[RunProcessInfo] = []


class MemoryInfo(BaseModel):
    total_mb: int
    used_mb: int
    available_mb: int
    percent: float


class CpuUsageResponse(BaseModel):
    per_core: list[float]
    avg: float
    processes: list[CoreProcessInfo] = []
    memory: MemoryInfo | None = None


class GpuProcessInfo(BaseModel):
    pid: int
    gpu_memory_mb: float
    process_name: str = ""
    session_id: str = ""
    run_id: str = ""


class GpuInfo(BaseModel):
    index: int
    name: str
    temperature: float
    utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    power_limit_w: float
    processes: list[GpuProcessInfo] = []


class GpuUsageResponse(BaseModel):
    gpus: list[GpuInfo]


class EnvInfo(BaseModel):
    env_id: str
    name: str
    obs_dim: int
    action_dim: int
    info_keys: dict[str, str]
    description: str
    engine: str = "mujoco"
    zoo_num_envs: int = 1


# ---------------------------------------------------------------------------
# VLM proxy
# ---------------------------------------------------------------------------


class VlmMessage(BaseModel):
    role: str
    content: str
    images: list[str] | None = None


class VlmChatRequest(BaseModel):
    model: str = "qwen3.5:27b"
    messages: list[VlmMessage]
    stream: bool = False
    think: bool = False
    options: dict | None = None


class VlmChatResponse(BaseModel):
    model: str
    message: dict
    done: bool


class VlmStatusResponse(BaseModel):
    available: bool
    model: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------


class SessionAnalysisResponse(BaseModel):
    """API mirror of contracts.SessionAnalysis."""

    session_id: str
    analysis_en: str
    key_findings: list[str]
    recommendations: list[str]
    tool_calls_used: int
    model: str
    created_at: str


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkCaseInfo(BaseModel):
    env_id: str
    category: str
    difficulty: str


class BenchmarkOptionsResponse(BaseModel):
    envs: list[str]
    categories: list[str]
    difficulties: list[str]
    cases: list[BenchmarkCaseInfo] = []
    csv_files: list[str] = []


class StageGateResult(BaseModel):
    """Gate evaluation result for a completed stage."""

    passed: bool
    avg_score: float
    success_rate: float
    completed: int
    total: int
    threshold: float


class StageDetail(BaseModel):
    """Runtime status of a single stage."""

    stage: int
    name: str
    status: str  # "pending" | "running" | "completed" | "gate_passed" | "gate_failed" | "skipped"
    gate_threshold: float
    max_parallel: int
    case_count: int
    case_indices: list[int] = []
    gate_result: StageGateResult | None = None


class BenchmarkGroupStats(BaseModel):
    total: int
    completed: int
    passed: int
    success_rate: float
    average_score: float
    cumulative_score: float


class BenchmarkTestCaseResult(BaseModel):
    index: int
    env_id: str
    instruction: str
    category: str
    difficulty: str
    session_id: str
    session_status: StatusLiteral
    best_score: float
    passed: bool
    iterations_completed: int
    video_urls: list[str]
    stage: int = 0
    iteration_scores: list[float] = []
    node_id: str = ""
    pids: list[int] = []
    max_iterations: int = 0
    judge_scores: dict[str, float | None] = {}
    judge_diagnoses: dict[str, str] = {}


class BenchmarkRunSummary(BaseModel):
    benchmark_id: str
    created_at: str
    completed_at: str | None = None
    status: StatusLiteral
    total_cases: int
    completed_cases: int
    passed_cases: int
    success_rate: float
    average_score: float
    cumulative_score: float
    mode: str = "flat"
    current_stage: int = 0
    total_stages: int = 0
    # User metadata
    alias: str = ""
    starred: bool = False
    tags: list[str] = []


class BenchmarkRunDetail(BenchmarkRunSummary):
    pass_threshold: float
    by_category: dict[str, BenchmarkGroupStats]
    by_difficulty: dict[str, BenchmarkGroupStats]
    by_env: dict[str, BenchmarkGroupStats]
    test_cases: list[BenchmarkTestCaseResult]
    stages: list[StageDetail] = []
    start_from_stage: int = 1
    max_iterations: int = 5


class StopBenchmarkResponse(BaseModel):
    stopped: bool
    stopped_sessions: int
    detail: str


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class EventSummary(BaseModel):
    seq: int
    timestamp: str
    event: str
    iteration: int | None = None
    data: dict = {}
    duration_ms: int | None = None
    has_full_content: bool = False


class EventDetail(BaseModel):
    seq: int
    timestamp: str
    event: str
    iteration: int | None = None
    data: dict = {}
    duration_ms: int | None = None


# ---------------------------------------------------------------------------
# Node resource monitoring
# ---------------------------------------------------------------------------


class NodeGpuInfo(BaseModel):
    index: int
    name: str
    utilization: float
    memory_used_mb: float
    memory_total_mb: float
    temperature: float
    power_draw_w: float
    power_limit_w: float


class NodeResourceSnapshot(BaseModel):
    node_id: str
    online: bool
    timestamp: str
    cpu_count: int = 0
    cpu_percent_avg: float = 0.0
    cpu_per_core: list[float] = []
    load_avg: list[float] = []
    mem_total_mb: int = 0
    mem_used_mb: int = 0
    mem_available_mb: int = 0
    gpus: list[NodeGpuInfo] = []
    error: str | None = None


class NodeResourcesResponse(BaseModel):
    nodes: list[NodeResourceSnapshot]
    poll_interval_s: int = 10


# ---------------------------------------------------------------------------
# Human labeling
# ---------------------------------------------------------------------------


class HumanLabelRequest(BaseModel):
    """Request to submit a human score for an iteration's eval videos."""

    session_id: str
    iteration: int
    annotator: str = Field(..., min_length=1, max_length=100)
    intent_score: float = Field(..., ge=0.0, le=1.0)
    video_url: str = ""


class HumanLabelResponse(BaseModel):
    status: str
    video_count: int


class LabelingStatusResponse(BaseModel):
    enabled: bool
    annotator: str = ""
