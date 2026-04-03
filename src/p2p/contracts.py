"""Explicit data contracts shared across modules.

Every dict that crosses module boundaries has a TypedDict here.
Producers MUST return these types. Consumers can rely on these fields.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from typing_extensions import NotRequired, Required, TypedDict

# ---------------------------------------------------------------------------
# Metric rounding utility
# ---------------------------------------------------------------------------


def round_metric(v: float | int | None, sig: int = 5) -> float | int | None:
    """Round a numeric value to *sig* significant figures.

    Handles any magnitude gracefully: large returns (1003.59… → 1003.6) and
    small scores (0.001234… → 0.0012346) both keep useful precision without
    serializing 15+ decimal digits.

    Non-float inputs (``None``, ``int``) pass through unchanged.
    """
    if not isinstance(v, float) or v == 0.0 or not math.isfinite(v):
        return v
    ndigits = sig - 1 - int(math.floor(math.log10(abs(v))))
    return round(v, ndigits)


# ---------------------------------------------------------------------------
# Intent elicitation
# ---------------------------------------------------------------------------


class IntentCriterion(TypedDict):
    """One behavioral criterion from intent elicitation."""

    title: str  # Short label, e.g. "Alternate legs symmetrically"
    description: str  # Full detail with measurable thresholds
    category: str  # "gait" | "posture" | "dynamics" | "stability" | "efficiency"
    default_on: bool  # LLM suggestion for default state


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

# Shared status values (used by both session and iteration):
#   pending, running, completed, error, cancelled
# Session-only values:
#   passed, max_iterations, rate_limited, auth_error, invalid_code
# Iteration-only values:
#   unknown
StatusLiteral = Literal[
    "pending",
    "queued",  # benchmark: scheduled but not yet launched
    "running",
    "completed",
    "error",
    "cancelled",
    "passed",  # session: loop passed threshold
    "max_iterations",  # session: loop exhausted iterations
    "rate_limited",  # session: API rate limit exhausted after retries
    "auth_error",  # session: API authentication failure
    "invalid_code",  # session: generated reward code has syntax error
    "failed",  # session: manual failure status
    "unknown",  # iteration: no status.json and no files to infer from
    "stale",  # benchmark: session data exists but process died without clean exit
    "gate_passed",  # benchmark: stage gate passed
    "gate_failed",  # benchmark: stage gate failed (early termination)
]

# ---------------------------------------------------------------------------
# Training backend output
# ---------------------------------------------------------------------------


class TrainSummary(TypedDict):
    """Returned by ppo.train() and sb3_trainer.train().

    Saved as summary.json by runner.run_training().
    """

    total_timesteps: int
    training_time_s: float
    final_episodic_return: float
    total_episodes: int
    algorithm: NotRequired[str]  # always "ppo"; present in some summary.json files


# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------


class EvalResult(TypedDict, total=False):
    """Returned by run_evaluation functions.

    Written to scalars.jsonl with added global_step and type="eval".
    When num_eval_rounds > 1, total_reward is the BEST episode's return
    and aggregate stats (mean_return, std_return, etc.) are included.
    """

    total_reward: Required[float]  # best episode return (backward compat)
    episode_length: Required[int]  # captured episode length (median if parallel)
    reward_terms: Required[dict[str, float]]  # captured episode per-step term means
    # Multi-episode aggregate stats (present when num_eval_rounds > 1)
    num_eval_rounds: int  # eval rounds (total episodes = num_envs * rounds)
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    median_return: float
    p10_return: float
    p90_return: float
    all_returns: list[float]
    per_episode_returns: list[float]  # backward compat (sequential path)


# ---------------------------------------------------------------------------
# Scalars.jsonl entries
# ---------------------------------------------------------------------------


class EvalScalar(TypedDict, total=False):
    """One eval entry in scalars.jsonl. Discriminated by type="eval"."""

    global_step: Required[int]
    type: Required[Literal["eval"]]
    total_reward: Required[float]  # best episode return (backward compat)
    episode_length: Required[int]
    reward_terms: Required[dict[str, float]]
    # Multi-episode aggregate stats
    num_eval_rounds: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    median_return: float
    p10_return: float
    p90_return: float
    all_returns: list[float]
    per_episode_returns: list[float]  # backward compat (sequential path)


# ---------------------------------------------------------------------------
# Judgment output
# ---------------------------------------------------------------------------

ScoringMethod = Literal[
    "vlm",
    "code_judge",
    "no_judge",
    "multi_rollout",
    "dual_judge+llm_synthesis",
    # Deprecated values (retained for reading existing sessions)
    "vlm+deterministic",
    "deterministic_only",
    "code_judge+deterministic",
    "llm_synthesis",
    "code_judge+llm_synthesis",
]


class SynthesisToolCall(TypedDict):
    """One tool call made during agentic synthesis."""

    tool_name: str  # "reask_vlm" | "run_trajectory_check"
    input: dict[str, str]  # {question: ...} or {python_code: ..., description: ...}
    output: str  # result text (truncated if long)


class JudgmentResult(TypedDict, total=False):
    """Returned by judge_all_checkpoints().

    Saved as judgment.json. Consumed by loop.py and services.py.
    Required fields are marked with Required[].
    """

    # Core fields (always present)
    intent_score: Required[float]
    passed: Required[bool]
    diagnosis: Required[str]
    failure_tags: Required[list[str]]

    # VLM fields
    evidence: list[str]
    vlm_criteria: str  # Turn 1 expectations (2-turn judging)
    criteria_scores: list[dict]  # Per-criterion assessment (criteria diagnosis)

    # Analysis metadata
    reward_term_analysis: dict
    vlm_score: float | None
    scoring_method: ScoringMethod

    # Synthesis traceability
    vlm_diagnosis: str

    # Code judge fields (present when scoring_method contains "code_judge")
    code_score: float | None
    code_diagnosis: str

    # Multi-checkpoint fields (from judge_all_checkpoints)
    best_checkpoint: str
    checkpoint_judgments: dict[str, JudgmentResult]

    # Agentic synthesis tool call traces
    synthesis_tool_calls: list[SynthesisToolCall]

    # Per-rollout judgment fields (from multi-rollout evaluation)
    rollout_judgments: list[RolloutJudgment]
    checkpoint_aggregate: CheckpointAggregateJudgment


# ---------------------------------------------------------------------------
# Per-rollout judgment (multi-rollout evaluation)
# ---------------------------------------------------------------------------


class RolloutJudgment(TypedDict, total=False):
    """Judgment for a single eval rollout episode."""

    episode_idx: Required[int]
    intent_score: Required[float]
    diagnosis: Required[str]
    failure_tags: Required[list[str]]
    eval_return: float  # absent when per-episode returns unavailable
    scoring_method: ScoringMethod
    rollout_label: str  # e.g. "p10", "median", "p90" for percentile videos
    # Component scores for debugging judge consistency
    code_diagnosis: str
    code_score: float
    vlm_diagnosis: str
    vlm_score: float
    # Agentic synthesis tool call traces
    synthesis_tool_calls: list[SynthesisToolCall]
    # VLM preview video filename (center-of-interval sampled at VLM fps)
    vlm_preview_filename: str
    # VLM criteria and per-criterion assessment
    vlm_criteria: str
    criteria_scores: list[dict]


class CheckpointAggregateJudgment(TypedDict, total=False):
    """Aggregate of all rollout judgments at one checkpoint."""

    step: Required[str]
    rollout_judgments: Required[list[RolloutJudgment]]
    mean_intent_score: Required[float]
    success_rate: Required[float]  # fraction with intent_score >= pass_threshold
    score_std: Required[float]
    common_failure_tags: Required[list[str]]  # representative failure tags for this checkpoint
    aggregate_diagnosis: str


# ---------------------------------------------------------------------------
# Reward spec
# ---------------------------------------------------------------------------


class RewardTerm(TypedDict, total=False):
    """One term in a reward function.

    Structured representation used in ``RewardSpec.terms``.
    """

    name: Required[str]
    latex: str  # per-term LaTeX equation
    description: str  # human-readable explanation
    weight: float  # term weight/coefficient


class RewardSpec(TypedDict, total=False):
    """Metadata about a reward function. Saved as reward_spec.json.

    ``terms`` is a list of :class:`RewardTerm` dicts.  Legacy JSON files
    may still store ``terms`` as ``dict[str, str]``; consumers should call
    :func:`normalize_reward_spec` to convert.
    """

    latex: Required[str]
    terms: Required[list[RewardTerm]]
    description: str


def normalize_reward_spec(spec: dict) -> RewardSpec:
    """Convert a raw reward_spec dict to the structured format.

    Handles backward compatibility: old files have ``terms: dict[str, str]``,
    new files have ``terms: list[RewardTerm]``.
    """
    terms_raw = spec.get("terms", [])
    if isinstance(terms_raw, dict):
        terms: list[RewardTerm] = []
        for k, v in terms_raw.items():
            term: RewardTerm = {"name": k}
            if isinstance(v, str):
                # Legacy format: value may contain "\n<equation>"
                parts = v.split("\n", 1)
                term["description"] = parts[0]
                if len(parts) > 1:
                    term["latex"] = parts[1].strip()
            else:
                term["description"] = str(v)
            terms.append(term)
        spec = dict(spec)
        spec["terms"] = terms
    return spec  # type: ignore[return-value]


class HumanLabelEntry(TypedDict, total=False):
    """Status of a human label submission for a single video.

    ``human_label.json`` maps video filenames to ``HumanLabelEntry`` dicts:
    ``{"eval_100000_median.mp4": {...}, "eval_100000_p10.mp4": {...}}``.
    """

    status: Required[Literal["sent", "scored", "error"]]
    annotator: Required[str]
    sent_at: Required[str]  # ISO 8601
    intent_score: float  # human score (0.0-1.0)
    scored_at: str  # ISO 8601, when score was confirmed by server
    error: str  # error message (status="error")
    labeling_server_result: dict  # raw response from labeling server


class StatusData(TypedDict):
    """Content of status.json (run or session level)."""

    status: StatusLiteral
    error: NotRequired[str]
    updated_at: NotRequired[str]  # ISO 8601 timestamp


# ---------------------------------------------------------------------------
# Multi-config hyperparameter search
# ---------------------------------------------------------------------------


class RunConfigEntry(TypedDict):
    """One hyperparameter configuration in a multi-config run."""

    config_id: str
    label: str  # human-readable, e.g. "lr=3e-4, ent=0.01"
    params: dict[str, Any]  # TrainConfig overrides


class RunAggregationEntry(TypedDict):
    """Aggregated statistics for one config across seeds."""

    mean_best_score: float
    std_best_score: float
    mean_final_return: float
    std_final_return: float
    per_seed: list[dict[str, float]]


# ---------------------------------------------------------------------------
# Iteration aggregation (multi-config×seeds per loop iteration)
# ---------------------------------------------------------------------------


class IterationAggregation(TypedDict):
    """Per-iteration aggregation across configs × seeds.

    Saved as ``aggregation.json`` inside each ``iter_N/`` directory.
    """

    best_config_id: str
    best_run_id: str  # e.g. "configA_seed1"
    configs: dict[str, RunAggregationEntry]
    config_judgments: NotRequired[dict[str, dict]]


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------


class EntityMetadata(TypedDict, total=False):
    """User-editable metadata for sessions and benchmarks.

    Stored as metadata.json in the entity directory.
    """

    alias: str
    starred: bool
    tags: list[str]
    deleted_at: str  # ISO 8601 timestamp, None/absent = not deleted


# ---------------------------------------------------------------------------
# Experiment lineage (cross-session experiment tree)
# ---------------------------------------------------------------------------


class LineageEntry(TypedDict, total=False):
    """One node in the experiment lineage tree.

    Key formats:
      - Single-config: ``"session_id/iter_N"``
      - Multi-config:  ``"session_id/iter_N/config_id"``

    In multi-config mode each config is a first-class node with its own
    score, lesson, diagnosis, and HP params.
    """

    parent: Required[str | None]  # parent node key or None for roots
    lesson: str  # distilled insight (revise agent for best, auto for others)
    score: float  # intent_score for quick lookup
    star: bool  # high-water mark across all iterations
    also_from: str  # secondary parent (cross-branch inspiration)
    # Rich fields (mirror LoopTimeline tooltips)
    diagnosis: str  # judge diagnosis text
    failure_tags: list[str]  # failure tag list
    final_return: float  # episodic return
    best_checkpoint: str  # best eval checkpoint step
    # Config-level fields (present only in multi-config mode)
    config_id: str  # e.g. "baseline", "config_0"
    config_label: str  # human-readable, e.g. "lr=1e-3, ent=0.02"
    hp_params: dict[str, Any]  # HP overrides for this config
    score_std: float  # score std across seeds
    return_std: float  # return std across seeds
    is_best: bool  # whether this config was selected as best in its iteration


LessonTier = Literal["HARD", "STRONG", "SOFT", "RETIRED"]


class StructuredLesson(TypedDict, total=False):
    """One lesson in the graduated lesson tier system.

    Tiers are managed by the revise agent via ``set_tier``.
    HARD is never auto-demoted; the agent must explicitly change it.
    """

    text: Required[str]
    tier: Required[LessonTier]
    learned_at: int  # iteration number when learned (0 = unknown/migrated)
    tier_reason: str  # why the tier was changed (audit trail)


class Lineage(TypedDict):
    """Full experiment lineage tree.

    Stored as ``lineage.json`` in the session directory.
    Keys are ``"session_id/iter_N"`` (single-config) or
    ``"session_id/iter_N/config_id"`` (multi-config).

    ``lessons`` may contain plain strings (legacy) or StructuredLesson dicts.
    ``load_lineage()`` auto-migrates plain strings on read.
    """

    iterations: dict[str, LineageEntry]
    lessons: list[StructuredLesson]  # accumulated global lessons (tiered)


# ---------------------------------------------------------------------------
# Revise agent output
# ---------------------------------------------------------------------------


class Phase1Result(TypedDict):
    """Phase 1 output from two-phase revision (diagnosis + plan, no code).

    Produced by ``_parse_phase1_response()`` in revise_agent.py.
    Consumed internally to drive Phase 2 code generation.
    """

    diagnosis: str
    lesson: str
    based_on: int
    planned_changes: str
    hp_changes: HPChanges
    hp_reasoning: str


class ReviseResult(TypedDict):
    """Result of the revise pipeline (LLM-parsed fields + injected context).

    Produced by revise_agent.py, consumed by loop.py and services.py.
    ``training_dynamics`` is injected by callers after parsing, not extracted
    from the LLM response — hence NotRequired.
    """

    reward_code: str
    reward_reasoning: str
    hp_changes: HPChanges
    hp_reasoning: str
    training_dynamics: NotRequired[str]
    diagnosis: str
    lesson: str  # distilled insight from this iteration (generated by revise agent)
    based_on: int  # iteration number whose reward code was used as the base (0 = initial)


# ---------------------------------------------------------------------------
# Hyperparameter changes (from revise agent)
# ---------------------------------------------------------------------------


class HPChanges(TypedDict, total=False):
    """Hyperparameter overrides proposed by the revise agent.

    Keys correspond to TrainConfig._TUNABLE_KEYS.
    All fields are optional since any subset may be changed per iteration.
    """

    learning_rate: float
    ent_coef: float
    vf_coef: float
    clip_coef: float
    max_grad_norm: float
    gae_lambda: float
    gamma: float
    num_steps: int
    update_epochs: int
    target_kl: float
    total_timesteps: int
    net_arch: list[int]
    normalize_obs: bool
    normalize_reward: bool
    reward_clip: float
    obs_clip: float
    max_episode_steps: int


# ---------------------------------------------------------------------------
# Session config (saved as session_config.json)
# ---------------------------------------------------------------------------


class SessionConfig(TypedDict, total=False):
    """Start configuration for a loop session.

    Derived from ``LoopConfig.to_json()`` with extra metadata (``prompt``).
    The ``train`` key holds all ``TrainConfig`` fields.

    Saved by process_manager.start_session(), read by services.get_session_config().
    """

    # Metadata (not part of LoopConfig)
    prompt: Required[str]

    # LoopConfig fields (auto-serialized via to_json)
    # Optional: some sessions store a flat config without a nested ``train`` dict.
    train: dict[str, Any]
    configs: list[dict[str, Any]] | None
    seeds: list[int] | None
    max_iterations: int
    pass_threshold: float
    vlm_model: str
    thinking_effort: str
    runs_dir: str
    cores_per_run: int
    max_parallel: int
    cores_pool: list[int] | None
    hp_tuning: bool
    use_code_judge: bool
    review_reward: bool
    review_judge: bool
    model: str
    judgment_select: str
    use_zoo_preset: bool
    elaborated_intent: str
    criteria_diagnosis: bool
    motion_trail_dual: bool


# ---------------------------------------------------------------------------
# Intermediate stage judgment (VLM / code judge)
# ---------------------------------------------------------------------------


class StageJudgment(TypedDict):
    """Intermediate result from a single judgment stage.

    Returned by VLM judge and code-based judge.
    Consumed by synthesis functions (_synthesize, _synthesize_dual_judges, etc.).
    """

    intent_score: float | None
    diagnosis: str
    failure_tags: list[str]
    evidence: NotRequired[list[str]]
    vlm_criteria: NotRequired[str]  # Turn 1 expectations (2-turn judging)
    criteria_scores: NotRequired[list[dict]]  # Per-criterion assessment (criteria diagnosis)


class FailureTagEntry(TypedDict, total=False):
    """A failure tag with cross-iteration tracking metadata.

    Computed from iteration history and passed to the synthesizer
    so it can track persistent failures across iterations.
    """

    tag: Required[str]
    count: Required[int]  # total iterations where this tag appeared
    first_seen: Required[int]  # iteration number when first observed
    last_seen: Required[int]  # iteration number when last observed
    status: Literal["active", "resolved"]


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------


class SessionAnalysis(TypedDict):
    """Result of an agentic LLM analysis of a session."""

    session_id: str
    analysis_en: str
    key_findings: list[str]
    recommendations: list[str]
    tool_calls_used: int
    model: str
    created_at: str


# ---------------------------------------------------------------------------
# Loop result (session-level output of run_loop)
# ---------------------------------------------------------------------------


class LoopResult(TypedDict):
    """Result of a complete loop session.

    Returned by ``loop.run_loop()``, saved as ``loop_history.json``
    via ``SessionRecord.save_history()``.
    """

    session_id: str
    prompt: str
    env_id: NotRequired[str]  # environment name (persisted at session start)
    status: StatusLiteral
    iterations: list[dict[str, Any]]  # serialized IterationData dicts
    best_iteration: int
    best_score: float
    pass_threshold: float
    error: NotRequired[str]


# ---------------------------------------------------------------------------
# Training dynamics (from training_dynamics.py analysis)
# ---------------------------------------------------------------------------

TrendLiteral = Literal["flat", "increasing", "decreasing"]


class TrainingDynamics(TypedDict):
    """Summary statistics computed from scalars.jsonl.

    Produced by training_dynamics.analyze_training_curves(),
    consumed by revise_agent and format_training_dynamics().
    All fields are always populated via _empty_dynamics().
    """

    entropy_initial: float
    entropy_final: float
    entropy_trend: TrendLiteral
    entropy_decay_rate: float
    entropy_too_fast: bool
    entropy_too_high: bool

    value_loss_initial: float
    value_loss_final: float
    value_loss_trend: TrendLiteral
    value_loss_stability: float  # CV of last 20%
    value_loss_diverging: bool

    policy_loss_initial: float
    policy_loss_final: float
    policy_loss_trend: TrendLiteral

    approx_kl_mean: float
    approx_kl_max: float
    approx_kl_spike_count: int

    clip_fraction_mean: float
    clip_fraction_trend: TrendLiteral

    explained_variance_final: float
    explained_variance_good: bool

    episodic_return_trend: TrendLiteral
    episodic_return_final: float
    episodic_return_max: float
    episodic_return_converged: bool
    episodic_return_improvement_pct: float

    sps_mean: float

    reward_term_stats: dict[str, dict]
    reward_term_avg_window: int

    num_entries: int


# ---------------------------------------------------------------------------
# API service return types (services.py → routes.py)
# ---------------------------------------------------------------------------


class IterationRunInfo(TypedDict):
    """Summary of a single sub-run within a multi-config iteration.

    Returned by ``services.get_iteration_runs()``.
    """

    config_id: str
    seed: int
    run_id: str
    status: StatusLiteral
    final_return: float | None
    intent_score: float | None
    video_urls: list[str]


class MeanStdSeries(TypedDict):
    """Mean and std arrays for one metric across global steps."""

    mean: list[float]
    std: list[float]


class AggregatedMetrics(TypedDict):
    """Cross-seed aggregated metrics for a single config.

    Returned by ``services.get_aggregated_metrics()``.
    """

    config_id: str
    seeds: list[int]
    available_metrics: list[str]
    global_steps: list[int]
    metrics: dict[str, MeanStdSeries]


class RunMetricsDetail(TypedDict):
    """Full metrics for a single sub-run.

    Returned by ``services.get_run_metrics()``.
    """

    run_id: str
    config_id: str
    seed: int
    training: list[dict[str, Any]]
    evaluation: list[dict[str, Any]]
    video_urls: list[str]
    checkpoint_scores: dict[str, float]
    checkpoint_diagnoses: dict[str, str]
    checkpoint_code_diagnoses: dict[str, str]
    checkpoint_vlm_diagnoses: dict[str, str]
    checkpoint_code_scores: dict[str, float]
    checkpoint_vlm_scores: dict[str, float]
    rollout_scores: dict[str, float]
    rollout_diagnoses: dict[str, str]
    rollout_code_diagnoses: dict[str, str]
    rollout_vlm_diagnoses: dict[str, str]
    rollout_code_scores: dict[str, float]
    rollout_vlm_scores: dict[str, float]
    rollout_synthesis_traces: NotRequired[dict[str, list[SynthesisToolCall]]]
    rollout_vlm_preview_urls: NotRequired[dict[str, str]]
    vlm_fps: NotRequired[int]
    best_checkpoint: str
    intent_score: float | None
    diagnosis: str
    config: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Resource status
# ---------------------------------------------------------------------------


class CpuAllocation(TypedDict):
    """A single CPU allocation entry."""

    run_id: str
    cores: list[int]


class ResourceStatus(TypedDict):
    """Returned by ``cpu_manager.CPUManager.status()``."""

    total_cores: int
    reserved_cores: int
    available_cores: int
    active_runs: int
    allocations: list[CpuAllocation]


# ---------------------------------------------------------------------------
# Event detail
# ---------------------------------------------------------------------------


class EventDetailRecord(TypedDict):
    """A single event from events.jsonl, returned by ``event_log.read_event_by_seq()``."""

    seq: int
    timestamp: str
    event: str
    data: dict[str, Any]
    iteration: NotRequired[int | None]
    duration_ms: NotRequired[int | None]
    has_full_content: NotRequired[bool]


# ---------------------------------------------------------------------------
# Resource allocation
# ---------------------------------------------------------------------------


class ResourceAllocation(TypedDict):
    """Optimal resource allocation returned by ``resource_auto.find_best_allocation``."""

    cores_per_run: int
    num_envs: int
    max_parallel: int
    total_runs: int
    num_batches: int
    time_score: float
    estimated_processes: int
    usable_cores: int


# ---------------------------------------------------------------------------
# Benchmark options
# ---------------------------------------------------------------------------


class BenchmarkCaseInfo(TypedDict):
    env_id: str
    category: str
    difficulty: str


class BenchmarkOptions(TypedDict):
    """Returned by ``benchmark_service.get_benchmark_options()``."""

    envs: list[str]
    categories: list[str]
    difficulties: list[str]
    cases: list[BenchmarkCaseInfo]
    csv_files: list[str]


class BenchmarkGroupStats(TypedDict):
    """Aggregated stats for a group of benchmark test cases.

    Used by scheduler and API layers. The API layer has a Pydantic mirror
    in ``api.schemas`` for response serialization.
    """

    total: int
    completed: int
    passed: int
    success_rate: float
    average_score: float
    cumulative_score: float


class TokenUsageByModel(TypedDict):
    """Token usage aggregated per model across benchmark sessions."""

    model: str
    input_tokens: int
    output_tokens: int
    call_count: int


class BenchmarkCostSummary(TypedDict):
    """Aggregated API token usage across all sessions in a benchmark job."""

    models: list[TokenUsageByModel]
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    sessions_counted: int  # sessions with events.jsonl
    sessions_total: int  # total sessions in the job


class BenchmarkTestCaseResult(TypedDict, total=False):
    """Result of a single benchmark test case.

    Used by scheduler and API layers. The API layer has a Pydantic mirror
    in ``api.schemas`` for response serialization.
    """

    index: Required[int]
    env_id: Required[str]
    instruction: Required[str]
    category: Required[str]
    difficulty: Required[str]
    session_id: Required[str]
    session_status: Required[StatusLiteral]
    best_score: Required[float]
    passed: Required[bool]
    iterations_completed: Required[int]
    video_urls: Required[list[str]]
    stage: int
    iteration_scores: list[float]
    node_id: str
    max_iterations: int
    judge_scores: dict[str, float | None]  # {"code", "vlm", "synthesizer"} from median rollout
    judge_diagnoses: dict[str, str]  # {"code", "vlm", "synthesizer"} diagnosis text
