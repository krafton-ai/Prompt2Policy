const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface IterationSummaryItem {
  iteration_id: string;
  session_id: string;
  env_id: string;
  status: "pending" | "running" | "completed" | "cancelled";
  created_at: string;
  total_timesteps: number;
  final_episodic_return: number | null;
  reward_latex: string;
  reward_description: string;
  video_urls: string[];
  progress: number | null;
}

export interface RewardTerm {
  name: string;
  description?: string;
  latex?: string;
  weight?: number;
}

export interface IterationDetail extends IterationSummaryItem {
  config: Record<string, unknown>;
  reward_spec: {
    latex: string;
    terms: RewardTerm[] | Record<string, string>;
    description: string;
  };
  reward_source: string;
  summary: {
    total_timesteps: number;
    training_time_s: number;
    final_episodic_return: number;
    total_episodes: number;
  } | null;
  eval_results: EvalResult[];
  judgment: {
    intent_score?: number;
    passed?: boolean;
    diagnosis?: string;
    failure_tags?: string[];
    best_checkpoint?: string;
    checkpoint_judgments?: Record<string, {
      intent_score?: number;
      diagnosis?: string;
    }>;
  } | null;
  training: TrainingEntry[];
}

export interface EvalResult {
  global_step: number;
  type: "eval";
  total_reward: number;
  episode_length: number;
  reward_terms: Record<string, number>;
  num_eval_rounds?: number;
  mean_return?: number;
  std_return?: number;
  min_return?: number;
  max_return?: number;
  median_return?: number;
  p10_return?: number;
  p90_return?: number;
  all_returns?: number[];
}

export interface TrainingEntry {
  global_step: number;
  iteration: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  approx_kl: number;
  clip_fraction: number;
  explained_variance: number;
  learning_rate: number;
  sps: number;
  episodic_return?: number;
  episodic_return_std?: number;
  episodic_return_min?: number;
  episodic_return_max?: number;
  episode_length?: number;
  episodes_per_rollout?: number;
  policy_std?: number;
  grad_norm?: number;
  rollout_time?: number;
  train_time?: number;
  elapsed_time?: number;
  // Dynamic reward_term_* keys (depend on reward function)
  [key: string]: number | string | undefined;
}

export interface Metrics {
  training: TrainingEntry[];
  evaluation: EvalResult[];
}

export interface LoopIterationSummary {
  iteration: number;
  iteration_dir: string;
  intent_score: number | null;
  best_checkpoint: string;
  checkpoint_scores: Record<string, number>;
  checkpoint_diagnoses: Record<string, string>;
  checkpoint_code_diagnoses: Record<string, string>;
  checkpoint_vlm_diagnoses: Record<string, string>;
  checkpoint_code_scores: Record<string, number>;
  checkpoint_vlm_scores: Record<string, number>;
  rollout_scores: Record<string, number>;
  rollout_diagnoses: Record<string, string>;
  rollout_code_diagnoses: Record<string, string>;
  rollout_vlm_diagnoses: Record<string, string>;
  rollout_code_scores: Record<string, number>;
  rollout_vlm_scores: Record<string, number>;
  diagnosis: string;
  failure_tags: string[];
  reward_code: string;
  reward_diff_summary: string;
  final_return: number | null;
  video_urls: string[];
  // Multi-config iteration fields
  is_multi_config?: boolean;
  aggregation?: Record<
    string,
    {
      mean_best_score: number;
      std_best_score: number;
      mean_final_return: number;
      std_final_return: number;
      per_seed: { seed: number; best_score: number; final_return: number }[];
    }
  >;
  best_config_id?: string;
  best_run_id?: string;
  // Video source info
  video_source_run_id?: string;
  video_source_return?: number | null;
  // Timing
  elapsed_time_s?: number | null;
  // Revise agent output
  reward_reasoning: string;
  hp_reasoning: string;
  hp_changes: Record<string, unknown>;
  training_dynamics: string;
  revise_diagnosis: string;
  based_on?: number;
  // Per-judge raw outputs
  code_diagnosis?: string;
  code_score?: number | null;
  vlm_diagnosis?: string;
  vlm_score?: number | null;
  vlm_criteria?: string;
  criteria_scores?: CriteriaScore[];
  scoring_method?: string;
  rollout_synthesis_traces?: Record<string, SynthesisToolTrace[]>;
  rollout_criteria_scores?: Record<string, CriteriaScore[]>;
  // VLM preview video URLs (center-of-interval sampled at VLM fps)
  rollout_vlm_preview_urls?: Record<string, string>;
  // Motion trail preview URLs (when --motion-trail-dual was enabled)
  rollout_motion_preview_urls?: Record<string, string>;
  vlm_fps?: number;
  // Human label status per video (from human_label.json, keyed by video filename)
  human_label?: Record<string, {
    status: "sent" | "scored" | "error";
    annotator: string;
    sent_at: string;
    intent_score?: number;
    scored_at?: string;
    error?: string;
  }> | null;
}

/** Per-criterion VLM assessment (Turn 2). */
export interface CriteriaScore {
  criterion: string;
  assessment: string;
  status?: string;
}

/** One tool call made during agentic synthesis. */
export interface SynthesisToolTrace {
  tool_name: string;
  input: {
    question?: string;
    description?: string;
    python_code?: string;
    start_time?: number;
    end_time?: number;
    fps?: number;
    [key: string]: unknown;
  };
  output: string;
}

export interface SessionDetail {
  session_id: string;
  prompt: string;
  status: "passed" | "max_iterations" | "plateau" | "error" | "running" | "cancelled";
  best_iteration: number;
  best_score: number;
  iterations: LoopIterationSummary[];
  error: string | null;
  env_id: string;
  created_at: string;
  total_timesteps: number;
  pass_threshold: number;
  is_stale: boolean;
  // User metadata
  alias: string;
  starred: boolean;
  tags: string[];
}

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function fetchIteration(
  iterationId: string,
  sessionId?: string,
): Promise<IterationDetail> {
  if (sessionId) {
    return fetchJSON(`/api/sessions/${sessionId}/iterations/${iterationId}`);
  }
  return fetchJSON(`/api/iterations/${iterationId}`);
}

export function fetchMetrics(
  iterationId: string,
  sessionId?: string,
): Promise<Metrics> {
  if (sessionId) {
    return fetchJSON(
      `/api/sessions/${sessionId}/iterations/${iterationId}/metrics`,
    );
  }
  return fetchJSON(`/api/iterations/${iterationId}/metrics`);
}

export function fetchEnvs(): Promise<EnvInfo[]> {
  return fetchJSON("/api/envs");
}

export interface ProviderStatus {
  anthropic: boolean;
  gemini: boolean;
  openai: boolean;
}

export function fetchProviders(): Promise<ProviderStatus> {
  return fetchJSON("/api/providers");
}

export async function startSession(data: {
  prompt: string;
  model?: string;
  total_timesteps: number;
  seed: number;
  max_iterations: number;
  pass_threshold: number;
  env_id?: string;
  num_envs?: number;
  vlm_model?: string;
  use_code_judge?: boolean;
  review_reward?: boolean;
  review_judge?: boolean;
  side_info?: boolean;
  use_zoo_preset?: boolean;
  hp_tuning?: boolean;
  configs?: { config_id: string; label?: string; params?: Record<string, unknown> }[];
  num_configs?: number;
  seeds?: number[];
  cores_per_run?: number;
  max_parallel?: number;
  num_evals?: number;
  trajectory_stride?: number;
  thinking_effort?: string;
  judgment_select?: string;
  elaborated_intent?: string;
  refined_initial_frame?: boolean;
  criteria_diagnosis?: boolean;
  motion_trail_dual?: boolean;
}): Promise<{ session_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Lineage
// ---------------------------------------------------------------------------

export interface LineageEntry {
  parent: string | null;
  lesson?: string;
  score?: number;
  star?: boolean;
  also_from?: string;
  diagnosis?: string;
  failure_tags?: string[];
  final_return?: number;
  best_checkpoint?: string;
  // Config-level fields (multi-config mode)
  config_id?: string;
  config_label?: string;
  hp_params?: Record<string, unknown>;
  score_std?: number;
  return_std?: number;
  is_best?: boolean;
}

export type LessonTier = "HARD" | "STRONG" | "SOFT" | "RETIRED";

export interface StructuredLesson {
  text: string;
  tier: LessonTier;
  learned_at?: number;
}

export interface Lineage {
  iterations: Record<string, LineageEntry>;
  lessons: (string | StructuredLesson)[];
}

export function fetchSessionLineage(sessionId: string): Promise<Lineage> {
  return fetchJSON(`/api/sessions/${sessionId}/lineage`);
}

// ---------------------------------------------------------------------------

export function fetchSessions(): Promise<SessionDetail[]> {
  return fetchJSON("/api/sessions");
}

export function fetchSession(sessionId: string): Promise<SessionDetail> {
  return fetchJSON(`/api/sessions/${sessionId}`);
}

export function fetchSessionConfig(sessionId: string): Promise<Record<string, unknown>> {
  return fetchJSON(`/api/sessions/${sessionId}/config`);
}

export function fetchSessionIterations(
  sessionId: string,
): Promise<LoopIterationSummary[]> {
  return fetchJSON(`/api/sessions/${sessionId}/loop-iterations`);
}

export function stopSession(
  sessionId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return postJSON(`/api/sessions/${sessionId}/stop`, {});
}

// Metadata operations
export interface UpdateMetadataRequest {
  alias?: string;
  starred?: boolean;
  tags?: string[];
}

export interface UpdateMetadataResponse {
  alias: string;
  starred: boolean;
  tags: string[];
}

export function updateSession(
  sessionId: string,
  data: UpdateMetadataRequest,
): Promise<UpdateMetadataResponse> {
  return patchJSON(`/api/sessions/${sessionId}`, data);
}

export function deleteSession(
  sessionId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return deleteJSON(`/api/sessions/${sessionId}`);
}

export function restoreSession(
  sessionId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return postJSON(`/api/sessions/${sessionId}/restore`, {});
}

export function updateBenchmark(
  benchmarkId: string,
  data: UpdateMetadataRequest,
): Promise<UpdateMetadataResponse> {
  return patchJSON(`/api/benchmarks/${benchmarkId}`, data);
}

export function deleteBenchmark(
  benchmarkId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return deleteJSON(`/api/benchmarks/${benchmarkId}`);
}

export function restoreBenchmark(
  benchmarkId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return postJSON(`/api/benchmarks/${benchmarkId}/restore`, {});
}



export interface TrashItem {
  entity_id: string;
  entity_type: "session" | "benchmark" | "job";
  alias: string;
  deleted_at: string;
  created_at: string;
  prompt: string;
  status: string;
}

export function fetchTrash(): Promise<TrashItem[]> {
  return fetchJSON("/api/trash");
}

export function hardDelete(
  entityId: string,
): Promise<{ stopped: boolean; detail: string }> {
  return deleteJSON(`/api/trash/${entityId}`);
}

export function hardDeleteAll(): Promise<{ stopped: boolean; detail: string }> {
  return deleteJSON("/api/trash");
}

export function staticUrl(path: string): string {
  return `${API_BASE}${path}`;
}

// ---------------------------------------------------------------------------
// EnvInfo
// ---------------------------------------------------------------------------

export interface EnvInfo {
  env_id: string;
  name: string;
  obs_dim: number;
  action_dim: number;
  info_keys: Record<string, string>;
  description: string;
  engine: string;
  zoo_num_envs: number;
}

// ---------------------------------------------------------------------------
// Intent elicitation
// ---------------------------------------------------------------------------

export interface IntentCriterion {
  title: string;
  description: string;
  category: string;
  default_on: boolean;
}

export interface ElaborateIntentResponse {
  criteria: IntentCriterion[];
}

export function elaborateIntent(
  prompt: string,
  envId: string,
  model?: string,
): Promise<ElaborateIntentResponse> {
  return postJSON("/api/elaborate-intent", { prompt, env_id: envId, ...(model && { model }) });
}

async function postJSON<T>(path: string, data: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function patchJSON<T>(path: string, data: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function deleteJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export interface ResourceStatus {
  total_cores: number;
  reserved_cores: number;
  available_cores: number;
  active_runs: number;
  allocations: { run_id: string; cores: number[] }[];
}

export function fetchResourceStatus(): Promise<ResourceStatus> {
  return fetchJSON("/api/resources/status");
}

export interface RunProcessInfo {
  run_id: string;
  pid: number;
  cores: number[];
}

export interface CoreProcessInfo {
  session_id: string;
  pid: number;
  cores: number[];
  runs: RunProcessInfo[];
}

export interface MemoryInfo {
  total_mb: number;
  used_mb: number;
  available_mb: number;
  percent: number;
}

export interface CpuUsage {
  per_core: number[];
  avg: number;
  processes: CoreProcessInfo[];
  memory?: MemoryInfo;
}

export function fetchCpuUsage(): Promise<CpuUsage> {
  return fetchJSON("/api/resources/cpu-usage");
}

export interface GpuProcessInfo {
  pid: number;
  gpu_memory_mb: number;
  process_name: string;
  session_id: string;
  run_id: string;
}

export interface GpuInfo {
  index: number;
  name: string;
  temperature: number;
  utilization: number;
  memory_utilization: number;
  memory_used_mb: number;
  memory_total_mb: number;
  power_draw_w: number;
  power_limit_w: number;
  processes: GpuProcessInfo[];
}

export interface GpuUsage {
  gpus: GpuInfo[];
}

export function fetchGpuUsage(): Promise<GpuUsage> {
  return fetchJSON("/api/resources/gpu-usage");
}

export interface ResourceAutoResult {
  cores_per_run: number;
  num_envs: number;
  max_parallel: number;
  total_runs: number;
  num_batches: number;
  time_score: number;
  estimated_processes: number;
  usable_cores: number;
}

export function fetchResourceAuto(
  numConfigs: number,
  numSeeds: number,
  envId?: string,
): Promise<ResourceAutoResult> {
  const params = new URLSearchParams({
    num_configs: String(numConfigs),
    num_seeds: String(numSeeds),
  });
  if (envId) params.set("env_id", envId);
  return fetchJSON(`/api/resources/auto?${params}`);
}

// ---------------------------------------------------------------------------
// Iteration sub-runs (multi-config × seeds)
// ---------------------------------------------------------------------------

export interface IterationRunEntry {
  config_id: string;
  seed: number;
  run_id: string;
  status: string;
  final_return: number | null;
  intent_score: number | null;
  video_urls: string[];
}

export interface MeanStdArray {
  mean: number[];
  std: number[];
}

export interface AggregatedMetrics {
  config_id: string;
  seeds: number[];
  available_metrics: string[];
  global_steps: number[];
  metrics: Record<string, MeanStdArray>;
}

export function fetchIterationRuns(
  sessionId: string,
  iterNum: number,
): Promise<IterationRunEntry[]> {
  return fetchJSON(`/api/sessions/${sessionId}/iterations/${iterNum}/runs`);
}

export function fetchAggregatedMetrics(
  sessionId: string,
  iterNum: number,
  configId: string,
): Promise<AggregatedMetrics> {
  return fetchJSON(
    `/api/sessions/${sessionId}/iterations/${iterNum}/configs/${configId}/aggregated-metrics`,
  );
}

export interface RunMetrics {
  run_id: string;
  config_id: string;
  seed: number;
  training: TrainingEntry[];
  evaluation: EvalResult[];
  video_urls: string[];
  // Per-checkpoint judgment data
  checkpoint_scores: Record<string, number>;
  checkpoint_diagnoses: Record<string, string>;
  checkpoint_code_diagnoses: Record<string, string>;
  checkpoint_vlm_diagnoses: Record<string, string>;
  checkpoint_code_scores: Record<string, number>;
  checkpoint_vlm_scores: Record<string, number>;
  rollout_scores: Record<string, number>;
  rollout_diagnoses: Record<string, string>;
  rollout_code_diagnoses: Record<string, string>;
  rollout_vlm_diagnoses: Record<string, string>;
  rollout_code_scores: Record<string, number>;
  rollout_vlm_scores: Record<string, number>;
  rollout_synthesis_traces?: Record<string, SynthesisToolTrace[]>;
  rollout_vlm_preview_urls?: Record<string, string>;
  rollout_motion_preview_urls?: Record<string, string>;
  vlm_fps?: number;
  best_checkpoint: string;
  intent_score: number | null;
  diagnosis: string;
  config: Record<string, unknown> | null;
}

export function fetchRunMetrics(
  sessionId: string,
  iterNum: number,
  runId: string,
): Promise<RunMetrics> {
  return fetchJSON(
    `/api/sessions/${sessionId}/iterations/${iterNum}/runs/${runId}/metrics`,
  );
}

// ---------------------------------------------------------------------------
// Session Analysis
// ---------------------------------------------------------------------------

export interface SessionAnalysis {
  session_id: string;
  analysis_en: string;
  key_findings: string[];
  recommendations: string[];
  tool_calls_used: number;
  model: string;
  created_at: string;
}

export function fetchSessionAnalysis(
  sessionId: string,
): Promise<SessionAnalysis> {
  return fetchJSON(`/api/sessions/${sessionId}/analysis`);
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

export interface BenchmarkGroupStats {
  total: number;
  completed: number;
  passed: number;
  success_rate: number;
  average_score: number;
  cumulative_score: number;
}

export interface StageGateResult {
  passed: boolean;
  avg_score: number;
  success_rate: number;
  completed: number;
  total: number;
  threshold: number;
}

export interface StageDetail {
  stage: number;
  name: string;
  status: string; // "pending" | "running" | "completed" | "gate_passed" | "gate_failed" | "skipped"
  gate_threshold: number;
  max_parallel: number;
  case_count: number;
  case_indices: number[];
  gate_result: StageGateResult | null;
}

export interface StageDefinition {
  stage: number;
  name: string;
  gate_threshold: number;
  max_parallel: number;
}

export interface BenchmarkTestCaseResult {
  index: number;
  env_id: string;
  instruction: string;
  category: string;
  difficulty: string;
  session_id: string;
  session_status: string;
  best_score: number;
  passed: boolean;
  iterations_completed: number;
  video_urls: string[];
  stage: number;
  iteration_scores: number[];
  node_id: string;
  pids: number[];
  max_iterations: number;
  judge_scores?: Record<string, number | null>;
  judge_diagnoses?: Record<string, string>;
}

export interface BenchmarkRunSummary {
  benchmark_id: string;
  created_at: string;
  completed_at: string | null;
  status: string;
  total_cases: number;
  completed_cases: number;
  passed_cases: number;
  success_rate: number;
  average_score: number;
  cumulative_score: number;
  mode: string;
  current_stage: number;
  total_stages: number;
  // User metadata
  alias: string;
  starred: boolean;
  tags: string[];
}

export interface TokenUsageByModel {
  model: string;
  input_tokens: number;
  output_tokens: number;
  call_count: number;
}

export interface BenchmarkCostSummary {
  models: TokenUsageByModel[];
  total_input_tokens: number;
  total_output_tokens: number;
  total_calls: number;
  sessions_counted: number;
  sessions_total: number;
}

export interface BenchmarkRunDetail extends BenchmarkRunSummary {
  pass_threshold: number;
  by_category: Record<string, BenchmarkGroupStats>;
  by_difficulty: Record<string, BenchmarkGroupStats>;
  by_env: Record<string, BenchmarkGroupStats>;
  test_cases: BenchmarkTestCaseResult[];
  stages: StageDetail[];
  start_from_stage: number;
  max_iterations: number;
  cost_summary?: BenchmarkCostSummary;
}

export interface StartBenchmarkRequest {
  model?: string;
  pass_threshold?: number;
  total_timesteps?: number;
  max_iterations?: number;
  seed?: number;
  num_configs?: number;
  seeds?: number[];
  num_envs?: number;
  vlm_model?: string;
  max_parallel?: number;
  cores_per_run?: number;
  filter_envs?: string[];
  filter_categories?: string[];
  filter_difficulties?: string[];
  mode?: string;
  stages?: StageDefinition[];
  start_from_stage?: number;
  num_stages?: number;
  gate_threshold?: number;
  side_info?: boolean;
  use_zoo_preset?: boolean;
  hp_tuning?: boolean;
  use_code_judge?: boolean;
  device?: string;
  csv_file?: string;
  backend?: string;
  thinking_effort?: string;
  criteria_diagnosis?: boolean;
  motion_trail_dual?: boolean;
}

export interface BenchmarkCaseInfo {
  env_id: string;
  category: string;
  difficulty: string;
}

export interface BenchmarkOptions {
  envs: string[];
  categories: string[];
  difficulties: string[];
  cases: BenchmarkCaseInfo[];
  csv_files: string[];
}

export interface StartBenchmarkResponse {
  benchmark_id: string;
  status: string;
  total_cases: number;
  mode: string;
  total_stages: number;
}

export interface StopBenchmarkResponse {
  stopped: boolean;
  stopped_sessions: number;
  detail: string;
}

export function fetchBenchmarkOptions(csvFile?: string): Promise<BenchmarkOptions> {
  const params = csvFile ? `?csv_file=${encodeURIComponent(csvFile)}` : "";
  return fetchJSON(`/api/benchmarks/options${params}`);
}

export function fetchBenchmarkConfig(id: string): Promise<Record<string, unknown>> {
  return fetchJSON(`/api/benchmarks/${id}/config`);
}

export function fetchBenchmarks(): Promise<BenchmarkRunSummary[]> {
  return fetchJSON("/api/benchmarks");
}

export function fetchBenchmark(id: string): Promise<BenchmarkRunDetail> {
  return fetchJSON(`/api/benchmarks/${id}`);
}

export function startBenchmark(
  data: StartBenchmarkRequest,
): Promise<StartBenchmarkResponse> {
  return postJSON("/api/benchmarks", data);
}

export function stopBenchmark(id: string): Promise<StopBenchmarkResponse> {
  return postJSON(`/api/benchmarks/${id}/stop`, {});
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------

export interface EventSummary {
  seq: number;
  timestamp: string;
  event: string;
  iteration: number | null;
  data: Record<string, unknown>;
  duration_ms: number | null;
  has_full_content: boolean;
}

export interface EventDetail {
  seq: number;
  timestamp: string;
  event: string;
  iteration: number | null;
  data: Record<string, unknown>;
  duration_ms: number | null;
}

export function fetchEvents(sessionId: string): Promise<EventSummary[]> {
  return fetchJSON(`/api/sessions/${sessionId}/events`);
}

export function fetchEventDetail(
  sessionId: string,
  seq: number,
): Promise<EventDetail> {
  return fetchJSON(`/api/sessions/${sessionId}/events/${seq}`);
}

// ---------------------------------------------------------------------------
// Node resource monitoring
// ---------------------------------------------------------------------------

export interface NodeGpuInfo {
  index: number;
  name: string;
  utilization: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature: number;
  power_draw_w: number;
  power_limit_w: number;
}

export interface NodeResourceSnapshot {
  node_id: string;
  online: boolean;
  timestamp: string;
  cpu_count: number;
  cpu_percent_avg: number;
  cpu_per_core: number[];
  load_avg: number[];
  mem_total_mb: number;
  mem_used_mb: number;
  mem_available_mb: number;
  gpus: NodeGpuInfo[];
  error: string | null;
}

export interface NodeResources {
  nodes: NodeResourceSnapshot[];
  poll_interval_s: number;
}

export function fetchNodeResources(): Promise<NodeResources> {
  return fetchJSON("/api/resources/nodes");
}

export function streamSessionAnalysis(
  sessionId: string,
  callbacks: {
    onStatus?: (message: string) => void;
    onAnalysis?: (analysis: SessionAnalysis) => void;
    onError?: (error: string) => void;
  },
): () => void {
  const controller = new AbortController();

  fetch(`${API_BASE}/api/sessions/${sessionId}/analyze`, {
    method: "POST",
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && eventType) {
            const data = JSON.parse(line.slice(6));
            if (eventType === "status") callbacks.onStatus?.(data.message);
            else if (eventType === "analysis")
              callbacks.onAnalysis?.(data as SessionAnalysis);
            else if (eventType === "error") callbacks.onError?.(data.error);
            eventType = "";
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError?.(err.message);
      }
    });

  return () => controller.abort();
}
