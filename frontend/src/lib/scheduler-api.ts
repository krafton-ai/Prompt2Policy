const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NodeResponse {
  node_id: string;
  host: string;
  user: string;
  port: number;
  base_dir: string;
  max_cores: number;
  num_gpus: number;
  gpu_memory_mb: number;
  enabled: boolean;
  used_cores: number;
  online: boolean;
  active_runs: number;
}

export interface NodeCreateRequest {
  node_id: string;
  host: string;
  user: string;
  port?: number;
  base_dir?: string;
  max_cores?: number;
  enabled?: boolean;
}

export interface NodeCheckResponse {
  node_id: string;
  online: boolean;
  gpu: string | null;
  mps_active: boolean;
  uv_available: boolean;
  error: string | null;
}

export type RunState =
  | "pending"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

export interface RunStatusResponse {
  run_id: string;
  state: RunState;
  node_id: string;
  pid: number | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface JobResponse {
  job_id: string;
  job_type: string;
  status: RunState;
  run_ids: string[];
  created_at: string;
  completed_at: string | null;
  error: string | null;
  metadata: Record<string, unknown> | null;
  backend: string | null;
  config: Record<string, unknown> | null;
}

// ---------------------------------------------------------------------------
// Job allocation helpers
// ---------------------------------------------------------------------------

export interface NodeAllocation {
  total: number;
  [state: string]: number;
}

export interface SessionSummary {
  session_group: string;
  case_index: number | null;
  node_id: string;
  total_runs: number;
  state_counts: Record<string, number>;
  run_ids: string[];
  env_id: string;
  instruction: string;
}

export function getJobAllocation(job: JobResponse) {
  const meta = job.metadata ?? {};
  return {
    stateCounts: (meta.state_counts ?? {}) as Record<string, number>,
    nodeAllocation: (meta.node_allocation ?? {}) as Record<
      string,
      NodeAllocation
    >,
    sessionAffinity: (meta.session_affinity ?? false) as boolean,
    affinityNode: (meta.affinity_node ?? null) as string | null,
    sessions: (meta.sessions ?? []) as SessionSummary[],
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON<T>(path: string, data?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data ? JSON.stringify(data) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function putJSON<T>(path: string, data: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function deleteJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Node API
// ---------------------------------------------------------------------------

export function fetchNodes(): Promise<NodeResponse[]> {
  return fetchJSON("/api/scheduler/nodes");
}

export function addNode(req: NodeCreateRequest): Promise<NodeResponse> {
  return postJSON("/api/scheduler/nodes", req);
}

export function updateNode(
  nodeId: string,
  updates: Partial<NodeCreateRequest>,
): Promise<NodeResponse> {
  return putJSON(`/api/scheduler/nodes/${nodeId}`, updates);
}

export function removeNode(nodeId: string): Promise<{ detail: string }> {
  return deleteJSON(`/api/scheduler/nodes/${nodeId}`);
}

export function checkNode(nodeId: string): Promise<NodeCheckResponse> {
  return postJSON(`/api/scheduler/nodes/${nodeId}/check`);
}

export function setupNode(
  nodeId: string,
): Promise<{ ok: boolean; steps?: Record<string, string>; error?: string }> {
  return postJSON(`/api/scheduler/nodes/${nodeId}/setup`);
}

// ---------------------------------------------------------------------------
// Job API
// ---------------------------------------------------------------------------

export function submitSessionJob(
  data: Record<string, unknown>,
): Promise<JobResponse> {
  return postJSON("/api/scheduler/jobs/session", data);
}

export function submitBenchmarkJob(
  data: Record<string, unknown>,
): Promise<JobResponse> {
  return postJSON("/api/scheduler/jobs/benchmark", data);
}

export function fetchJobs(): Promise<{ jobs: JobResponse[] }> {
  return fetchJSON("/api/scheduler/jobs");
}

export function fetchJob(jobId: string): Promise<JobResponse> {
  return fetchJSON(`/api/scheduler/jobs/${jobId}`);
}

export function cancelJob(jobId: string): Promise<{ detail: string }> {
  return postJSON(`/api/scheduler/jobs/${jobId}/cancel`);
}

export function deleteJob(jobId: string): Promise<{ detail: string }> {
  return deleteJSON(`/api/scheduler/jobs/${jobId}`);
}

export function bulkTrashJobs(
  jobIds: string[],
): Promise<{ trashed: number; failed: string[]; detail: string }> {
  return postJSON("/api/scheduler/jobs/bulk-trash", { job_ids: jobIds });
}

export function restoreJob(jobId: string): Promise<{ detail: string }> {
  return postJSON(`/api/scheduler/jobs/${jobId}/restore`);
}

// ---------------------------------------------------------------------------
// Run API
// ---------------------------------------------------------------------------

export function fetchRuns(): Promise<RunStatusResponse[]> {
  return fetchJSON("/api/scheduler/runs");
}

export function fetchRun(runId: string): Promise<RunStatusResponse> {
  return fetchJSON(`/api/scheduler/runs/${runId}`);
}

export interface RunLogResponse {
  run_id: string;
  log: string;
  available: boolean;
}

export function fetchRunLog(runId: string): Promise<RunLogResponse> {
  return fetchJSON(`/api/scheduler/runs/${runId}/log`);
}

// ---------------------------------------------------------------------------
// Benchmark detail (aggregated from job manifest)
// ---------------------------------------------------------------------------

export function fetchJobBenchmark(jobId: string): Promise<import("@/lib/api").BenchmarkRunDetail> {
  return fetchJSON(`/api/scheduler/jobs/${jobId}/benchmark`);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Disk usage
// ---------------------------------------------------------------------------

export interface JobDiskUsage {
  total_bytes: number;
  trajectory_bytes: number;
  video_bytes: number;
  checkpoint_bytes: number;
}

export function fetchJobDiskUsage(jobId: string): Promise<JobDiskUsage> {
  return fetchJSON(`/api/scheduler/jobs/${jobId}/disk-usage`);
}

export function syncJob(jobId: string): Promise<{ synced: number; failed: number; skipped: number }> {
  return postJSON(`/api/scheduler/jobs/${jobId}/sync`);
}

export function syncRun(
  runId: string,
  opts?: { jobId?: string; mode?: "lite" | "full" },
): Promise<{ synced: boolean; mode: string; error: string | null }> {
  const sp = new URLSearchParams();
  if (opts?.jobId) sp.set("job_id", opts.jobId);
  if (opts?.mode) sp.set("mode", opts.mode);
  const qs = sp.toString();
  return postJSON(`/api/scheduler/runs/${runId}/sync${qs ? `?${qs}` : ""}`);
}
