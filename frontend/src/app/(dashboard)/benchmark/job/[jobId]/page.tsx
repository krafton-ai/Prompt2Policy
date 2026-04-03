"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import useSWR from "swr";
import Link from "next/link";
import { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import StatusBadge from "@/components/StatusBadge";
import ProgressBar from "@/components/ProgressBar";
import GroupStatsTable from "@/components/GroupStatsTable";
import ScoreProgressionChart from "@/components/ScoreProgressionChart";
import StagePipeline from "@/components/StagePipeline";
import {
  fetchJob,
  fetchRunLog,
  cancelJob,
  fetchJobBenchmark,
  syncJob,
  syncRun,
  type JobResponse,
  type RunLogResponse,
} from "@/lib/scheduler-api";
import {
  staticUrl,
  type BenchmarkRunDetail,
  type BenchmarkCostSummary,
} from "@/lib/api";
import { computeModelCost, formatCost } from "@/lib/pricing";

/** Extended detail type with scheduler-specific sync_status field. */
type SchedulerBenchmarkDetail = BenchmarkRunDetail & {
  sync_status?: Record<string, boolean>;
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

type BreakdownTab = "category" | "difficulty" | "env" | "stage";
type SortKey = "index" | "env_id" | "instruction" | "category" | "difficulty" | "best_score" | "session_status" | "stage";
type SortDir = "asc" | "desc";

const DIFFICULTY_ORDER: Record<string, number> = { easy: 0, medium: 1, hard: 2 };
const WARN_SCORE_THRESHOLD = 0.4;

function JudgeScoreBadge({ label, score, hasDetail, expanded, onToggle }: {
  label: string; score?: number | null; hasDetail?: boolean; expanded?: boolean; onToggle?: () => void;
}) {
  if (score == null) return null;
  const color = score >= 0.7 ? "text-green-700 bg-green-50 border-green-200" : score >= 0.4 ? "text-yellow-700 bg-yellow-50 border-yellow-200" : "text-red-700 bg-red-50 border-red-200";
  return (
    <button
      type="button"
      onClick={(e) => { if (hasDetail && onToggle) { e.preventDefault(); e.stopPropagation(); onToggle(); } }}
      className={`inline-flex items-center gap-0.5 px-1 py-0.5 rounded border text-[9px] font-medium ${color} ${hasDetail ? "cursor-pointer hover:opacity-80" : "cursor-default"} ${expanded ? "ring-1 ring-offset-0 ring-gray-300" : ""}`}
    >
      <span className="opacity-60">{label}</span> {score.toFixed(2)}
      {hasDetail && (
        <span className={`ml-0.5 transition-transform inline-block text-[8px] leading-none ${expanded ? "rotate-180" : ""}`}>&#9660;</span>
      )}
    </button>
  );
}

function VideoPreviewCard({ tc }: { tc: import("@/lib/api").BenchmarkTestCaseResult }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const js = tc.judge_scores ?? {};
  const jd = tc.judge_diagnoses ?? {};
  const toggle = (key: string) => setExpanded(expanded === key ? null : key);

  return (
    <div className="rounded-lg border border-gray-200 overflow-hidden hover:border-blue-400 hover:shadow-md transition-all">
      <Link href={`/e2e/${tc.session_id}`} className="block">
        <video
          src={staticUrl(tc.video_urls[0])}
          muted
          loop
          playsInline
          preload="metadata"
          onMouseEnter={(e) => (e.target as HTMLVideoElement).play().catch(() => {})}
          onMouseLeave={(e) => {
            const v = e.target as HTMLVideoElement;
            v.pause();
            v.currentTime = 0;
          }}
          className="w-full aspect-video bg-gray-900 object-contain"
        />
      </Link>
      <div className="px-2 py-1.5 bg-gray-50">
        <div className="flex items-center gap-1">
          <span className="text-xs font-mono text-gray-500">#{tc.index + 1}</span>
          <span className="text-xs text-gray-700">{tc.env_id}</span>
          <span className={`ml-auto text-[10px] font-medium ${tc.passed ? "text-green-600" : tc.best_score >= WARN_SCORE_THRESHOLD ? "text-yellow-600" : "text-gray-400"}`}>
            {tc.best_score.toFixed(2)}
          </span>
        </div>
        {Object.keys(js).length > 0 && (
          <div className="flex items-center gap-1.5 mt-1">
            <JudgeScoreBadge label="Code" score={js.code} hasDetail={!!jd.code} expanded={expanded === "code"} onToggle={() => toggle("code")} />
            <JudgeScoreBadge label="VLM" score={js.vlm} hasDetail={!!jd.vlm} expanded={expanded === "vlm"} onToggle={() => toggle("vlm")} />
            <JudgeScoreBadge label="Synth" score={js.synthesizer} hasDetail={!!jd.synthesizer} expanded={expanded === "synthesizer"} onToggle={() => toggle("synthesizer")} />
          </div>
        )}
        {expanded && jd[expanded] && (
          <div className="mt-1.5 p-1.5 bg-white border border-gray-200 rounded text-[10px] text-gray-600 leading-relaxed max-h-32 overflow-y-auto whitespace-pre-wrap">
            {jd[expanded]}
          </div>
        )}
        <p className="text-[10px] text-gray-500 mt-0.5 max-h-12 overflow-y-auto leading-tight" title={tc.instruction}>{tc.instruction}</p>
      </div>
    </div>
  );
}

function Elapsed({ since }: { since: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const secs = Math.floor((now - new Date(since).getTime()) / 1000);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return (
    <span className="font-mono text-sm text-blue-600">
      {m}:{s.toString().padStart(2, "0")}
    </span>
  );
}

function RunLogPanel({ runId, isRunning }: { runId: string; isRunning: boolean }) {
  const { data: logData } = useSWR<RunLogResponse>(
    `scheduler-run-log-${runId}`,
    () => fetchRunLog(runId),
    { refreshInterval: isRunning ? 3000 : 0 },
  );

  if (!logData?.available) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
        <span className="text-xs font-medium text-gray-500 uppercase">
          Log: {runId}
        </span>
      </div>
      <pre className="p-4 text-xs bg-gray-900 text-gray-300 overflow-auto max-h-96 font-mono whitespace-pre-wrap">
        {logData.log || "(empty)"}
      </pre>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Job config panel
// ---------------------------------------------------------------------------

function JobConfigPanel({
  backend,
  config,
}: {
  backend: string | null;
  config: Record<string, unknown>;
}) {
  const mode = config.mode as string | undefined;
  const maxParallel = config.max_parallel as number | undefined;
  const seeds = config.seeds as number[] | undefined;
  const totalTimesteps = config.total_timesteps as number | undefined;
  const maxIterations = config.max_iterations as number | undefined;
  const numEnvs = config.num_envs as number | undefined;
  const llmModel = config.model as string | undefined;
  const vlmModel = config.vlm_model as string | undefined;
  const coresPerRun = config.cores_per_run as number | undefined;
  const device = config.device as string | undefined;
  const passThreshold = config.pass_threshold as number | undefined;
  const gateThreshold = config.gate_threshold as number | undefined;
  const numStages = config.num_stages as number | undefined;
  const sideInfo = config.side_info as boolean | undefined;
  const useZoo = config.use_zoo_preset as boolean | undefined;
  const hpTuning = config.hp_tuning as boolean | undefined;
  const useCodeJudge = config.use_code_judge as boolean | undefined;
  const reviewReward = config.review_reward as boolean | undefined;
  const reviewJudge = config.review_judge as boolean | undefined;
  const csvFile = config.csv_file as string | undefined;
  const thinkingEffort = config.thinking_effort as string | undefined;
  const refinedInitialFrame = config.refined_initial_frame as boolean | undefined;
  const criteriaDiagnosis = config.criteria_diagnosis as boolean | undefined;
  const motionTrailDual = config.motion_trail_dual as boolean | undefined;
  const judgmentSelect = config.judgment_select as string | undefined;
  const numEvals = config.num_evals as number | undefined;
  const trajectoryStride = config.trajectory_stride as number | undefined;
  const startFromStage = config.start_from_stage as number | undefined;

  const numConfigs = (config.num_configs as number | undefined) ?? (hpTuning ? (config.configs as unknown[])?.length ?? 1 : 1);
  const numSeeds = seeds?.length ?? 1;
  const runsPerCase = numConfigs * numSeeds;
  const effectiveCores = coresPerRun || 4;
  const effectiveEnvs = numEnvs || effectiveCores * 4;
  const effectiveParallel = maxParallel || 0;
  const totalProcs = effectiveParallel * runsPerCase * effectiveEnvs;

  const items: [string, string][] = [
    ["Backend", backend ?? "local"],
    ["Mode", mode ?? "-"],
  ];

  if (csvFile) items.push(["CSV", csvFile]);
  if (seeds?.length) items.push(["Seeds", seeds.join(", ")]);
  if (device) items.push(["Device", device]);

  const tsLabel = totalTimesteps !== undefined
    ? (totalTimesteps >= 1_000_000 ? `${(totalTimesteps / 1_000_000).toFixed(1)}M` : totalTimesteps.toLocaleString())
    : undefined;
  if (tsLabel) items.push(["Timesteps", tsLabel]);
  if (maxIterations !== undefined) items.push(["Max Iter", String(maxIterations)]);
  if (passThreshold !== undefined) items.push(["Pass Threshold", String(passThreshold)]);
  if (mode === "staged") {
    if (numStages !== undefined) items.push(["Stages", String(numStages)]);
    if (gateThreshold !== undefined) items.push(["Gate Threshold", String(gateThreshold)]);
  }
  items.push(["Num Envs", numEnvs !== undefined ? String(numEnvs) : "auto"]);
  items.push(["Cores/Run", coresPerRun ? String(coresPerRun) : "auto"]);
  if (effectiveParallel > 0) items.push(["Max Parallel", String(effectiveParallel)]);
  items.push(["Num Configs", String(numConfigs)]);
  items.push(["LLM Model", llmModel || DEFAULT_LLM_MODEL]);
  if (vlmModel) items.push(["VLM Model", vlmModel]);
  items.push(["Thinking Effort", thinkingEffort ?? "None"]);
  items.push(["Judgment Select", judgmentSelect ?? "None"]);
  if (numEvals !== undefined) items.push(["Num Evals", String(numEvals)]);
  if (trajectoryStride !== undefined && trajectoryStride !== 1) items.push(["Trajectory Stride", String(trajectoryStride)]);
  if (mode === "staged" && startFromStage !== undefined && startFromStage > 1) items.push(["Start from Stage", String(startFromStage)]);

  const enabledFlags: string[] = [];
  const disabledFlags: string[] = [];
  if (sideInfo) enabledFlags.push("Side Info");
  if (useZoo) enabledFlags.push("Zoo Preset");
  if (hpTuning) enabledFlags.push("HP Tuning");
  if (refinedInitialFrame) enabledFlags.push("Refined IF");
  if (criteriaDiagnosis) enabledFlags.push("Criteria Diagnosis");
  if (motionTrailDual) enabledFlags.push("Trail Dual");
  if (useCodeJudge) enabledFlags.push("Code Judge");
  if (reviewReward === true) enabledFlags.push("Review Reward");
  else if (reviewReward === false) disabledFlags.push("Review Reward");
  if (reviewJudge === true) enabledFlags.push("Review Judge");
  else if (reviewJudge === false) disabledFlags.push("Review Judge");

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 mb-6">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
        Job Configuration
      </h3>

      {/* Resource allocation summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-2.5 mb-3">
        <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide mb-1">
          Resource Allocation
        </p>
        <p className="text-xs text-blue-600">
          {effectiveCores} core / {effectiveEnvs} env per run
          {runsPerCase > 1 && ` · ${runsPerCase} runs/case`}
          {effectiveParallel > 0 && ` · max ${effectiveParallel} cases parallel`}
          {totalProcs > 0 && ` · ~${totalProcs} procs`}
        </p>
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {items.map(([label, value]) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className="text-xs text-gray-400">{label}:</span>
            <span className="text-xs font-medium text-gray-700 font-mono">{value}</span>
          </div>
        ))}
      </div>
      {(enabledFlags.length > 0 || disabledFlags.length > 0) && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {enabledFlags.map((flag) => (
            <span
              key={flag}
              className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700"
            >
              {flag}
            </span>
          ))}
          {disabledFlags.map((flag) => (
            <span
              key={flag}
              className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-gray-100 text-gray-400 line-through"
            >
              {flag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cost section
// ---------------------------------------------------------------------------

function CostSection({ cost }: { cost: BenchmarkCostSummary }) {
  if (cost.total_calls === 0) return null;

  const totalCost = cost.models.reduce(
    (sum, m) => sum + computeModelCost(m.model, m.input_tokens, m.output_tokens),
    0,
  );

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">API Cost</h3>

      {/* Total cost */}
      <div className="flex items-baseline gap-3 mb-3">
        <span className="text-2xl font-bold text-gray-900">{formatCost(totalCost)}</span>
        <span className="text-xs text-gray-500">
          {cost.total_calls.toLocaleString()} calls across {cost.sessions_counted} sessions
          {cost.sessions_counted < cost.sessions_total && (
            <> ({cost.sessions_counted} of {cost.sessions_total} synced)</>
          )}
        </span>
      </div>

      {/* Per-model breakdown */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-400 border-b border-gray-100">
              <th className="pb-2 pr-4 font-medium">Model</th>
              <th className="pb-2 pr-4 font-medium text-right">Calls</th>
              <th className="pb-2 pr-4 font-medium text-right">Input Tokens</th>
              <th className="pb-2 pr-4 font-medium text-right">Output Tokens</th>
              <th className="pb-2 font-medium text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {cost.models.map((m) => {
              const mc = computeModelCost(m.model, m.input_tokens, m.output_tokens);
              return (
                <tr key={m.model} className="border-b border-gray-50">
                  <td className="py-1.5 pr-4 font-mono text-gray-700">{m.model}</td>
                  <td className="py-1.5 pr-4 text-right text-gray-600">{m.call_count.toLocaleString()}</td>
                  <td className="py-1.5 pr-4 text-right text-gray-600">{m.input_tokens.toLocaleString()}</td>
                  <td className="py-1.5 pr-4 text-right text-gray-600">{m.output_tokens.toLocaleString()}</td>
                  <td className="py-1.5 text-right font-medium text-gray-900">{formatCost(mc)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Benchmark view (used when job_type === "benchmark")
// ---------------------------------------------------------------------------

function BenchmarkJobView({
  job,
  isRunning,
  onCancel,
  cancelling,
}: {
  job: JobResponse;
  isRunning: boolean;
  onCancel: () => void;
  cancelling: boolean;
}) {
  const benchmarkId = (job.metadata?.benchmark_id as string) ?? "";

  const { data: detail, mutate: mutateDetail } = useSWR<SchedulerBenchmarkDetail>(
    `scheduler-job-benchmark-${job.job_id}`,
    () => fetchJobBenchmark(job.job_id),
    { refreshInterval: 5000 },
  );

  const [tab, setTab] = useState<BreakdownTab>("category");
  const [showCases, setShowCases] = useState(false);
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [stageFilter, setStageFilter] = useState<number | null>(null);
  const [hoveredStage, setHoveredStage] = useState<number | null>(null);
  const [showVideos, setShowVideos] = useState<boolean | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncingCase, setSyncingCase] = useState<number | null>(null);

  const syncStatus = detail?.sync_status ?? {};

  async function handleSyncAll() {
    setSyncing(true);
    try {
      await syncJob(job.job_id);
      await mutateDetail();
    } finally {
      setSyncing(false);
    }
  }

  async function handleSyncCase(tc: { index: number; session_id: string }) {
    if (!tc.session_id) return;
    setSyncingCase(tc.index);
    try {
      await syncRun(tc.session_id, { jobId: job.job_id });
      await mutateDetail();
    } finally {
      setSyncingCase(null);
    }
  }

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "best_score" ? "desc" : "asc");
    }
  }

  const casesWithVideos = detail
    ? detail.test_cases.filter((tc) => tc.video_urls.length > 0)
    : [];

  // Auto-expand Video Preview when <= 20 videos, collapse for larger sets
  const videosExpanded = showVideos ?? casesWithVideos.length <= 20;

  const isStaged = detail ? detail.mode === "staged" && detail.stages.length > 0 : false;
  const hasNodeInfo = detail?.test_cases.some((tc) => tc.node_id && tc.node_id !== "") ?? false;

  const breakdownData = detail
    ? tab === "category"
      ? detail.by_category
      : tab === "difficulty"
        ? detail.by_difficulty
        : tab === "env"
          ? detail.by_env
          : (() => {
              const groups: Record<string, { total: number; completed: number; passed: number; success_rate: number; average_score: number; cumulative_score: number }> = {};
              for (const s of detail.stages) {
                const stageCases = detail.test_cases.filter((tc) => tc.stage === s.stage);
                const completed = stageCases.filter(
                  (tc) => !["running", "pending", "queued"].includes(tc.session_status),
                );
                const passed = stageCases.filter((tc) => tc.passed);
                const cumulative = stageCases.reduce((sum, tc) => sum + tc.best_score, 0);
                groups[`Stage ${s.stage}: ${s.name}`] = {
                  total: stageCases.length,
                  completed: completed.length,
                  passed: passed.length,
                  success_rate: completed.length > 0 ? passed.length / completed.length : 0,
                  average_score: completed.length > 0 ? cumulative / completed.length : 0,
                  cumulative_score: cumulative,
                };
              }
              return groups;
            })()
    : {};

  const filteredCases = detail
    ? detail.test_cases.filter((tc) => {
        if (stageFilter !== null && tc.stage !== stageFilter) return false;
        if (!filter) return true;
        const q = filter.toLowerCase();
        const re = new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i");
        return re.test(tc.env_id) || re.test(tc.category) || re.test(tc.difficulty) || re.test(tc.instruction);
      })
    : [];

  const sortedCases = [...filteredCases].sort((a, b) => {
    let cmp = 0;
    switch (sortKey) {
      case "index": cmp = a.index - b.index; break;
      case "best_score": cmp = a.best_score - b.best_score; break;
      case "stage": cmp = a.stage - b.stage; break;
      case "difficulty":
        cmp = (DIFFICULTY_ORDER[a.difficulty] ?? 99) - (DIFFICULTY_ORDER[b.difficulty] ?? 99);
        break;
      default:
        cmp = String(a[sortKey]).localeCompare(String(b[sortKey]));
    }
    return sortDir === "asc" ? cmp : -cmp;
  });

  const tabOptions: [BreakdownTab, string][] = [
    ["category", "By Category"],
    ["difficulty", "By Difficulty"],
    ["env", "By Environment"],
  ];
  if (isStaged) {
    tabOptions.push(["stage", "By Stage"]);
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/benchmark" className="text-gray-400 hover:text-gray-600 text-sm">
          &larr; Back
        </Link>
        <h1 className="text-2xl font-bold text-gray-900">{benchmarkId || job.job_id}</h1>
        <span className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
          {job.job_type}
        </span>
        <StatusBadge status={job.status} />
        {isStaged && (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-100 text-indigo-700">
            Staged
          </span>
        )}
        {isRunning && <Elapsed since={job.created_at} />}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={handleSyncAll}
            disabled={syncing}
            className="px-4 py-1.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 disabled:opacity-50 transition-colors"
          >
            {syncing ? "Syncing..." : "Sync All"}
          </button>
          {isRunning && (
            <button
              onClick={onCancel}
              disabled={cancelling}
              className="px-4 py-1.5 rounded-full text-xs font-medium bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50 transition-colors"
            >
              {cancelling ? "Cancelling..." : "Cancel Job"}
            </button>
          )}
        </div>
      </div>

      {/* Job Config */}
      {job.config && (
        <JobConfigPanel backend={job.backend} config={job.config} />
      )}

      {/* Video Preview */}
      {casesWithVideos.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden mb-6">
          <button
            onClick={() => setShowVideos(!videosExpanded)}
            className="w-full flex items-center gap-2 px-6 py-4 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className={`transition-transform text-xs ${videosExpanded ? "rotate-90" : ""}`}>
              &#9654;
            </span>
            Video Preview ({casesWithVideos.length})
          </button>
          {videosExpanded && (
            <div className="border-t border-gray-100 p-6 max-h-[70vh] overflow-y-auto">
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {casesWithVideos.map((tc) => (
                  <VideoPreviewCard key={tc.index} tc={tc} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stage Pipeline (staged mode only) */}
      {isStaged && detail && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Stage Pipeline
          </h3>
          <StagePipeline
            stages={detail.stages}
            activeStage={stageFilter}
            hoveredStage={hoveredStage}
            onHover={setHoveredStage}
            onStageClick={(stage) => {
              setStageFilter(stageFilter === stage ? null : stage);
              setShowCases(true);
            }}
          />
          {stageFilter !== null && (
            <button
              type="button"
              onClick={() => setStageFilter(null)}
              className="mt-2 text-xs text-gray-500 hover:text-gray-700"
            >
              Clear stage filter
            </button>
          )}
        </div>
      )}

      {/* Overview Card */}
      {detail ? (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
          <div className="grid grid-cols-3 gap-6 mb-4">
            <div className="text-center">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                Cumulative Score
              </p>
              <p className="text-2xl font-bold font-mono text-gray-900">
                {detail.cumulative_score.toFixed(1)}
                <span className="text-sm text-gray-400 font-normal"> / {detail.total_cases}</span>
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                Pass Rate
              </p>
              <p className="text-2xl font-bold font-mono text-gray-900">
                {detail.passed_cases}
                <span className="text-sm text-gray-400 font-normal"> / {detail.completed_cases} passed</span>
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                Success Rate
              </p>
              <p className="text-2xl font-bold font-mono text-gray-900">
                {(detail.success_rate * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          <ProgressBar completed={detail.completed_cases} total={detail.total_cases} />
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
          <p className="text-sm text-gray-400 text-center">Loading benchmark data...</p>
        </div>
      )}

      {/* Score Progression Chart */}
      {detail && (
        <ScoreProgressionChart
          testCases={detail.test_cases}
          stages={detail.stages}
          maxIterations={detail.max_iterations ?? 5}
        />
      )}

      {/* Breakdown Tabs */}
      {detail && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
          <div className="flex gap-1 mb-4">
            {tabOptions.map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                  tab === key
                    ? "bg-gray-900 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <GroupStatsTable
            groups={breakdownData}
            label={
              tab === "env"
                ? "Environment"
                : tab === "category"
                  ? "Category"
                  : tab === "stage"
                    ? "Stage"
                    : "Difficulty"
            }
          />
        </div>
      )}

      {/* Test Cases Table */}
      {detail && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm mb-6">
          <button
            onClick={() => setShowCases(!showCases)}
            className="w-full flex items-center gap-2 px-6 py-4 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className={`transition-transform text-xs ${showCases ? "rotate-90" : ""}`}>
              &#9654;
            </span>
            Test Cases ({detail.test_cases.length})
            {stageFilter !== null && (
              <span className="text-xs text-blue-600 ml-1">
                (Stage {stageFilter} filter active)
              </span>
            )}
          </button>

          {showCases && (
            <div className="border-t border-gray-100 p-6">
              <div className="flex items-center gap-3 mb-4">
                <input
                  type="text"
                  placeholder="Filter by instruction, env, category..."
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="w-full max-w-md rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                {isStaged && (
                  <div className="flex gap-1">
                    {detail.stages.map((s) => (
                      <button
                        key={s.stage}
                        type="button"
                        onClick={() => setStageFilter(stageFilter === s.stage ? null : s.stage)}
                        className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                          stageFilter === s.stage
                            ? "bg-blue-100 text-blue-800 border border-blue-300"
                            : "bg-gray-100 text-gray-500 border border-gray-200 hover:border-gray-300"
                        }`}
                      >
                        S{s.stage}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                      {(
                        [
                          ["index", "#"],
                          ...(isStaged ? [["stage", "Stage"] as [SortKey, string]] : []),
                          ["env_id", "Env"],
                          ["instruction", "Instruction"],
                          ["category", "Category"],
                          ["difficulty", "Difficulty"],
                          ["best_score", "Score"],
                          ["session_status", "Status"],
                        ] as [SortKey, string][]
                      ).map(([key, label]) => (
                        <th
                          key={key}
                          onClick={() => toggleSort(key)}
                          className="py-2 pr-3 cursor-pointer select-none hover:text-gray-700 transition-colors"
                        >
                          {label}
                          {sortKey === key && (
                            <span className="ml-1 text-gray-400">
                              {sortDir === "asc" ? "\u25B2" : "\u25BC"}
                            </span>
                          )}
                        </th>
                      ))}
                      {hasNodeInfo && <th className="py-2 pr-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Node (PID)</th>}
                      <th className="py-2 pr-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Iter</th>
                      <th className="py-2 pr-3">Detail</th>
                      <th className="py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedCases.map((tc) => (
                      <tr key={tc.index} className="border-b border-gray-50 hover:bg-gray-50">
                        <td className="py-2 pr-3 font-mono text-gray-400">{tc.index + 1}</td>
                        {isStaged && (
                          <td className="py-2 pr-3">
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-50 text-indigo-600">
                              S{tc.stage}
                            </span>
                          </td>
                        )}
                        <td className="py-2 pr-3 whitespace-nowrap">
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-700">
                            {tc.env_id}
                          </span>
                        </td>
                        <td className="py-2 pr-3 text-gray-700 max-w-md">{tc.instruction}</td>
                        <td className="py-2 pr-3 text-gray-500">{tc.category}</td>
                        <td className="py-2 pr-3">
                          <span
                            className={`text-xs font-medium ${
                              tc.difficulty === "hard"
                                ? "text-red-600"
                                : tc.difficulty === "medium"
                                  ? "text-yellow-600"
                                  : "text-green-600"
                            }`}
                          >
                            {tc.difficulty}
                          </span>
                        </td>
                        <td className="py-2 pr-3">
                          <span
                            className={`font-mono font-medium ${
                              tc.passed
                                ? "text-green-600"
                                : tc.best_score >= WARN_SCORE_THRESHOLD
                                  ? "text-yellow-600"
                                  : "text-gray-500"
                            }`}
                          >
                            {tc.best_score.toFixed(2)}
                          </span>
                        </td>
                        <td className="py-2 pr-3">
                          <StatusBadge status={tc.session_status} />
                        </td>
                        {hasNodeInfo && (
                          <td className="py-2 pr-3">
                            <span className="text-xs text-gray-500 font-mono">
                              {tc.node_id || "local"}
                              {tc.pids?.length > 0 && (
                                <span className="ml-1 text-gray-400">
                                  ({tc.pids.join(", ")})
                                </span>
                              )}
                            </span>
                          </td>
                        )}
                        <td className="py-2 pr-3">
                          <span className="text-xs font-mono text-gray-500">
                            {tc.iterations_completed}/{tc.max_iterations || "?"}
                          </span>
                        </td>
                        <td className="py-2 pr-3">
                          {tc.video_urls.length > 0 && (
                            <a
                              href={staticUrl(tc.video_urls[0])}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-500 hover:text-blue-700 text-xs"
                            >
                              &#9654; Play
                            </a>
                          )}
                        </td>
                        <td className="py-2">
                          {syncStatus[String(tc.index)] ? (
                            <Link
                              href={`/e2e/${tc.session_id}`}
                              className="text-blue-500 hover:text-blue-700 text-xs"
                            >
                              Detail &rarr;
                            </Link>
                          ) : tc.session_id ? (
                            <button
                              onClick={() => handleSyncCase(tc)}
                              disabled={syncingCase === tc.index}
                              className="text-xs text-gray-400 hover:text-blue-600 disabled:opacity-50"
                            >
                              {syncingCase === tc.index ? "Syncing..." : "Sync & View"}
                            </button>
                          ) : (
                            <span className="text-xs text-gray-300">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cost summary */}
      {detail?.cost_summary && <CostSection cost={detail.cost_summary} />}

      {/* Logs (collapsible) */}
      {job.run_ids.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="w-full flex items-center gap-2 px-6 py-4 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className={`transition-transform text-xs ${showLogs ? "rotate-90" : ""}`}>
              &#9654;
            </span>
            Logs ({job.run_ids.length})
          </button>
          {showLogs && (
            <div className="border-t border-gray-100 space-y-0">
              {job.run_ids.map((runId) => (
                <RunLogPanel key={runId} runId={runId} isRunning={isRunning} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error display */}
      {job.error && (
        <div className="p-3 rounded-lg bg-red-50 border border-red-200">
          <p className="text-sm font-medium text-red-800">Error</p>
          <p className="text-sm text-red-600 mt-1">{job.error}</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic job view (session / run-group / etc.)
// ---------------------------------------------------------------------------

function GenericJobView({
  job,
  isRunning,
  onCancel,
  cancelling,
}: {
  job: JobResponse;
  isRunning: boolean;
  onCancel: () => void;
  cancelling: boolean;
}) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/benchmark" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Back
        </Link>
      </div>

      {/* Job overview card */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-gray-900 font-mono">{job.job_id}</h1>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
              {job.job_type}
            </span>
            <StatusBadge status={job.status} />
            {isRunning && <Elapsed since={job.created_at} />}
          </div>
          {isRunning && (
            <button
              onClick={onCancel}
              disabled={cancelling}
              className="px-4 py-1.5 text-sm rounded-md bg-red-50 hover:bg-red-100 text-red-700 disabled:opacity-50"
            >
              {cancelling ? "Cancelling..." : "Cancel Job"}
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Created</span>
            <p className="text-gray-700">{new Date(job.created_at).toLocaleString()}</p>
          </div>
          {job.completed_at && (
            <div>
              <span className="text-gray-400">Completed</span>
              <p className="text-gray-700">{new Date(job.completed_at).toLocaleString()}</p>
            </div>
          )}
          <div>
            <span className="text-gray-400">Runs</span>
            <p className="text-gray-700">{job.run_ids.length}</p>
          </div>
          {job.metadata && Object.keys(job.metadata).length > 0 && (
            <div>
              <span className="text-gray-400">Metadata</span>
              <pre className="text-xs text-gray-600 mt-1 bg-gray-50 rounded p-2 overflow-auto">
                {JSON.stringify(job.metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {job.error && (
          <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-200">
            <p className="text-sm font-medium text-red-800">Error</p>
            <p className="text-sm text-red-600 mt-1">{job.error}</p>
          </div>
        )}
      </div>


      {/* Logs */}
      {job.run_ids.map((runId) => (
        <RunLogPanel key={runId} runId={runId} isRunning={isRunning} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function JobDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const [cancelling, setCancelling] = useState(false);

  const { data: job, mutate } = useSWR<JobResponse>(
    `scheduler-job-${jobId}`,
    () => fetchJob(jobId),
    { refreshInterval: 3000 },
  );

  if (!job) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-400">Loading job...</p>
      </div>
    );
  }

  const isRunning = job.status === "running";

  const handleCancel = async () => {
    if (!confirm("Cancel this job?")) return;
    setCancelling(true);
    try {
      await cancelJob(jobId);
      await mutate();
    } finally {
      setCancelling(false);
    }
  };

  const isBenchmarkJob = job.job_type === "benchmark" && !!job.metadata?.benchmark_id;

  if (isBenchmarkJob) {
    return (
      <BenchmarkJobView
        job={job}
        isRunning={isRunning}
        onCancel={handleCancel}
        cancelling={cancelling}
      />
    );
  }

  return (
    <GenericJobView
      job={job}
      isRunning={isRunning}
      onCancel={handleCancel}
      cancelling={cancelling}
    />
  );
}
