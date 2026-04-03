"use client";

import { use, useState, useCallback, useEffect, useMemo } from "react";
import useSWR, { useSWRConfig } from "swr";
import {
  fetchSession,
  fetchSessionIterations,
  fetchSessionConfig,
  fetchIteration,
  fetchSessionLineage,
  fetchEvents,
  stopSession,
  type SessionDetail,
  type LoopIterationSummary,
  type IterationDetail,
  type Lineage,
  type EventSummary,
  type RunMetrics,
} from "@/lib/api";
import ReactMarkdown from "react-markdown";
import { savePreset, type E2EPresetParams } from "@/lib/presets";
import { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import { isThinkingEffort } from "@/components/ThinkingEffortSelector";
import StatusBadge from "@/components/StatusBadge";
import SavePresetButton from "@/components/SavePresetButton";
import LoopTimeline from "@/components/LoopTimeline";
import LineageTree from "@/components/LineageTree";
import JudgmentCard from "@/components/JudgmentCard";
import VideoPlayer from "@/components/VideoPlayer";
import RewardFormula from "@/components/RewardFormula";
import RewardDiff from "@/components/RewardDiff";
import SessionSummaryCard from "@/components/SessionSummaryCard";
import SessionAnalysisCard from "@/components/SessionAnalysisCard";
import EvalTimeline from "@/components/EvalTimeline";
import TermContributionBar from "@/components/TermContributionBar";
import RewardChart from "@/components/RewardChart";
import TrainingChart from "@/components/TrainingChart";
import EventTimeline from "@/components/EventTimeline";
import IterationMeanStdChart from "@/components/IterationMeanStdChart";
import RunListPanel from "@/components/RunListPanel";
import CollapsibleCard from "@/components/CollapsibleCard";
import { syncRun } from "@/lib/scheduler-api";
import { fetchLabelingStatus, type LabelingStatus } from "@/lib/labeling-api";

/**
 * Extract LaTeX string from a reward function's docstring.
 * Looks for lines like `latex = "..."` or `LaTeX: ...`.
 */
function extractLatex(rewardCode: string): string {
  if (!rewardCode) return "";
  // Match: latex = "..." or latex = '...'
  const assignMatch = rewardCode.match(/latex\s*=\s*["']([^"']+)["']/i);
  if (assignMatch) return assignMatch[1];
  // Match: LaTeX: <expression> in docstring
  const docMatch = rewardCode.match(/LaTeX:\s*(.+)/i);
  if (docMatch) return docMatch[1].trim();
  return "";
}

function iterationIdFromDir(iterationDir: string): string {
  const parts = iterationDir.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1];
}

export default function SessionDetailPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = use(params);
  const [selectedIteration, setSelectedIteration] = useState<number | null>(
    null,
  );
  const [pendingConfigId, setPendingConfigId] = useState<string | null>(null);
  const [selectedRunMetrics, setSelectedRunMetrics] = useState<RunMetrics | null>(null);
  const handleRunSelect = useCallback((_: string | null, metrics: RunMetrics | null) => {
    setSelectedRunMetrics(metrics);
  }, []);
  const [stopping, setStopping] = useState(false);
  const [syncing, setSyncing] = useState<"lite" | "full" | null>(null);
  const [syncResult, setSyncResult] = useState<{ ok: boolean; mode: string; error?: string } | null>(null);
  const { mutate: globalMutate } = useSWRConfig();

  const handleSync = useCallback(async (mode: "lite" | "full") => {
    setSyncing(mode);
    setSyncResult(null);
    try {
      const res = await syncRun(sessionId, { mode });
      setSyncResult({ ok: res.synced, mode, error: res.error ?? undefined });
      if (res.synced) {
        // Revalidate all SWR keys for this session so charts/data refresh
        globalMutate((key: string) => typeof key === "string" && key.includes(sessionId), undefined, { revalidate: true });
      }
    } catch (err) {
      setSyncResult({ ok: false, mode, error: err instanceof Error ? err.message : "Sync failed" });
    } finally {
      setSyncing(null);
    }
  }, [sessionId, globalMutate]);

  const { data: session, error, mutate } = useSWR<SessionDetail>(
    `session-${sessionId}`,
    () => fetchSession(sessionId),
    { refreshInterval: 3000 },
  );

  // Fetch iterations separately (parallel with session meta)
  const { data: iterations } = useSWR<LoopIterationSummary[]>(
    `session-${sessionId}-iters`,
    () => fetchSessionIterations(sessionId),
    { refreshInterval: 3000 },
  );

  // Fetch events while waiting for first iteration (live progress)
  const waitingForIter = session?.status === "running" && !session?.is_stale && (iterations?.length ?? 0) === 0;
  const { data: events } = useSWR<EventSummary[]>(
    waitingForIter ? `session-${sessionId}-events` : null,
    () => fetchEvents(sessionId),
    { refreshInterval: 3000 },
  );

  // Fetch session lineage (always loaded — details panel starts open)
  const { data: lineage } = useSWR<Lineage>(
    `lineage-${sessionId}`,
    () => fetchSessionLineage(sessionId),
    { refreshInterval: 10000 },
  );

  // Fetch session config for multi-config info (config IDs)
  const { data: sessionConfig } = useSWR<Record<string, unknown>>(
    `session-config-${sessionId}`,
    () => fetchSessionConfig(sessionId),
  );

  // Fetch labeling server status (for human scoring UI)
  const { data: labelingStatus } = useSWR<LabelingStatus>(
    "labeling-status",
    fetchLabelingStatus,
  );
  const labelingEnabled = labelingStatus?.enabled ?? false;
  const labelingAnnotator = labelingStatus?.annotator ?? "";

  // Compute active iteration (null-safe — data may not be loaded yet)
  const activeIt: LoopIterationSummary | null = iterations
    ? selectedIteration !== null
      ? (iterations.find((it) => it.iteration === selectedIteration) ?? null)
      : (iterations[iterations.length - 1] ?? null)
    : null;

  const activeIterationId = activeIt?.iteration_dir
    ? iterationIdFromDir(activeIt.iteration_dir)
    : null;

  // Reset stale run metrics when switching iterations
  useEffect(() => { setSelectedRunMetrics(null); }, [activeIterationId]);

  // Lazy-load full iteration detail (includes training metrics) for selected iteration
  const { data: iterDetail } = useSWR<IterationDetail>(
    activeIterationId ? `iter-detail-${sessionId}-${activeIterationId}` : null,
    () => fetchIteration(activeIterationId!, sessionId),
  );

  // For multi-config iterations, override eval/training/config when a specific run is selected
  const effectiveEvalResults = useMemo(() => {
    if (activeIt?.is_multi_config && selectedRunMetrics?.evaluation?.length) {
      return selectedRunMetrics.evaluation;
    }
    return iterDetail?.eval_results ?? [];
  }, [activeIt?.is_multi_config, selectedRunMetrics?.evaluation, iterDetail?.eval_results]);

  const effectiveTraining = useMemo(() => {
    if (activeIt?.is_multi_config && selectedRunMetrics?.training?.length) {
      return selectedRunMetrics.training;
    }
    return iterDetail?.training ?? [];
  }, [activeIt?.is_multi_config, selectedRunMetrics?.training, iterDetail?.training]);

  const effectiveConfig = useMemo(() => {
    if (activeIt?.is_multi_config && selectedRunMetrics?.config) {
      return selectedRunMetrics.config;
    }
    return iterDetail?.config ?? null;
  }, [activeIt?.is_multi_config, selectedRunMetrics?.config, iterDetail?.config]);

  async function handleStop() {
    setStopping(true);
    try {
      await stopSession(sessionId);
      await mutate();
    } catch {
      // SWR polling will pick up the new status
    } finally {
      setStopping(false);
    }
  }

  async function handleSavePreset(name: string) {
    const config = await fetchSessionConfig(sessionId);
    const params: E2EPresetParams = {
      prompt: (config.prompt as string) ?? session?.prompt ?? "",
      numConfigs: (config.num_configs as number) ?? 2,
      seeds: ((config.seeds as number[]) ?? [1, 2, 3]).join(", "),
      timesteps: (config.total_timesteps as number) ?? 1_000_000,
      maxIterations: (config.max_iterations as number) ?? 100,
      passThreshold: (config.pass_threshold as number) ?? 0.7,
      envId: (config.env_id as string) ?? "HalfCheetah-v5",
      numEnvs: (config.num_envs as number) ?? 128,
      vlmModel: (config.vlm_model as string) ?? "gemini-3.1-pro-preview",
      model: (config.model as string) ?? DEFAULT_LLM_MODEL,
      numEvals: (config.num_evals as number) ?? 4,
      useCodeJudge: (config.use_code_judge as boolean) ?? true,
      coresPerRun: (config.cores_per_run as number) ?? 10,
      device: ((config.device as "auto" | "cpu")) ?? "auto",
      thinkingEffort: isThinkingEffort(config.thinking_effort) ? config.thinking_effort : "max",
    };
    savePreset("e2e", name, params);
  }

  if (error) {
    return <p className="text-red-600">Failed to load session.</p>;
  }

  if (!session) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="flex items-start justify-between">
          <div>
            <div className="h-6 w-48 bg-gray-200 rounded" />
            <div className="h-4 w-64 bg-gray-100 rounded mt-2" />
          </div>
          <div className="h-6 w-20 bg-gray-200 rounded-full" />
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <div className="grid grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i}>
                <div className="h-4 w-20 bg-gray-100 rounded mb-2" />
                <div className="h-7 w-16 bg-gray-200 rounded" />
              </div>
            ))}
          </div>
        </div>
        <div className="h-24 bg-gray-100 rounded-xl" />
      </div>
    );
  }

  const itersLoaded = iterations !== undefined;
  const iters = iterations ?? [];

  const prevIt: LoopIterationSummary | null =
    activeIt && activeIt.iteration > 1
      ? (iters.find(
          (it) => it.iteration === activeIt.iteration - 1,
        ) ?? null)
      : null;

  // Use iteration detail's reward_spec when available, fallback to extractLatex
  const rewardLatex = iterDetail?.reward_spec?.latex || extractLatex(activeIt?.reward_code ?? "");
  const rawTerms = iterDetail?.reward_spec?.terms;
  const rewardTerms = rawTerms ?? {};
  // Flat dict for components that need Record<string, string> (charts, etc.)
  const termDescriptionsFlat: Record<string, string> = Array.isArray(rawTerms)
    ? Object.fromEntries(rawTerms.map((t: { name: string; description?: string }) => [t.name, t.description ?? ""]))
    : (rawTerms ?? {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">{sessionId}</h1>
          <p className="text-sm text-gray-500 mt-1">
            {session.prompt || "No prompt"}
          </p>
          {typeof sessionConfig?.["elaborated_intent"] === "string" && sessionConfig["elaborated_intent"] !== session.prompt && (
            <details className="mt-2">
              <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                Behavioral criteria
              </summary>
              <pre className="text-xs text-gray-400 mt-1 whitespace-pre-wrap font-sans leading-relaxed">
                {sessionConfig["elaborated_intent"]}
              </pre>
            </details>
          )}
        </div>
        <div className="flex items-center gap-2">
          <SavePresetButton onSave={handleSavePreset} />
          {/* Sync buttons (for remote sessions) */}
          <div className="flex gap-1">
            <button
              onClick={() => handleSync("lite")}
              disabled={syncing !== null}
              className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 disabled:opacity-50 transition-colors"
              title="Sync metrics, configs, judgments (no videos/trajectories)"
            >
              {syncing === "lite" ? "Syncing..." : "Sync Lite"}
            </button>
            <button
              onClick={() => handleSync("full")}
              disabled={syncing !== null}
              className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-50 transition-colors"
              title="Sync everything including videos and trajectories"
            >
              {syncing === "full" ? "Syncing..." : "Sync Full"}
            </button>
          </div>
          {syncResult && (
            <span className={`text-xs ${syncResult.ok ? "text-green-600" : "text-red-600"}`}>
              {syncResult.ok ? `${syncResult.mode} synced` : syncResult.error}
            </span>
          )}
          {session.status === "running" && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50 transition-colors"
            >
              {stopping ? "Stopping..." : "Stop"}
            </button>
          )}
          <StatusBadge status={session.status} isStale={session.is_stale} />
        </div>
      </div>

      {/* Session config summary (LLM / VLM / thinking / env) */}
      {sessionConfig && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {session.env_id && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="font-medium">Env</span>
              <span className="text-emerald-600">{session.env_id}</span>
            </span>
          )}
          {typeof sessionConfig.model === "string" && sessionConfig.model && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-violet-50 text-violet-700 border border-violet-200">
              <span className="font-medium">LLM</span>
              <span className="text-violet-600">{sessionConfig.model}</span>
            </span>
          )}
          {typeof sessionConfig.vlm_model === "string" && sessionConfig.vlm_model && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
              <span className="font-medium">VLM</span>
              <span className="text-sky-600">{sessionConfig.vlm_model}</span>
            </span>
          )}
          {typeof sessionConfig.thinking_effort === "string" && sessionConfig.thinking_effort && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
              <span className="font-medium">Thinking</span>
              <span className="text-amber-600">{sessionConfig.thinking_effort}</span>
            </span>
          )}
          {sessionConfig.criteria_diagnosis === true && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
              <span className="font-medium">Criteria Diagnosis</span>
            </span>
          )}
          {sessionConfig.motion_trail_dual === true && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-fuchsia-50 text-fuchsia-700 border border-fuchsia-200">
              <span className="font-medium">Motion Trail</span>
            </span>
          )}
        </div>
      )}

      {/* Best score summary */}
      {iters.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Best Score</p>
              <p className="font-mono font-semibold text-lg">
                {session.best_score.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Best Iteration</p>
              <p className="font-mono font-semibold text-lg">
                v{session.best_iteration}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Iterations</p>
              <p className="font-mono font-semibold text-lg">
                {iters.length}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stale session warning */}
      {session.is_stale && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <p className="text-sm font-medium text-amber-800">No Response</p>
          <p className="text-sm text-amber-600 mt-1">
            Session has not responded for over 5 minutes. The process may have terminated.
          </p>
        </div>
      )}

      {/* Session summary (frontend aggregation) */}
      {iters.length >= 2 && (
        <SessionSummaryCard
          iterations={iters}
          status={session.status}
          bestScore={session.best_score}
        />
      )}

      {/* Error display */}
      {session.error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-sm font-medium text-red-800">Error</p>
          <p className="text-sm text-red-600 mt-1">{session.error}</p>
        </div>
      )}

      {/* Waiting for first iteration — show live event progress */}
      {session.status === "running" && !session.is_stale && itersLoaded && iters.length === 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 text-center">
          <div className="inline-flex items-center gap-2 text-blue-700 text-sm font-medium">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            {(() => {
              const lastEvent = events?.[events.length - 1];
              if (!lastEvent) return "Preparing training...";
              const labels: Record<string, string> = {
                "session.started": "Session started",
                "judge_code.generate.start": "Generating judge code (LLM thinking...)",
                "judge_code.generate.end": "Judge code generated",
                "reward.provided": "Reward function loaded",
                "reward.generate.start": "Generating reward function (LLM thinking...)",
                "reward.generate.end": "Reward function generated",
                "iteration.started": "Starting iteration...",
              };
              return labels[lastEvent.event] ?? lastEvent.event.replace(/\./g, " ");
            })()}
          </div>
          {(() => {
            const lastEvent = events?.[events.length - 1];
            if (!lastEvent) return <p className="text-xs text-blue-500 mt-1">Results will appear when the first iteration starts.</p>;
            return (
              <p className="text-xs text-blue-500 mt-1">
                Step {events!.length}
              </p>
            );
          })()}
        </div>
      )}

      {/* Timeline */}
      <LoopTimeline
        iterations={iters}
        activeIteration={activeIt?.iteration ?? null}
        onSelect={setSelectedIteration}
        lineage={lineage}
      />

      {/* Experiment Lineage Tree */}
      <CollapsibleCard
        defaultOpen
        title={<>
          Experiment Lineage
          {lineage && (
            <span className="ml-2 text-xs font-normal text-gray-400">
              {Object.keys(lineage.iterations).length} iterations{lineage.lessons.length > 0 ? ` \u00b7 ${lineage.lessons.length} lessons` : ""}
            </span>
          )}
        </>}
      >
        {lineage ? (
          <LineageTree
            lineage={lineage}
            elapsedMap={Object.fromEntries(
              (iterations ?? [])
                .filter((it) => it.elapsed_time_s != null)
                .map((it) => [it.iteration, it.elapsed_time_s!])
            )}
            isRunning={session.status === "running"}
            activeIterationKey={activeIt ? `${sessionId}/iter_${activeIt.iteration}` : undefined}
            onSelectIteration={(key) => {
              const m = key.match(/iter_(\d+)(?:\/(.+))?/);
              if (m) {
                setSelectedIteration(parseInt(m[1], 10));
                setPendingConfigId(m[2] ?? null);
              }
            }}
            ghostBasedOn={iters.length > 0 ? iters[iters.length - 1].based_on : undefined}
            configInfos={(() => {
              const cfgs = sessionConfig?.configs;
              if (Array.isArray(cfgs) && cfgs.length > 1) {
                return cfgs.map((c: Record<string, unknown>) => ({
                  config_id: String(c.config_id ?? ""),
                  label: String(c.label ?? c.config_id ?? ""),
                  params:
                    typeof c.params === "object" && c.params !== null
                      ? (c.params as Record<string, unknown>)
                      : {},
                }));
              }
              return undefined;
            })()}
          />
        ) : (
          <p className="text-sm text-gray-400">Loading lineage...</p>
        )}
      </CollapsibleCard>

      {/* Event Timeline (debug observability) */}
      <details className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <summary className="px-5 py-3 text-sm font-semibold text-gray-900 cursor-pointer hover:bg-gray-50 rounded-xl">
          Event Timeline (LLM Traces)
        </summary>
        <div className="px-0 pb-0">
          <EventTimeline sessionId={sessionId} isRunning={session.status === "running"} />
        </div>
      </details>

      {/* MeanStd Training Curves (multi-config iterations) */}
      {activeIt?.is_multi_config && activeIt.aggregation && (
        <IterationMeanStdChart
          sessionId={sessionId}
          iterNum={activeIt.iteration}
          configIds={Object.keys(activeIt.aggregation)}
          totalTimesteps={session.total_timesteps}
          isRunning={session.status === "running"}
        />
      )}

      {/* LLM Analysis (on-demand, SSE streaming) */}
      {session.status !== "running" && iters.length >= 2 && (
        <SessionAnalysisCard sessionId={sessionId} />
      )}

      {/* Selected iteration detail */}
      {activeIt && (
        <>
          {/* Best eval videos (iteration-level, from best run) */}
          {activeIt.video_urls && activeIt.video_urls.length > 0 && (
            <VideoPlayer
              title="Best Evaluation Videos"
              urls={activeIt.video_urls}
              checkpointScores={activeIt.checkpoint_scores}
              checkpointDiagnoses={activeIt.checkpoint_diagnoses}
              checkpointCodeDiagnoses={activeIt.checkpoint_code_diagnoses}
              checkpointVlmDiagnoses={activeIt.checkpoint_vlm_diagnoses}
              checkpointCodeScores={activeIt.checkpoint_code_scores}
              checkpointVlmScores={activeIt.checkpoint_vlm_scores}
              rolloutScores={activeIt.rollout_scores}
              rolloutDiagnoses={activeIt.rollout_diagnoses}
              rolloutCodeDiagnoses={activeIt.rollout_code_diagnoses}
              rolloutVlmDiagnoses={activeIt.rollout_vlm_diagnoses}
              rolloutCodeScores={activeIt.rollout_code_scores}
              rolloutVlmScores={activeIt.rollout_vlm_scores}
              rolloutSynthesisTraces={activeIt.rollout_synthesis_traces}
              rolloutCriteriaScores={activeIt.rollout_criteria_scores}
              rolloutVlmPreviewUrls={activeIt.rollout_vlm_preview_urls}
              rolloutMotionPreviewUrls={activeIt.rollout_motion_preview_urls}
              vlmFps={activeIt.vlm_fps}
              vlmCriteria={activeIt.vlm_criteria}
              criteriaScores={activeIt.criteria_scores}
              bestCheckpoint={activeIt.best_checkpoint}
              sourceRunId={activeIt.video_source_run_id || activeIt.best_run_id}
              sourceReturn={activeIt.video_source_return}
              sessionId={sessionId}
              iteration={activeIt.iteration}
              labelingEnabled={labelingEnabled}
              labelingAnnotator={labelingAnnotator}
              humanLabels={activeIt.human_label}
            />
          )}

          <JudgmentCard iteration={activeIt} />

          {/* Per-run drill-down (multi-config iterations) */}
          {activeIt.is_multi_config && (
            <RunListPanel
              sessionId={sessionId}
              iterNum={activeIt.iteration}
              bestRunId={activeIt.best_run_id}
              onRunSelect={handleRunSelect}
              targetConfigId={pendingConfigId}
              onTargetConfigConsumed={() => setPendingConfigId(null)}
            />
          )}

          {/* Revise Agent Analysis */}
          {(activeIt.revise_diagnosis || activeIt.reward_reasoning || activeIt.hp_reasoning) && (
            <CollapsibleCard
              defaultOpen
              title={`Revise Agent Analysis (v${activeIt.iteration} → v${activeIt.iteration + 1})`}
            >
              <div className="space-y-4 text-sm">
                {activeIt.revise_diagnosis && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-1">Diagnosis</h4>
                    <div className="prose prose-sm prose-gray max-w-none bg-gray-50 p-3 rounded-lg">
                      <ReactMarkdown>{activeIt.revise_diagnosis}</ReactMarkdown>
                    </div>
                  </div>
                )}
                {activeIt.reward_reasoning && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-1">Reward Reasoning</h4>
                    <div className="prose prose-sm prose-gray max-w-none bg-blue-50 p-3 rounded-lg">
                      <ReactMarkdown>{activeIt.reward_reasoning}</ReactMarkdown>
                    </div>
                  </div>
                )}
                {activeIt.hp_reasoning && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-1">HP Reasoning</h4>
                    <div className="prose prose-sm prose-gray max-w-none bg-amber-50 p-3 rounded-lg">
                      <ReactMarkdown>{activeIt.hp_reasoning}</ReactMarkdown>
                    </div>
                  </div>
                )}
                {activeIt.hp_changes && Object.keys(activeIt.hp_changes).length > 0 && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-1">HP Changes</h4>
                    <pre className="text-xs font-mono text-gray-600 bg-amber-50 p-3 rounded-lg">
                      {JSON.stringify(activeIt.hp_changes, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </CollapsibleCard>
          )}

          {/* Lazy-loaded iteration details (single API call) */}
          {iterDetail ? (
            <>
              {/* Iteration Summary — key stats at a glance */}
              {iterDetail.summary && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">Iteration Summary</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Final Return</p>
                      <p className="font-mono font-semibold">
                        {iterDetail.summary.final_episodic_return.toFixed(1)}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">Episodes</p>
                      <p className="font-mono font-semibold">
                        {iterDetail.summary.total_episodes}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">Total Steps</p>
                      <p className="font-mono font-semibold">
                        {iterDetail.summary.total_timesteps.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">Training Time</p>
                      <p className="font-mono font-semibold">
                        {(iterDetail.summary.training_time_s / 60).toFixed(1)} min
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeIt?.is_multi_config && selectedRunMetrics && (
                <p className="text-xs text-gray-500 mb-1">
                  Showing data for: <span className="font-mono font-medium">{selectedRunMetrics.run_id}</span>
                </p>
              )}

              <CollapsibleCard title={activeIt?.is_multi_config && selectedRunMetrics?.config ? `Config (${selectedRunMetrics.run_id})` : "Config"}>
                <pre className="text-xs font-mono text-gray-600 overflow-x-auto">
                  {JSON.stringify(effectiveConfig, null, 2)}
                </pre>
              </CollapsibleCard>

              <EvalTimeline evalResults={effectiveEvalResults} />

              {rewardLatex && (
                <RewardFormula latex={rewardLatex} terms={rewardTerms} />
              )}

              <RewardDiff
                prev={prevIt?.reward_code ?? null}
                current={activeIt.reward_code}
                diffSummary={activeIt.reward_diff_summary}
              />

              <TermContributionBar
                evalResults={effectiveEvalResults}
                termDescriptions={termDescriptionsFlat}
              />

              {/* Reward & Return charts — grouped together */}
              <CollapsibleCard defaultOpen title="Reward &amp; Return">
                <div className="space-y-4">
                  <RewardChart
                    evalResults={effectiveEvalResults}
                    termDescriptions={termDescriptionsFlat}
                  />

                  {effectiveTraining.length > 0 && (
                    <>
                      <TrainingChart
                        data={effectiveTraining}
                        lines={["episodic_return"]}
                        title="Episodic Return"
                      />

                      {(() => {
                        const termKeys = Array.from(
                          new Set(
                            effectiveTraining.flatMap((e) =>
                              Object.keys(e).filter((k) => k.startsWith("reward_term_"))
                            )
                          )
                        ).sort();
                        if (termKeys.length === 0) return null;
                        const colsClass = termKeys.length <= 2 ? "md:grid-cols-2" : "md:grid-cols-3";
                        return (
                          <div>
                            <h2 className="text-sm font-semibold text-gray-500 mb-2">Reward Terms (Training)</h2>
                            <div className={`grid grid-cols-1 ${colsClass} gap-4`}>
                              {termKeys.map((key) => (
                                <TrainingChart
                                  key={key}
                                  data={effectiveTraining}
                                  lines={[key]}
                                  title={key.replace("reward_term_", "").replaceAll("_", " ")}
                                  height={200}
                                />
                              ))}
                            </div>
                          </div>
                        );
                      })()}
                    </>
                  )}
                </div>
              </CollapsibleCard>

              {effectiveTraining.length > 0 && (
                <>
                  <CollapsibleCard title="Episode Stats">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <TrainingChart data={effectiveTraining} lines={["episodic_return_std"]} title="Return Std" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["episodic_return_min", "episodic_return_max"]} title="Return Min / Max" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["episode_length"]} title="Episode Length" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["episodes_per_rollout"]} title="Episodes / Rollout" height={200} />
                    </div>
                  </CollapsibleCard>

                  <CollapsibleCard title="Loss">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <TrainingChart data={effectiveTraining} lines={["policy_loss"]} title="Clip Loss" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["value_loss"]} title="Value Loss" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["entropy"]} title="Exploration Loss" height={200} />
                    </div>
                  </CollapsibleCard>

                  <CollapsibleCard title="PPO Diagnostic">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <TrainingChart data={effectiveTraining} lines={["clip_fraction"]} title="Clip Fraction" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["approx_kl"]} title="Approx KL" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["explained_variance"]} title="Explained Variance" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["kl_mean_term", "kl_var_term"]} title="KL Decomposition" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["mean_shift_normalized"]} title="Mean Shift (Normalized)" height={200} />
                    </div>
                  </CollapsibleCard>

                  <CollapsibleCard title="Policy &amp; Gradient Health">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <TrainingChart data={effectiveTraining} lines={["policy_std"]} title="Policy Std (Action)" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["grad_norm"]} title="Gradient Norm (L2)" height={200} />
                    </div>
                  </CollapsibleCard>

                  <CollapsibleCard title="Throughput &amp; General">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <TrainingChart data={effectiveTraining} lines={["learning_rate"]} title="Learning Rate" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["rollout_time", "train_time"]} title="Worker vs Learner (s)" height={200} />
                      <TrainingChart data={effectiveTraining} lines={["sps"]} title="Steps Per Second" height={200} />
                    </div>
                  </CollapsibleCard>
                </>
              )}

            </>
          ) : activeIterationId ? (
            /* Loading skeleton for lazy-loaded content */
            <div className="space-y-4 animate-pulse">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
                  <div className="h-4 w-32 bg-gray-200 rounded mb-3" />
                  <div className="h-24 bg-gray-100 rounded" />
                </div>
              ))}
            </div>
          ) : (
            /* No iteration detail available — show RewardDiff from session data */
            <RewardDiff
              prev={prevIt?.reward_code ?? null}
              current={activeIt.reward_code}
              diffSummary={activeIt.reward_diff_summary}
            />
          )}
        </>
      )}
    </div>
  );
}
