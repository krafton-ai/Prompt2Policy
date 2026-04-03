"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import useSWR from "swr";
import {
  fetchIterationRuns,
  fetchRunMetrics,
  type IterationRunEntry,
  type RunMetrics,
} from "@/lib/api";
import type { TrainingEntry } from "@/lib/api";
import TrainingChart from "@/components/TrainingChart";
import VideoPlayer from "@/components/VideoPlayer";

function RunDetail({ metrics }: { metrics: RunMetrics }) {
  const hasTraining = metrics.training.length > 0;
  const hasEval = metrics.evaluation.length > 0;

  // Convert eval entries to TrainingEntry-compatible format for charting
  const evalAsTraining: TrainingEntry[] = useMemo(() => {
    if (hasTraining || !hasEval) return [];
    return metrics.evaluation.map((e) => ({
      global_step: e.global_step,
      iteration: 0,
      policy_loss: 0,
      value_loss: 0,
      entropy: 0,
      approx_kl: 0,
      clip_fraction: 0,
      explained_variance: 0,
      learning_rate: 0,
      sps: 0,
      episodic_return: e.total_reward,
    }));
  }, [metrics.evaluation, hasTraining, hasEval]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-500">
          Showing:{" "}
          <span className="font-mono font-medium text-gray-900">
            {metrics.run_id}
          </span>
        </p>
        {metrics.intent_score != null && (
          <span className={`text-sm font-mono font-semibold ${
            metrics.intent_score >= 0.7 ? "text-green-700" :
            metrics.intent_score >= 0.42 ? "text-yellow-700" : "text-red-600"
          }`}>
            Intent: {metrics.intent_score.toFixed(2)}
          </span>
        )}
      </div>

      {metrics.diagnosis && (
        <p className="text-xs text-gray-500 bg-gray-50 rounded-lg px-3 py-2">{metrics.diagnosis}</p>
      )}

      {/* Episodic return chart */}
      {hasTraining && (
        <TrainingChart
          data={metrics.training}
          lines={["episodic_return"]}
        />
      )}

      {/* Eval reward chart (fallback when no training data) */}
      {!hasTraining && evalAsTraining.length > 0 && (
        <TrainingChart data={evalAsTraining} lines={["episodic_return"]} />
      )}

      {/* Videos with judgment scores */}
      {metrics.video_urls.length > 0 && (
        <VideoPlayer
          urls={metrics.video_urls}
          checkpointScores={metrics.checkpoint_scores}
          checkpointDiagnoses={metrics.checkpoint_diagnoses}
          checkpointCodeDiagnoses={metrics.checkpoint_code_diagnoses}
          checkpointVlmDiagnoses={metrics.checkpoint_vlm_diagnoses}
          checkpointCodeScores={metrics.checkpoint_code_scores}
          checkpointVlmScores={metrics.checkpoint_vlm_scores}
          rolloutScores={metrics.rollout_scores}
          rolloutDiagnoses={metrics.rollout_diagnoses}
          rolloutCodeDiagnoses={metrics.rollout_code_diagnoses}
          rolloutVlmDiagnoses={metrics.rollout_vlm_diagnoses}
          rolloutCodeScores={metrics.rollout_code_scores}
          rolloutVlmScores={metrics.rollout_vlm_scores}
          rolloutSynthesisTraces={metrics.rollout_synthesis_traces}
          rolloutVlmPreviewUrls={metrics.rollout_vlm_preview_urls}
          rolloutMotionPreviewUrls={metrics.rollout_motion_preview_urls}
          vlmFps={metrics.vlm_fps}
          bestCheckpoint={metrics.best_checkpoint}
          sourceRunId={metrics.run_id}
        />
      )}
    </div>
  );
}

interface Props {
  sessionId: string;
  iterNum: number;
  bestRunId?: string;
  onRunSelect?: (runId: string | null, metrics: RunMetrics | null) => void;
  targetConfigId?: string | null;
  onTargetConfigConsumed?: () => void;
}

export default function RunListPanel({ sessionId, iterNum, bestRunId, onRunSelect, targetConfigId, onTargetConfigConsumed }: Props) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const { data: runs } = useSWR<IterationRunEntry[]>(
    `runs-${sessionId}-${iterNum}`,
    () => fetchIterationRuns(sessionId, iterNum),
    { refreshInterval: 5000 },
  );

  // Determine best run (from prop or by highest return) — single source of truth
  const effectiveBestRunId = useMemo(() => {
    if (bestRunId) return bestRunId;
    if (!runs || runs.length === 0) return null;
    return runs.reduce((a, b) =>
      (b.final_return ?? -Infinity) > (a.final_return ?? -Infinity) ? b : a,
    ).run_id;
  }, [runs, bestRunId]);

  // Stable ref for targetConfigId consumed callback
  const onTargetConfigConsumedRef = useRef(onTargetConfigConsumed);
  useEffect(() => { onTargetConfigConsumedRef.current = onTargetConfigConsumed; }, [onTargetConfigConsumed]);

  // Reset selection and clear stale targetConfigId when iteration changes
  // Skip initial mount — pendingConfigId may still be needed for first runs load
  const prevIterNumRef = useRef<number | null>(null);
  useEffect(() => {
    if (prevIterNumRef.current !== null && prevIterNumRef.current !== iterNum) {
      setSelectedRunId(null);
      onTargetConfigConsumedRef.current?.();
    }
    prevIterNumRef.current = iterNum;
  }, [iterNum]);

  // Auto-select best run on initial load only (selectedRunId === null)
  useEffect(() => {
    if (!runs || runs.length === 0 || selectedRunId !== null) return;
    setSelectedRunId(effectiveBestRunId);
  }, [runs, selectedRunId, effectiveBestRunId]);

  useEffect(() => {
    if (!targetConfigId || !runs?.length) return;
    const match = runs.find((r) => r.config_id === targetConfigId);
    if (match) setSelectedRunId(match.run_id);
    onTargetConfigConsumedRef.current?.();
  }, [targetConfigId, runs]);

  const { data: runMetrics } = useSWR<RunMetrics>(
    selectedRunId
      ? `run-metrics-${sessionId}-${iterNum}-${selectedRunId}`
      : null,
    () => fetchRunMetrics(sessionId, iterNum, selectedRunId!),
    { refreshInterval: 10000 },
  );

  // Track latest callback ref to avoid re-triggering on unstable references
  const onRunSelectRef = useRef(onRunSelect);
  useEffect(() => { onRunSelectRef.current = onRunSelect; }, [onRunSelect]);

  // Notify parent when selected run metrics change
  useEffect(() => {
    onRunSelectRef.current?.(selectedRunId ?? null, runMetrics ?? null);
  }, [selectedRunId, runMetrics]);

  // Group runs by config_id
  const grouped = useMemo(() => {
    if (!runs) return {};
    const map: Record<string, IterationRunEntry[]> = {};
    for (const r of runs) {
      (map[r.config_id] ??= []).push(r);
    }
    return map;
  }, [runs]);

  if (!runs || runs.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
      <h2 className="text-sm font-semibold text-gray-900">
        Individual Runs
        <span className="ml-2 text-xs font-normal text-gray-400">
          {runs.length} runs across {Object.keys(grouped).length} config{Object.keys(grouped).length > 1 ? "s" : ""}
        </span>
      </h2>

      {/* Run table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-200 text-gray-500">
              <th className="text-left py-2 pr-3 font-medium">Config</th>
              <th className="text-left py-2 pr-3 font-medium">Seed</th>
              <th className="text-right py-2 pr-3 font-medium">
                Final Return
              </th>
              <th className="text-right py-2 pr-3 font-medium">
                Judge Score
              </th>
              <th className="text-left py-2 pr-3 font-medium">Status</th>
              <th className="text-right py-2 pr-3 font-medium">Videos</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(grouped).map(([, configRuns]) =>
              configRuns.map((run) => {
                const isSelected = selectedRunId === run.run_id;
                const isBest = run.run_id === effectiveBestRunId;
                return (
                  <tr
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`border-b border-gray-100 cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-blue-50 ring-1 ring-blue-200"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    <td className="py-2 pr-3 font-medium text-gray-900">
                      {run.config_id}
                      {isBest && (
                        <span className="ml-1.5 text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded-full">
                          best
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-3 font-mono">{run.seed}</td>
                    <td className="py-2 pr-3 text-right font-mono text-gray-900">
                      {run.final_return?.toFixed(1) ?? "..."}
                    </td>
                    <td className={`py-2 pr-3 text-right font-mono font-semibold ${
                      run.intent_score == null ? "text-gray-400" :
                      run.intent_score >= 0.7 ? "text-green-700" :
                      run.intent_score >= 0.42 ? "text-yellow-700" : "text-red-600"
                    }`}>
                      {run.intent_score?.toFixed(2) ?? "..."}
                    </td>
                    <td className="py-2 pr-3 text-gray-500">{run.status}</td>
                    <td className="py-2 pr-3 text-right text-gray-400">
                      {run.video_urls?.length ?? 0}
                    </td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </div>

      {/* Selected run detail */}
      {selectedRunId && runMetrics && (
        <RunDetail metrics={runMetrics} />
      )}
    </div>
  );
}
