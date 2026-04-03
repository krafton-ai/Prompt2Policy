"use client";

import useSWR from "swr";
import { fetchAggregatedMetrics, type AggregatedMetrics } from "@/lib/api";
import MeanStdChart from "./MeanStdChart";

const COLORS = [
  "#2563eb",
  "#dc2626",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
];

interface Props {
  sessionId: string;
  iterNum: number;
  configIds: string[];
  totalTimesteps?: number;
  /** Poll while training is still running. */
  isRunning?: boolean;
}

export default function IterationMeanStdChart({
  sessionId,
  iterNum,
  configIds,
  totalTimesteps,
  isRunning,
}: Props) {
  // Fetch aggregated metrics for each config in parallel
  const results = configIds.map((configId) => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const { data } = useSWR<AggregatedMetrics>(
      `agg-${sessionId}-${iterNum}-${configId}`,
      () => fetchAggregatedMetrics(sessionId, iterNum, configId),
      { refreshInterval: isRunning ? 10000 : 0 },
    );
    return { configId, data };
  });

  const loaded = results.filter(
    (r): r is { configId: string; data: AggregatedMetrics } => r.data != null,
  );

  if (loaded.length === 0) return null;

  // Only show if at least one config has actual metric data
  const hasData = loaded.some(
    (r) => r.data.global_steps.length > 0 && r.data.available_metrics.length > 0,
  );
  if (!hasData) return null;

  const configs = loaded.map((r, i) => ({
    config_id: r.configId,
    label: r.configId,
    color: COLORS[i % COLORS.length],
    data: r.data,
  }));

  return (
    <MeanStdChart
      configs={configs}
      totalTimesteps={totalTimesteps}
      title="Training Curves — Config Comparison (mean ± std across seeds)"
    />
  );
}
