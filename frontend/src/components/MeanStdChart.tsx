"use client";

import { useState, useMemo } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { AggregatedMetrics } from "@/lib/api";
import { formatWallTime } from "@/lib/format";

interface ConfigSeries {
  config_id: string;
  label: string;
  color: string;
  data: AggregatedMetrics;
}

interface Props {
  configs: ConfigSeries[];
  defaultMetric?: string;
  totalTimesteps?: number;
  /** When set, locks to this metric (no dropdown). */
  fixedMetric?: string;
  /** Override heading text. */
  title?: string;
  /** Chart height in px (default 350). */
  height?: number;
}

export default function MeanStdChart({
  configs,
  defaultMetric = "episodic_return",
  totalTimesteps,
  fixedMetric,
  title,
  height: chartHeight = 350,
}: Props) {
  // Collect all available metrics across all configs
  const availableMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const c of configs) {
      for (const k of c.data.available_metrics) {
        keys.add(k);
      }
    }
    return Array.from(keys).sort();
  }, [configs]);

  const [metricState, setMetricState] = useState(
    availableMetrics.includes(defaultMetric)
      ? defaultMetric
      : availableMetrics[0] || "",
  );

  const metric = fixedMetric && availableMetrics.includes(fixedMetric)
    ? fixedMetric
    : metricState;

  const [hidden, setHidden] = useState<Set<string>>(new Set());

  // Build merged data array: { step, configA_mean, configA_upper, configA_lower, ... }
  const chartData = useMemo(() => {
    if (!metric || configs.length === 0) return [];

    // Collect all unique steps
    const allSteps = new Set<number>();
    for (const c of configs) {
      for (const s of c.data.global_steps) {
        allSteps.add(s);
      }
    }
    const steps = Array.from(allSteps).sort((a, b) => a - b);

    // Downsample if needed
    const maxPoints = 200;
    let sampled = steps;
    if (steps.length > maxPoints) {
      const stepSize = steps.length / maxPoints;
      sampled = [];
      for (let i = 0; i < maxPoints; i++) {
        sampled.push(steps[Math.floor(i * stepSize)]);
      }
      if (sampled[sampled.length - 1] !== steps[steps.length - 1]) {
        sampled.push(steps[steps.length - 1]);
      }
    }

    // Pre-cache per-config metric series and elapsed_time
    const configLookups = configs.map((c) => ({
      config_id: c.config_id,
      metric: c.data.metrics[metric],
      elapsed: c.data.metrics["elapsed_time"],
      gs: c.data.global_steps,
    }));

    return sampled.map((step) => {
      const point: Record<string, number | [number, number]> = { step };
      for (const { config_id, metric: m, elapsed, gs } of configLookups) {
        if (!m) continue;
        if (gs.length === 0 || step > gs[gs.length - 1]) continue;
        const idx = findClosestIndex(gs, step);
        if (idx < 0 || idx >= m.mean.length) continue;
        const mean = m.mean[idx];
        const std = m.std[idx];
        point[`${config_id}_mean`] = mean;
        point[`${config_id}_range`] = [mean - std, mean + std];
        if (elapsed && idx < elapsed.mean.length) {
          point[`${config_id}_walltime`] = elapsed.mean[idx];
        }
      }
      return point;
    });
  }, [configs, metric]);

  const toggleConfig = (configId: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(configId)) {
        next.delete(configId);
      } else {
        next.add(configId);
      }
      return next;
    });
  };

  if (configs.length === 0 || availableMetrics.length === 0) {
    return null;
  }

  // If fixedMetric specified but not available in any config, skip
  if (fixedMetric && !availableMetrics.includes(fixedMetric)) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-900">
          {title || (fixedMetric ? fixedMetric : "Training Curves (mean ± std)")}
        </h2>
        {!fixedMetric && (
          <select
            value={metric}
            onChange={(e) => setMetricState(e.target.value)}
            className="text-xs border border-gray-300 rounded px-2 py-1 bg-white text-gray-700"
          >
            {availableMetrics.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Config toggle legend */}
      <div className="flex flex-wrap gap-2 mb-3">
        {configs.map((c) => (
          <button
            key={c.config_id}
            onClick={() => toggleConfig(c.config_id)}
            className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
              hidden.has(c.config_id)
                ? "border-gray-300 text-gray-400 bg-gray-50"
                : "border-current text-white"
            }`}
            style={
              hidden.has(c.config_id)
                ? undefined
                : { backgroundColor: c.color, borderColor: c.color }
            }
          >
            {c.label || c.config_id}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={chartHeight}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="step"
            type="number"
            domain={[0, totalTimesteps && totalTimesteps > 0 ? totalTimesteps : "dataMax"]}
            tickFormatter={(v: number) =>
              v >= 1_000_000
                ? `${(v / 1_000_000).toFixed(1)}M`
                : `${(v / 1000).toFixed(0)}k`
            }
            fontSize={11}
          />
          <YAxis fontSize={11} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              // Build lookup map for O(1) access by dataKey
              const byKey = new Map(payload.map((p) => [String(p.dataKey ?? ""), p]));
              const dataPoint = payload[0]?.payload;
              const entries: { configId: string; label: string; color: string; mean: number; range?: [number, number]; walltime?: number }[] = [];
              for (const c of configs) {
                if (hidden.has(c.config_id)) continue;
                const meanEntry = byKey.get(`${c.config_id}_mean`);
                if (!meanEntry) continue;
                const rangeEntry = byKey.get(`${c.config_id}_range`);
                const wt = dataPoint?.[`${c.config_id}_walltime`];
                entries.push({
                  configId: c.config_id,
                  label: c.label || c.config_id,
                  color: c.color,
                  mean: Number(meanEntry.value ?? 0),
                  range: Array.isArray(rangeEntry?.value) ? rangeEntry.value as [number, number] : undefined,
                  walltime: typeof wt === "number" ? wt : undefined,
                });
              }
              return (
                <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs shadow-md">
                  <p className="font-medium text-gray-700 mb-1">Step {Number(label).toLocaleString()}</p>
                  {entries.map((e) => (
                    <div key={e.configId} className="mb-1 last:mb-0">
                      <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: e.color }} />
                      <span className="font-medium">{e.label}</span>
                      <span className="text-gray-600">: {e.mean.toFixed(3)}</span>
                      {e.range && (
                        <span className="text-gray-400 ml-1">({e.range[0].toFixed(3)} ~ {e.range[1].toFixed(3)})</span>
                      )}
                      {e.walltime != null && (
                        <span className="text-gray-500 ml-1">{formatWallTime(e.walltime)}</span>
                      )}
                    </div>
                  ))}
                </div>
              );
            }}
          />
          <Legend content={() => null} />

          {configs.map((c) => {
            if (hidden.has(c.config_id)) return null;
            const meanKey = `${c.config_id}_mean`;
            const rangeKey = `${c.config_id}_range`;

            return (
              <g key={c.config_id}>
                <Area
                  dataKey={rangeKey}
                  stroke="none"
                  fill={c.color}
                  fillOpacity={0.12}
                  isAnimationActive={false}
                />
                <Line
                  dataKey={meanKey}
                  stroke={c.color}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </g>
            );
          })}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function findClosestIndex(arr: number[], target: number): number {
  if (arr.length === 0) return -1;
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  // Check if prev is closer
  if (lo > 0 && Math.abs(arr[lo - 1] - target) < Math.abs(arr[lo] - target)) {
    return lo - 1;
  }
  return lo;
}
