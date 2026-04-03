"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { BenchmarkTestCaseResult, StageDetail } from "@/lib/api";

const COLORS = [
  "#2563eb", // blue
  "#dc2626", // red
  "#10b981", // green
  "#f59e0b", // amber
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#6366f1", // indigo
  "#84cc16", // lime
];

interface Props {
  testCases: BenchmarkTestCaseResult[];
  stages: StageDetail[];
  maxIterations: number;
}

// Counts: reached / total for each series at each iteration
interface IterCounts {
  overall: { reached: number; total: number };
  [key: string]: { reached: number; total: number };
}

interface DataPoint {
  iteration: number;
  overall: number | null;
  [key: string]: number | null;
}

export default function ScoreProgressionChart({
  testCases,
  stages,
  maxIterations,
}: Props) {
  const data = useMemo(() => {
    // Build a map: stage -> list of test cases with scores
    const stageMap = new Map<number, BenchmarkTestCaseResult[]>();
    for (const tc of testCases) {
      if (!tc.iteration_scores || tc.iteration_scores.length === 0) continue;
      const list = stageMap.get(tc.stage) || [];
      list.push(tc);
      stageMap.set(tc.stage, list);
    }

    // Only include stages that have at least one test case with scores
    const activeStages = stages.filter((s) => stageMap.has(s.stage));
    if (activeStages.length === 0)
      return { points: [], stageKeys: [], countsMap: new Map<number, IterCounts>() };

    // All test cases with scores (for overall line)
    const allWithScores = testCases.filter(
      (tc) => tc.iteration_scores && tc.iteration_scores.length > 0,
    );

    // Pre-compute cumulative best (running max) per test case
    const cumulativeBest = new Map<BenchmarkTestCaseResult, number[]>();
    for (const tc of allWithScores) {
      const bests: number[] = [];
      let runMax = 0;
      for (const s of tc.iteration_scores) {
        runMax = Math.max(runMax, s);
        bests.push(runMax);
      }
      cumulativeBest.set(tc, bests);
    }

    const points: DataPoint[] = [];
    const countsMap = new Map<number, IterCounts>();

    for (let i = 0; i < maxIterations; i++) {
      const point: DataPoint = { iteration: i + 1, overall: null };

      // Skip this iteration if no test case has actually reached it
      const anyReached = allWithScores.some(
        (tc) => cumulativeBest.get(tc)!.length > i,
      );
      if (!anyReached) continue;

      // Overall average of cumulative best at this iteration
      // Cases that haven't reached this iteration carry forward their last known best
      let totalScore = 0;
      let overallReached = 0;
      for (const tc of allWithScores) {
        const bests = cumulativeBest.get(tc)!;
        totalScore += bests[Math.min(i, bests.length - 1)];
        if (bests.length > i) overallReached++;
      }
      point.overall = totalScore / allWithScores.length;

      const counts: IterCounts = {
        overall: { reached: overallReached, total: allWithScores.length },
      };

      // Per-stage average of cumulative best at this iteration
      for (const s of activeStages) {
        const cases = stageMap.get(s.stage) || [];
        let stageTotal = 0;
        let stageCount = 0;
        let stageReached = 0;
        for (const tc of cases) {
          const bests = cumulativeBest.get(tc);
          if (bests && bests.length > 0) {
            stageTotal += bests[Math.min(i, bests.length - 1)];
            stageCount++;
            if (bests.length > i) stageReached++;
          }
        }
        point[`stage_${s.stage}`] =
          stageCount > 0 ? stageTotal / stageCount : null;
        counts[`stage_${s.stage}`] = {
          reached: stageReached,
          total: stageCount,
        };
      }

      countsMap.set(i + 1, counts);
      points.push(point);
    }

    // Trim trailing points where everything is null
    let lastValid = -1;
    for (let i = points.length - 1; i >= 0; i--) {
      if (points[i].overall !== null) {
        lastValid = i;
        break;
      }
    }

    const stageKeys = activeStages.map((s) => ({
      key: `stage_${s.stage}`,
      label: `Stage ${s.stage}`,
      stage: s.stage,
    }));

    return {
      points: points.slice(0, lastValid + 1),
      stageKeys,
      countsMap,
    };
  }, [testCases, stages, maxIterations]);

  if (data.points.length === 0) {
    return null;
  }

  // Show per-stage lines only if there are multiple stages with data
  const showStages = data.stageKeys.length > 1;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Score Progression by Iteration
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.points}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="iteration"
            label={{
              value: "Iteration",
              position: "insideBottom",
              offset: -5,
              style: { fontSize: 12, fill: "#6b7280" },
            }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Avg Best Score",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              style: { fontSize: 12, fill: "#6b7280" },
            }}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const iter = label as number;
              const counts = data.countsMap.get(iter);
              return (
                <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 text-xs">
                  <p className="font-semibold text-gray-700 mb-1">
                    Iteration {iter}
                  </p>
                  {payload.map((entry) => {
                    const key = entry.dataKey as string;
                    const c = counts?.[key];
                    return (
                      <p key={key} style={{ color: entry.color }}>
                        {entry.name}:{" "}
                        <span className="font-mono font-medium">
                          {entry.value != null
                            ? (entry.value as number).toFixed(3)
                            : "—"}
                        </span>
                        {c && (
                          <span className="text-gray-400 ml-1">
                            ({c.reached}/{c.total} done)
                          </span>
                        )}
                      </p>
                    );
                  })}
                </div>
              );
            }}
          />
          {showStages && <Legend />}
          <Line
            type="monotone"
            dataKey="overall"
            name="Overall"
            stroke="#111827"
            strokeWidth={2.5}
            dot={{ r: 3 }}
            connectNulls
          />
          {showStages &&
            data.stageKeys.map((s, i) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={1.5}
                strokeDasharray="4 2"
                dot={{ r: 2 }}
                connectNulls
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-2">
        Average cumulative best score across test cases at each iteration.
        {showStages && " Dashed lines show per-stage breakdown."}
      </p>
    </div>
  );
}
