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
import type { EvalResult } from "@/lib/api";

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
  evaluation: EvalResult[];
}

export default function EvalRewardChart({ evaluation }: Props) {
  const { data, termKeys } = useMemo(() => {
    if (evaluation.length === 0) return { data: [], termKeys: [] };

    // Collect all reward term keys
    const keys = new Set<string>();
    for (const e of evaluation) {
      if (e.reward_terms) {
        for (const k of Object.keys(e.reward_terms)) {
          keys.add(k);
        }
      }
    }
    const termKeys = Array.from(keys).sort();

    // Build chart data
    const data = evaluation.map((e) => ({
      global_step: e.global_step,
      total_reward: e.total_reward,
      ...e.reward_terms,
    }));

    return { data, termKeys };
  }, [evaluation]);

  if (data.length === 0 || termKeys.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Reward Terms
      </h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="global_step"
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
            fontSize={11}
          />
          <YAxis fontSize={11} />
          <Tooltip
            labelFormatter={(v) => `Step ${Number(v).toLocaleString()}`}
          />
          <Legend />
          {termKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
