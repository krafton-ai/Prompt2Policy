"use client";

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

const TERM_COLORS = [
  "#2563eb",
  "#dc2626",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
];

interface Props {
  evalResults: EvalResult[];
  termDescriptions?: Record<string, string>;
}

export default function RewardChart({ evalResults, termDescriptions }: Props) {
  if (evalResults.length === 0) return null;

  const termNames = Object.keys(evalResults[0]?.reward_terms || {});

  const chartData = evalResults.map((e) => ({
    step: e.global_step,
    total_reward: e.total_reward,
    ...e.reward_terms,
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Per-Term Rewards (Eval)
      </h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="step"
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
            fontSize={11}
          />
          <YAxis fontSize={11} />
          <Tooltip
            labelFormatter={(v) => `Step ${Number(v).toLocaleString()}`}
            content={({ active, payload, label }) => {
              if (!active || !payload) return null;
              return (
                <div className="bg-white border border-gray-200 shadow-lg rounded-lg p-3 text-xs max-w-sm">
                  <p className="font-semibold mb-1">
                    Step {Number(label).toLocaleString()}
                  </p>
                  {payload.map((entry) => {
                    const desc =
                      termDescriptions?.[entry.dataKey as string] || "";
                    return (
                      <div key={entry.dataKey} className="flex items-start gap-1 py-0.5">
                        <span
                          className="inline-block w-2 h-2 rounded-full mt-1 shrink-0"
                          style={{ backgroundColor: entry.color }}
                        />
                        <span className="font-mono">{entry.dataKey}:</span>
                        <span className="font-semibold">
                          {Number(entry.value).toFixed(2)}
                        </span>
                        {desc && (
                          <span className="text-gray-400 ml-1">({desc})</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              );
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="total_reward"
            stroke="#111827"
            strokeWidth={2}
            dot
          />
          {termNames.map((name, i) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={TERM_COLORS[i % TERM_COLORS.length]}
              dot
              strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
