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
import type { TrainingEntry } from "@/lib/api";

const PALETTE = [
  "#2563eb", "#dc2626", "#f59e0b", "#10b981", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
];

const COLORS: Record<string, string> = {
  episodic_return: "#2563eb",
  policy_loss: "#dc2626",
  value_loss: "#f59e0b",
  entropy: "#10b981",
  approx_kl: "#8b5cf6",
  clip_fraction: "#ec4899",
  learning_rate: "#06b6d4",
  sps: "#84cc16",
  rollout_time: "#f97316",
  train_time: "#6366f1",
  episodic_return_std: "#f59e0b",
  episodic_return_min: "#ef4444",
  episodic_return_max: "#22c55e",
  episode_length: "#0ea5e9",
  policy_std: "#a855f7",
  grad_norm: "#f43f5e",
  explained_variance: "#14b8a6", // teal
  episodes_per_rollout: "#78716c", // stone
  kl_mean_term: "#7c3aed", // violet
  kl_var_term: "#db2777", // pink
  mean_shift_normalized: "#0891b2", // cyan
};

function getColor(key: string, index: number): string {
  return COLORS[key] || PALETTE[index % PALETTE.length];
}

interface Props {
  data: TrainingEntry[];
  lines?: string[];
  title?: string;
  height?: number;
}

export default function TrainingChart({
  data,
  lines = ["episodic_return"],
  title = "Training Curves",
  height = 300,
}: Props) {
  if (data.length === 0) return null;

  // Downsample for performance (max 200 points)
  const step = Math.max(1, Math.floor(data.length / 200));
  const sampled = data.filter((_, i) => i % step === 0);

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        {title}
      </h2>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={sampled}>
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
          {lines.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={getColor(key, i)}
              dot={false}
              strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
