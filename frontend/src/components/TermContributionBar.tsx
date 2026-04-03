"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import type { EvalResult } from "@/lib/api";

interface Props {
  evalResults: EvalResult[];
  termDescriptions: Record<string, string>;
}

export default function TermContributionBar({
  evalResults,
  termDescriptions,
}: Props) {
  if (evalResults.length === 0) return null;

  // Use the latest eval step
  const latest = evalResults[evalResults.length - 1];
  const terms = latest.reward_terms;
  if (!terms || Object.keys(terms).length === 0) return null;

  const total = Object.values(terms).reduce(
    (sum, v) => sum + Math.abs(v),
    0,
  );

  const data = Object.entries(terms)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([name, value]) => ({
      name,
      value: Number(value.toFixed(2)),
      pct: total > 0 ? ((Math.abs(value) / total) * 100).toFixed(1) : "0",
      description: termDescriptions[name] || "",
    }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-1">
        Term Contributions
      </h2>
      <p className="text-xs text-gray-400 mb-3">
        Final evaluation (step{" "}
        {latest.global_step.toLocaleString()})
      </p>
      <ResponsiveContainer width="100%" height={Math.max(180, data.length * 40)}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
          <XAxis type="number" fontSize={11} />
          <YAxis
            type="category"
            dataKey="name"
            width={180}
            fontSize={11}
            tick={{ fontFamily: "monospace" }}
          />
          <Tooltip
            content={({ payload }) => {
              if (!payload || payload.length === 0) return null;
              const d = payload[0].payload;
              return (
                <div className="bg-white border border-gray-200 shadow-lg rounded-lg p-3 text-xs max-w-xs">
                  <p className="font-mono font-semibold">{d.name}</p>
                  {d.description && (
                    <p className="text-gray-500 mt-0.5">{d.description}</p>
                  )}
                  <p className="mt-1">
                    Value: <span className="font-mono font-semibold">{d.value}</span>
                    {" "}({d.pct}% of total)
                  </p>
                </div>
              );
            }}
          />
          <ReferenceLine x={0} stroke="#9ca3af" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.value >= 0 ? "#10b981" : "#ef4444"}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {/* Term legend with descriptions */}
      <div className="mt-3 space-y-1 border-t border-gray-100 pt-3">
        {data.map((d) => (
          <div key={d.name} className="flex items-start gap-2 text-xs">
            <span
              className={`inline-block w-2 h-2 rounded-full mt-1 shrink-0 ${d.value >= 0 ? "bg-green-500" : "bg-red-500"}`}
            />
            <code className="font-mono text-gray-800 shrink-0">{d.name}</code>
            <span className="text-gray-500">{d.description}</span>
            <span className="ml-auto font-mono text-gray-700 shrink-0">
              {d.value} ({d.pct}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
