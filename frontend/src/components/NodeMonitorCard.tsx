"use client";

import type { NodeResourceSnapshot } from "@/lib/api";
import { timeAgo } from "@/lib/format";
import BarStat from "@/components/BarStat";

export default function NodeMonitorCard({
  node,
}: {
  node: NodeResourceSnapshot;
}) {
  const memPct =
    node.mem_total_mb > 0
      ? ((node.mem_used_mb) / node.mem_total_mb) * 100
      : 0;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-gray-900">{node.node_id}</h3>
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
              node.online
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
            }`}
          >
            {node.online ? "online" : "offline"}
          </span>
        </div>
        <span className="text-xs text-gray-400">{timeAgo(node.timestamp)}</span>
      </div>

      {!node.online ? (
        <p className="text-sm text-red-600">{node.error || "Unreachable"}</p>
      ) : (
        <div className="space-y-3">
          {/* CPU */}
          <div>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-2xl font-bold text-gray-900">
                {node.cpu_percent_avg.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">
                CPU ({node.cpu_count} cores)
              </span>
            </div>
            {node.load_avg.length >= 3 && (
              <p className="text-xs text-gray-500 font-mono">
                Load: {node.load_avg[0].toFixed(2)} / {node.load_avg[1].toFixed(2)} / {node.load_avg[2].toFixed(2)}
              </p>
            )}
          </div>

          {/* Memory */}
          <BarStat
            label="Memory"
            value={`${(node.mem_used_mb / 1024).toFixed(1)} / ${(node.mem_total_mb / 1024).toFixed(1)} GB`}
            pct={memPct}
          />

          {/* GPUs */}
          {node.gpus.length > 0 && (
            <div className="space-y-2 pt-1 border-t border-gray-100">
              {node.gpus.map((gpu) => {
                const gpuMemPct =
                  gpu.memory_total_mb > 0
                    ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
                    : 0;
                return (
                  <div key={gpu.index}>
                    <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                      <span className="font-medium">
                        GPU {gpu.index}: {gpu.name}
                      </span>
                      <span>{gpu.temperature}&deg;C</span>
                    </div>
                    <BarStat
                      label="Util"
                      value={`${gpu.utilization}%`}
                      pct={gpu.utilization}
                    />
                    <div className="mt-1">
                      <BarStat
                        label="VRAM"
                        value={`${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`}
                        pct={gpuMemPct}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {node.error && (
            <p className="text-xs text-amber-600">{node.error}</p>
          )}
        </div>
      )}
    </div>
  );
}
