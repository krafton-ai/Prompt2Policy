"use client";

import { useState } from "react";
import useSWR from "swr";
import {
  fetchResourceStatus,
  fetchCpuUsage,
  fetchGpuUsage,
  fetchNodeResources,
  type ResourceStatus,
  type CpuUsage,
  type GpuUsage,
  type NodeResources,
} from "@/lib/api";
import BarStat from "@/components/BarStat";
import NodeMonitorCard from "@/components/NodeMonitorCard";
import {
  CoreGrid,
  AllocationTable,
  COLOR_HEX,
  adjustColorTone,
} from "@/components/CoreGrid";

function formatRange(cores: number[]): string {
  if (cores.length === 0) return "-";
  const sorted = [...cores].sort((a, b) => a - b);
  const ranges: string[] = [];
  let start = sorted[0];
  let end = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] === end + 1) {
      end = sorted[i];
    } else {
      ranges.push(start === end ? `${start}` : `${start}-${end}`);
      start = sorted[i];
      end = sorted[i];
    }
  }
  ranges.push(start === end ? `${start}` : `${start}-${end}`);
  return ranges.join(", ");
}

function formatMB(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)}G`;
  return `${mb}M`;
}

export default function MonitorPage() {
  const { data } = useSWR<ResourceStatus>("resources", fetchResourceStatus, {
    refreshInterval: 2000,
  });
  const { data: usage } = useSWR<CpuUsage>("cpu-usage", fetchCpuUsage, {
    refreshInterval: 2000,
  });
  const { data: gpuData } = useSWR<GpuUsage>("gpu-usage", fetchGpuUsage, {
    refreshInterval: 2000,
  });
  const { data: nodeData } = useSWR<NodeResources>(
    "node-resources",
    fetchNodeResources,
    { refreshInterval: 5000 },
  );

  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const totalCores = data?.total_cores ?? 64;
  const reservedCores = data?.reserved_cores ?? 2;
  const allocations = data?.allocations ?? [];
  const processes = usage?.processes ?? [];

  const allocatable = totalCores - reservedCores;
  const avgUsage = usage?.avg ?? 0;

  // Derive used cores from both CPUManager allocations and psutil processes
  // (CPUManager state is lost on server reload, so psutil is the fallback)
  const allocUsedCores = allocations.reduce((sum, a) => sum + a.cores.length, 0);
  const procUsedCores = new Set(processes.flatMap((p) => p.cores)).size;
  const usedCores = Math.max(allocUsedCores, procUsedCores);
  const sessionCount = Math.max(data?.active_runs ?? 0, processes.length);

  // Count how many cores are actively busy (>10% usage)
  const busyCores = usage?.per_core.filter((u) => u > 10).length ?? 0;

  // Total run count across all sessions
  const totalRunCount = processes.reduce((s, p) => s + p.runs.length, 0);

  // Build session color map (same logic as CoreGrid)
  const sessionColorMap = new Map<string, number>();
  allocations.forEach((a) => {
    if (!sessionColorMap.has(a.run_id))
      sessionColorMap.set(a.run_id, sessionColorMap.size % COLOR_HEX.length);
  });
  processes.forEach((p) => {
    if (!sessionColorMap.has(p.session_id))
      sessionColorMap.set(p.session_id, sessionColorMap.size % COLOR_HEX.length);
  });

  const toggleExpand = (sid: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(sid)) next.delete(sid);
      else next.add(sid);
      return next;
    });
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Monitor</h1>

      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-gray-600 mb-6">
        <span>
          Allocated{" "}
          <span className="font-semibold text-gray-900">
            {usedCores}/{allocatable}
          </span>{" "}
          cores ({reservedCores} reserved)
        </span>
        <span>
          <span className="font-semibold text-gray-900">{sessionCount}</span>{" "}
          active session{sessionCount !== 1 ? "s" : ""}
        </span>
        <span>
          Avg usage{" "}
          <span className="font-semibold text-gray-900">
            {avgUsage.toFixed(1)}%
          </span>
        </span>
        <span>
          <span className="font-semibold text-gray-900">{busyCores}</span>{" "}
          busy cores (&gt;10%)
        </span>
        {processes.length > 0 && (
          <span>
            <span className="font-semibold text-gray-900">
              {processes.length}
            </span>{" "}
            session{processes.length !== 1 ? "s" : ""}
            {totalRunCount > 0 && (
              <>
                {" / "}
                <span className="font-semibold text-gray-900">
                  {totalRunCount}
                </span>{" "}
                run{totalRunCount !== 1 ? "s" : ""}
              </>
            )}
          </span>
        )}
      </div>

      {/* System Memory */}
      {usage?.memory && (
        <div className="mb-6 max-w-md">
          <BarStat
            label="Memory"
            value={`${formatMB(usage.memory.used_mb)} / ${formatMB(usage.memory.total_mb)} (${usage.memory.percent.toFixed(1)}%)`}
            pct={usage.memory.percent}
          />
        </div>
      )}

      <div className="flex flex-col lg:flex-row gap-8">
        <div>
          <h2 className="text-sm font-medium text-gray-500 mb-3">
            Core Grid ({totalCores} cores)
          </h2>
          <CoreGrid
            totalCores={totalCores}
            reservedCores={reservedCores}
            allocations={allocations}
            cpuUsage={usage?.per_core}
            processes={processes}
          />
          <div className="flex flex-wrap gap-4 mt-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded bg-gray-300" />
              Reserved
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded bg-gray-50 border border-gray-200" />
              Free
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded bg-blue-400" />
              Session
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded"
                style={{ backgroundColor: "rgba(74,222,128,0.55)" }}
              />
              Low
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded"
                style={{ backgroundColor: "rgba(250,204,21,0.6)" }}
              />
              Medium
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded"
                style={{ backgroundColor: "rgba(248,113,113,0.7)" }}
              />
              High
            </span>
          </div>
        </div>

        <div className="flex-1 min-w-0">
          {processes.length > 0 && (
            <>
              <h2 className="text-sm font-medium text-gray-500 mb-3">
                Active Sessions
              </h2>
              <div className="overflow-x-auto mb-6">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-gray-200">
                      <th className="pb-2 pr-4 font-medium">Session / Run</th>
                      <th className="pb-2 pr-4 font-medium">PID</th>
                      <th className="pb-2 pr-4 font-medium">Cores</th>
                      <th className="pb-2 font-medium">Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {processes.map((p) => {
                      const colorIdx =
                        sessionColorMap.get(p.session_id) ?? 0;
                      const baseColor = COLOR_HEX[colorIdx];
                      const hasRuns = p.runs.length > 0;
                      const isExpanded = expanded.has(p.session_id);

                      return (
                        <SessionRows
                          key={p.session_id}
                          sessionId={p.session_id}
                          pid={p.pid}
                          cores={p.cores}
                          runs={p.runs}
                          baseColor={baseColor}
                          hasRuns={hasRuns}
                          isExpanded={isExpanded}
                          onToggle={() => toggleExpand(p.session_id)}
                        />
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h2 className="text-sm font-medium text-gray-500 mb-3">
            CPUManager Allocations
          </h2>
          <AllocationTable allocations={allocations} />
        </div>
      </div>

      {/* GPU Monitor */}
      {gpuData && gpuData.gpus.length > 0 && (
        <div className="mt-10">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            GPU Monitor
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {gpuData.gpus.map((gpu) => (
              <GpuCard
                key={gpu.index}
                gpu={gpu}
                sessionColorMap={sessionColorMap}
              />
            ))}
          </div>
        </div>
      )}

      {/* Remote Nodes */}
      {nodeData && nodeData.nodes.length > 0 && (
        <div className="mt-10">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Remote Nodes
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {nodeData.nodes.map((node) => (
              <NodeMonitorCard key={node.node_id} node={node} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function GpuCard({
  gpu,
  sessionColorMap,
}: {
  gpu: GpuUsage["gpus"][number];
  sessionColorMap: Map<string, number>;
}) {
  const memPct =
    gpu.memory_total_mb > 0
      ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
      : 0;
  const powerPct =
    gpu.power_limit_w > 0
      ? (gpu.power_draw_w / gpu.power_limit_w) * 100
      : 0;

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-semibold text-gray-900">
          GPU {gpu.index}: {gpu.name}
        </span>
        <span className="text-sm text-gray-500">{gpu.temperature}°C</span>
      </div>

      <div className="space-y-2.5">
        <BarStat
          label="GPU Util"
          value={`${gpu.utilization}%`}
          pct={gpu.utilization}
        />
        <BarStat
          label="Memory"
          value={`${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`}
          pct={memPct}
        />
        <BarStat
          label="Power"
          value={`${gpu.power_draw_w.toFixed(0)} / ${gpu.power_limit_w.toFixed(0)} W`}
          pct={powerPct}
        />
      </div>

      {gpu.processes.length > 0 && (
        <div className="mt-3 border-t border-gray-100 pt-2">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1.5">
            <span>Processes ({gpu.processes.length})</span>
            <span className="font-mono">
              {gpu.processes.reduce((s, p) => s + p.gpu_memory_mb, 0)} MiB
            </span>
          </div>
          <div className="space-y-1">
            {[...gpu.processes]
              .sort((a, b) =>
                (a.run_id || a.process_name).localeCompare(
                  b.run_id || b.process_name,
                ),
              )
              .map((proc) => {
                const colorIdx = sessionColorMap.get(proc.session_id);
                return (
                  <div
                    key={proc.pid}
                    className="flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-1.5 min-w-0">
                      {colorIdx !== undefined ? (
                        <span
                          className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                          style={{
                            backgroundColor: COLOR_HEX[colorIdx],
                          }}
                        />
                      ) : (
                        <span className="inline-block w-2 h-2 flex-shrink-0" />
                      )}
                      <span className="font-mono text-gray-400 flex-shrink-0">
                        {proc.pid}
                      </span>
                      <span className="text-gray-700 truncate max-w-[180px]">
                        {proc.run_id || proc.session_id || proc.process_name}
                      </span>
                    </div>
                    <span className="font-mono text-gray-600 flex-shrink-0 ml-2">
                      {proc.gpu_memory_mb} MiB
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}


function SessionRows({
  sessionId,
  pid,
  cores,
  runs,
  baseColor,
  hasRuns,
  isExpanded,
  onToggle,
}: {
  sessionId: string;
  pid: number;
  cores: number[];
  runs: { run_id: string; pid: number; cores: number[] }[];
  baseColor: string;
  hasRuns: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr className="border-b border-gray-100">
        <td className="py-2 pr-4">
          <div className="flex items-center gap-2">
            {hasRuns ? (
              <button
                onClick={onToggle}
                className="text-gray-400 hover:text-gray-600 w-4 text-center flex-shrink-0"
              >
                {isExpanded ? "\u25BC" : "\u25B6"}
              </button>
            ) : (
              <span className="w-4 flex-shrink-0" />
            )}
            <span
              className="inline-block w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: baseColor }}
            />
            <span className="font-mono text-gray-800 truncate max-w-[220px]">
              {sessionId}
            </span>
            {hasRuns && (
              <span className="text-xs text-gray-400">
                ({runs.length} run{runs.length !== 1 ? "s" : ""})
              </span>
            )}
          </div>
        </td>
        <td className="py-2 pr-4 text-gray-600 font-mono">{pid}</td>
        <td className="py-2 pr-4 text-gray-600">{cores.length}</td>
        <td className="py-2 font-mono text-gray-600 text-xs">
          {formatRange(cores)}
        </td>
      </tr>
      {isExpanded &&
        runs.map((r, idx) => (
          <tr
            key={r.run_id}
            className="border-b border-gray-50 bg-gray-50/50"
          >
            <td className="py-1.5 pr-4">
              <div className="flex items-center gap-2 pl-6">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor: adjustColorTone(
                      baseColor,
                      idx,
                      runs.length,
                    ),
                  }}
                />
                <span className="font-mono text-gray-600 text-xs truncate max-w-[200px]">
                  {r.run_id}
                </span>
              </div>
            </td>
            <td className="py-1.5 pr-4 text-gray-500 font-mono text-xs">
              {r.pid}
            </td>
            <td className="py-1.5 pr-4 text-gray-500 text-xs">
              {r.cores.length}
            </td>
            <td className="py-1.5 font-mono text-gray-500 text-xs">
              {formatRange(r.cores)}
            </td>
          </tr>
        ))}
    </>
  );
}
