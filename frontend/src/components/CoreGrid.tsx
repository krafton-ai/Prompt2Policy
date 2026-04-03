"use client";

const COLORS = [
  "bg-blue-400",
  "bg-emerald-400",
  "bg-amber-400",
  "bg-rose-400",
  "bg-violet-400",
  "bg-cyan-400",
  "bg-orange-400",
  "bg-pink-400",
  "bg-lime-400",
  "bg-indigo-400",
];

export const COLOR_HEX = [
  "#60a5fa",
  "#34d399",
  "#fbbf24",
  "#fb7185",
  "#a78bfa",
  "#22d3ee",
  "#fb923c",
  "#f472b6",
  "#a3e635",
  "#818cf8",
];

/** Map utilization 0-100 to a green→yellow→red color. */
function utilizationColor(pct: number): string {
  if (pct < 30) return "rgba(74,222,128,0.55)"; // green
  if (pct < 70) return "rgba(250,204,21,0.6)"; // yellow
  return "rgba(248,113,113,0.7)"; // red
}

/** Adjust hex color lightness for run-level tone variation within a session.
 *  First run is 20% darker than base, last run is 50% toward white. */
export function adjustColorTone(hex: string, step: number, total: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  if (total <= 1) return hex;
  // t: 0 (first run) → 1 (last run)
  // shift: -0.2 (darker) → +0.5 (lighter toward white)
  const t = step / (total - 1);
  const shift = -0.2 + t * 0.7;
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  if (shift < 0) {
    const f = 1 + shift;
    return `rgb(${clamp(r * f)},${clamp(g * f)},${clamp(b * f)})`;
  }
  return `rgb(${clamp(r + (255 - r) * shift)},${clamp(g + (255 - g) * shift)},${clamp(b + (255 - b) * shift)})`;
}

interface RunInfo {
  session_id: string;
  run_id: string;
  cores: number[];
}

interface ProcessWithRuns {
  session_id: string;
  pid: number;
  cores: number[];
  runs: { run_id: string; pid: number; cores: number[] }[];
}

interface CoreGridProps {
  totalCores: number;
  reservedCores: number;
  allocations: { run_id: string; cores: number[] }[];
  /** Per-core CPU utilization percentages (0-100). Length should match totalCores. */
  cpuUsage?: number[];
  /** Process→core mapping from OS-level cpu_affinity with run-level detail. */
  processes?: ProcessWithRuns[];
}

export function CoreGrid({
  totalCores,
  reservedCores,
  allocations,
  cpuUsage,
  processes,
}: CoreGridProps) {
  // Build core -> owner map from CPUManager allocations
  const coreOwnerMap = new Map<number, string>(); // core -> session/run id
  for (const alloc of allocations) {
    for (const core of alloc.cores) {
      coreOwnerMap.set(core, alloc.run_id);
    }
  }

  // Build core -> run detail map from process tree
  const coreRunMap = new Map<number, RunInfo>();
  if (processes) {
    for (const proc of processes) {
      // Map individual run cores first (more specific)
      for (const run of proc.runs) {
        for (const core of run.cores) {
          coreRunMap.set(core, {
            session_id: proc.session_id,
            run_id: run.run_id,
            cores: run.cores,
          });
        }
      }
      // Fill remaining session-level cores
      for (const core of proc.cores) {
        if (!coreOwnerMap.has(core)) {
          coreOwnerMap.set(core, proc.session_id);
        }
        if (!coreRunMap.has(core)) {
          coreRunMap.set(core, {
            session_id: proc.session_id,
            run_id: "",
            cores: proc.cores,
          });
        }
      }
    }
  }

  // Build session -> color index map
  const sessionColorMap = new Map<string, number>();
  allocations.forEach((a) => {
    if (!sessionColorMap.has(a.run_id))
      sessionColorMap.set(a.run_id, sessionColorMap.size % COLORS.length);
  });
  processes?.forEach((p) => {
    if (!sessionColorMap.has(p.session_id))
      sessionColorMap.set(p.session_id, sessionColorMap.size % COLORS.length);
  });

  // Build run -> tone step within each session
  const sessionRunIndex = new Map<string, Map<string, number>>();
  processes?.forEach((p) => {
    if (p.runs.length > 0) {
      const runMap = new Map<string, number>();
      p.runs.forEach((r, idx) => runMap.set(r.run_id, idx));
      sessionRunIndex.set(p.session_id, runMap);
    }
  });

  return (
    <div className="grid grid-cols-8 gap-1.5 max-w-md">
      {Array.from({ length: totalCores }, (_, i) => {
        const isReserved = i >= totalCores - reservedCores;
        const ownerId = coreOwnerMap.get(i);
        const runInfo = coreRunMap.get(i);
        const usage = cpuUsage?.[i];

        // Determine color: session base color, with tone variation per run
        const sessionId = runInfo?.session_id ?? ownerId;
        const colorIdx =
          sessionId !== undefined
            ? (sessionColorMap.get(sessionId) ?? 0) % COLORS.length
            : -1;

        let cellStyle: React.CSSProperties = {};
        let cellClass =
          "relative w-full aspect-square rounded-md text-[10px] flex items-center justify-center font-mono transition-colors ";

        if (isReserved) {
          cellClass += "bg-gray-300 text-gray-500";
        } else if (ownerId !== undefined || runInfo) {
          // Use tone variation if this core belongs to a known run within a session
          const sid = runInfo?.session_id;
          const rid = runInfo?.run_id;
          const runSteps = sid ? sessionRunIndex.get(sid) : undefined;
          if (rid && runSteps && runSteps.has(rid)) {
            const step = runSteps.get(rid)!;
            cellStyle = {
              backgroundColor: adjustColorTone(
                COLOR_HEX[colorIdx],
                step,
                runSteps.size,
              ),
            };
            cellClass += "text-white cursor-default";
          } else {
            cellClass += `${COLORS[colorIdx]} text-white cursor-default`;
          }
        } else {
          cellClass += "bg-gray-50 border border-gray-200 text-gray-400";
        }

        // Build tooltip with run detail
        const tooltipLines: string[] = [];
        if (runInfo?.session_id) {
          tooltipLines.push(runInfo.session_id);
          if (runInfo.run_id) {
            tooltipLines.push(
              `${runInfo.run_id} → cores [${runInfo.cores.join(",")}]`,
            );
          }
        } else if (ownerId !== undefined) {
          tooltipLines.push(ownerId);
        }
        if (usage !== undefined) tooltipLines.push(`${Math.round(usage)}%`);
        const tooltip = tooltipLines.length > 0 ? tooltipLines : undefined;

        return (
          <div key={i} className="group relative">
            <div className={cellClass} style={cellStyle}>
              {/* Utilization overlay for non-reserved, non-allocated cores */}
              {!isReserved &&
                ownerId === undefined &&
                !runInfo &&
                usage !== undefined &&
                usage > 5 && (
                  <div
                    className="absolute inset-0 rounded-md"
                    style={{ backgroundColor: utilizationColor(usage) }}
                  />
                )}
              <span className="relative z-10">{i}</span>
              {isReserved && (
                <svg
                  className="absolute inset-0 w-full h-full rounded-md"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <pattern
                      id="hatch"
                      width="6"
                      height="6"
                      patternUnits="userSpaceOnUse"
                      patternTransform="rotate(45)"
                    >
                      <line
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="6"
                        stroke="#9ca3af"
                        strokeWidth="1.5"
                      />
                    </pattern>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#hatch)" />
                </svg>
              )}
              {/* Usage badge on allocated cores */}
              {(ownerId !== undefined || runInfo) &&
                usage !== undefined && (
                  <span className="absolute -top-1.5 -right-1.5 z-20 bg-gray-900/80 text-white text-[10px] rounded px-1 py-0.5 leading-none font-semibold">
                    {Math.round(usage)}
                  </span>
                )}
            </div>
            {tooltip && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-30 whitespace-nowrap bg-gray-900 text-white text-xs rounded px-2 py-1 pointer-events-none">
                {tooltip.map((line, li) => (
                  <div key={li}>{line}</div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function AllocationTable({
  allocations,
}: {
  allocations: { run_id: string; cores: number[] }[];
}) {
  const runColorMap = new Map<string, number>();
  allocations.forEach((a) => {
    if (!runColorMap.has(a.run_id)) {
      runColorMap.set(a.run_id, runColorMap.size % COLORS.length);
    }
  });

  if (allocations.length === 0) {
    return (
      <p className="text-sm text-gray-400 mt-4">No active allocations.</p>
    );
  }

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

  return (
    <div className="mt-6 overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b border-gray-200">
            <th className="pb-2 pr-4 font-medium">Run</th>
            <th className="pb-2 pr-4 font-medium">Cores</th>
            <th className="pb-2 font-medium">Range</th>
          </tr>
        </thead>
        <tbody>
          {allocations.map((a) => {
            const colorIdx = runColorMap.get(a.run_id)!;
            return (
              <tr key={a.run_id} className="border-b border-gray-100">
                <td className="py-2 pr-4 flex items-center gap-2">
                  <span
                    className="inline-block w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: COLOR_HEX[colorIdx] }}
                  />
                  <span className="font-mono text-gray-800 truncate max-w-[200px]">
                    {a.run_id}
                  </span>
                </td>
                <td className="py-2 pr-4 text-gray-600">{a.cores.length}</td>
                <td className="py-2 font-mono text-gray-600 text-xs">
                  {formatRange(a.cores)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
