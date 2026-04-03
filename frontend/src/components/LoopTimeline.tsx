"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import type { Lineage, LoopIterationSummary } from "@/lib/api";

function scoreColor(score: number | null, passThreshold: number): string {
  if (score === null) return "bg-gray-200 text-gray-600";
  if (score >= passThreshold) return "bg-green-100 text-green-800 border-green-300";
  if (score >= Math.max(0, passThreshold - 0.3)) return "bg-yellow-100 text-yellow-800 border-yellow-300";
  return "bg-red-100 text-red-800 border-red-300";
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
}

function deltaText(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}`;
}

function deltaColor(val: number): string {
  if (val > 0) return "text-green-400";
  if (val < 0) return "text-red-400";
  return "text-gray-400";
}

// ---------------------------------------------------------------------------
// Floating tooltip (portal-based to avoid overflow clipping)
// ---------------------------------------------------------------------------

function FloatingTooltip({
  anchor,
  children,
}: {
  anchor: DOMRect | null;
  children: React.ReactNode;
}) {
  if (!anchor) return null;

  const top = anchor.bottom + 8;
  const left = anchor.left + anchor.width / 2;

  return createPortal(
    <div
      className="fixed z-50 pointer-events-none"
      style={{ top, left, transform: "translateX(-50%)" }}
    >
      <div className="bg-gray-900 text-gray-100 text-xs rounded-xl shadow-2xl p-4 max-w-xs w-max leading-relaxed pointer-events-auto">
        {children}
      </div>
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------
// Node tooltip content
// ---------------------------------------------------------------------------

function NodeTooltipContent({ it, passThreshold, lesson }: { it: LoopIterationSummary; passThreshold: number; lesson?: string }) {
  return (
    <>
      <p className="font-semibold mb-2">Iteration v{it.iteration}</p>
      <div className="space-y-1">
        {it.intent_score != null && (
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Score</span>
            <span className={`font-mono font-medium ${it.intent_score >= passThreshold ? "text-green-400" : it.intent_score >= Math.max(0, passThreshold - 0.3) ? "text-yellow-400" : "text-red-400"}`}>
              {it.intent_score.toFixed(2)}
            </span>
          </div>
        )}
        {it.final_return != null && (
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Final Return</span>
            <span className="font-mono">{it.final_return.toFixed(1)}</span>
          </div>
        )}
        {it.best_checkpoint && (
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Best Checkpoint</span>
            <span className="font-mono">{it.best_checkpoint}</span>
          </div>
        )}
        {it.failure_tags.length > 0 && (
          <div>
            <span className="text-gray-400">Failure Tags</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {it.failure_tags.map((tag) => (
                <span key={tag} className="px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 text-[10px]">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
        {it.diagnosis && (
          <div className="mt-1">
            <span className="text-gray-400">Diagnosis</span>
            <p className="mt-0.5 text-gray-300 line-clamp-3">{it.diagnosis}</p>
          </div>
        )}
        {lesson && (
          <div className="mt-1">
            <span className="text-gray-400">Lesson</span>
            <p className="mt-0.5 text-amber-300">{lesson}</p>
          </div>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Edge tooltip content
// ---------------------------------------------------------------------------

function EdgeTooltipContent({
  prev,
  curr,
}: {
  prev: LoopIterationSummary;
  curr: LoopIterationSummary;
}) {
  const scoreDelta = prev.intent_score != null && curr.intent_score != null
    ? curr.intent_score - prev.intent_score
    : null;
  const returnDelta = prev.final_return != null && curr.final_return != null
    ? curr.final_return - prev.final_return
    : null;

  return (
    <>
      <p className="font-semibold mb-2">v{prev.iteration} &rarr; v{curr.iteration}</p>
      <div className="space-y-1">
        {scoreDelta != null && (
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Score Delta</span>
            <span className={`font-mono font-medium ${deltaColor(scoreDelta)}`}>
              {deltaText(scoreDelta)}
            </span>
          </div>
        )}
        {returnDelta != null && (
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Return Delta</span>
            <span className={`font-mono font-medium ${deltaColor(returnDelta)}`}>
              {deltaText(returnDelta)}
            </span>
          </div>
        )}
        {curr.reward_diff_summary && (
          <div className="mt-1">
            <span className="text-gray-400">Change Summary</span>
            <p className="mt-0.5 text-gray-300 line-clamp-3">{curr.reward_diff_summary}</p>
          </div>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function LoopTimeline({
  iterations,
  activeIteration,
  onSelect,
  passThreshold = 0.7,
  sessionStatus,
  lineage,
}: {
  iterations: LoopIterationSummary[];
  activeIteration: number | null;
  onSelect: (iteration: number) => void;
  passThreshold?: number;
  sessionStatus?: string;
  lineage?: Lineage;
}) {
  const [tooltip, setTooltip] = useState<{
    type: "node" | "edge";
    key: number;
    rect: DOMRect;
  } | null>(null);

  const isRunning = sessionStatus === "running";

  if (iterations.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <h2 className="text-sm font-semibold text-gray-900 mb-3">
          Iteration Timeline
        </h2>
        {isRunning ? (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse" />
            <p className="text-sm text-gray-500">Training first iteration...</p>
          </div>
        ) : (
          <p className="text-sm text-gray-400">Waiting for first iteration...</p>
        )}
      </div>
    );
  }

  // Detect best iteration
  const bestScore = Math.max(...iterations.filter((it) => it.intent_score != null).map((it) => it.intent_score!));

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Iteration Timeline
      </h2>
      <div className="flex items-center gap-2 overflow-x-auto pb-2">
        {iterations.map((it, idx) => {
          const isActive = activeIteration === it.iteration;
          const isLast = idx === iterations.length - 1;
          const isBest = it.intent_score != null && it.intent_score === bestScore;
          return (
            <div key={it.iteration_dir || `it-${it.iteration}`} className="flex items-center">
              <button
                onClick={() => onSelect(it.iteration)}
                onMouseEnter={(e) => setTooltip({ type: "node", key: it.iteration, rect: e.currentTarget.getBoundingClientRect() })}
                onMouseLeave={() => setTooltip(null)}
                className={`relative flex flex-col items-center px-3 py-2 rounded-lg border text-xs font-medium transition-all ${scoreColor(it.intent_score, passThreshold)} ${
                  isActive
                    ? "ring-2 ring-blue-500 ring-offset-1"
                    : "hover:ring-1 hover:ring-gray-300"
                }`}
              >
                <span className="font-semibold">
                  v{it.iteration}
                  {isBest && <span className="ml-0.5 text-amber-500" title="Best score">&#9733;</span>}
                </span>
                <span className="font-mono">
                  {it.intent_score !== null
                    ? it.intent_score.toFixed(2)
                    : "..."}
                </span>
                {it.elapsed_time_s != null && (
                  <span className="mt-0.5 text-[10px] text-gray-500 font-normal">
                    {formatElapsed(it.elapsed_time_s)}
                  </span>
                )}
                {it.intent_score !== null && it.intent_score >= passThreshold && (
                  <span className="mt-0.5">pass</span>
                )}
              </button>
              {!isLast && (
                <div
                  onMouseEnter={(e) => setTooltip({ type: "edge", key: idx, rect: e.currentTarget.getBoundingClientRect() })}
                  onMouseLeave={() => setTooltip(null)}
                  className="w-6 h-2 flex items-center justify-center mx-0.5 cursor-default"
                >
                  <div className="w-full h-px bg-gray-300" />
                </div>
              )}
              {/* In-progress glow node after the last completed iteration */}
              {isLast && isRunning && (
                <>
                  <div className="w-6 h-2 flex items-center justify-center mx-0.5">
                    <div className="w-full h-px bg-gray-300" />
                  </div>
                  <div className="flex flex-col items-center px-3 py-2 rounded-lg border border-blue-300 bg-blue-50 text-xs text-blue-600 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.4)]">
                    <span className="font-semibold">v{it.iteration + 1}</span>
                    <span className="font-mono text-[10px]">training...</span>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* Tooltip */}
      {tooltip?.type === "node" && (() => {
        const it = iterations.find((i) => i.iteration === tooltip.key)!;
        let lesson: string | undefined;
        if (lineage) {
          const suffix = `/iter_${it.iteration}`;
          const entry = Object.entries(lineage.iterations).find(([k]) => k.endsWith(suffix));
          lesson = entry?.[1]?.lesson;
        }
        return (
          <FloatingTooltip anchor={tooltip.rect}>
            <NodeTooltipContent it={it} passThreshold={passThreshold} lesson={lesson} />
          </FloatingTooltip>
        );
      })()}
      {tooltip?.type === "edge" && iterations[tooltip.key + 1] && (
        <FloatingTooltip anchor={tooltip.rect}>
          <EdgeTooltipContent
            prev={iterations[tooltip.key]}
            curr={iterations[tooltip.key + 1]}
          />
        </FloatingTooltip>
      )}
    </div>
  );
}
