"use client";

import type { StageDetail } from "@/lib/api";

const STAGE_STATUS_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  pending: { bg: "bg-gray-50", text: "text-gray-500", border: "border-gray-200" },
  running: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-300" },
  completed: { bg: "bg-green-50", text: "text-green-700", border: "border-green-300" },
  gate_passed: { bg: "bg-green-50", text: "text-green-700", border: "border-green-300" },
  gate_failed: { bg: "bg-red-50", text: "text-red-700", border: "border-red-300" },
  skipped: { bg: "bg-gray-50", text: "text-gray-400", border: "border-gray-100" },
};

export default function StagePipeline({
  stages,
  onStageClick,
  activeStage,
  hoveredStage,
  onHover,
}: {
  stages: StageDetail[];
  onStageClick?: (stage: number) => void;
  activeStage?: number | null;
  hoveredStage?: number | null;
  onHover?: (stage: number | null) => void;
}) {
  const tooltipStage = stages.find((s) => s.stage === hoveredStage);

  return (
    <div className="relative">
      <div className="flex flex-wrap gap-1.5 pb-2">
        {stages.map((s) => {
          const styles = STAGE_STATUS_STYLES[s.status] ?? STAGE_STATUS_STYLES.pending;
          const isActive = activeStage === s.stage;
          const scoreText = s.gate_result
            ? s.gate_result.avg_score.toFixed(2)
            : "";
          return (
            <button
              key={s.stage}
              type="button"
              onClick={() => onStageClick?.(s.stage)}
              onMouseEnter={() => onHover?.(s.stage)}
              onMouseLeave={() => onHover?.(null)}
              className={`relative w-9 h-9 rounded-lg border-2 flex flex-col items-center justify-center transition-all ${styles.bg} ${styles.border} ${
                isActive ? "ring-2 ring-blue-400 ring-offset-1" : ""
              } ${s.status === "skipped" ? "opacity-40" : "hover:shadow-sm cursor-pointer"}`}
              title={`${s.name}: ${s.case_count} cases — ${s.status.replace("_", " ")}${scoreText ? ` (avg: ${scoreText})` : ""}`}
            >
              <span className={`text-[10px] font-bold leading-none ${styles.text}`}>
                {s.stage}
              </span>
              {s.gate_result && (
                <span
                  className={`text-[8px] font-mono leading-none mt-0.5 ${
                    s.gate_result.passed ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {s.gate_result.avg_score.toFixed(1)}
                </span>
              )}
              {s.status === "running" && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              )}
            </button>
          );
        })}
      </div>
      {/* Tooltip for hovered stage */}
      {tooltipStage && (
        <div className="mt-1 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg inline-block">
          <span className="font-medium">{tooltipStage.name}</span>
          {" \u00B7 "}
          {tooltipStage.case_count} cases
          {" \u00B7 "}
          {tooltipStage.status.replace("_", " ")}
          {tooltipStage.gate_threshold > 0 && (
            <span>
              {" \u00B7 gate \u2265 "}
              {tooltipStage.gate_threshold}
            </span>
          )}
          {tooltipStage.gate_result && (
            <span>
              {" \u00B7 avg: "}
              <span className={tooltipStage.gate_result.passed ? "text-green-300" : "text-red-300"}>
                {tooltipStage.gate_result.avg_score.toFixed(3)}
              </span>
              {" / "}
              {tooltipStage.gate_result.threshold}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
