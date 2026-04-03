import type { LoopIterationSummary } from "@/lib/api";

function tagColor(tag: string): string {
  const warm = ["falling_over", "flipping", "energy_too_high"];
  if (warm.includes(tag)) return "bg-red-50 text-red-700";
  return "bg-orange-50 text-orange-700";
}

function DiagnosisText({ text, className = "" }: { text: string; className?: string }) {
  return (
    <div className={`whitespace-pre-wrap ${className}`}>
      {text}
    </div>
  );
}

export default function JudgmentCard({
  iteration,
}: {
  iteration: LoopIterationSummary;
}) {
  const hasCodeJudge = !!iteration.code_diagnosis;
  const hasVlmJudge = !!iteration.vlm_diagnosis;
  const hasDualJudge = hasCodeJudge && hasVlmJudge;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">
          Iteration {iteration.iteration} Judgment
        </h2>
        <span className="font-mono text-sm font-semibold">
          Score:{" "}
          {iteration.intent_score !== null
            ? iteration.intent_score.toFixed(2)
            : "N/A"}
        </span>
      </div>

      {/* Per-checkpoint scores */}
      {iteration.checkpoint_scores &&
        Object.keys(iteration.checkpoint_scores).length > 1 && (
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">
              Checkpoint Scores
            </p>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(iteration.checkpoint_scores)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([step, score]) => (
                  <span
                    key={step}
                    className={`px-2 py-0.5 rounded text-xs font-mono ${
                      step === iteration.best_checkpoint
                        ? "bg-green-100 text-green-800 font-semibold ring-1 ring-green-300"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    Step {step}: {score.toFixed(2)}
                  </span>
                ))}
            </div>
          </div>
        )}

      {/* Code-Based Judge */}
      {hasCodeJudge && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-gray-500">
            Code-Based Judge
            {iteration.code_score != null && (
              <span className="ml-1 font-mono">
                (Score: {iteration.code_score.toFixed(2)})
              </span>
            )}
          </p>
          <DiagnosisText
            text={iteration.code_diagnosis!}
            className="text-xs font-mono bg-slate-50 p-3 rounded-lg text-gray-700"
          />
        </div>
      )}

      {/* VLM Judge */}
      {hasVlmJudge && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-gray-500">
            VLM Judge
            {iteration.vlm_score != null && (
              <span className="ml-1 font-mono">
                (Score: {iteration.vlm_score.toFixed(2)})
              </span>
            )}
          </p>
          <DiagnosisText
            text={iteration.vlm_diagnosis!}
            className="text-sm bg-blue-50 p-3 rounded-lg text-gray-700"
          />
          {iteration.vlm_criteria && (
            <details className="mt-1">
              <summary className="text-xs font-medium text-gray-400 cursor-pointer hover:text-gray-600">
                Turn 1 — Visual Criteria
              </summary>
              <DiagnosisText
                text={iteration.vlm_criteria}
                className="text-xs bg-blue-50/50 p-3 rounded-lg text-gray-500 mt-1"
              />
            </details>
          )}
          {iteration.criteria_scores && iteration.criteria_scores.length > 0 && (
            <details className="mt-1">
              <summary className="text-xs font-medium text-blue-500 cursor-pointer hover:text-blue-700">
                Turn 2 — Criteria Assessment ({iteration.criteria_scores.length} criteria)
              </summary>
              <div className="mt-1 space-y-1.5">
                {iteration.criteria_scores.map((cs, i) => {
                  const status = (cs.status || "").toLowerCase();
                  const txt = (cs.assessment || "").trim().toLowerCase();
                  let badge = null;
                  if (status === "not_met" || (!status && txt.startsWith("not met"))) {
                    badge = <span className="inline-block text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-100 text-red-700 mr-1.5">NOT MET</span>;
                  } else if (status === "partially_met" || (!status && txt.startsWith("partially"))) {
                    badge = <span className="inline-block text-[10px] font-bold px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 mr-1.5">PARTIAL</span>;
                  } else if (status === "met" || (!status && txt.startsWith("met"))) {
                    badge = <span className="inline-block text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-100 text-green-700 mr-1.5">MET</span>;
                  }
                  return (
                    <div key={i} className="bg-blue-50/30 p-2 rounded border border-blue-100">
                      <p className="text-xs font-semibold text-gray-700">{cs.criterion}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{badge}{cs.assessment || ""}</p>
                    </div>
                  );
                })}
              </div>
            </details>
          )}
        </div>
      )}

      {/* Synthesized / Final / Diagnosis */}
      {iteration.diagnosis && (
        <div className="space-y-2">
          <div className="space-y-1">
            <p className="text-xs font-medium text-gray-500">
              {hasDualJudge
                ? "Synthesized Judgment"
                : hasCodeJudge || hasVlmJudge
                  ? "Final Judgment"
                  : "Diagnosis"}
            </p>
            <DiagnosisText
              text={iteration.diagnosis}
              className="text-sm bg-gray-50 p-3 rounded-lg text-gray-700"
            />
          </div>
        </div>
      )}

      {/* Failure Tags */}
      {iteration.failure_tags.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">
            Failure Tags
          </p>
          <div className="flex flex-wrap gap-1.5">
            {iteration.failure_tags.map((tag) => (
              <span
                key={tag}
                className={`px-2 py-0.5 rounded text-xs font-medium ${tagColor(tag)}`}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Final Return */}
      {iteration.final_return !== null && (
        <div className="text-sm">
          <span className="text-gray-500">Final Return: </span>
          <span className="font-mono font-medium">
            {iteration.final_return.toFixed(1)}
          </span>
        </div>
      )}
    </div>
  );
}
