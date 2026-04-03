"use client";

import { useState, useMemo, useCallback } from "react";
import { staticUrl, type CriteriaScore, type SynthesisToolTrace } from "@/lib/api";
import HumanScoreInput, { type HumanLabel } from "@/components/HumanScoreInput";

function scoreColor(score: number, passThreshold: number): string {
  if (score >= passThreshold) return "text-green-700";
  if (score >= passThreshold * 0.6) return "text-yellow-700";
  return "text-red-600";
}

/** Parse eval video URL into step and suffix (p10/median/p90/ep0/etc). */
function parseVideoUrl(url: string): {
  step: string;
  suffix: string;
  rolloutKey: string | null;
} {
  const label =
    url
      .split("/")
      .pop()
      ?.replace(".mp4", "")
      .replace("eval_", "") ?? "";
  // Extract numeric step prefix
  const stepMatch = label.match(/^(\d+)/);
  const step = stepMatch ? stepMatch[1] : label;
  // Extract suffix after step (e.g. "median", "p10", "p90", "ep0")
  const suffixMatch = label.match(/^\d+_(.+)$/);
  const suffix = suffixMatch ? suffixMatch[1] : "";
  // Legacy rollout key for ep-based naming
  const rolloutKey = suffix ? label : null;
  return { step, suffix, rolloutKey };
}

function parseRunId(runId: string): { config: string; seed: string } | null {
  const m = runId.match(/^(.+)_seed_(\d+)$/);
  if (!m) return null;
  return { config: m[1], seed: m[2] };
}

/** Resolve score: per-rollout first, then checkpoint. */
function resolveScore(
  url: string,
  rolloutScores?: Record<string, number>,
  checkpointScores?: Record<string, number>,
): number | undefined {
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutScores?.[rolloutKey] !== undefined)
    return rolloutScores[rolloutKey];
  return checkpointScores?.[step];
}

/** Resolve diagnosis: per-rollout first, then checkpoint. */
function resolveDiagnosis(
  url: string,
  rolloutDiagnoses?: Record<string, string>,
  checkpointDiagnoses?: Record<string, string>,
): string | undefined {
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutDiagnoses?.[rolloutKey])
    return rolloutDiagnoses[rolloutKey];
  return checkpointDiagnoses?.[step];
}

/** Resolve code diagnosis: per-rollout first, then checkpoint. */
function resolveCodeDiagnosis(
  url: string,
  checkpointCodeDiagnoses?: Record<string, string>,
  rolloutCodeDiagnoses?: Record<string, string>,
): string | undefined {
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutCodeDiagnoses?.[rolloutKey])
    return rolloutCodeDiagnoses[rolloutKey];
  return checkpointCodeDiagnoses?.[step];
}

/** Resolve VLM diagnosis: per-rollout first, then checkpoint. */
function resolveVlmDiagnosis(
  url: string,
  checkpointVlmDiagnoses?: Record<string, string>,
  rolloutVlmDiagnoses?: Record<string, string>,
): string | undefined {
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutVlmDiagnoses?.[rolloutKey])
    return rolloutVlmDiagnoses[rolloutKey];
  return checkpointVlmDiagnoses?.[step];
}

/** Resolve per-judge score: per-rollout first, then checkpoint. */
function resolveJudgeScore(
  url: string,
  checkpointScores?: Record<string, number>,
  rolloutScores?: Record<string, number>,
): number | undefined {
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutScores?.[rolloutKey] !== undefined)
    return rolloutScores[rolloutKey];
  return checkpointScores?.[step];
}

/** Resolve a preview URL from per-rollout map, falling back to checkpoint key. */
function resolvePreviewUrl(
  url: string,
  rolloutUrls?: Record<string, string>,
): string | undefined {
  if (!rolloutUrls) return undefined;
  const { step, rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutUrls[rolloutKey])
    return rolloutUrls[rolloutKey];
  return rolloutUrls[step];
}

/** Resolve per-rollout criteria scores, falling back to iteration-level. */
function resolveCriteriaScores(
  url: string,
  rolloutCriteriaScores?: Record<string, CriteriaScore[]>,
  iterationCriteriaScores?: CriteriaScore[],
): CriteriaScore[] | undefined {
  const { rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutCriteriaScores?.[rolloutKey])
    return rolloutCriteriaScores[rolloutKey];
  return iterationCriteriaScores;
}

/** Resolve synthesis tool call traces for a video URL (per-rollout only). */
function resolveSynthesisTraces(
  url: string,
  rolloutSynthesisTraces?: Record<string, SynthesisToolTrace[]>,
): SynthesisToolTrace[] | undefined {
  if (!rolloutSynthesisTraces) return undefined;
  const { rolloutKey } = parseVideoUrl(url);
  if (rolloutKey && rolloutSynthesisTraces[rolloutKey]) return rolloutSynthesisTraces[rolloutKey];
  return undefined;
}

interface Props {
  urls: string[];
  checkpointScores?: Record<string, number>;
  checkpointDiagnoses?: Record<string, string>;
  rolloutScores?: Record<string, number>;
  rolloutDiagnoses?: Record<string, string>;
  rolloutCodeDiagnoses?: Record<string, string>;
  rolloutVlmDiagnoses?: Record<string, string>;
  rolloutCodeScores?: Record<string, number>;
  rolloutVlmScores?: Record<string, number>;
  checkpointCodeDiagnoses?: Record<string, string>;
  checkpointVlmDiagnoses?: Record<string, string>;
  checkpointCodeScores?: Record<string, number>;
  checkpointVlmScores?: Record<string, number>;
  bestCheckpoint?: string;
  passThreshold?: number;
  sourceRunId?: string;
  sourceReturn?: number | null;
  title?: string;
  rolloutSynthesisTraces?: Record<string, SynthesisToolTrace[]>;
  rolloutCriteriaScores?: Record<string, CriteriaScore[]>;
  rolloutVlmPreviewUrls?: Record<string, string>;
  rolloutMotionPreviewUrls?: Record<string, string>;
  vlmFps?: number;
  vlmCriteria?: string;
  criteriaScores?: CriteriaScore[];
  sessionId?: string;
  iteration?: number;
  labelingEnabled?: boolean;
  labelingAnnotator?: string;
  humanLabels?: Record<string, HumanLabel> | null;
}

/** Render text with proper line breaks. */
function DiagnosisText({ text, className }: { text: string; className?: string }) {
  const lines = text.split("\n");
  return (
    <p className={className}>
      {lines.map((line, i) => (
        <span key={i}>
          {line}
          {i < lines.length - 1 && <br />}
        </span>
      ))}
    </p>
  );
}

type JudgeVariant = "code" | "vlm" | "synthesized";

const JUDGE_STYLES: Record<JudgeVariant, { bg: string; label: string; text: string; mono: boolean; name: string }> = {
  code:        { bg: "bg-slate-50",  label: "text-slate-400",  text: "text-slate-700",  mono: true,  name: "Code Judge" },
  vlm:         { bg: "bg-blue-50",   label: "text-blue-400",   text: "text-blue-700",   mono: false, name: "VLM Judge" },
  synthesized: { bg: "bg-purple-50", label: "text-purple-400", text: "text-purple-700", mono: false, name: "Synthesized" },
};

/** Collapsible judge card. */
function JudgeCard({
  variant,
  score,
  diagnosis,
  passThreshold,
  vlmPreviewUrl,
  motionPreviewUrl,
  vlmCriteria,
  criteriaScores,
  vlmFps,
}: {
  variant: JudgeVariant;
  score?: number;
  diagnosis: string;
  passThreshold: number;
  vlmPreviewUrl?: string;
  motionPreviewUrl?: string;
  vlmCriteria?: string;
  criteriaScores?: CriteriaScore[];
  vlmFps?: number;
}) {
  const s = JUDGE_STYLES[variant];
  return (
    <details className={`group ${s.bg} rounded overflow-hidden`}>
      <summary className="list-none px-1.5 py-1 cursor-pointer select-none text-[10px] flex items-center justify-between [&::-webkit-details-marker]:hidden">
        <span>
          <span className="text-gray-400 mr-1 inline-block transition-transform group-open:rotate-90">&#9656;</span>
          <span className={`font-medium ${s.label}`}>{s.name}</span>
        </span>
        {score !== undefined && (
          <span className={`font-mono font-semibold ${scoreColor(score, passThreshold)}`}>
            {score.toFixed(2)}
          </span>
        )}
      </summary>
      <div className="px-1.5 pb-1.5">
        {variant === "vlm" && vlmPreviewUrl && (
          <div className="mb-1.5">
            <p className="text-[9px] text-blue-400 mb-0.5">VLM input ({vlmFps ?? "?"}fps center-of-interval)</p>
            <video
              src={staticUrl(vlmPreviewUrl)}
              controls
              preload="metadata"
              className="w-full rounded bg-black"
            />
          </div>
        )}
        {variant === "vlm" && motionPreviewUrl && (
          <div className="mb-1.5">
            <p className="text-[9px] text-orange-400 mb-0.5">Motion Trail ({vlmFps ?? "?"}fps)</p>
            <video
              src={staticUrl(motionPreviewUrl)}
              controls
              preload="metadata"
              className="w-full rounded bg-black"
            />
          </div>
        )}
        {variant === "vlm" && vlmCriteria && (
          <details className="mt-1.5">
            <summary className="list-none cursor-pointer select-none text-[10px] text-gray-400 hover:text-gray-600 [&::-webkit-details-marker]:hidden">
              <span className="text-gray-400 mr-1 inline-block transition-transform [details[open]>&]:rotate-90">&#9656;</span>
              Turn 1 — Visual Criteria
            </summary>
            <DiagnosisText
              text={vlmCriteria}
              className="text-[10px] leading-relaxed text-gray-400 mt-1 bg-blue-50/50 p-1.5 rounded"
            />
          </details>
        )}
        {variant === "vlm" && criteriaScores && criteriaScores.length > 0 && (
          <details className="mt-1.5">
            <summary className="list-none cursor-pointer select-none text-[10px] text-blue-500 hover:text-blue-700 font-medium [&::-webkit-details-marker]:hidden">
              <span className="text-blue-400 mr-1 inline-block transition-transform [details[open]>&]:rotate-90">&#9656;</span>
              Turn 2 — Criteria Assessment ({criteriaScores.length})
            </summary>
            <div className="mt-1 space-y-1">
              {criteriaScores.map((cs, i) => {
                const st = (cs.status || "").toLowerCase();
                let badge = null;
                if (st === "not_met") badge = <span className="inline-block text-[9px] font-bold px-1 py-0.5 rounded bg-red-100 text-red-700 mr-1">NOT MET</span>;
                else if (st === "partially_met") badge = <span className="inline-block text-[9px] font-bold px-1 py-0.5 rounded bg-yellow-100 text-yellow-700 mr-1">PARTIAL</span>;
                else if (st === "met") badge = <span className="inline-block text-[9px] font-bold px-1 py-0.5 rounded bg-green-100 text-green-700 mr-1">MET</span>;
                return (
                  <div key={i} className="bg-blue-50/40 p-1.5 rounded border border-blue-100">
                    <p className="text-[10px] font-semibold text-gray-700">{badge}{cs.criterion}</p>
                    <p className="text-[10px] text-gray-500 mt-0.5">{cs.assessment}</p>
                  </div>
                );
              })}
            </div>
          </details>
        )}
        <DiagnosisText
          text={diagnosis}
          className={`text-xs leading-relaxed ${s.text}${s.mono ? " font-mono" : ""}`}
        />
      </div>
    </details>
  );
}

/** Collapsible amber card showing agentic synthesis tool call traces. */
function SynthesisToolsCard({ traces }: { traces: SynthesisToolTrace[] }) {
  if (!traces || traces.length === 0) return null;
  return (
    <details className="group bg-amber-50 rounded overflow-hidden">
      <summary className="list-none px-1.5 py-1 cursor-pointer select-none text-[10px] flex items-center justify-between [&::-webkit-details-marker]:hidden">
        <span>
          <span className="text-gray-400 mr-1 inline-block transition-transform group-open:rotate-90">
            &#9656;
          </span>
          <span className="font-medium text-amber-600">
            Synthesis Tools ({traces.length} call
            {traces.length > 1 ? "s" : ""})
          </span>
        </span>
      </summary>
      <div className="px-1.5 pb-1.5 space-y-1.5">
        {traces.map((tc, i) => (
          <div key={i} className="text-[10px]">
            <span className="inline-block bg-amber-200 text-amber-800 rounded px-1 py-0.5 font-mono font-medium">
              {tc.tool_name}
            </span>
            <p className="text-amber-700 mt-0.5 break-words">
              {tc.tool_name === "reask_vlm"
                ? <>
                    {tc.input.question}
                    {(tc.input.start_time != null || tc.input.end_time != null || tc.input.fps != null) && (
                      <span className="ml-1 text-amber-500 font-mono">
                        {(() => {
                          const parts: string[] = [];
                          if (tc.input.start_time != null && tc.input.end_time != null)
                            parts.push(`${tc.input.start_time}s–${tc.input.end_time}s`);
                          else if (tc.input.start_time != null)
                            parts.push(`from ${tc.input.start_time}s`);
                          else if (tc.input.end_time != null)
                            parts.push(`to ${tc.input.end_time}s`);
                          if (tc.input.fps != null) parts.push(`@${tc.input.fps}fps`);
                          return `[${parts.join(" ")}]`;
                        })()}
                      </span>
                    )}
                  </>
                : tc.input.description || tc.input.python_code?.slice(0, 120)}
            </p>
            <p className="text-amber-600 mt-0.5 break-words font-mono text-[9px]">
              {tc.output.length > 300
                ? tc.output.slice(0, 300) + "..."
                : tc.output}
            </p>
          </div>
        ))}
      </div>
    </details>
  );
}

function VideoCard({
  url,
  score,
  codeScore,
  vlmScore,
  codeDiagnosis,
  vlmDiagnosis,
  vlmCriteria,
  synthesisDiagnosis,
  synthesisTraces,
  vlmPreviewUrl,
  motionPreviewUrl,
  criteriaScores,
  vlmFps,
  isBest,
  passThreshold,
  sessionId,
  iteration,
  labelingEnabled,
  labelingAnnotator,
  humanLabel,
}: {
  url: string;
  score?: number;
  codeScore?: number;
  vlmScore?: number;
  codeDiagnosis?: string;
  vlmDiagnosis?: string;
  vlmCriteria?: string;
  synthesisDiagnosis?: string;
  synthesisTraces?: SynthesisToolTrace[];
  vlmPreviewUrl?: string;
  motionPreviewUrl?: string;
  criteriaScores?: CriteriaScore[];
  vlmFps?: number;
  isBest: boolean;
  passThreshold: number;
  sessionId?: string;
  iteration?: number;
  labelingEnabled?: boolean;
  labelingAnnotator?: string;
  humanLabel?: HumanLabel | null;
}) {
  const { step, suffix } = parseVideoUrl(url);
  return (
    <div className={`rounded-lg ${isBest ? "ring-2 ring-green-400" : ""}`}>
      <video
        src={staticUrl(url)}
        controls
        preload="metadata"
        className="w-full rounded-t-lg bg-black"
      />
      <div className="px-2 py-1.5 space-y-1">
        <div className="flex items-center justify-between">
          <p className="text-xs text-gray-500">
            Step {Number(step).toLocaleString()}
            {suffix && <span className="ml-1 text-gray-400">({suffix})</span>}
          </p>
          <div className="flex items-center gap-1.5">
            {codeScore !== undefined && (
              <span className={`text-[10px] font-mono ${scoreColor(codeScore, passThreshold)}`}>
                C:{codeScore.toFixed(2)}
              </span>
            )}
            {vlmScore !== undefined && (
              <span className={`text-[10px] font-mono ${scoreColor(vlmScore, passThreshold)}`}>
                V:{vlmScore.toFixed(2)}
              </span>
            )}
            {score !== undefined && (
              <span className={`text-xs font-mono font-semibold ${scoreColor(score, passThreshold)}`}>
                {isBest && "★ "}{score.toFixed(2)}
              </span>
            )}
          </div>
        </div>

        {codeDiagnosis && (
          <JudgeCard variant="code" score={codeScore} diagnosis={codeDiagnosis} passThreshold={passThreshold} />
        )}

        {vlmDiagnosis && (
          <JudgeCard variant="vlm" score={vlmScore} diagnosis={vlmDiagnosis} passThreshold={passThreshold} vlmPreviewUrl={vlmPreviewUrl} motionPreviewUrl={motionPreviewUrl} vlmCriteria={vlmCriteria} criteriaScores={criteriaScores} vlmFps={vlmFps} />
        )}

        {synthesisTraces && <SynthesisToolsCard traces={synthesisTraces} />}

        {/* Dual-judge: show synthesized card when both individual judges exist.
            Single-judge fallback: when neither individual diagnosis exists,
            show as the appropriate single judge (code if codeScore present, else vlm). */}
        {synthesisDiagnosis && codeDiagnosis && vlmDiagnosis && (
          <JudgeCard variant="synthesized" score={score} diagnosis={synthesisDiagnosis} passThreshold={passThreshold} />
        )}
        {synthesisDiagnosis && !codeDiagnosis && !vlmDiagnosis && (
          <JudgeCard
            variant={codeScore !== undefined ? "code" : "vlm"}
            score={score}
            diagnosis={synthesisDiagnosis}
            passThreshold={passThreshold}
          />
        )}
        {labelingEnabled && labelingAnnotator && sessionId && iteration !== undefined && (
          <HumanScoreInput
            sessionId={sessionId}
            iteration={iteration}
            videoUrl={url}
            annotator={labelingAnnotator}
            humanLabel={humanLabel}
          />
        )}
      </div>
    </div>
  );
}

const TOP_N = 3;

type SortField = "score" | "step";
type SortOrder = "asc" | "desc";

/** Compact toggle button for sort controls. Highlights with inverted colors when active. */
function SortButton({
  label,
  active,
  ariaLabel,
  onClick,
}: {
  label: string;
  active: boolean;
  ariaLabel?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      aria-label={ariaLabel}
      onClick={onClick}
      className={`px-2 py-0.5 text-[11px] rounded border transition-colors ${
        active
          ? "bg-gray-800 text-white border-gray-800"
          : "bg-white text-gray-500 border-gray-300 hover:border-gray-400 hover:text-gray-700"
      }`}
    >
      {label}
    </button>
  );
}

export default function VideoPlayer({
  urls,
  checkpointScores,
  checkpointDiagnoses,
  rolloutScores,
  rolloutDiagnoses,
  rolloutCodeDiagnoses,
  rolloutVlmDiagnoses,
  rolloutCodeScores,
  rolloutVlmScores,
  checkpointCodeDiagnoses,
  checkpointVlmDiagnoses,
  checkpointCodeScores,
  checkpointVlmScores,
  bestCheckpoint,
  passThreshold = 0.7,
  sourceRunId,
  sourceReturn,
  title = "Evaluation Videos",
  rolloutSynthesisTraces,
  rolloutCriteriaScores,
  rolloutVlmPreviewUrls,
  rolloutMotionPreviewUrls,
  vlmFps,
  vlmCriteria,
  criteriaScores,
  sessionId,
  iteration,
  labelingEnabled,
  labelingAnnotator,
  humanLabels,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const [sortBy, setSortBy] = useState<SortField>("score");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSortScore = useCallback(() => {
    if (sortBy === "score") setSortOrder((o) => (o === "desc" ? "asc" : "desc"));
    else { setSortBy("score"); setSortOrder("desc"); }
  }, [sortBy]);

  const handleSortStep = useCallback(() => {
    if (sortBy === "step") setSortOrder((o) => (o === "desc" ? "asc" : "desc"));
    else { setSortBy("step"); setSortOrder("desc"); }
  }, [sortBy]);

  // Pre-compute per-URL metadata to avoid redundant parsing in sort/render
  const metaMap = useMemo(
    () =>
      new Map(
        urls.map((u) => [
          u,
          {
            score: resolveScore(u, rolloutScores, checkpointScores) ?? -1,
            diagnosis: resolveDiagnosis(u, rolloutDiagnoses, checkpointDiagnoses),
            codeDiagnosis: resolveCodeDiagnosis(u, checkpointCodeDiagnoses, rolloutCodeDiagnoses),
            vlmDiagnosis: resolveVlmDiagnosis(u, checkpointVlmDiagnoses, rolloutVlmDiagnoses),
            codeScore: resolveJudgeScore(u, checkpointCodeScores, rolloutCodeScores),
            vlmScore: resolveJudgeScore(u, checkpointVlmScores, rolloutVlmScores),
            synthesisTraces: resolveSynthesisTraces(u, rolloutSynthesisTraces),
            vlmPreviewUrl: resolvePreviewUrl(u, rolloutVlmPreviewUrls),
            motionPreviewUrl: resolvePreviewUrl(u, rolloutMotionPreviewUrls),
            criteriaScores: resolveCriteriaScores(u, rolloutCriteriaScores, criteriaScores),
            step: Number(parseVideoUrl(u).step),
          },
        ]),
      ),
    [urls, rolloutScores, checkpointScores, rolloutDiagnoses, checkpointDiagnoses, checkpointCodeDiagnoses, checkpointVlmDiagnoses, rolloutCodeDiagnoses, rolloutVlmDiagnoses, checkpointCodeScores, checkpointVlmScores, rolloutCodeScores, rolloutVlmScores, rolloutSynthesisTraces, rolloutCriteriaScores, criteriaScores, rolloutVlmPreviewUrls, rolloutMotionPreviewUrls],
  );

  // Sort videos by the selected field and order
  const ranked = useMemo(() => {
    const dir = sortOrder === "asc" ? 1 : -1;
    return [...urls].sort((a, b) => {
      const ma = metaMap.get(a)!;
      const mb = metaMap.get(b)!;
      if (sortBy === "score") {
        if (ma.score !== mb.score) return (ma.score - mb.score) * dir;
        // Secondary: step (same direction)
        return (ma.step - mb.step) * dir;
      }
      // sortBy === "step"
      if (ma.step !== mb.step) return (ma.step - mb.step) * dir;
      // Secondary: score (same direction)
      return (ma.score - mb.score) * dir;
    });
  }, [urls, metaMap, sortBy, sortOrder]);

  // All hooks above — early return after to satisfy rules-of-hooks
  if (urls.length === 0) return null;

  const topVideos = ranked.slice(0, TOP_N);
  const restVideos = ranked.slice(TOP_N);
  const hasMore = restVideos.length > 0;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-900">
          {title}
        </h2>
        <div className="flex items-center gap-3">
          {/* Sort controls: click to select, click again to toggle order */}
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-gray-400 mr-0.5">Sort:</span>
            <SortButton
              label={`Score${sortBy === "score" ? (sortOrder === "desc" ? " ↓" : " ↑") : ""}`}
              active={sortBy === "score"}
              ariaLabel={sortBy === "score" ? `Sort by score ${sortOrder === "desc" ? "descending" : "ascending"}` : "Sort by score"}
              onClick={handleSortScore}
            />
            <SortButton
              label={`Step${sortBy === "step" ? (sortOrder === "desc" ? " ↓" : " ↑") : ""}`}
              active={sortBy === "step"}
              ariaLabel={sortBy === "step" ? `Sort by step ${sortOrder === "desc" ? "descending" : "ascending"}` : "Sort by step"}
              onClick={handleSortStep}
            />
          </div>
          {sourceRunId && (() => {
            const parsed = parseRunId(sourceRunId);
            const label = parsed
              ? <><span className="font-mono font-medium text-gray-700">{parsed.config}</span>{" · "}seed <span className="font-mono font-medium text-gray-700">{parsed.seed}</span></>
              : <span className="font-mono font-medium text-gray-700">{sourceRunId}</span>;
            return (
              <span className="text-xs text-gray-500">
                {label}
                {sourceReturn != null && (
                  <> · return <span className="font-mono font-semibold text-gray-900">{sourceReturn.toFixed(1)}</span></>
                )}
              </span>
            );
          })()}
          {hasMore && (
            <span className="text-xs text-gray-400">
              Top {TOP_N} of {urls.length} videos
            </span>
          )}
        </div>
      </div>

      {/* Top 3 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {topVideos.map((url) => {
          const m = metaMap.get(url)!;
          return (
            <VideoCard
              key={url}
              url={url}
              score={m.score >= 0 ? m.score : undefined}
              codeScore={m.codeScore}
              vlmScore={m.vlmScore}
              codeDiagnosis={m.codeDiagnosis}
              vlmDiagnosis={m.vlmDiagnosis}
              vlmCriteria={vlmCriteria}
              criteriaScores={m.criteriaScores}
              synthesisDiagnosis={m.diagnosis}
              synthesisTraces={m.synthesisTraces}
              vlmPreviewUrl={m.vlmPreviewUrl}
              motionPreviewUrl={m.motionPreviewUrl}
              vlmFps={vlmFps}
              isBest={bestCheckpoint === String(m.step)}
              passThreshold={passThreshold}
              sessionId={sessionId}
              iteration={iteration}
              labelingEnabled={labelingEnabled}
              labelingAnnotator={labelingAnnotator}
              humanLabel={humanLabels?.[url.split("/").pop() ?? ""]}
            />
          );
        })}
      </div>

      {/* Collapsed rest */}
      {hasMore && (
        <details
          open={expanded}
          onToggle={(e) =>
            setExpanded((e.target as HTMLDetailsElement).open)
          }
          className="mt-4"
        >
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
            {expanded ? "Hide" : "Show"} {restVideos.length} more video
            {restVideos.length > 1 ? "s" : ""}
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-3">
            {restVideos.map((url) => {
              const m = metaMap.get(url)!;
              return (
                <VideoCard
                  key={url}
                  url={url}
                  score={m.score >= 0 ? m.score : undefined}
                  codeScore={m.codeScore}
                  vlmScore={m.vlmScore}
                  codeDiagnosis={m.codeDiagnosis}
                  vlmDiagnosis={m.vlmDiagnosis}
                  vlmCriteria={vlmCriteria}
                  criteriaScores={m.criteriaScores}
                  synthesisDiagnosis={m.diagnosis}
                  synthesisTraces={m.synthesisTraces}
                  vlmPreviewUrl={m.vlmPreviewUrl}
                  motionPreviewUrl={m.motionPreviewUrl}
                  vlmFps={vlmFps}
                  isBest={bestCheckpoint === String(m.step)}
                  passThreshold={passThreshold}
                  sessionId={sessionId}
                  iteration={iteration}
                  labelingEnabled={labelingEnabled}
                  labelingAnnotator={labelingAnnotator}
                  humanLabel={humanLabels?.[url.split("/").pop() ?? ""]}
                />
              );
            })}
          </div>
        </details>
      )}

    </div>
  );
}
