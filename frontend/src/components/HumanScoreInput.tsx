"use client";

import { useState, useEffect, useCallback } from "react";
import { submitHumanLabel } from "@/lib/labeling-api";

const PRESETS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] as const;

const RUBRIC = [
  { range: "0.0", desc: "No evidence of the intended behavior" },
  { range: "0.2", desc: "Wrong behavior (clearly different from the task)" },
  { range: "0.4", desc: "Partial attempt (some relevant elements, but far from the goal)" },
  { range: "0.6", desc: "Mostly achieved with notable flaws" },
  { range: "0.8", desc: "Good execution with minor imperfections" },
  { range: "1.0", desc: "Perfect match to the stated intent" },
] as const;
const RUBRIC_HINT = "Use 0.1, 0.3, 0.5, 0.7, 0.9 when behavior falls between two anchors.";

export interface HumanLabel {
  status: "sent" | "scored" | "error";
  annotator: string;
  sent_at: string;
  intent_score?: number;
  video_url?: string;
  video_count?: number;
  scored_at?: string;
  error?: string;
}

interface HumanScoreInputProps {
  sessionId: string;
  iteration: number;
  videoUrl: string;
  annotator: string;
  humanLabel?: HumanLabel | null;
}

export default function HumanScoreInput({
  sessionId,
  iteration,
  videoUrl,
  annotator,
  humanLabel,
}: HumanScoreInputProps) {
  const [score, setScore] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRubric, setShowRubric] = useState(false);

  // Initialize score from humanLabel when available
  useEffect(() => {
    setSubmitted(false);
    setError(null);
    if (humanLabel?.intent_score !== undefined) {
      setScore(humanLabel.intent_score);
    } else {
      setScore(null);
    }
  }, [iteration, videoUrl, humanLabel?.intent_score]);

  const handleSubmit = useCallback(async () => {
    if (score === null || !annotator) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitHumanLabel({
        session_id: sessionId,
        iteration,
        annotator,
        intent_score: score,
        video_url: videoUrl,
      });
      setSubmitted(true);
      setTimeout(() => setSubmitted(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Submission failed");
    } finally {
      setSubmitting(false);
    }
  }, [score, annotator, sessionId, iteration, videoUrl]);

  // Status badge from server-side human_label
  const serverStatus = humanLabel?.status;

  return (
    <div className="mt-1 pt-1 border-t border-gray-100">
      {/* Previous submission status */}
      {serverStatus === "scored" && humanLabel?.intent_score !== undefined && (
        <div className="flex items-center gap-1.5 mb-1">
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-medium bg-green-100 text-green-700">
            Scored {humanLabel.intent_score.toFixed(1)}
          </span>
          <span className="text-[9px] text-gray-400">
            by {humanLabel.annotator}
          </span>
        </div>
      )}
      {serverStatus === "sent" && (
        <div className="flex items-center gap-1.5 mb-1">
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-medium bg-yellow-100 text-yellow-700">
            Pending...
          </span>
        </div>
      )}
      {serverStatus === "error" && (
        <div className="flex items-center gap-1.5 mb-1">
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-medium bg-red-100 text-red-700">
            Error
          </span>
          {humanLabel?.error && (
            <span className="text-[9px] text-red-400 truncate max-w-[150px]" title={humanLabel.error}>
              {humanLabel.error}
            </span>
          )}
        </div>
      )}

      <div className="flex items-center gap-0.5 flex-wrap">
        {PRESETS.map((v) => (
          <button
            key={v}
            onClick={() => setScore(v)}
            className={`px-1 py-0 rounded text-[9px] font-mono transition-colors ${
              score === v
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-500 hover:bg-gray-200"
            }`}
          >
            {v.toFixed(1)}
          </button>
        ))}
        <button
          onClick={handleSubmit}
          disabled={score === null || !annotator || submitting || submitted}
          className={`px-1.5 py-0 rounded text-[10px] font-medium transition-colors ${
            submitted
              ? "bg-green-100 text-green-700"
              : "bg-blue-50 text-blue-700 hover:bg-blue-100 disabled:opacity-40 disabled:cursor-not-allowed"
          }`}
        >
          {submitted ? "\u2713" : submitting ? "..." : serverStatus === "scored" ? "Resend" : "Send"}
        </button>
        <button
          onClick={() => setShowRubric((v) => !v)}
          className="px-1 py-0 rounded text-[10px] text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          title="Scoring rubric"
        >
          ?
        </button>
        {error && <span className="text-[10px] text-red-500">!</span>}
      </div>
      {showRubric && (
        <div className="mt-1 text-[9px] bg-gray-50 rounded p-1.5">
          {RUBRIC.map((r) => (
            <div key={r.range} className="flex gap-1.5 py-0.5">
              <span className="font-mono font-semibold text-gray-500 w-6 text-right shrink-0">{r.range}</span>
              <span className="text-gray-600">{r.desc}</span>
            </div>
          ))}
          <div className="mt-1 pt-1 border-t border-gray-200 text-gray-400 italic">{RUBRIC_HINT}</div>
        </div>
      )}
    </div>
  );
}
