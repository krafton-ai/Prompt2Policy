"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchSessionAnalysis,
  streamSessionAnalysis,
  type SessionAnalysis,
} from "@/lib/api";

export default function SessionAnalysisCard({
  sessionId,
}: {
  sessionId: string;
}) {
  const [analysis, setAnalysis] = useState<SessionAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState("");
  const cancelRef = useRef<(() => void) | null>(null);

  // Check cache on mount
  useEffect(() => {
    fetchSessionAnalysis(sessionId)
      .then(setAnalysis)
      .catch((err) => {
        // 404 = no cache, that's expected; log other errors
        if (!String(err).includes("404")) {
          console.warn("Failed to fetch cached analysis:", err);
        }
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  // Cleanup streaming on unmount
  useEffect(() => {
    return () => {
      cancelRef.current?.();
    };
  }, []);

  const runAnalysis = useCallback(() => {
    cancelRef.current?.();
    setStreaming(true);
    setError("");
    setStatusMsg("");

    cancelRef.current = streamSessionAnalysis(sessionId, {
      onStatus: (msg) => setStatusMsg(msg),
      onAnalysis: (result) => {
        setAnalysis(result);
        setStreaming(false);
        setStatusMsg("");
        cancelRef.current = null;
      },
      onError: (err) => {
        setError(err);
        setStreaming(false);
        setStatusMsg("");
        cancelRef.current = null;
      },
    });
  }, [sessionId]);

  if (loading) return null;

  // No analysis yet — show button
  if (!analysis && !streaming) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-900">
            Session Analysis
          </h2>
          <button
            onClick={runAnalysis}
            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors"
          >
            Analyze Session
          </button>
        </div>
        {error && (
          <p className="text-sm text-red-600 mt-2">{error}</p>
        )}
      </div>
    );
  }

  // Streaming in progress
  if (streaming) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-3">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          <h2 className="text-sm font-semibold text-gray-900">
            Analyzing Session...
          </h2>
        </div>
        {statusMsg && (
          <p className="text-sm text-gray-500 font-mono">{statusMsg}</p>
        )}
      </div>
    );
  }

  // Analysis loaded — render results
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">
          Session Analysis
        </h2>
        <button
          onClick={runAnalysis}
          className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
        >
          Re-analyze
        </button>
      </div>

      {/* Analysis text */}
      {analysis?.analysis_en && (
        <div className="space-y-2">
          <p className="text-sm text-gray-700 whitespace-pre-wrap">
            {analysis.analysis_en}
          </p>
        </div>
      )}

      {/* Key findings */}
      {analysis?.key_findings && analysis.key_findings.length > 0 && (
        <div className="space-y-2">
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">
              Key Findings
            </p>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-0.5">
              {analysis.key_findings.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Recommendations */}
      {analysis?.recommendations && analysis.recommendations.length > 0 && (
        <div className="space-y-2">
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">
              Recommendations
            </p>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-0.5">
              {analysis.recommendations.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Meta */}
      {analysis && (
        <p className="text-xs text-gray-400">
          {analysis.model} · {analysis.tool_calls_used} tool calls · {new Date(analysis.created_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}
