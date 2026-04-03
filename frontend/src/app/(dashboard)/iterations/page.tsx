"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetchSessions, type SessionDetail } from "@/lib/api";
import SessionCard from "@/components/SessionCard";

type SortKey = "newest" | "oldest" | "score";

const sortLabels: Record<SortKey, string> = {
  newest: "Newest",
  oldest: "Oldest",
  score: "Score",
};

function sortSessions(
  sessions: SessionDetail[],
  key: SortKey,
): SessionDetail[] {
  const sorted = [...sessions];
  switch (key) {
    case "newest":
      sorted.sort((a, b) =>
        (b.created_at || "").localeCompare(a.created_at || ""),
      );
      break;
    case "oldest":
      sorted.sort((a, b) =>
        (a.created_at || "").localeCompare(b.created_at || ""),
      );
      break;
    case "score":
      sorted.sort((a, b) => b.best_score - a.best_score);
      break;
  }
  return sorted;
}

export default function IterationsPage() {
  const [sortKey, setSortKey] = useState<SortKey>("newest");
  const {
    data: sessions,
    error,
    isLoading,
  } = useSWR<SessionDetail[]>("sessions", fetchSessions, {
    refreshInterval: 5000,
  });

  if (isLoading) {
    return <p className="text-gray-500">Loading...</p>;
  }

  if (error) {
    return (
      <p className="text-red-600">
        Failed to load experiments. Is the API server running?
      </p>
    );
  }

  if (!sessions || sessions.length === 0) {
    return (
      <div className="text-center py-20">
        <h2 className="text-xl font-semibold text-gray-700 mb-2">
          No experiments yet
        </h2>
        <p className="text-gray-500">Start a new experiment to get going.</p>
      </div>
    );
  }

  const sorted = sortSessions(sessions, sortKey);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
        <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
          {(Object.keys(sortLabels) as SortKey[]).map((key) => (
            <button
              key={key}
              onClick={() => setSortKey(key)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                sortKey === key
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {sortLabels[key]}
            </button>
          ))}
        </div>
      </div>
      <div className="grid gap-4">
        {sorted.map((s) => (
          <SessionCard key={s.session_id} session={s} />
        ))}
      </div>
    </div>
  );
}
