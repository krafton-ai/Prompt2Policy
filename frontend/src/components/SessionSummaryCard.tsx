import type { LoopIterationSummary } from "@/lib/api";

type Trend = "improving" | "declining" | "plateau";

function computeTrend(iterations: LoopIterationSummary[]): Trend {
  const scores = iterations
    .map((it) => it.intent_score)
    .filter((s): s is number => s !== null);
  if (scores.length < 2) return "plateau";
  const mid = Math.floor(scores.length / 2);
  const firstHalf = scores.slice(0, mid);
  const secondHalf = scores.slice(mid);
  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const diff = avg(secondHalf) - avg(firstHalf);
  if (diff > 0.05) return "improving";
  if (diff < -0.05) return "declining";
  return "plateau";
}

function topFailureTags(
  iterations: LoopIterationSummary[],
  n: number = 5,
): [string, number][] {
  const counts = new Map<string, number>();
  for (const it of iterations) {
    for (const tag of it.failure_tags) {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}

const trendConfig: Record<Trend, { label: string; color: string; icon: string }> = {
  improving: { label: "Improving", color: "text-green-700 bg-green-50", icon: "↑" },
  declining: { label: "Declining", color: "text-red-700 bg-red-50", icon: "↓" },
  plateau: { label: "Plateau", color: "text-yellow-700 bg-yellow-50", icon: "→" },
};

export default function SessionSummaryCard({
  iterations,
  status,
  bestScore,
}: {
  iterations: LoopIterationSummary[];
  status: string;
  bestScore: number;
}) {
  const trend = computeTrend(iterations);
  const tags = topFailureTags(iterations);
  const tc = trendConfig[trend];

  const scores = iterations
    .map((it) => it.intent_score)
    .filter((s): s is number => s !== null);
  const bestIdx = iterations.findIndex(
    (it) => it.intent_score !== null && it.intent_score === Math.max(...scores),
  );

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">Session Summary</h2>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${tc.color}`}>
          {tc.icon} {tc.label}
        </span>
      </div>

      {/* One-line summary */}
      <p className="text-sm text-gray-700">
        {status === "passed" ? "Session passed" : `Session ended (${status})`}
        {" — "}best score {bestScore.toFixed(2)} at iteration v{bestIdx >= 0 ? iterations[bestIdx].iteration : "?"}
        {" — "}trend: {trend}
      </p>

      {/* Top failure tags */}
      {tags.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">
            Top Failure Tags
          </p>
          <div className="flex flex-wrap gap-1.5">
            {tags.map(([tag, count]) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded text-xs font-medium bg-orange-50 text-orange-700"
              >
                {tag} ({count})
              </span>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
