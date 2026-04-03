import Link from "next/link";
import type { BenchmarkRunSummary } from "@/lib/api";
import { timeAgo } from "@/lib/format";
import StatusBadge from "./StatusBadge";
import ProgressBar from "./ProgressBar";

export default function BenchmarkRunCard({
  run,
}: {
  run: BenchmarkRunSummary;
}) {
  const isStaged = run.mode === "staged" && run.total_stages > 0;

  return (
    <Link href={`/benchmark/${run.benchmark_id}`}>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow p-5 cursor-pointer">
        <div className="flex items-center gap-2 mb-2">
          <p className="text-xs font-mono text-gray-400">{run.benchmark_id}</p>
          <span className="text-xs text-gray-400">{timeAgo(run.created_at)}</span>
          {isStaged && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-50 text-indigo-600">
              Staged
            </span>
          )}
        </div>

        <div className="mb-3">
          <ProgressBar completed={run.completed_cases} total={run.total_cases} />
        </div>

        <div className="flex items-center gap-3 text-sm">
          <StatusBadge status={run.status} />
          {isStaged && run.current_stage > 0 && (
            <span className="text-gray-500 text-xs">
              Stage{" "}
              <span className="font-mono font-medium text-gray-700">
                {run.current_stage}/{run.total_stages}
              </span>
            </span>
          )}
          <span className="text-gray-500">
            Pass{" "}
            <span className="font-mono font-medium text-gray-900">
              {run.passed_cases}/{run.completed_cases}
            </span>
          </span>
          <span className="text-gray-500">
            Rate{" "}
            <span className="font-mono font-medium text-gray-900">
              {(run.success_rate * 100).toFixed(1)}%
            </span>
          </span>
          <span className="text-gray-500">
            Score{" "}
            <span className="font-mono font-medium text-gray-900">
              {run.cumulative_score.toFixed(1)}/{run.total_cases}
            </span>
          </span>
        </div>
      </div>
    </Link>
  );
}
