import type { BenchmarkGroupStats } from "@/lib/api";

function rateColor(rate: number): string {
  if (rate >= 0.7) return "bg-green-500";
  if (rate >= 0.4) return "bg-yellow-500";
  return "bg-red-500";
}

export default function GroupStatsTable({
  groups,
  label,
}: {
  groups: Record<string, BenchmarkGroupStats>;
  label: string;
}) {
  const entries = Object.entries(groups).sort(
    ([, a], [, b]) => b.success_rate - a.success_rate,
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
            <th className="py-2 pr-4">{label}</th>
            <th className="py-2 pr-4">Total</th>
            <th className="py-2 pr-4">Passed</th>
            <th className="py-2 pr-4 min-w-[140px]">Success Rate</th>
            <th className="py-2 pr-4">Avg Score</th>
            <th className="py-2">Cumul. Score</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, stats]) => (
            <tr key={name} className="border-b border-gray-100">
              <td className="py-2.5 pr-4 font-medium text-gray-900">{name}</td>
              <td className="py-2.5 pr-4 font-mono text-gray-600">
                {stats.total}
              </td>
              <td className="py-2.5 pr-4 font-mono text-gray-600">
                {stats.passed}/{stats.completed}
              </td>
              <td className="py-2.5 pr-4">
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden max-w-[80px]">
                    <div
                      className={`h-full rounded-full ${rateColor(stats.success_rate)}`}
                      style={{
                        width: `${(stats.success_rate * 100).toFixed(0)}%`,
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs text-gray-600">
                    {(stats.success_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </td>
              <td className="py-2.5 pr-4 font-mono text-gray-600">
                {stats.average_score.toFixed(2)}
              </td>
              <td className="py-2.5 font-mono text-gray-600">
                {stats.cumulative_score.toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
