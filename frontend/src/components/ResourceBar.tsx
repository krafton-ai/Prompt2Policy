import type { ResourceStatus } from "@/lib/api";

export default function ResourceBar({ status }: { status: ResourceStatus }) {
  const used =
    status.total_cores - status.reserved_cores - status.available_cores;
  const usable = status.total_cores - status.reserved_cores;
  const pct = usable > 0 ? (used / usable) * 100 : 0;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 mb-6">
      <div className="flex items-center justify-between text-sm mb-2">
        <span className="text-gray-500">CPU Cores</span>
        <span className="font-mono text-gray-700">
          {used} / {usable} used ({status.reserved_cores} reserved)
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
