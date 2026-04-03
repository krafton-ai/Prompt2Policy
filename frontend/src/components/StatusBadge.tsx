const colors: Record<string, string> = {
  completed: "bg-green-100 text-green-800",
  passed: "bg-green-100 text-green-800",
  running: "bg-blue-100 text-blue-800",
  pending: "bg-yellow-100 text-yellow-800",
  max_iterations: "bg-yellow-100 text-yellow-800",
  plateau: "bg-orange-100 text-orange-800",
  error: "bg-red-100 text-red-800",
  cancelled: "bg-gray-100 text-gray-800",
  queued: "bg-gray-100 text-gray-500",
  stale: "bg-amber-100 text-amber-800",
  gate_failed: "bg-red-100 text-red-800",
};

export default function StatusBadge({
  status,
  isStale,
}: {
  status: string;
  isStale?: boolean;
}) {
  const displayStatus = isStale ? "stale" : status;
  const colorClass = isStale
    ? colors.stale
    : (colors[status] || "bg-gray-100 text-gray-800");

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}`}
    >
      {status === "running" && !isStale && (
        <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-1.5 animate-pulse" />
      )}
      {displayStatus}
    </span>
  );
}
