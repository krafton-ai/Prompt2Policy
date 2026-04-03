export default function BarStat({
  label,
  value,
  pct,
}: {
  label: string;
  value: string;
  pct: number;
}) {
  const color =
    pct < 50
      ? "bg-emerald-400"
      : pct < 80
        ? "bg-amber-400"
        : "bg-rose-400";
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-600 mb-0.5">
        <span>{label}</span>
        <span className="font-mono">{value}</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}
