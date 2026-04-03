"use client";

interface ConfigAggregation {
  mean_best_score: number;
  std_best_score: number;
  mean_final_return: number;
  std_final_return: number;
  per_seed: { seed: number; best_score: number; final_return: number }[];
}

interface Props {
  aggregation: Record<string, ConfigAggregation>;
  configParams: Record<string, Record<string, unknown>>;
  bestConfigId: string;
}

// Keys to always exclude from diff (they differ by definition or are noise)
const EXCLUDED_KEYS = new Set([
  "seed",
  "iteration_id",
  "batch_size",
  "minibatch_size",
  "num_iterations",
]);

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) return v.toExponential(2);
    return v.toPrecision(4);
  }
  if (v === null || v === undefined) return "-";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export default function ConfigDiffTable({
  aggregation,
  configParams,
  bestConfigId,
}: Props) {
  if (
    !aggregation ||
    !configParams ||
    Object.keys(configParams).length < 2
  )
    return null;

  const configIds = Object.keys(aggregation).sort();
  const allConfigs = configIds.map((id) => configParams[id] || {});

  // Find keys that differ across configs
  const allKeys = new Set<string>();
  for (const cfg of allConfigs) {
    for (const k of Object.keys(cfg)) {
      if (!EXCLUDED_KEYS.has(k)) allKeys.add(k);
    }
  }

  const diffKeys: string[] = [];
  for (const key of allKeys) {
    const values = allConfigs.map((c) => JSON.stringify(c[key] ?? null));
    const unique = new Set(values);
    if (unique.size > 1) diffKeys.push(key);
  }

  // If no differences found (bug case), show a warning
  if (diffKeys.length === 0) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <p className="text-sm font-medium text-amber-800">
          All configs have identical hyperparameters
        </p>
        <p className="text-xs text-amber-600 mt-1">
          Config comparison is not meaningful when all parameters are the same.
        </p>
      </div>
    );
  }

  // Use baseline as reference for highlighting
  const baselineParams = configParams[configIds[0]] || {};

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-1">
        Config Comparison
      </h2>
      <p className="text-xs text-gray-400 mb-3">
        Showing {diffKeys.length} differing parameter{diffKeys.length > 1 ? "s" : ""} only
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-200 text-gray-500">
              <th className="text-left py-2 pr-3 font-medium">Parameter</th>
              {configIds.map((id) => (
                <th key={id} className="text-right py-2 px-3 font-medium">
                  <span>{id}</span>
                  {id === bestConfigId && (
                    <span className="ml-1 text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded-full">
                      best
                    </span>
                  )}
                </th>
              ))}
              <th className="text-right py-2 pl-3 font-medium border-l border-gray-200">
                Mean Return
              </th>
            </tr>
          </thead>
          <tbody>
            {diffKeys.sort().map((key) => (
              <tr key={key} className="border-b border-gray-100">
                <td className="py-1.5 pr-3 font-mono text-gray-600">
                  {key}
                </td>
                {configIds.map((id) => {
                  const val = configParams[id]?.[key];
                  const baseVal = baselineParams[key];
                  const isDiff =
                    JSON.stringify(val ?? null) !==
                    JSON.stringify(baseVal ?? null);
                  return (
                    <td
                      key={id}
                      className={`py-1.5 px-3 text-right font-mono ${
                        isDiff
                          ? "text-blue-700 bg-blue-50 font-semibold"
                          : "text-gray-500"
                      }`}
                    >
                      {formatValue(val)}
                    </td>
                  );
                })}
                <td className="py-1.5 pl-3 border-l border-gray-200" />
              </tr>
            ))}
            {/* Mean Return row */}
            <tr className="border-t-2 border-gray-300 font-medium">
              <td className="py-2 pr-3 text-gray-700">Mean Return</td>
              {configIds.map((id) => {
                const agg = aggregation[id];
                const isBest = id === bestConfigId;
                return (
                  <td
                    key={id}
                    className={`py-2 px-3 text-right font-mono ${
                      isBest ? "text-green-700 font-semibold" : "text-gray-900"
                    }`}
                  >
                    {(agg?.mean_final_return ?? 0).toFixed(1)}
                    <span className="text-gray-400 font-normal">
                      {" "}
                      ± {(agg?.std_final_return ?? 0).toFixed(1)}
                    </span>
                  </td>
                );
              })}
              <td className="py-2 pl-3 border-l border-gray-200 text-right font-mono text-gray-500">
                {configIds.length} seeds
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
