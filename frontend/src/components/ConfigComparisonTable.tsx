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
  bestConfigId: string;
  configs?: { config_id: string; label?: string; params?: Record<string, unknown> }[];
}

export default function ConfigComparisonTable({
  aggregation,
  bestConfigId,
  configs,
}: Props) {
  if (!aggregation || Object.keys(aggregation).length === 0) return null;

  const configIds = Object.keys(aggregation).sort();

  const getLabel = (configId: string) => {
    const cfg = configs?.find((c) => c.config_id === configId);
    return cfg?.label || configId;
  };

  const getParams = (configId: string) => {
    const cfg = configs?.find((c) => c.config_id === configId);
    if (!cfg?.params || Object.keys(cfg.params).length === 0) return "-";
    return Object.entries(cfg.params)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Config Comparison
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-200 text-gray-500">
              <th className="text-left py-2 pr-3 font-medium">Config</th>
              <th className="text-left py-2 pr-3 font-medium">Parameters</th>
              <th className="text-right py-2 pr-3 font-medium">
                Mean Return
              </th>
              <th className="text-right py-2 pr-3 font-medium">Seeds</th>
            </tr>
          </thead>
          <tbody>
            {configIds.map((configId) => {
              const agg = aggregation[configId];
              const isBest = configId === bestConfigId;
              return (
                <tr
                  key={configId}
                  className={`border-b border-gray-100 ${
                    isBest ? "bg-green-50" : ""
                  }`}
                >
                  <td className="py-2 pr-3">
                    <span className="font-medium text-gray-900">
                      {getLabel(configId)}
                    </span>
                    {isBest && (
                      <span className="ml-1.5 text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded-full">
                        best
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-gray-500 font-mono">
                    {getParams(configId)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-gray-900">
                    {(agg.mean_final_return ?? 0).toFixed(1)}
                    <span className="text-gray-400">
                      {" "}
                      ± {(agg.std_final_return ?? 0).toFixed(1)}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right text-gray-500">
                    {agg.per_seed?.length ?? 0}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
