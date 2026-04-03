import type { EvalResult } from "@/lib/api";

interface Props {
  evalResults: EvalResult[];
}

export default function EvalTimeline({ evalResults }: Props) {
  if (evalResults.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Evaluation Timeline
      </h2>
      <div className="flex items-center gap-2 overflow-x-auto pb-2">
        {evalResults.map((e, i) => (
          <div key={`${e.global_step}-${i}`} className="flex items-center">
            <div className="text-center">
              <div
                className={`rounded-lg px-3 py-2 text-sm font-mono ${
                  i === evalResults.length - 1
                    ? "bg-blue-100 text-blue-800 border border-blue-300"
                    : "bg-gray-100 text-gray-700"
                }`}
              >
                <div className="flex flex-col gap-0.5 text-left">
                  {e.p10_return != null && (
                    <span>
                      <span className="text-[10px] text-gray-400">p10{"  "}</span>
                      {e.p10_return.toFixed(0)}
                    </span>
                  )}
                  {e.median_return != null ? (
                    <span>
                      <span className="text-[10px] text-gray-400">med{"  "}</span>
                      {e.median_return.toFixed(0)}
                    </span>
                  ) : (
                    <span>
                      <span className="text-[10px] text-gray-400">ret{"  "}</span>
                      {e.total_reward.toFixed(0)}
                    </span>
                  )}
                  {e.p90_return != null && (
                    <span>
                      <span className="text-[10px] text-gray-400">p90{"  "}</span>
                      {e.p90_return.toFixed(0)}
                    </span>
                  )}
                  {e.std_return != null && (
                    <span>
                      <span className="text-[10px] text-gray-400">std{"  "}</span>
                      {e.std_return.toFixed(0)}
                    </span>
                  )}
                </div>
              </div>
              <p className="text-[10px] text-gray-400 mt-1">
                {(e.global_step / 1000).toFixed(0)}k
              </p>
            </div>
            {i < evalResults.length - 1 && (
              <div className="w-8 h-px bg-gray-300 mx-1" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
