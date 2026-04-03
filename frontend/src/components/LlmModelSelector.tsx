"use client";

export { MODEL_OPTIONS, DEFAULT_LLM_MODEL, getModelProvider } from "@/lib/model-options";
export type { LlmModelId } from "@/lib/model-options";
import { MODEL_OPTIONS } from "@/lib/model-options";

interface LlmModelSelectorProps {
  value: string;
  onChange: (model: string) => void;
}

export default function LlmModelSelector({
  value,
  onChange,
}: LlmModelSelectorProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        LLM Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
      >
        {MODEL_OPTIONS.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {opt.name}
          </option>
        ))}
      </select>
    </div>
  );
}
