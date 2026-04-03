"use client";

export { VLM_OPTIONS } from "@/lib/model-options";
import { VLM_OPTIONS } from "@/lib/model-options";

interface VlmSelectorProps {
  value: string;
  onChange: (model: string) => void;
}

export default function VlmSelector({ value, onChange }: VlmSelectorProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        VLM Provider
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
      >
        {VLM_OPTIONS.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {opt.name} — {opt.provider}
          </option>
        ))}
      </select>
    </div>
  );
}
