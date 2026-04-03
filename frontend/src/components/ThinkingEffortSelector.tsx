"use client";

import { useEffect } from "react";
import { DEFAULT_LLM_MODEL, getModelProvider } from "@/lib/model-options";
export { isThinkingEffort } from "@/lib/model-options";
export type { ThinkingEffort } from "@/lib/model-options";
import type { ThinkingEffort } from "@/lib/model-options";

interface EffortOption {
  id: string;
  name: string;
  description: string;
}

const ALL_EFFORT_OPTIONS: EffortOption[] = [
  {
    id: "max",
    name: "Max",
    description:
      "Deepest reasoning with no constraints on thinking. Claude Opus 4.6 only.",
  },
  {
    id: "xhigh",
    name: "xHigh",
    description:
      "Highest reasoning effort for OpenAI GPT-5.x models.",
  },
  {
    id: "high",
    name: "High",
    description:
      "Always thinks deeply. Best for complex reasoning and coding tasks.",
  },
  {
    id: "medium",
    name: "Medium",
    description: "Balanced thinking. May skip thinking for simple queries.",
  },
  {
    id: "low",
    name: "Low",
    description: "Minimal thinking. Fastest responses.",
  },
  {
    id: "minimal",
    name: "Minimal",
    description: "Least thinking. Only for trivial tasks.",
  },
];

/** Effort levels available per provider category. */
const EFFORTS_BY_PROVIDER: Record<string, string[]> = {
  "anthropic-opus": ["max", "high", "medium", "low"],
  anthropic: ["high", "medium", "low"],
  openai: ["xhigh", "high", "medium", "low"],
  "gemini-pro": ["high", "medium", "low"],
  "gemini-flash": ["high", "medium", "low", "minimal"],
};

/** Default effort when switching to a provider that doesn't support the current value. */
const DEFAULT_BY_PROVIDER: Record<string, string> = {
  "anthropic-opus": "max",
  anthropic: "high",
  openai: "high",
  "gemini-pro": "high",
  "gemini-flash": "high",
};

interface ThinkingEffortSelectorProps {
  value: ThinkingEffort;
  onChange: (effort: ThinkingEffort) => void;
  /** LLM model ID — used to filter available effort levels. */
  model?: string;
}

export default function ThinkingEffortSelector({
  value,
  onChange,
  model = DEFAULT_LLM_MODEL,
}: ThinkingEffortSelectorProps) {
  const provider = getModelProvider(model);
  const allowedIds =
    EFFORTS_BY_PROVIDER[provider] ?? EFFORTS_BY_PROVIDER["anthropic-opus"];
  const options = ALL_EFFORT_OPTIONS.filter((o) => allowedIds.includes(o.id));

  // Auto-adjust value when model changes and current effort is not available
  useEffect(() => {
    if (value && !allowedIds.includes(value)) {
      onChange((DEFAULT_BY_PROVIDER[provider] ?? "high") as ThinkingEffort);
    }
  }, [model]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Thinking Effort
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ThinkingEffort)}
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
      >
        {options.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {opt.name}
          </option>
        ))}
      </select>
    </div>
  );
}
