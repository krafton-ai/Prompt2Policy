/**
 * Shared model/VLM/effort constants and types.
 * Lives in lib/ so both lib/ and components/ can import without cycles.
 */

// ── LLM models ──

export const MODEL_OPTIONS = [
  {
    id: "claude-opus-4-6",
    name: "Claude Opus 4.6",
    description: "Most capable Anthropic model. Supports max thinking effort.",
  },
  {
    id: "claude-sonnet-4-6",
    name: "Claude Sonnet 4.6",
    description: "Fast and capable. Thinking up to high effort.",
  },
  {
    id: "gpt-5.4",
    name: "GPT-5.4",
    description: "Latest OpenAI flagship model.",
  },
  {
    id: "gpt-5.3-codex",
    name: "GPT-5.3 Codex",
    description: "OpenAI code-optimized reasoning model.",
  },
  {
    id: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro",
    description: "Google flagship with deep thinking.",
  },
  {
    id: "gemini-3-flash-preview",
    name: "Gemini 3 Flash",
    description: "Google fast and cost-efficient model.",
  },
] as const;

export type LlmModelId = (typeof MODEL_OPTIONS)[number]["id"];

export const DEFAULT_LLM_MODEL: LlmModelId = "gemini-3.1-pro-preview";

/**
 * Derive the provider category from a model ID string.
 * Used by ThinkingEffortSelector to filter available effort levels.
 */
export function getModelProvider(
  model: string,
): "anthropic-opus" | "anthropic" | "openai" | "gemini-pro" | "gemini-flash" {
  if (model.includes("opus")) return "anthropic-opus";
  if (model.startsWith("claude")) return "anthropic";
  if (
    model.startsWith("gpt-") ||
    model.startsWith("o1") ||
    model.startsWith("o3") ||
    model.startsWith("o4")
  )
    return "openai";
  if (model.startsWith("gemini") && model.includes("pro"))
    return "gemini-pro";
  if (model.startsWith("gemini")) return "gemini-flash";
  return "anthropic-opus"; // default
}

// ── VLM models ──

export const VLM_OPTIONS = [
  {
    id: "",
    name: "No VLM",
    provider: "",
    description:
      "Disable VLM-based judging. Requires code-based judge to be enabled.",
  },
  {
    id: "vllm-Qwen/Qwen3.5-27B",
    name: "Qwen3.5-VL 27B (Video)",
    provider: "Remote vLLM (AWS)",
    description:
      "Native video input via remote vLLM on AWS g5.12xlarge (4×A10G). Videos sent as base64.",
  },
  {
    id: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro (Video)",
    provider: "Google",
    description:
      "Native video input via Google API. Sends MP4 inline. Uses GEMINI_API_KEY.",
  },
  {
    id: "claude-opus-4-6",
    name: "Claude Opus 4.6 (Image)",
    provider: "Anthropic",
    description:
      "Composite image via Anthropic API. Uses ANTHROPIC_API_KEY.",
  },
] as const;

// ── Thinking effort ──

export type ThinkingEffort = "max" | "xhigh" | "high" | "medium" | "low" | "minimal" | "";

const VALID_EFFORTS: readonly string[] = ["max", "xhigh", "high", "medium", "low", "minimal"];
export function isThinkingEffort(v: unknown): v is ThinkingEffort {
  return v === "" || (typeof v === "string" && VALID_EFFORTS.includes(v));
}
