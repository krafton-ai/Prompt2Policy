/** Shared pricing utilities for LLM/VLM cost calculation.
 *
 * Only models registered in LlmModelSelector are listed.
 * Unrecognized models fall back to DEFAULT_PRICING.
 *
 * Sources (updated 2026-03-20):
 *   Anthropic  — https://platform.claude.com/docs/en/about-claude/pricing
 *   OpenAI     — https://developers.openai.com/api/docs/pricing
 *   Google     — https://ai.google.dev/gemini-api/docs/pricing
 */

export const MODEL_PRICING: { pattern: string; inputPerMTok: number; outputPerMTok: number }[] = [
  { pattern: "opus-4-6", inputPerMTok: 5, outputPerMTok: 25 },
  { pattern: "sonnet-4-6", inputPerMTok: 3, outputPerMTok: 15 },
  { pattern: "gpt-5.4", inputPerMTok: 2.5, outputPerMTok: 15 },
  { pattern: "gpt-5.3-codex", inputPerMTok: 1.75, outputPerMTok: 14 },
  { pattern: "gemini-3.1-pro", inputPerMTok: 2, outputPerMTok: 12 },
  { pattern: "gemini-3-flash", inputPerMTok: 0.5, outputPerMTok: 3 },
];

export const DEFAULT_PRICING = { pattern: "", inputPerMTok: 3, outputPerMTok: 15 };

export function getModelPricing(model: string) {
  return MODEL_PRICING.find((e) => model.includes(e.pattern)) ?? DEFAULT_PRICING;
}

export function formatCost(cost: number): string {
  if (cost < 0.01) return "<$0.01";
  if (cost >= 10) return `$${cost.toFixed(1)}`;
  return `$${cost.toFixed(2)}`;
}

export function computeModelCost(model: string, inputTokens: number, outputTokens: number): number {
  const p = getModelPricing(model);
  return (inputTokens * p.inputPerMTok + outputTokens * p.outputPerMTok) / 1_000_000;
}
