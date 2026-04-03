/**
 * Persist user's LLM / VLM / thinking-effort preferences in localStorage
 * so they survive page refreshes across all pages.
 */

import type { ThinkingEffort } from "@/lib/model-options";
import { isThinkingEffort, MODEL_OPTIONS, VLM_OPTIONS } from "@/lib/model-options";

const STORAGE_KEY = "p2p:model-prefs";

const VALID_LLM_IDS: Set<string> = new Set(MODEL_OPTIONS.map((o) => o.id));
const VALID_VLM_IDS: Set<string> = new Set(VLM_OPTIONS.map((o) => o.id));

export interface ModelPrefs {
  llm?: string;
  vlm?: string;
  thinkingEffort?: ThinkingEffort;
}

/** Read cached preferences (SSR-safe — returns {} on server).
 *  Drops values that no longer match the current model/effort lists. */
export function loadModelPrefs(): ModelPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as ModelPrefs;
    const prefs: ModelPrefs = {};
    if (parsed.llm && VALID_LLM_IDS.has(parsed.llm)) prefs.llm = parsed.llm;
    if (parsed.vlm && VALID_VLM_IDS.has(parsed.vlm)) prefs.vlm = parsed.vlm;
    if (isThinkingEffort(parsed.thinkingEffort)) prefs.thinkingEffort = parsed.thinkingEffort;
    return prefs;
  } catch {
    return {};
  }
}

/** Merge partial updates into the cache. */
export function saveModelPrefs(partial: Partial<ModelPrefs>): void {
  if (typeof window === "undefined") return;
  try {
    const prev = loadModelPrefs();
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...prev, ...partial }));
  } catch {
    // localStorage full or blocked — silently ignore
  }
}
