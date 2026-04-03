/**
 * localStorage-based preset management for Benchmark and E2E forms.
 *
 * Each preset stores a partial snapshot of form params.
 * On load, the snapshot is spread over the form defaults.
 */

import type { ThinkingEffort } from "@/lib/model-options";

export type PresetTarget = "benchmark" | "e2e" | "scheduler-benchmark";

export interface Preset<T = Record<string, unknown>> {
  name: string;
  params: T;
  created_at: string;
}

const STORAGE_KEY_PREFIX = "p2p:presets:";

/** When loading presets for a target, also include presets from the fallback target (deduplicated by name). */
const FALLBACK_MAP: Partial<Record<PresetTarget, PresetTarget>> = {
  "scheduler-benchmark": "benchmark",
};

function storageKey(target: PresetTarget): string {
  return `${STORAGE_KEY_PREFIX}${target}`;
}

export function loadPresets<T = Record<string, unknown>>(
  target: PresetTarget,
): Preset<T>[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(storageKey(target));
    const primary: Preset<T>[] = raw ? JSON.parse(raw) : [];

    const fallbackTarget = FALLBACK_MAP[target];
    if (fallbackTarget) {
      const fallbackRaw = localStorage.getItem(storageKey(fallbackTarget));
      const fallback: Preset<T>[] = fallbackRaw ? JSON.parse(fallbackRaw) : [];
      const primaryNames = new Set(primary.map((p) => p.name));
      for (const fp of fallback) {
        if (!primaryNames.has(fp.name)) {
          primary.push(fp);
        }
      }
    }

    return primary;
  } catch {
    return [];
  }
}

export function savePreset<T = Record<string, unknown>>(
  target: PresetTarget,
  name: string,
  params: T,
): Preset<T> {
  const presets = loadPresets<T>(target);
  const existing = presets.findIndex((p) => p.name === name);
  const preset: Preset<T> = {
    name,
    params,
    created_at: new Date().toISOString(),
  };
  if (existing >= 0) {
    presets[existing] = preset;
  } else {
    presets.push(preset);
  }
  localStorage.setItem(storageKey(target), JSON.stringify(presets));
  return preset;
}

export function deletePreset(target: PresetTarget, name: string): void {
  // Delete from the primary target
  const raw = localStorage.getItem(storageKey(target));
  const primary: Preset[] = raw ? JSON.parse(raw) : [];
  const filtered = primary.filter((p) => p.name !== name);
  localStorage.setItem(storageKey(target), JSON.stringify(filtered));

  // Also delete from the fallback target if present
  const fallbackTarget = FALLBACK_MAP[target];
  if (fallbackTarget) {
    const fbRaw = localStorage.getItem(storageKey(fallbackTarget));
    const fallback: Preset[] = fbRaw ? JSON.parse(fbRaw) : [];
    const fbFiltered = fallback.filter((p) => p.name !== name);
    if (fbFiltered.length !== fallback.length) {
      localStorage.setItem(storageKey(fallbackTarget), JSON.stringify(fbFiltered));
    }
  }
}

// ---------------------------------------------------------------------------
// Preset param types (shared between list pages and detail pages)
// ---------------------------------------------------------------------------

export interface E2EPresetParams {
  model?: string;
  prompt: string;
  numConfigs: number;
  seeds: string;
  timesteps: number;
  maxIterations: number;
  passThreshold: number;
  envId: string;
  numEnvs: number;
  vlmModel: string;
  numEvals: number;
  useCodeJudge: boolean;
  coresPerRun: number;
  device: "auto" | "cpu";
  thinkingEffort: ThinkingEffort;
}

export interface BenchmarkPresetParams {
  model?: string;
  timesteps: number;
  numConfigs: number;
  seedsStr: string;
  maxIterations: number;
  passThreshold: number;
  numEnvs: number;
  maxParallel: number;
  coresPerRun: number;
  vlmModel: string;
  useCodeJudge: boolean;
  thinkingEffort: ThinkingEffort;
  filterEnvs: string[];
  filterCategories: string[];
  filterDifficulties: string[];
  device: "auto" | "cpu";
  csvFile: string;
  backend: "local" | "ssh";
}
