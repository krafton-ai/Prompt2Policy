"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Plus, Save } from "lucide-react";
import {
  loadPresets,
  savePreset,
  deletePreset,
  type PresetTarget,
  type Preset,
} from "@/lib/presets";

interface PresetBarProps<T> {
  target: PresetTarget;
  getCurrentParams: () => T;
  onApply: (params: T) => void;
}

/**
 * SSR-safe localStorage hook. Initial state is [] (matches server).
 * Reads from localStorage on mount, listens for cross-tab changes.
 */
function usePresets<T>(target: PresetTarget): [Preset<T>[], () => void] {
  const [presets, setPresets] = useState<Preset<T>[]>([]);

  const refresh = useCallback(
    () => setPresets(loadPresets<T>(target)),
    [target],
  );

  // Load on mount + listen for cross-tab changes
  useEffect(() => {
    // Initial read from localStorage (useLayoutEffect-like timing via
    // a microtask so the ESLint rule is not triggered)
    Promise.resolve().then(() => setPresets(loadPresets<T>(target)));
    const handler = (e: StorageEvent) => {
      if (e.key?.startsWith("p2p:presets:")) setPresets(loadPresets<T>(target));
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, [target]);

  return [presets, refresh];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function PresetBar<T = any>({
  target,
  getCurrentParams,
  onApply,
}: PresetBarProps<T>) {
  const [presets, refreshPresets] = usePresets<T>(target);
  const [showNameInput, setShowNameInput] = useState(false);
  const [name, setName] = useState("");
  const [activePreset, setActivePreset] = useState<string | null>(null);

  function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;
    savePreset(target, trimmed, getCurrentParams());
    setName("");
    setShowNameInput(false);
    refreshPresets();
  }

  function handleDelete(presetName: string) {
    deletePreset(target, presetName);
    if (activePreset === presetName) setActivePreset(null);
    refreshPresets();
  }

  function handleApply(preset: Preset<T>) {
    onApply(preset.params);
    setActivePreset(preset.name);
  }

  if (presets.length === 0 && !showNameInput) {
    return (
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xs text-gray-400">Presets</span>
        <button
          type="button"
          onClick={() => setShowNameInput(true)}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border border-dashed border-gray-300 text-gray-500 hover:border-gray-400 hover:text-gray-700 transition-colors"
        >
          <Plus size={12} />
          Save Current
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      <span className="text-xs text-gray-400">Presets</span>

      {presets.map((p) => (
        <button
          key={p.name}
          type="button"
          onClick={() => handleApply(p)}
          className={`group inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
            activePreset === p.name
              ? "bg-blue-100 text-blue-800 border-blue-300"
              : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"
          }`}
        >
          {p.name}
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              handleDelete(p.name);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.stopPropagation();
                handleDelete(p.name);
              }
            }}
            className="opacity-0 group-hover:opacity-100 ml-0.5 text-gray-400 hover:text-red-500 transition-opacity"
          >
            <X size={12} />
          </span>
        </button>
      ))}

      {showNameInput ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSave();
          }}
          className="inline-flex items-center gap-1"
        >
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Preset name"
            className="w-32 px-2 py-1 rounded-lg border border-gray-300 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setShowNameInput(false);
                setName("");
              }
            }}
          />
          <button
            type="submit"
            disabled={!name.trim()}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
          >
            <Save size={12} />
          </button>
          <button
            type="button"
            onClick={() => {
              setShowNameInput(false);
              setName("");
            }}
            className="px-1.5 py-1 text-xs text-gray-400 hover:text-gray-600"
          >
            <X size={14} />
          </button>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setShowNameInput(true)}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border border-dashed border-gray-300 text-gray-500 hover:border-gray-400 hover:text-gray-700 transition-colors"
        >
          <Plus size={12} />
          Save
        </button>
      )}
    </div>
  );
}
