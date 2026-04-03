"use client";

import { useState } from "react";
import Tooltip from "@/components/Tooltip";
import LlmModelSelector, { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import ThinkingEffortSelector, { type ThinkingEffort } from "@/components/ThinkingEffortSelector";
import VlmSelector from "@/components/VlmSelector";
import type { BenchmarkOptions } from "@/lib/api";

export interface BenchmarkFormState {
  model: string;
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
  filterEnvs: Set<string>;
  filterCategories: Set<string>;
  filterDifficulties: Set<string>;
  device: "auto" | "cpu";
  csvFile: string;
  backend: "local" | "ssh";
  thinkingEffort: ThinkingEffort;
  criteriaDiagnosis: boolean;
  motionTrailDual: boolean;
}

export const BENCHMARK_DEFAULTS: BenchmarkFormState = {
  model: DEFAULT_LLM_MODEL,
  timesteps: 10_000_000,
  numConfigs: 3,
  seedsStr: "1",
  maxIterations: 10,
  passThreshold: 0.9,
  numEnvs: 32,
  maxParallel: 0,
  coresPerRun: 6,
  vlmModel: "gemini-3.1-pro-preview",
  useCodeJudge: true,
  filterEnvs: new Set(),
  filterCategories: new Set(),
  filterDifficulties: new Set(),
  device: "auto",
  csvFile: "test_cases_exotic_ant_halfcheetah_humanoid.csv",
  backend: "local",
  thinkingEffort: "max",
  criteriaDiagnosis: true,
  motionTrailDual: true,
};

function toggleSet(prev: Set<string>, value: string): Set<string> {
  const next = new Set(prev);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

function ToggleChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${active ? "bg-blue-100 text-blue-800 border-blue-300" : "bg-white text-gray-500 border-gray-200 hover:border-gray-300"}`}>
      {label}
    </button>
  );
}

function FilterSection({ title, options, selected, onToggle }: { title: string; options: string[]; selected: Set<string>; onToggle: (v: string) => void }) {
  const allSelected = selected.size === 0;
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="block text-sm font-medium text-gray-700">{title}</label>
        {!allSelected && (
          <button type="button" onClick={() => { for (const o of [...selected]) onToggle(o); }} className="text-xs text-gray-400 hover:text-gray-600">Clear</button>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => (
          <ToggleChip key={opt} label={opt} active={allSelected || selected.has(opt)} onClick={() => onToggle(opt)} />
        ))}
      </div>
      {allSelected && <p className="text-xs text-gray-400 mt-1">All selected — click to pick specific ones</p>}
    </div>
  );
}

export default function BenchmarkFormFields({
  state,
  onChange,
  options,
  idPrefix = "",
}: {
  state: BenchmarkFormState;
  onChange: <K extends keyof BenchmarkFormState>(key: K, value: BenchmarkFormState[K]) => void;
  options?: BenchmarkOptions;
  idPrefix?: string;
}) {
  const [showSettings, setShowSettings] = useState(false);
  const pfx = idPrefix ? `${idPrefix}-` : "";

  const totalProcesses = state.maxParallel * state.numEnvs;
  const isIsaacLab = state.csvFile.toLowerCase().includes("isaaclab");
  const processLimitExceeded = !isIsaacLab && totalProcesses > 640;

  return (
    <>
      {/* Backend Toggle */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Backend</label>
        <div className="flex gap-1">
          <button type="button" onClick={() => onChange("backend", "local")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-colors ${state.backend === "local" ? "bg-green-100 text-green-800 border border-green-300" : "bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200"}`}>Local</button>
          <button type="button" onClick={() => onChange("backend", "ssh")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-colors ${state.backend === "ssh" ? "bg-orange-100 text-orange-800 border border-orange-300" : "bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200"}`}>SSH (Distributed)</button>
        </div>
        <p className="text-xs text-gray-400 mt-1">
          {state.backend === "local" ? "Run all sessions on this server" : "Distribute sessions across registered SSH nodes"}
        </p>
      </div>

      {/* CSV File Selector */}
      {options && options.csv_files && options.csv_files.length > 1 && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Test Suite</label>
          <select
            value={state.csvFile}
            onChange={(e) => {
              const csv = e.target.value;
              onChange("csvFile", csv);
              onChange("filterEnvs", new Set());
              onChange("filterCategories", new Set());
              onChange("filterDifficulties", new Set());
              // Auto-set IsaacLab defaults when switching to an isaaclab suite
              if (csv.toLowerCase().includes("isaaclab")) {
                onChange("numEnvs", 4096);
                onChange("timesteps", 50_000_000);
              } else {
                onChange("numEnvs", BENCHMARK_DEFAULTS.numEnvs);
                onChange("timesteps", BENCHMARK_DEFAULTS.timesteps);
              }
            }}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          >
            {options.csv_files.map((f) => (
              <option key={f} value={f}>
                {f.replace(".csv", "").replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Filters */}
      {options && (
        <div className="space-y-3">
          <FilterSection title="Environments" options={options.envs} selected={state.filterEnvs} onToggle={(v) => onChange("filterEnvs", toggleSet(state.filterEnvs, v))} />
          <FilterSection title="Categories" options={options.categories} selected={state.filterCategories} onToggle={(v) => onChange("filterCategories", toggleSet(state.filterCategories, v))} />
          <FilterSection title="Difficulty" options={options.difficulties} selected={state.filterDifficulties} onToggle={(v) => onChange("filterDifficulties", toggleSet(state.filterDifficulties, v))} />
        </div>
      )}

      {/* Settings toggle */}
      <button type="button" onClick={() => setShowSettings(!showSettings)} className="flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-gray-700">
        <span className={`transition-transform text-xs ${showSettings ? "rotate-90" : ""}`}>&#9654;</span>
        Settings
      </button>

      {showSettings && (
        <div className="space-y-4 border-t border-gray-100 pt-4">
          <LlmModelSelector value={state.model} onChange={(v) => onChange("model", v)} />

          <ThinkingEffortSelector value={state.thinkingEffort} onChange={(v) => onChange("thinkingEffort", v)} model={state.model} />

          <VlmSelector value={state.vlmModel} onChange={(v) => onChange("vlmModel", v)} />

          <div className="flex items-center gap-2">
            <input type="checkbox" id={`${pfx}useCodeJudge`} checked={state.useCodeJudge} onChange={(e) => onChange("useCodeJudge", e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
            <label htmlFor={`${pfx}useCodeJudge`} className="text-sm font-medium text-gray-700">Code-Based Judge</label>
            <p className="text-xs text-gray-400">Generate and use a code-based judge alongside VLM</p>
          </div>

          {/* Device Toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Device</label>
            <div className="flex gap-1">
              <button type="button" onClick={() => onChange("device", "auto")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-colors ${state.device === "auto" ? "bg-emerald-100 text-emerald-800 border border-emerald-300" : "bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200"}`}>Auto</button>
              <button type="button" onClick={() => onChange("device", "cpu")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-colors ${state.device === "cpu" ? "bg-gray-900 text-white border border-gray-900" : "bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200"}`}>CPU</button>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {state.device === "auto" ? "SB3 selects best available device (GPU if present)" : "Force CPU-only training (no GPU)"}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Total Timesteps<Tooltip content="Total environment steps per training run. Higher = longer training but potentially better policy." /></label>
            <input type="text" value={state.timesteps.toLocaleString()} onChange={(e) => { const v = Number(e.target.value.replace(/,/g, "")); if (!isNaN(v) && v >= 10000) onChange("timesteps", v); }} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Num Configs<Tooltip content="1 = baseline only, 2+ = baseline + perturbed" /></label>
              <input type="number" value={state.numConfigs} onChange={(e) => onChange("numConfigs", Number(e.target.value))} min={1} max={10} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Seeds (comma-separated)</label>
              <input type="text" value={state.seedsStr} onChange={(e) => onChange("seedsStr", e.target.value)} placeholder="1" className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Iterations<Tooltip content="Maximum reward revision cycles. The loop stops early if the pass threshold is reached." /></label>
              <input type="number" value={state.maxIterations} onChange={(e) => onChange("maxIterations", Number(e.target.value))} min={1} max={20} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Pass Threshold<Tooltip content="Intent score (0-1) required to declare success and stop iterating." /></label>
              <input type="number" value={state.passThreshold} onChange={(e) => onChange("passThreshold", Number(e.target.value))} min={0} max={1} step={0.1} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Num Envs<Tooltip content="Number of parallel environments per training run. More envs = faster data collection but more CPU/GPU usage." /></label>
            <input type="number" value={state.numEnvs} onChange={(e) => onChange("numEnvs", Number(e.target.value))} min={0} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Parallel Sessions</label>
              <input type="number" value={state.maxParallel} onChange={(e) => onChange("maxParallel", Math.max(0, Number(e.target.value)))} min={0} max={64} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cores/Run<Tooltip content="CPU cores allocated per training run. Controls taskset pinning and determines max parallel runs." /></label>
              <input type="number" value={state.coresPerRun} onChange={(e) => onChange("coresPerRun", Math.max(0, Number(e.target.value)))} min={0} max={64} className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
          </div>
          {processLimitExceeded && (
            <p className="text-xs text-red-600">
              max_parallel ({state.maxParallel}) &times; num_envs ({state.numEnvs}) = {totalProcesses} &gt; 640 limit
            </p>
          )}
        </div>
      )}
    </>
  );
}
