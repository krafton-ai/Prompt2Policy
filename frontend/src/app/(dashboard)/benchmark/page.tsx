// Benchmark tab — job scheduler for benchmark runs.
// See src/p2p/scheduler/__init__.py.
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import NodeCard from "@/components/NodeCard";
import RemoteJobCard from "@/components/RemoteJobCard";
import { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import type { ThinkingEffort } from "@/components/ThinkingEffortSelector";
import BenchmarkFormFields, { type BenchmarkFormState, BENCHMARK_DEFAULTS } from "@/components/BenchmarkFormFields";
import {
  fetchNodes,
  fetchJobs,
  addNode,
  submitBenchmarkJob,
  bulkTrashJobs,
  type NodeResponse,
  type JobResponse,
} from "@/lib/scheduler-api";
import { fetchBenchmarks, fetchBenchmarkOptions, fetchResourceAuto, type BenchmarkRunSummary, type BenchmarkOptions, type ResourceAutoResult } from "@/lib/api";
import BenchmarkRunCard from "@/components/BenchmarkRunCard";
import PresetBar from "@/components/PresetBar";
import { type BenchmarkPresetParams } from "@/lib/presets";
import { buildBenchmarkCli } from "@/lib/cli-builder";
import { loadModelPrefs, saveModelPrefs } from "@/lib/model-cache";

// ---------------------------------------------------------------------------
// CLI Preview
// ---------------------------------------------------------------------------

function CliPreview({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = command;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative rounded-lg bg-gray-900 text-gray-100 text-xs font-mono">
      <button
        type="button"
        onClick={handleCopy}
        className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-sans bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
      >
        {copied ? "Copied!" : "Copy"}
      </button>
      <pre className="p-4 pr-20 max-h-96 overflow-auto whitespace-pre-wrap break-all">
        {command}
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Add Node Form
// ---------------------------------------------------------------------------

function AddNodeForm({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false);
  const [nodeId, setNodeId] = useState("");
  const [host, setHost] = useState("");
  const [user, setUser] = useState("");
  const [port, setPort] = useState(22);
  const [baseDir, setBaseDir] = useState("/home/user/p2p");
  const [maxCores, setMaxCores] = useState(60);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await addNode({
        node_id: nodeId, host, user, port,
        base_dir: baseDir, max_cores: maxCores,
      });
      setNodeId(""); setHost(""); setUser("");
      setPort(22); setBaseDir("/home/user/p2p"); setMaxCores(60);
      setOpen(false);
      onAdded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add node");
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} className="px-4 py-2 text-sm rounded-lg bg-gray-900 text-white hover:bg-gray-800">
        + Add Node
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Node ID</label>
          <input value={nodeId} onChange={(e) => setNodeId(e.target.value)} required placeholder="gpu-node-1" className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Host</label>
          <input value={host} onChange={(e) => setHost(e.target.value)} required placeholder="192.168.1.10" className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">User</label>
          <input value={user} onChange={(e) => setUser(e.target.value)} required placeholder="researcher" className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Port</label>
          <input type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Base Dir</label>
          <input value={baseDir} onChange={(e) => setBaseDir(e.target.value)} className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Max Cores</label>
          <input type="number" min={1} value={maxCores} onChange={(e) => setMaxCores(Number(e.target.value))} className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        </div>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      <div className="flex gap-2">
        <button type="submit" disabled={submitting} className="px-4 py-1.5 text-sm rounded-md bg-gray-900 text-white hover:bg-gray-800 disabled:opacity-50">
          {submitting ? "Adding..." : "Add"}
        </button>
        <button type="button" onClick={() => setOpen(false)} className="px-4 py-1.5 text-sm rounded-md bg-gray-100 text-gray-700 hover:bg-gray-200">
          Cancel
        </button>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Submit Benchmark Job Form
// ---------------------------------------------------------------------------

function SubmitJobForm({ onSubmitted }: { onSubmitted: () => void }) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const [bmForm, setBmForm] = useState<BenchmarkFormState>({ ...BENCHMARK_DEFAULTS });
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from localStorage after mount to avoid SSR mismatch.
  useEffect(() => {
    const cached = loadModelPrefs();
    setBmForm((prev) => ({
      ...prev,
      ...(cached.llm ? { model: cached.llm } : {}),
      ...(cached.vlm ? { vlmModel: cached.vlm } : {}),
      ...(cached.thinkingEffort ? { thinkingEffort: cached.thinkingEffort } : {}),
    }));
    setHydrated(true);
  }, []);
  const updateBm = <K extends keyof BenchmarkFormState>(key: K, value: BenchmarkFormState[K]) => {
    setBmForm((prev) => ({ ...prev, [key]: value }));
    if (key === "model") saveModelPrefs({ llm: value as string });
    else if (key === "vlmModel") saveModelPrefs({ vlm: value as string });
    else if (key === "thinkingEffort") saveModelPrefs({ thinkingEffort: value as ThinkingEffort });
  };

  const backend = bmForm.backend;

  const { data: benchmarkOptions } = useSWR<BenchmarkOptions>(
    `scheduler-benchmark-options-${bmForm.csvFile}`,
    () => fetchBenchmarkOptions(bmForm.csvFile),
  );

  // --- Auto resource ---
  const resolveEnvs = (numEnvs: number, coresPerRun: number) =>
    numEnvs > 0 ? numEnvs : (coresPerRun > 0 ? coresPerRun : 4);

  const parsedBmSeeds = bmForm.seedsStr.split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s)).map(Number);
  const bmNumSeeds = Math.max(1, parsedBmSeeds.length);
  const runsPerCase = bmForm.numConfigs * bmNumSeeds;

  const { data: autoResource } = useSWR<ResourceAutoResult>(
    `scheduler-resource-auto-${bmForm.numConfigs}-${bmNumSeeds}`,
    () => fetchResourceAuto(bmForm.numConfigs, bmNumSeeds),
  );

  const { data: nodesForAuto } = useSWR<NodeResponse[]>(
    backend === "ssh" ? "scheduler-nodes-for-auto" : null,
    fetchNodes,
  );

  const bmDisplayEnvs = resolveEnvs(bmForm.numEnvs, bmForm.coresPerRun);
  const bmDisplayCores = bmForm.coresPerRun > 0 ? bmForm.coresPerRun : Math.max(2, bmDisplayEnvs);
  const bmCaseCores = bmDisplayCores * bmForm.numConfigs * bmNumSeeds;

  // Per-node parallel capacity
  // MuJoCo: CPU-only, no GPU constraint → limited by cores only
  // IsaacLab: 1 session per GPU → limited by min(cores, GPUs)
  const isIsaacLab = bmForm.csvFile.toLowerCase().includes("isaaclab");
  const perNodeCapacity = (sessionCores: number) =>
    (nodesForAuto ?? []).filter((n) => n.enabled).map((n) => {
      const cpuParallel = sessionCores > 0 ? Math.floor(n.max_cores / sessionCores) : 1;
      const gpuParallel = (isIsaacLab && n.num_gpus > 0) ? n.num_gpus : Infinity;
      return {
        id: n.node_id,
        max: n.max_cores,
        num_gpus: n.num_gpus ?? 0,
        parallel: Math.min(cpuParallel, gpuParallel),
      };
    });

  const bmPerNode = perNodeCapacity(bmCaseCores);
  const sshAutoParallel = bmPerNode.reduce((sum, n) => sum + n.parallel, 0);

  const localAutoParallel = (() => {
    const usableCores = autoResource?.usable_cores ?? 62;
    const totalConcurrent = bmDisplayCores > 0
      ? Math.floor(usableCores / bmDisplayCores)
      : (autoResource?.max_parallel ?? 10);
    return Math.max(1, Math.floor(totalConcurrent / runsPerCase));
  })();
  const autoParallel = backend === "ssh" && sshAutoParallel > 0
    ? sshAutoParallel
    : localAutoParallel;
  const displayParallel = bmForm.maxParallel > 0 ? bmForm.maxParallel : autoParallel;

  // CLI preview command
  const cliCommand = useMemo(() => {
    const seeds = bmForm.seedsStr.split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s)).map(Number);
    const enabledNodes = (nodesForAuto ?? []).filter((n) => n.enabled).map((n) => n.node_id);
    return buildBenchmarkCli({
      backend,
      csvFile: bmForm.csvFile,
      totalTimesteps: bmForm.timesteps,
      seed: seeds[0] ?? 1,
      seeds,
      maxIterations: bmForm.maxIterations,
      passThreshold: bmForm.passThreshold,
      numEnvs: bmDisplayEnvs,
      numConfigs: bmForm.numConfigs,
      model: bmForm.model,
      vlmModel: bmForm.vlmModel,
      maxParallel: bmForm.maxParallel,
      coresPerRun: bmDisplayCores,
      useCodeJudge: bmForm.useCodeJudge,
      filterEnvs: [...bmForm.filterEnvs],
      filterCategories: [...bmForm.filterCategories],
      filterDifficulties: [...bmForm.filterDifficulties],
      device: bmForm.device,
      thinkingEffort: bmForm.thinkingEffort,
      criteriaDiagnosis: bmForm.criteriaDiagnosis,
      motionTrailDual: bmForm.motionTrailDual,
      nodes: enabledNodes,
    });
  }, [backend, nodesForAuto, bmForm, bmDisplayEnvs, bmDisplayCores]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const parsed = bmForm.seedsStr.split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s)).map(Number);
      if (parsed.length === 0) { setError("At least one valid seed is required"); setSubmitting(false); return; }
      await submitBenchmarkJob({
        backend,
        model: bmForm.model || undefined,
        csv_file: bmForm.csvFile !== "test_cases.csv" ? bmForm.csvFile : undefined,
        total_timesteps: bmForm.timesteps,
        seed: parsed[0],
        seeds: parsed,
        max_iterations: bmForm.maxIterations,
        pass_threshold: bmForm.passThreshold,
        num_envs: bmDisplayEnvs,
        num_configs: bmForm.numConfigs,
        vlm_model: bmForm.vlmModel || undefined,
        max_parallel: displayParallel,
        cores_per_run: bmDisplayCores,
        mode: "flat",
        filter_envs: bmForm.filterEnvs.size > 0 ? [...bmForm.filterEnvs] : undefined,
        filter_categories: bmForm.filterCategories.size > 0 ? [...bmForm.filterCategories] : undefined,
        filter_difficulties: bmForm.filterDifficulties.size > 0 ? [...bmForm.filterDifficulties] : undefined,
        side_info: true,
        use_zoo_preset: true,
        hp_tuning: true,
        use_code_judge: bmForm.useCodeJudge,
        review_reward: true,
        review_judge: true,
        device: bmForm.device,
        trajectory_stride: 1,
        thinking_effort: bmForm.thinkingEffort,
        refined_initial_frame: true,
        criteria_diagnosis: bmForm.criteriaDiagnosis,
        motion_trail_dual: bmForm.motionTrailDual,
      });
      onSubmitted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit job");
    } finally {
      setSubmitting(false);
    }
  };

  // Defer render until localStorage preferences are loaded to avoid flash.
  if (!hydrated) {
    return (
      <div className="animate-pulse space-y-4 p-6">
        <div className="h-8 w-48 bg-gray-200 rounded" />
        <div className="h-64 bg-gray-100 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
      <PresetBar<BenchmarkPresetParams>
        target="scheduler-benchmark"
        getCurrentParams={() => ({
          ...bmForm,
          filterEnvs: [...bmForm.filterEnvs],
          filterCategories: [...bmForm.filterCategories],
          filterDifficulties: [...bmForm.filterDifficulties],
        })}
        onApply={(p) => setBmForm({
          ...BENCHMARK_DEFAULTS,
          ...p,
          model: p.model ?? DEFAULT_LLM_MODEL,
          filterEnvs: new Set(p.filterEnvs ?? []),
          filterCategories: new Set(p.filterCategories ?? []),
          filterDifficulties: new Set(p.filterDifficulties ?? []),
        })}
      />
      <form onSubmit={handleSubmit} className="space-y-4">
        <BenchmarkFormFields state={bmForm} onChange={updateBm} options={benchmarkOptions} idPrefix="ssh-bm" />

        {/* Resource Allocation Summary */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 space-y-1">
          <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide">
            Resource Allocation
          </p>
          <p className="text-xs text-blue-600">
            1 case = {bmDisplayCores} core &times; {bmForm.numConfigs} config &times; {bmNumSeeds} seed = <span className="font-semibold">{bmCaseCores} cores</span>
          </p>
          {backend === "ssh" && bmPerNode.length > 0 ? (
            <>
              <p className="text-xs text-blue-600">
                Per node: {bmPerNode.map((n) => `${n.id}(${n.max}) → ${n.parallel}`).join(", ")}
              </p>
              <p className="text-xs text-blue-700 font-semibold">
                {bmPerNode.length} nodes → {displayParallel} cases parallel
                {bmForm.maxParallel > 0 && ` (manual)`}
              </p>
            </>
          ) : (
            <p className="text-xs text-blue-700 font-semibold">
              max {displayParallel} cases parallel
            </p>
          )}
        </div>

        {/* CLI Preview */}
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">CLI Command</p>
          <CliPreview command={cliCommand} />
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button type="submit" disabled={submitting} className="w-full bg-blue-600 text-white py-2.5 rounded-lg font-medium text-sm hover:bg-blue-700 disabled:bg-blue-300 transition-colors">
          {submitting ? "Submitting..." : "Submit Benchmark Job"}
        </button>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function SchedulerPage() {
  const { data: nodes, mutate: mutateNodes } = useSWR<NodeResponse[]>(
    "scheduler-nodes", fetchNodes, { refreshInterval: 10000 },
  );
  const { data: jobsData, mutate: mutateJobs } = useSWR<{ jobs: JobResponse[] }>(
    "scheduler-jobs", fetchJobs, { refreshInterval: 5000 },
  );
  const { data: benchmarks } = useSWR<BenchmarkRunSummary[]>(
    "benchmarks", fetchBenchmarks, { refreshInterval: 5000 },
  );
  const jobs = (jobsData?.jobs ?? []).filter((j) => j.job_type !== "session");
  const [sortBy, setSortBy] = useState<"newest" | "oldest" | "name" | "status">("newest");

  // Bulk selection state for jobs
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkLoading, setBulkLoading] = useState(false);
  const selectAllRef = useRef<HTMLInputElement>(null);

  // Prune stale IDs when jobs list changes (skip while SWR hasn't loaded yet)
  useEffect(() => {
    if (!jobsData) return;
    const validIds = new Set(jobs.map((j) => j.job_id));
    setSelected((prev) => {
      const pruned = new Set([...prev].filter((id) => validIds.has(id)));
      return pruned.size === prev.size ? prev : pruned;
    });
  }, [jobs, jobsData]);

  // Indeterminate state for Select All checkbox
  const allSelected = useMemo(
    () => jobs.length > 0 && jobs.every((j) => selected.has(j.job_id)),
    [jobs, selected],
  );
  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = selected.size > 0 && !allSelected;
    }
  }, [selected, allSelected]);

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    setSelected((prev) => {
      const allIds = new Set(jobs.map((j) => j.job_id));
      const isAll = allIds.size > 0 && [...allIds].every((id) => prev.has(id));
      return isAll ? new Set() : allIds;
    });
  }, [jobs]);

  const handleBulkTrash = useCallback(async () => {
    if (selected.size === 0) return;
    const confirmed = window.confirm(
      `Move ${selected.size} job(s) to Trash?`,
    );
    if (!confirmed) return;
    setBulkLoading(true);
    try {
      await bulkTrashJobs([...selected]);
    } catch (err) {
      console.error(err);
      alert("Failed to move some jobs to trash.");
    } finally {
      await mutateJobs();
      setSelected(new Set());
      setBulkLoading(false);
    }
  }, [selected, mutateJobs]);

  const sortedJobs = useMemo(() => {
    const sorted = [...jobs];
    switch (sortBy) {
      case "newest": sorted.sort((a, b) => b.created_at.localeCompare(a.created_at)); break;
      case "oldest": sorted.sort((a, b) => a.created_at.localeCompare(b.created_at)); break;
      case "name":   sorted.sort((a, b) => a.job_id.localeCompare(b.job_id)); break;
      case "status": sorted.sort((a, b) => a.status.localeCompare(b.status)); break;
    }
    return sorted;
  }, [jobs, sortBy]);

  return (
    <div className="space-y-8">
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Nodes</h2>
          <AddNodeForm onAdded={() => mutateNodes()} />
        </div>
        {nodes && nodes.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {nodes.map((node) => (
              <NodeCard key={node.node_id} node={node} onRemoved={() => mutateNodes()} onUpdated={() => mutateNodes()} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No nodes registered. Add a node to get started.</p>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Job Submission</h2>
        <SubmitJobForm onSubmitted={mutateJobs} />
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-gray-900">Job List</h2>
            {jobs.length > 0 && (
              <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer select-none">
                <input
                  ref={selectAllRef}
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleAll}
                  className="rounded border-gray-300"
                />
                Select All
              </label>
            )}
          </div>
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="rounded-md border border-gray-300 px-3 py-1.5 text-sm"
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="name">Name (A-Z)</option>
              <option value="status">Status</option>
            </select>
            {jobs.length > 0 && (
              <button
                onClick={handleBulkTrash}
                disabled={bulkLoading || selected.size === 0}
                className="px-3 py-1.5 text-sm rounded-lg bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
              >
                {bulkLoading ? "Moving..." : selected.size > 0 ? `Move ${selected.size} to Trash` : "Move to Trash"}
              </button>
            )}
          </div>
        </div>
        <div className="space-y-3">
          {sortedJobs.length > 0 ? (
            sortedJobs.map((job) => (
              <RemoteJobCard
                key={job.job_id}
                job={job}
                onCancelled={() => mutateJobs()}
                onDeleted={() => mutateJobs()}
                selectable
                selected={selected.has(job.job_id)}
                onToggle={() => toggleSelect(job.job_id)}
              />
            ))
          ) : (
            <p className="text-sm text-gray-500">No jobs submitted yet.</p>
          )}
        </div>
      </section>

      {/* Local Benchmarks */}
      {benchmarks && benchmarks.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Local Benchmarks</h2>
          <div className="grid gap-4">
            {benchmarks.map((r) => (
              <BenchmarkRunCard key={r.benchmark_id} run={r} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
