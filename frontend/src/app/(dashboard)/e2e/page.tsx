"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { Star } from "lucide-react";
import Tooltip from "@/components/Tooltip";
import {
  startSession,
  elaborateIntent,
  fetchSessions,
  fetchEnvs,
  fetchResourceAuto,
  stopSession,
  type SessionDetail,
  type ResourceAutoResult,
  type IntentCriterion,
  type EnvInfo,
} from "@/lib/api";
import { parseSeeds } from "@/lib/format";
import EnvSelector from "@/components/EnvSelector";
import LlmModelSelector, { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import VlmSelector from "@/components/VlmSelector";
import ThinkingEffortSelector, { type ThinkingEffort } from "@/components/ThinkingEffortSelector";
import SessionCard from "@/components/SessionCard";
import PresetBar from "@/components/PresetBar";
import { type E2EPresetParams } from "@/lib/presets";
import { loadModelPrefs, saveModelPrefs } from "@/lib/model-cache";

export default function E2EPage() {
  const router = useRouter();
  const [model, setModelRaw] = useState<string>(DEFAULT_LLM_MODEL);
  const [prompt, setPrompt] = useState("");
  const [numConfigs, setNumConfigs] = useState(3);
  const [seeds, setSeeds] = useState("1");
  const [timesteps, setTimesteps] = useState(10_000_000);
  const [maxIterations, setMaxIterations] = useState(20);
  const [passThreshold, setPassThreshold] = useState(0.9);
  const [envId, setEnvId] = useState("Humanoid-v5");
  const [envEngine, setEnvEngine] = useState("mujoco");
  const [numEnvs, setNumEnvs] = useState(32);
  const [vlmModel, setVlmModelRaw] = useState("gemini-3.1-pro-preview");
  const [numEvals, setNumEvals] = useState(4);
  const [useCodeJudge, setUseCodeJudge] = useState(true);
  const [coresPerRun, setCoresPerRun] = useState(8);
  const [thinkingEffort, setThinkingEffortRaw] = useState<ThinkingEffort>("max");
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from localStorage after mount to avoid SSR mismatch.
  useEffect(() => {
    const cached = loadModelPrefs();
    if (cached.llm) setModelRaw(cached.llm);
    if (cached.vlm) setVlmModelRaw(cached.vlm);
    if (cached.thinkingEffort) setThinkingEffortRaw(cached.thinkingEffort);
    setHydrated(true);
  }, []);

  const setModel = useCallback((v: string) => { setModelRaw(v); saveModelPrefs({ llm: v }); }, []);
  const setVlmModel = useCallback((v: string) => { setVlmModelRaw(v); saveModelPrefs({ vlm: v }); }, []);
  const setThinkingEffort = useCallback((v: ThinkingEffort) => { setThinkingEffortRaw(v); saveModelPrefs({ thinkingEffort: v }); }, []);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [stopError, setStopError] = useState("");
  const [elicitPhase, setElicitPhase] = useState<"idle" | "criteria">("idle");
  const [criteria, setCriteria] = useState<IntentCriterion[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [customCriteria, setCustomCriteria] = useState<string[]>([]);
  const [customInput, setCustomInput] = useState("");
  const [elaborating, setElaborating] = useState(false);
  const [starredOnly, setStarredOnly] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  function resetElicitation() {
    setElicitPhase("idle");
    setCriteria([]);
    setSelectedIndices(new Set());
    setCustomCriteria([]);
    setCustomInput("");
  }

  function addCustomCriterion() {
    const trimmed = customInput.trim();
    if (trimmed) {
      setCustomCriteria((prev) => [...prev, trimmed]);
      setCustomInput("");
    }
  }

  const {
    data: sessions,
    error: sessionsError,
    mutate: mutateSessions,
  } = useSWR<SessionDetail[]>("sessions", fetchSessions, {
    refreshInterval: 5000,
  });

  const { data: allEnvs } = useSWR<EnvInfo[]>("envs", fetchEnvs);

  // Sync envEngine whenever envId or env list changes
  useEffect(() => {
    const env = allEnvs?.find((e) => e.env_id === envId);
    if (env) setEnvEngine(env.engine);
  }, [envId, allEnvs]);

  // Derive seed list once for SWR key + submit
  const parsedSeeds = useMemo(() => parseSeeds(seeds), [seeds]);
  const parsedSeedCount = parsedSeeds.length;

  // Auto-compute resource allocation when configs/seeds/env change
  const { data: autoResource } = useSWR<ResourceAutoResult>(
    parsedSeedCount > 0 && numConfigs > 0
      ? `resource-auto-${numConfigs}-${parsedSeedCount}-${envId}`
      : null,
    () => fetchResourceAuto(numConfigs, parsedSeedCount, envId),
  );

  const displayCores = coresPerRun > 0 ? coresPerRun : (autoResource?.cores_per_run ?? 4);
  const autoDerivedEnvs = coresPerRun > 0 ? coresPerRun * 4 : (autoResource?.num_envs ?? 16);
  const usableCores = autoResource?.usable_cores ?? 62;
  const displayEnvs = numEnvs > 0 ? numEnvs : autoDerivedEnvs;
  const displayParallel = coresPerRun > 0 ? Math.floor(usableCores / coresPerRun) : (autoResource?.max_parallel ?? 10);
  // IsaacLab runs all envs in 1 GPU process; MuJoCo uses 1 OS process per env
  const isGpuVec = envEngine === "isaaclab";
  const totalProcesses = isGpuVec ? displayParallel : displayParallel * displayEnvs;
  const processLimitExceeded = !isGpuVec && totalProcesses > 700;

  async function handleStop(sessionId: string) {
    setStopError("");
    try {
      await stopSession(sessionId);
      await mutateSessions();
    } catch (err) {
      setStopError(
        `Failed to stop session: ${err instanceof Error ? err.message : "unknown error"}`,
      );
    }
  }

  async function handleElaborate() {
    setError("");
    setElaborating(true);
    try {
      const res = await elaborateIntent(prompt, envId, model);
      setCriteria(res.criteria);
      const defaults = new Set<number>();
      res.criteria.forEach((c, i) => { if (c.default_on) defaults.add(i); });
      setSelectedIndices(defaults);
      setCustomCriteria([]);
      setCustomInput("");
      setElicitPhase("criteria");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to elaborate intent");
    } finally {
      setElaborating(false);
    }
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setStopError("");
    setSubmitting(true);
    try {
      const rawParts = seeds.split(",").map((s) => s.trim()).filter((s) => s !== "");
      const seedList = parsedSeeds;

      if (seedList.length === 0) {
        setError("At least one valid seed is required (non-negative integers)");
        setSubmitting(false);
        return;
      }
      if (seedList.length > 100) {
        setError("Maximum 100 seeds allowed");
        setSubmitting(false);
        return;
      }
      if (seedList.length < rawParts.length) {
        setError(`${rawParts.length - seedList.length} invalid seed value(s) removed — check input`);
        setSubmitting(false);
        return;
      }

      // Build elaborated_intent from selected criteria.
      // If no criteria selected, leave undefined so the backend falls back to the original prompt.
      let elaboratedIntent: string | undefined;
      if (elicitPhase === "criteria") {
        const selected = criteria
          .filter((_, i) => selectedIndices.has(i))
          .map((c) => c.description);
        const allCriteria = [...selected, ...customCriteria];
        if (allCriteria.length > 0) {
          elaboratedIntent = `${prompt}\n\nBehavioral criteria:\n${allCriteria.map((c) => `- ${c}`).join("\n")}`;
        }
      }

      const { session_id } = await startSession({
        prompt,
        model,
        total_timesteps: timesteps,
        seed: seedList[0],
        max_iterations: maxIterations,
        pass_threshold: passThreshold,
        env_id: envId,
        num_envs: numEnvs,
        vlm_model: vlmModel,
        use_code_judge: useCodeJudge,
        review_reward: true,
        review_judge: true,
        num_configs: numConfigs,
        seeds: seedList,
        cores_per_run: coresPerRun,
        num_evals: numEvals,
        side_info: true,
        use_zoo_preset: true,
        hp_tuning: true,
        trajectory_stride: 1,
        thinking_effort: thinkingEffort,
        judgment_select: "last",
        elaborated_intent: elaboratedIntent,
        refined_initial_frame: true,
        criteria_diagnosis: true,
        motion_trail_dual: true,
      });
      router.push(`/e2e/${session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start");
      setSubmitting(false);
    }
  }

  function getE2EParams(): E2EPresetParams {
    return {
      model, prompt, numConfigs, seeds, timesteps, maxIterations, passThreshold,
      envId, numEnvs, vlmModel, numEvals, useCodeJudge, coresPerRun, device: "auto", thinkingEffort,
    };
  }

  function applyE2EPreset(p: E2EPresetParams) {
    setModel(p.model ?? DEFAULT_LLM_MODEL);
    setPrompt(p.prompt);
    setNumConfigs(p.numConfigs);
    setSeeds(p.seeds);
    setTimesteps(p.timesteps);
    setMaxIterations(p.maxIterations);
    setPassThreshold(p.passThreshold);
    setEnvId(p.envId);
    setNumEnvs(p.numEnvs);
    setVlmModel(p.vlmModel);
    setNumEvals(p.numEvals);
    setUseCodeJudge(p.useCodeJudge ?? true);
    setCoresPerRun(p.coresPerRun);
    setThinkingEffort(p.thinkingEffort ?? "max");
  }

  const filteredSessions = useMemo(
    () => (sessions ? (starredOnly ? sessions.filter((s) => s.starred) : sessions) : []),
    [sessions, starredOnly],
  );

  const groupedCriteria = useMemo(
    () =>
      Object.entries(
        criteria.reduce<Record<string, { criterion: IntentCriterion; index: number }[]>>(
          (groups, c, i) => {
            const cat = c.category || "other";
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push({ criterion: c, index: i });
            return groups;
          },
          {},
        ),
      ),
    [criteria],
  );

  const handleStopCb = useCallback(
    (sessionId: string) => handleStop(sessionId),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mutateSessions],
  );

  // Defer render until localStorage preferences are loaded to avoid flash.
  if (!hydrated) {
    return (
      <div className="animate-pulse space-y-4 p-8">
        <div className="h-8 w-48 bg-gray-200 rounded" />
        <div className="h-64 bg-gray-100 rounded-xl" />
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">
        E2E Experimental
      </h1>

      {/* Start Session Form + Criteria Panel */}
      <div className="flex gap-6 mb-10 items-start">
        <div className="w-[28rem] shrink-0 bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
            Start Session
          </h2>
          <PresetBar<E2EPresetParams>
            target="e2e"
            getCurrentParams={getE2EParams}
            onApply={applyE2EPreset}
          />
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Intent
              </label>
              <textarea
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value);
                  if (elicitPhase === "criteria") resetElicitation();
                }}
                placeholder="e.g., Run forward as fast as possible"
                rows={3}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <EnvSelector value={envId} onChange={(v, env) => {
              setEnvId(v);
              // Only auto-set numEnvs for IsaacLab (where 4096 vs 1 matters).
              // MuJoCo zoo presets have zoo_num_envs=1 which is too low for our
              // SubprocVecEnv pipeline — keep the user's current value.
              if (env?.engine === "isaaclab" && env.zoo_num_envs > 1) {
                setNumEnvs(env.zoo_num_envs);
                setTimesteps(50_000_000);
              } else if (env?.engine === "mujoco") {
                setTimesteps(10_000_000);
              }
              if (elicitPhase === "criteria") resetElicitation();
            }} />

            <button
              type="button"
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-gray-700"
            >
              <span
                className={`transition-transform text-xs ${showSettings ? "rotate-90" : ""}`}
              >
                &#9654;
              </span>
              Settings
            </button>

            {showSettings && (
              <div className="space-y-4 border-t border-gray-100 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <LlmModelSelector value={model} onChange={setModel} />
                  <ThinkingEffortSelector value={thinkingEffort} onChange={setThinkingEffort} model={model} />
                </div>

                <VlmSelector value={vlmModel} onChange={setVlmModel} />

                <label htmlFor="useCodeJudge" className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="useCodeJudge"
                    checked={useCodeJudge}
                    onChange={(e) => setUseCodeJudge(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Code-Based Judge</span>
                  <Tooltip content="Generate and use a code-based judge alongside VLM" />
                </label>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Num Configs
                      <Tooltip content="1 = baseline only, 2+ = baseline + perturbed" />
                    </label>
                    <input
                      type="number"
                      value={numConfigs}
                      onChange={(e) => setNumConfigs(Number(e.target.value))}
                      min={1}
                      max={10}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Seeds (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={seeds}
                      onChange={(e) => setSeeds(e.target.value)}
                      placeholder="1, 2, 3"
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Total Timesteps
                      <Tooltip content="Total environment steps per training run. Higher = longer training but potentially better policy." />
                    </label>
                    <input
                      type="text"
                      value={timesteps.toLocaleString()}
                      onChange={(e) => { const v = Number(e.target.value.replace(/,/g, "")); if (!isNaN(v) && v >= 10000) setTimesteps(v); }}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Num Envs
                      <Tooltip content="Number of parallel environments per training run. More envs = faster data collection but more CPU/GPU usage." />
                    </label>
                    <input
                      type="number"
                      value={numEnvs}
                      onChange={(e) => setNumEnvs(Number(e.target.value))}
                      min={0}
                      max={8192}
                      placeholder={autoResource ? String(autoResource.num_envs) : "auto"}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Max Iterations
                      <Tooltip content="Maximum reward revision cycles. The loop stops early if the pass threshold is reached." />
                    </label>
                    <input
                      type="number"
                      value={maxIterations}
                      onChange={(e) => setMaxIterations(Number(e.target.value))}
                      min={1}
                      max={1000}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Pass Threshold
                      <Tooltip content="Intent score (0-1) required to declare success and stop iterating." />
                    </label>
                    <input
                      type="number"
                      value={passThreshold}
                      onChange={(e) => setPassThreshold(Number(e.target.value))}
                      min={0}
                      max={1}
                      step={0.1}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Evals
                      <Tooltip content="Evenly-spaced checkpoints to evaluate. Each produces a rollout video." />
                    </label>
                    <input
                      type="number"
                      value={numEvals}
                      onChange={(e) => setNumEvals(parseInt(e.target.value, 10) || 4)}
                      min={1}
                      max={50}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Cores/Run
                      <Tooltip content="CPU cores allocated per training run. Controls taskset pinning and determines max parallel runs." />
                    </label>
                    <input
                      type="number"
                      value={coresPerRun}
                      onChange={(e) => setCoresPerRun(Number(e.target.value))}
                      min={0}
                      max={128}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                {processLimitExceeded && (
                  <p className="text-xs text-red-600">
                    max_parallel ({displayParallel}) &times; num_envs ({displayEnvs}) = {totalProcesses} &gt; 700 limit
                  </p>
                )}
              </div>
            )}

            {/* Auto Resource Allocation */}
            {autoResource && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3">
                <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide mb-2">
                  Auto Resource Allocation
                </p>
                <p className={`text-xs mt-1 ${processLimitExceeded ? "text-red-600 font-medium" : "text-blue-600"}`}>
                  {displayCores} core / {displayEnvs} env per run
                  {isGpuVec && " (GPU vectorized)"}
                  {" · "}max {displayParallel} runs concurrent
                  {!isGpuVec && <>{" · "}~{totalProcesses} procs</>}
                  {processLimitExceeded && " ⚠ exceeds 700"}
                </p>
              </div>
            )}

            {error && <p className="text-sm text-red-600">{error}</p>}

            <div className="grid grid-cols-2 gap-3">
              <button
                type="submit"
                disabled={submitting || elaborating || processLimitExceeded}
                className="bg-blue-600 text-white py-2.5 rounded-lg font-medium text-sm hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
              >
                {submitting ? "Starting..." : "Start Session"}
              </button>
              <button
                type="button"
                onClick={handleElaborate}
                disabled={submitting || elaborating || !prompt.trim()}
                className="bg-gray-100 text-gray-700 py-2.5 rounded-lg font-medium text-sm hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 transition-colors border border-gray-300"
              >
                {elaborating ? "Analyzing..." : "Elaborate Intent"}
              </button>
            </div>
          </form>
        </div>

        {/* Criteria Panel (right side) */}
        {elicitPhase === "criteria" && criteria.length > 0 && (
          <div className="flex-1 min-w-0 bg-white rounded-xl border border-blue-200 shadow-sm p-6 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                Behavioral Criteria
              </h2>
              <button
                type="button"
                onClick={resetElicitation}
                className="text-xs text-gray-400 hover:text-gray-600 underline"
              >
                Clear
              </button>
            </div>
            {groupedCriteria.map(([category, items]) => (
              <div key={category}>
                <p className="text-xs font-medium text-blue-600 uppercase tracking-wide mb-1">
                  {category}
                </p>
                {items.map(({ criterion, index }) => (
                  <div key={index} className="py-1">
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedIndices.has(index)}
                        onChange={() => {
                          setSelectedIndices((prev) => {
                            const next = new Set(prev);
                            if (next.has(index)) next.delete(index);
                            else next.add(index);
                            return next;
                          });
                        }}
                        className="h-4 w-4 mt-0.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-800">{criterion.title || criterion.description.split(":")[0]}</span>
                    </label>
                    <details className="ml-6 mt-0.5">
                      <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">detail</summary>
                      <p className="text-xs text-gray-500 mt-1">{criterion.description}</p>
                    </details>
                  </div>
                ))}
              </div>
            ))}
            {/* Custom criteria */}
            <div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value)}
                  placeholder="Add custom criterion..."
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addCustomCriterion();
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={addCustomCriterion}
                  className="px-3 py-1.5 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200"
                >
                  Add
                </button>
              </div>
              {customCriteria.length > 0 && (
                <div className="mt-2 space-y-1">
                  {customCriteria.map((c, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-gray-700">
                      <span>- {c}</span>
                      <button
                        type="button"
                        onClick={() => setCustomCriteria((prev) => prev.filter((_, j) => j !== i))}
                        className="text-xs text-red-500 hover:text-red-700"
                      >
                        remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {stopError && (
        <p className="text-sm text-red-600 mb-4">{stopError}</p>
      )}

      {/* Session List */}
      {sessionsError ? (
        <p className="text-center text-sm text-red-600 py-12">
          Failed to load sessions. Is the API server running?
        </p>
      ) : sessions === undefined ? null : sessions.length > 0 ? (
        <div>
          <div className="flex items-center gap-3 mb-3">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
              Sessions
            </h2>
            <button
              onClick={() => setStarredOnly(!starredOnly)}
              className={`cursor-pointer flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border transition-colors ${
                starredOnly
                  ? "bg-yellow-50 border-yellow-300 text-yellow-700"
                  : "border-gray-200 text-gray-400 hover:border-gray-300"
              }`}
            >
              <Star
                size={12}
                className={starredOnly ? "fill-yellow-400 text-yellow-400" : ""}
              />
              {starredOnly ? "Starred" : "All"}
            </button>
            <span className="text-xs text-gray-400">
              {filteredSessions.length} / {sessions.length}
            </span>
          </div>
          <div className="grid gap-4 max-h-[calc(100vh-200px)] overflow-y-auto">
            {filteredSessions.map((session) => (
              <SessionCard
                key={session.session_id}
                session={session}
                onStop={handleStopCb}
                onMutate={mutateSessions}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="text-center py-12 text-gray-400 text-sm">
          No sessions yet. Start one above.
        </div>
      )}
    </div>
  );
}
