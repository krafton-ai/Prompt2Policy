"use client";

import { useState, useMemo, useEffect, useLayoutEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import useSWR from "swr";
import {
  ArrowRight,
  X,
  RefreshCcw,
  ChevronRight,
  AlertCircle,
  LayoutDashboard,
} from "lucide-react";
import {
  startSession,
  fetchSessions,
  fetchEnvs,
  fetchProviders,
  type SessionDetail,
  type EnvInfo,
  type ProviderStatus,
} from "@/lib/api";
import { timeAgo } from "@/lib/format";
import { MODEL_OPTIONS } from "@/components/LlmModelSelector";
import { VLM_OPTIONS } from "@/components/VlmSelector";
import StatusBadge from "@/components/StatusBadge";
import { loadModelPrefs, saveModelPrefs } from "@/lib/model-cache";

/* ── Sample-run showcase chips ── */
const SAMPLE_RUNS = [
  {
    label: "Breakdance ground spin",
    intent: "Perform a breakdance-style spin lower the body close to the ground, then rotate rapidly around the vertical axis while staying low. Complete several full rotations in a row without stopping.",
    env: "Ant-v5",
    video: "/sample_runs/Perform a breakdance-style spin lower the body close to the ground, then rotate rapidly around the vertical axis while staying low. Complete several full rotations in a row without stopping..mp4",
  },
  {
    label: "Continuous forward tumbling",
    intent: "Perform continuous forward tumbling on the ground. The body should roll forward repeatedly, using the leg to drive rotation while maintaining contact or near-contact with the ground. Chain multiple forward rolls.",
    env: "Hopper-v5",
    video: "/sample_runs/Perform continuous forward tumbling on the ground. The body should roll forward repeatedly, using the leg to drive rotation while maintaining contact or near-contact with the ground. Chain multiple forwar.mp4",
  },
  {
    label: "Forward roll",
    intent: "Perform a forward roll tuck the head and roll the body forward over the ground through a full somersault, then return to a standing or crouching posture. Complete at least one full roll.",
    env: "Humanoid-v5",
    video: "/sample_runs/Perform a forward roll tuck the head and roll the body forward over the ground through a full somersault, then return to a standing or crouching posture. Complete at least one full roll..mp4",
  },
  {
    label: "Lunge hold",
    intent: "Hold a lunge position step one leg forward with the knee bent and the rear leg extended behind with the knee close to the ground. Hold this stance steadily with the torso upright.",
    env: "Humanoid-v5",
    video: "/sample_runs/Hold a lunge position step one leg forward with the knee bent and the rear leg extended behind with the knee close to the ground. Hold this stance steadily with the torso upright..mp4",
  },
  {
    label: "Rear legs stand",
    intent: "Stand on the rear two legs only with the front two legs lifted off the ground. Hold this pose steadily.",
    env: "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    video: "/sample_runs/Stand on the rear two legs only with the front two legs lifted off the ground. Hold this pose steadily..mp4",
  },
];

/* Derive {value, label} lists from canonical model arrays */
const LLM_ITEMS = MODEL_OPTIONS.map((o) => ({ value: o.id, label: o.name }));
const VLM_ITEMS = VLM_OPTIONS.map((o) => ({ value: o.id, label: o.name }));

/* Map model ID to provider key in ProviderStatus */
function modelProvider(id: string): keyof ProviderStatus | null {
  if (id.startsWith("claude") || id.includes("opus") || id.includes("sonnet")) return "anthropic";
  if (id.startsWith("gpt-") || id.startsWith("o1") || id.startsWith("o3") || id.startsWith("o4")) return "openai";
  if (id.startsWith("gemini")) return "gemini";
  return null; // vLLM, "No VLM", etc. — always available
}

/* ── Score ring SVG ── */
function ScoreRing({ score, size = 36 }: { score: number; size?: number }) {
  const r = (size - 4) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - Math.max(0, Math.min(1, score)));
  const color =
    score >= 0.8 ? "#16a34a" : score >= 0.5 ? "#ca8a04" : "#dc2626";
  return (
    <svg width={size} height={size} className="shrink-0 -rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="#f1f5f9"
        strokeWidth={3}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={3}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        className="score-ring"
      />
      <text
        x={size / 2}
        y={size / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill={color}
        fontSize={size * 0.28}
        fontWeight={600}
        className="rotate-90"
        style={{ transformOrigin: "center" }}
      >
        {(score * 100).toFixed(0)}
      </text>
    </svg>
  );
}

/* ── Sample chip with video popover ── */
function SampleChip({
  sample,
  disabled,
  onSelect,
}: {
  sample: (typeof SAMPLE_RUNS)[number];
  disabled: boolean;
  onSelect: (s: (typeof SAMPLE_RUNS)[number]) => void;
}) {
  const [show, setShow] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
  }, []);

  function handleEnter() {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setShow(true);
  }
  function handleLeave() {
    timeoutRef.current = setTimeout(() => setShow(false), 150);
  }

  return (
    <span className="relative" onMouseEnter={handleEnter} onMouseLeave={handleLeave}>
      <button
        onClick={() => onSelect(sample)}
        disabled={disabled}
        className="chip flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-gray-200 text-sm text-gray-500 cursor-pointer shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`/env-thumbnails/${sample.env}.png`} alt={sample.env} width={20} height={20} className="rounded-sm" onError={(e) => { e.currentTarget.style.display = "none"; }} />
        {sample.label}
      </button>
      {show && sample.video && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 z-50 animate-fade-in">
          <div className="rounded-xl overflow-hidden shadow-xl w-64">
            <video
              src={encodeURI(sample.video)}
              autoPlay
              loop
              muted
              playsInline
              className="w-full aspect-square object-contain bg-black rounded-xl"
            />
          </div>
          <div className="absolute left-1/2 -translate-x-1/2 -bottom-1.5 w-3 h-3 rotate-45 bg-black" />
        </div>
      )}
    </span>
  );
}

/* ── Resolve model defaults from available providers ── */
function resolveModels(providers: ProviderStatus | undefined) {
  if (!providers) return null;
  if (providers.gemini) {
    return { model: "gemini-3.1-pro-preview", vlm: "gemini-3.1-pro-preview" } as const;
  }
  if (providers.anthropic) {
    return { model: "claude-opus-4-6", vlm: "claude-opus-4-6" } as const;
  }
  if (providers.openai) {
    return { model: "gpt-5.4", vlm: "" } as const;
  }
  return null;
}

/* ── Landing page ── */
export default function LandingPage() {
  const router = useRouter();

  /* ── Form state ── */
  const [prompt, setPrompt] = useState("");
  const [envId, setEnvId] = useState("Humanoid-v5");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [thumbError, setThumbError] = useState<Set<string>>(new Set());
  const [showEnvPreview, setShowEnvPreview] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, []);

  /* Resize textarea whenever prompt changes (e.g. chip click) */
  useLayoutEffect(() => {
    autoResize();
  }, [prompt, autoResize]);

  /* ── Data ── */
  const { data: sessions } = useSWR<SessionDetail[]>(
    "sessions",
    fetchSessions,
    { refreshInterval: 5000 },
  );
  const { data: allEnvs } = useSWR<EnvInfo[]>("envs", fetchEnvs);
  const { data: providers } = useSWR<ProviderStatus>("providers", fetchProviders);

  const defaults = useMemo(() => resolveModels(providers), [providers]);
  const noKeys = providers && !defaults;

  const [model, setModelRaw] = useState("");
  const [vlmModel, setVlmModelRaw] = useState("");

  const setModel = useCallback((v: string) => {
    setModelRaw(v);
    saveModelPrefs({ llm: v });
  }, []);
  const setVlmModel = useCallback((v: string) => {
    setVlmModelRaw(v);
    saveModelPrefs({ vlm: v });
  }, []);

  /* Initialize from cache, then fill gaps from detected providers */
  useEffect(() => {
    const cached = loadModelPrefs();
    if (cached.llm) setModelRaw(cached.llm);
    if (cached.vlm) setVlmModelRaw(cached.vlm);
  }, []);

  useEffect(() => {
    if (!defaults) return;
    setModelRaw((prev) => prev || defaults.model);
    setVlmModelRaw((prev) => prev || defaults.vlm);
  }, [defaults]);

  /* Track env-dependent defaults */
  const [timesteps, setTimesteps] = useState(10_000_000);
  const [numEnvs, setNumEnvs] = useState(32);

  /* Sync timesteps/numEnvs when switching env */
  useEffect(() => {
    const env = allEnvs?.find((e) => e.env_id === envId);
    if (env?.engine === "isaaclab" && env.zoo_num_envs > 1) {
      setNumEnvs(env.zoo_num_envs);
      setTimesteps(50_000_000);
    } else if (env?.engine === "mujoco") {
      setTimesteps(10_000_000);
    }
  }, [envId, allEnvs]);

  /* ── Grouped envs for selector ── */
  const envGroups = useMemo(() => {
    if (!allEnvs) return [];
    const groups: Record<string, EnvInfo[]> = {};
    for (const env of allEnvs) {
      const key =
        env.engine === "mujoco"
          ? "MuJoCo"
          : `IsaacLab / ${env.description || "Other"}`;
      (groups[key] ??= []).push(env);
    }
    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "MuJoCo") return -1;
      if (b === "MuJoCo") return 1;
      return a.localeCompare(b);
    });
  }, [allEnvs]);

  /* ── Recent sessions (latest 6) ── */
  const recentSessions = useMemo(
    () => (sessions ? sessions.slice(0, 6) : []),
    [sessions],
  );

  /* ── Submit ── */
  async function handleSubmit() {
    if (!prompt.trim() || !model) return;
    setError("");
    setSubmitting(true);
    try {
      const { session_id } = await startSession({
        prompt,
        model,
        total_timesteps: timesteps,
        seed: 1,
        max_iterations: 20,
        pass_threshold: 0.9,
        env_id: envId,
        num_envs: numEnvs,
        vlm_model: vlmModel,
        use_code_judge: true,
        review_reward: true,
        review_judge: true,
        num_configs: 3,
        seeds: [1],
        cores_per_run: 8,
        num_evals: 4,
        side_info: true,
        use_zoo_preset: true,
        hp_tuning: true,
        trajectory_stride: 1,
        thinking_effort: loadModelPrefs().thinkingEffort || "high",
        judgment_select: "last",
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

  function applySuggestion(s: (typeof SAMPLE_RUNS)[number]) {
    setPrompt(s.intent);
    setEnvId(s.env);
  }

  return (
    <div className="landing-bg min-h-screen relative z-10">
      {/* ── Top bar ── */}
      <header className="animate-fade-in relative z-20 flex items-center justify-between px-8 py-5 max-w-5xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/20">
            <RefreshCcw size={16} className="text-white" />
          </div>
          <span className="text-lg font-bold text-gray-900 tracking-tight">
            Prompt2Policy
          </span>
        </div>
        <Link
          href="/e2e"
          className="flex items-center gap-2 text-sm text-gray-500 hover:text-indigo-600 transition-colors font-medium"
        >
          <LayoutDashboard size={14} />
          Dashboard
        </Link>
      </header>

      {/* ── Hero ── */}
      <div className="relative z-10 flex flex-col items-center pt-12 pb-8 px-6">
        <h1 className="animate-fade-up text-4xl sm:text-5xl font-bold text-gray-900 text-center leading-tight max-w-2xl whitespace-nowrap">
          What should your agent learn?
        </h1>

        <p className="animate-fade-up-delay-1 mt-4 text-gray-500 text-center max-w-2xl text-base leading-relaxed">
          <span className="font-semibold">Write a{" "}
          <span className="bg-gradient-to-r from-purple-400 via-indigo-500 to-indigo-700 bg-clip-text text-transparent" style={{backgroundSize: "300% 100%", backgroundPosition: "0% 50%"}}>prompt</span>
          , get a trained{" "}
          <span className="bg-gradient-to-r from-purple-400 via-indigo-500 to-indigo-700 bg-clip-text text-transparent" style={{backgroundSize: "300% 100%", backgroundPosition: "100% 50%"}}>policy</span>
          .</span>
          <br />
          The agent writes the reward, trains RL, judges the result, and iterates until it works.
        </p>
        {model && (
          <div className="animate-fade-up-delay-1 mt-2 flex items-center gap-1 text-xs text-gray-400">
            <span>LLM</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-transparent text-gray-500 text-xs border-none focus:outline-none cursor-pointer hover:text-indigo-600 transition-colors font-medium appearance-none text-center"
              style={{ width: `${(LLM_ITEMS.find((o) => o.value === model)?.label.length ?? 10) * 6.2 + 8}px` }}
            >
              {LLM_ITEMS.map((o) => {
                const prov = modelProvider(o.value);
                const available = !prov || !providers || providers[prov];
                return (
                  <option key={o.value} value={o.value} disabled={!available}>
                    {o.label}{!available ? " (no key)" : ""}
                  </option>
                );
              })}
            </select>
            <span className="ml-2">VLM</span>
            <select
              value={vlmModel}
              onChange={(e) => setVlmModel(e.target.value)}
              className="bg-transparent text-gray-500 text-xs border-none focus:outline-none cursor-pointer hover:text-indigo-600 transition-colors font-medium appearance-none text-center"
              style={{ width: `${(VLM_ITEMS.find((o) => o.value === vlmModel)?.label.length ?? 10) * 6.2 + 8}px` }}
            >
              {VLM_ITEMS.map((o) => {
                const prov = modelProvider(o.value);
                const available = !prov || !providers || providers[prov];
                return (
                  <option key={o.value} value={o.value} disabled={!available}>
                    {o.label}{!available ? " (no key)" : ""}
                  </option>
                );
              })}
            </select>
          </div>
        )}

        {/* ── No API keys warning ── */}
        {noKeys && (
          <div className="animate-fade-up mt-8 flex items-center gap-3 px-5 py-3 rounded-xl bg-amber-50 border border-amber-200 text-sm text-amber-800 max-w-2xl w-full">
            <AlertCircle size={18} className="shrink-0 text-amber-500" />
            <div>
              <p className="font-medium">API keys required</p>
              <p className="text-xs text-amber-600 mt-0.5">
                Add <code className="font-mono bg-amber-100 px-1 rounded">ANTHROPIC_API_KEY</code> and/or{" "}
                <code className="font-mono bg-amber-100 px-1 rounded">GEMINI_API_KEY</code> to your{" "}
                <code className="font-mono bg-amber-100 px-1 rounded">.env</code> file, then restart the server.
              </p>
            </div>
          </div>
        )}

        {/* ── Main input ── */}
        <div className="animate-fade-up-delay-2 mt-10 w-full max-w-2xl">
          {/* ── Env avatar ── */}
          <div className="flex flex-col items-center mb-4 gap-1.5">
            <button
              key={envId}
              type="button"
              onClick={() => setShowEnvPreview(true)}
              className="avatar-pop shrink-0 w-36 h-36 rounded-2xl overflow-hidden border-2 border-gray-200 bg-gray-50 cursor-pointer hover:border-indigo-400 hover:shadow-lg transition-all"
            >
              {!thumbError.has(envId) ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={`/env-thumbnails/${envId}.png`}
                  alt={envId}
                  onError={() => setThumbError((prev) => new Set(prev).add(envId))}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400 text-lg">?</div>
              )}
            </button>
            <div className="relative inline-flex items-center">
              <select
                value={envId}
                onChange={(e) => { setEnvId(e.target.value); setShowEnvPreview(false); }}
                className="bg-transparent text-gray-500 text-xs border-none focus:outline-none cursor-pointer hover:text-gray-700 transition-colors text-center appearance-none pl-1"
                style={{ width: `${envId.length * 6.5 + 28}px` }}
              >
              {envGroups.map(([group, envs]) => (
                <optgroup key={group} label={group}>
                  {envs.map((env) => (
                    <option key={env.env_id} value={env.env_id}>
                      {env.env_id}
                    </option>
                  ))}
                </optgroup>
              ))}
              </select>
              <svg className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 text-gray-400" width="8" height="5" viewBox="0 0 10 6"><path d="M0 0l5 6 5-6z" fill="currentColor"/></svg>
            </div>
          </div>

          <div className={`input-ring card-soft rounded-2xl p-1.5 ${noKeys ? "opacity-50 pointer-events-none" : "animate-pulse-shadow"}`}>
            <div className="flex items-end gap-2">
              <textarea
                ref={textareaRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="Describe a behavior in detail — the more specific your intent (gait, posture, rhythm, constraints), the better the result."
                rows={2}
                disabled={!!noKeys}
                className="flex-1 bg-transparent text-gray-900 placeholder-gray-400 px-4 py-3 text-base resize-none focus:outline-none disabled:cursor-not-allowed max-h-48 overflow-y-auto"
              />
              <button
                onClick={handleSubmit}
                disabled={submitting || !prompt.trim() || !!noKeys || !model}
                className="shrink-0 mb-1.5 mr-1.5 w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-200 disabled:text-gray-400 text-white flex items-center justify-center transition-all shadow-md shadow-indigo-600/20 disabled:shadow-none cursor-pointer disabled:cursor-default"
              >
                {submitting ? (
                  <div className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                ) : (
                  <ArrowRight size={18} />
                )}
              </button>
            </div>
          </div>

          {error && (
            <p className="mt-3 text-sm text-red-500 text-center">{error}</p>
          )}
        </div>

        {/* ── Sample-run chips with video popover ── */}
        <div className="animate-fade-up-delay-3 mt-8 flex flex-wrap justify-center gap-2 max-w-2xl">
          {SAMPLE_RUNS.map((s) => (
            <SampleChip key={s.label} sample={s} disabled={!!noKeys} onSelect={applySuggestion} />
          ))}
        </div>

        <Link
          href="/e2e"
          className="animate-fade-up-delay-4 mt-6 text-xs text-gray-400 hover:text-indigo-500 transition-colors"
        >
          Need fine-grained control? <span className="underline underline-offset-2">Open full dashboard</span> <ChevronRight size={10} className="inline" />
        </Link>
      </div>

      {/* ── Recent sessions ── */}
      {recentSessions.length > 0 && (
        <div className="animate-fade-up-delay-4 relative z-10 max-w-4xl mx-auto px-6 pt-16 pb-20">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Recent Sessions
            </h2>
            <Link
              href="/e2e"
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-indigo-600 transition-colors font-medium"
            >
              View all <ChevronRight size={12} />
            </Link>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {recentSessions.map((s) => (
              <Link
                key={s.session_id}
                href={`/e2e/${s.session_id}`}
                className="card-soft rounded-xl p-4 group"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-gray-900 font-medium truncate group-hover:text-indigo-600 transition-colors">
                      {s.alias || s.prompt || s.session_id.slice(0, 8)}
                    </p>
                    <p className="text-xs text-gray-400 mt-1 truncate">
                      {s.env_id}
                    </p>
                  </div>
                  <ScoreRing score={s.best_score} />
                </div>
                <div className="flex items-center gap-3 mt-3">
                  <StatusBadge status={s.status} />
                  <span className="text-xs text-gray-400">
                    {s.iterations.length} iter
                  </span>
                  <span className="text-xs text-gray-400 ml-auto">
                    {timeAgo(s.created_at)}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* ── Env image lightbox ── */}
      {showEnvPreview && !thumbError.has(envId) && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in cursor-pointer"
          onClick={() => setShowEnvPreview(false)}
        >
          <div
            className="relative max-w-lg w-full mx-4 card-soft rounded-2xl overflow-hidden animate-fade-up"
            onClick={(e) => e.stopPropagation()}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/env-thumbnails/${envId}.png`}
              alt={envId}
              className="w-full"
            />
            <div className="flex items-center justify-between px-4 py-3">
              <span className="text-sm font-medium text-gray-900">{envId}</span>
              <button
                onClick={() => setShowEnvPreview(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="h-12" />
    </div>
  );
}
