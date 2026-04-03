"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import useSWR from "swr";
import {
  fetchBenchmark,
  fetchBenchmarkConfig,
  stopBenchmark,
  staticUrl,
  type BenchmarkRunDetail,
} from "@/lib/api";
import { savePreset, type BenchmarkPresetParams } from "@/lib/presets";
import { DEFAULT_LLM_MODEL } from "@/components/LlmModelSelector";
import { isThinkingEffort } from "@/components/ThinkingEffortSelector";
import StatusBadge from "@/components/StatusBadge";
import SavePresetButton from "@/components/SavePresetButton";
import ProgressBar from "@/components/ProgressBar";
import GroupStatsTable from "@/components/GroupStatsTable";
import ScoreProgressionChart from "@/components/ScoreProgressionChart";
import StagePipeline from "@/components/StagePipeline";

type BreakdownTab = "category" | "difficulty" | "env" | "stage";
type SortKey = "index" | "env_id" | "instruction" | "category" | "difficulty" | "best_score" | "session_status" | "stage";
type SortDir = "asc" | "desc";

const DIFFICULTY_ORDER: Record<string, number> = { easy: 0, medium: 1, hard: 2 };

function formatDuration(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function Elapsed({ since, until }: { since: string; until?: string | null }) {
  const [now, setNow] = useState(() => Date.now());
  const isLive = !until;
  useEffect(() => {
    if (!isLive) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isLive]);
  const endMs = until ? new Date(until).getTime() : now;
  const startMs = new Date(since).getTime();
  const secs = Number.isFinite(endMs) && Number.isFinite(startMs)
    ? Math.max(0, Math.floor((endMs - startMs) / 1000))
    : 0;
  return (
    <span className={`font-mono text-sm ${isLive ? "text-blue-600" : "text-gray-500"}`}>
      {formatDuration(secs)}
    </span>
  );
}

export default function BenchmarkDetailPage() {
  const { benchmarkId } = useParams<{ benchmarkId: string }>();
  const [tab, setTab] = useState<BreakdownTab>("category");
  const [showCases, setShowCases] = useState(false);
  const [filter, setFilter] = useState("");
  const [stopping, setStopping] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [stageFilter, setStageFilter] = useState<number | null>(null);
  const [hoveredStage, setHoveredStage] = useState<number | null>(null);

  const { data: detail, mutate } = useSWR<BenchmarkRunDetail>(
    benchmarkId ? `benchmark-${benchmarkId}` : null,
    () => fetchBenchmark(benchmarkId),
    { refreshInterval: 5000 },
  );

  async function handleStop() {
    setStopping(true);
    try {
      await stopBenchmark(benchmarkId);
      await mutate();
    } catch {
      // polling picks up
    } finally {
      setStopping(false);
    }
  }

  async function handleSavePreset(name: string) {
    const config = await fetchBenchmarkConfig(benchmarkId);
    const params: BenchmarkPresetParams = {
      model: (config.model as string) ?? DEFAULT_LLM_MODEL,
      timesteps: (config.total_timesteps as number) ?? 1_000_000,
      numConfigs: (config.num_configs as number) ?? 1,
      seedsStr: ((config.seeds as number[]) ?? [1]).join(", "),
      maxIterations: (config.max_iterations as number) ?? 5,
      passThreshold: (config.pass_threshold as number) ?? 0.9,
      numEnvs: (config.num_envs as number) ?? 16,
      maxParallel: (config.max_parallel as number) ?? 10,
      coresPerRun: (config.cores_per_run as number) ?? 0,
      vlmModel: (config.vlm_model as string) ?? "gemini-3.1-pro-preview",
      useCodeJudge: (config.use_code_judge as boolean) ?? true,
      thinkingEffort: isThinkingEffort(config.thinking_effort) ? config.thinking_effort : "max",
      filterEnvs: (config.filter_envs as string[]) ?? [],
      filterCategories: (config.filter_categories as string[]) ?? [],
      filterDifficulties: (config.filter_difficulties as string[]) ?? [],
      device: ((config.device as "auto" | "cpu")) ?? "auto",
      csvFile: (config.csv_file as string) ?? "test_cases.csv",
      backend: ((config.backend as "local" | "ssh")) ?? "local",
    };
    savePreset("benchmark", name, params);
  }

  if (!detail) {
    return (
      <div className="text-gray-500 text-sm py-10 text-center">Loading...</div>
    );
  }

  const isStaged = detail.mode === "staged" && detail.stages.length > 0;
  const hasNodeInfo = detail.test_cases.some((tc) => tc.node_id && tc.node_id !== "");

  const breakdownData =
    tab === "category"
      ? detail.by_category
      : tab === "difficulty"
        ? detail.by_difficulty
        : tab === "env"
          ? detail.by_env
          : (() => {
              // By Stage breakdown
              const groups: Record<string, typeof detail.by_category[string]> = {};
              for (const s of detail.stages) {
                const stageCases = detail.test_cases.filter((tc) => tc.stage === s.stage);
                const completed = stageCases.filter(
                  (tc) => !["running", "pending", "queued"].includes(tc.session_status),
                );
                const passed = stageCases.filter((tc) => tc.passed);
                const cumulative = stageCases.reduce((sum, tc) => sum + tc.best_score, 0);
                groups[`Stage ${s.stage}: ${s.name}`] = {
                  total: stageCases.length,
                  completed: completed.length,
                  passed: passed.length,
                  success_rate: completed.length > 0 ? passed.length / completed.length : 0,
                  average_score: completed.length > 0 ? cumulative / completed.length : 0,
                  cumulative_score: cumulative,
                };
              }
              return groups;
            })();

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "best_score" ? "desc" : "asc");
    }
  }

  const filteredCases = detail.test_cases.filter((tc) => {
    // Stage filter
    if (stageFilter !== null && tc.stage !== stageFilter) return false;

    if (!filter) return true;
    const q = filter.toLowerCase();
    const re = new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i");
    return (
      re.test(tc.env_id) ||
      re.test(tc.category) ||
      re.test(tc.difficulty) ||
      re.test(tc.instruction)
    );
  });

  const sortedCases = [...filteredCases].sort((a, b) => {
    let cmp = 0;
    switch (sortKey) {
      case "index": cmp = a.index - b.index; break;
      case "best_score": cmp = a.best_score - b.best_score; break;
      case "stage": cmp = a.stage - b.stage; break;
      case "difficulty":
        cmp = (DIFFICULTY_ORDER[a.difficulty] ?? 99) - (DIFFICULTY_ORDER[b.difficulty] ?? 99);
        break;
      default:
        cmp = String(a[sortKey]).localeCompare(String(b[sortKey]));
    }
    return sortDir === "asc" ? cmp : -cmp;
  });

  const tabOptions: [BreakdownTab, string][] = [
    ["category", "By Category"],
    ["difficulty", "By Difficulty"],
    ["env", "By Environment"],
  ];
  if (isStaged) {
    tabOptions.push(["stage", "By Stage"]);
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link
          href="/benchmark"
          className="text-gray-400 hover:text-gray-600 text-sm"
        >
          &larr; Back
        </Link>
        <h1 className="text-2xl font-bold text-gray-900">{detail.benchmark_id}</h1>
        <StatusBadge status={detail.status} />
        {detail.created_at && (
          <Elapsed since={detail.created_at} until={detail.completed_at} />
        )}
        {isStaged && (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-100 text-indigo-700">
            Staged
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <SavePresetButton onSave={handleSavePreset} />
          {detail.status === "running" && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="px-4 py-1.5 rounded-full text-xs font-medium bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50 transition-colors"
            >
              {stopping ? "Stopping..." : "Stop All"}
            </button>
          )}
        </div>
      </div>

      {/* Stage Pipeline (staged mode only) */}
      {isStaged && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Stage Pipeline
          </h3>
          <StagePipeline
            stages={detail.stages}
            activeStage={stageFilter}
            hoveredStage={hoveredStage}
            onHover={setHoveredStage}
            onStageClick={(stage) => {
              setStageFilter(stageFilter === stage ? null : stage);
              setShowCases(true);
            }}
          />
          {stageFilter !== null && (
            <button
              type="button"
              onClick={() => setStageFilter(null)}
              className="mt-2 text-xs text-gray-500 hover:text-gray-700"
            >
              Clear stage filter
            </button>
          )}
        </div>
      )}

      {/* Overview Card */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
        <div className="grid grid-cols-3 gap-6 mb-4">
          <div className="text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
              Cumulative Score
            </p>
            <p className="text-2xl font-bold font-mono text-gray-900">
              {detail.cumulative_score.toFixed(1)}
              <span className="text-sm text-gray-400 font-normal">
                {" "}
                / {detail.total_cases}
              </span>
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
              Pass Rate
            </p>
            <p className="text-2xl font-bold font-mono text-gray-900">
              {detail.passed_cases}
              <span className="text-sm text-gray-400 font-normal">
                {" "}
                / {detail.completed_cases} passed
              </span>
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
              Success Rate
            </p>
            <p className="text-2xl font-bold font-mono text-gray-900">
              {(detail.success_rate * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <ProgressBar
          completed={detail.completed_cases}
          total={detail.total_cases}
        />
      </div>

      {/* Score Progression Chart */}
      <ScoreProgressionChart
        testCases={detail.test_cases}
        stages={detail.stages}
        maxIterations={detail.max_iterations ?? 5}
      />

      {/* Breakdown Tabs */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
        <div className="flex gap-1 mb-4">
          {tabOptions.map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                tab === key
                  ? "bg-gray-900 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <GroupStatsTable
          groups={breakdownData}
          label={
            tab === "env"
              ? "Environment"
              : tab === "category"
                ? "Category"
                : tab === "stage"
                  ? "Stage"
                  : "Difficulty"
          }
        />
      </div>

      {/* Test Cases Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <button
          onClick={() => setShowCases(!showCases)}
          className="w-full flex items-center gap-2 px-6 py-4 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
        >
          <span
            className={`transition-transform text-xs ${showCases ? "rotate-90" : ""}`}
          >
            &#9654;
          </span>
          Test Cases ({detail.test_cases.length})
          {stageFilter !== null && (
            <span className="text-xs text-blue-600 ml-1">
              (Stage {stageFilter} filter active)
            </span>
          )}
        </button>

        {showCases && (
          <div className="border-t border-gray-100 p-6">
            <div className="flex items-center gap-3 mb-4">
              <input
                type="text"
                placeholder="Filter by instruction, env, category..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="w-full max-w-md rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {isStaged && (
                <div className="flex gap-1">
                  {detail.stages.map((s) => (
                    <button
                      key={s.stage}
                      type="button"
                      onClick={() => setStageFilter(stageFilter === s.stage ? null : s.stage)}
                      className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                        stageFilter === s.stage
                          ? "bg-blue-100 text-blue-800 border border-blue-300"
                          : "bg-gray-100 text-gray-500 border border-gray-200 hover:border-gray-300"
                      }`}
                    >
                      S{s.stage}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                    {(
                      [
                        ["index", "#"],
                        ...(isStaged ? [["stage", "Stage"] as [SortKey, string]] : []),
                        ["env_id", "Env"],
                        ["instruction", "Instruction"],
                        ["category", "Category"],
                        ["difficulty", "Difficulty"],
                        ["best_score", "Score"],
                        ["session_status", "Status"],
                      ] as [SortKey, string][]
                    ).map(([key, label]) => (
                      <th
                        key={key}
                        onClick={() => toggleSort(key)}
                        className="py-2 pr-3 cursor-pointer select-none hover:text-gray-700 transition-colors"
                      >
                        {label}
                        {sortKey === key && (
                          <span className="ml-1 text-gray-400">
                            {sortDir === "asc" ? "\u25B2" : "\u25BC"}
                          </span>
                        )}
                      </th>
                    ))}
                    {hasNodeInfo && <th className="py-2 pr-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Node</th>}
                    <th className="py-2 pr-3">Video</th>
                    <th className="py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {sortedCases.map((tc) => (
                    <tr
                      key={tc.index}
                      className="border-b border-gray-50 hover:bg-gray-50"
                    >
                      <td className="py-2 pr-3 font-mono text-gray-400">
                        {tc.index + 1}
                      </td>
                      {isStaged && (
                        <td className="py-2 pr-3">
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-50 text-indigo-600">
                            S{tc.stage}
                          </span>
                        </td>
                      )}
                      <td className="py-2 pr-3 whitespace-nowrap">
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-700">
                          {tc.env_id}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-gray-700 max-w-md">
                        {tc.instruction}
                      </td>
                      <td className="py-2 pr-3 text-gray-500">{tc.category}</td>
                      <td className="py-2 pr-3">
                        <span
                          className={`text-xs font-medium ${
                            tc.difficulty === "hard"
                              ? "text-red-600"
                              : tc.difficulty === "medium"
                                ? "text-yellow-600"
                                : "text-green-600"
                          }`}
                        >
                          {tc.difficulty}
                        </span>
                      </td>
                      <td className="py-2 pr-3">
                        <span
                          className={`font-mono font-medium ${
                            tc.passed
                              ? "text-green-600"
                              : tc.best_score >= 0.4
                                ? "text-yellow-600"
                                : "text-gray-500"
                          }`}
                        >
                          {tc.best_score.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-2 pr-3">
                        <StatusBadge status={tc.session_status} />
                      </td>
                      {hasNodeInfo && (
                        <td className="py-2 pr-3">
                          <span className="text-xs text-gray-500 font-mono">
                            {tc.node_id || "local"}
                          </span>
                        </td>
                      )}
                      <td className="py-2 pr-3">
                        {tc.video_urls.length > 0 && (
                          <a
                            href={staticUrl(tc.video_urls[0])}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-500 hover:text-blue-700 text-xs"
                          >
                            &#9654; Play
                          </a>
                        )}
                      </td>
                      <td className="py-2">
                        <Link
                          href={`/e2e/${tc.session_id}`}
                          className="text-blue-500 hover:text-blue-700 text-xs"
                        >
                          Detail &rarr;
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
