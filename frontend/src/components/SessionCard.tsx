"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Star, Trash2, Pencil, ChevronRight, RotateCcw } from "lucide-react";
import type { SessionDetail } from "@/lib/api";
import { updateSession, deleteSession, fetchSessionConfig, startSession } from "@/lib/api";
import { formatDate, timeAgo } from "@/lib/format";
import StatusBadge from "./StatusBadge";

function iterationIdFromDir(iterationDir: string): string {
  const parts = iterationDir.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1];
}

export default function SessionCard({
  session,
  onStop,
  onMutate,
}: {
  session: SessionDetail;
  onStop?: (id: string) => void;
  onMutate?: () => void;
}) {
  const router = useRouter();
  const [expanded, setExpanded] = useState(false);
  const [editingAlias, setEditingAlias] = useState(false);
  const [aliasValue, setAliasValue] = useState(session.alias || "");
  const [rerunning, setRerunning] = useState(false);
  const iterCount = session.iterations.length;
  const multiConfig = session.iterations.some((it) => it.is_multi_config);
  const created = formatDate(session.created_at);

  async function handleRerun(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Start a new session with the same configuration?")) return;
    setRerunning(true);
    try {
      const config = await fetchSessionConfig(session.session_id);
      // Flatten nested train config for startSession API
      const train = (config.train as Record<string, unknown>) ?? {};
      const flat = {
        ...config,
        env_id: train.env_id ?? config.env_id,
        total_timesteps: train.total_timesteps ?? config.total_timesteps,
        seed: train.seed ?? config.seed ?? 1,
        num_envs: train.num_envs ?? config.num_envs,
        side_info: train.side_info ?? config.side_info,
        num_evals: train.num_evals ?? config.num_evals,
        trajectory_stride: train.trajectory_stride ?? config.trajectory_stride,
      };
      const result = await startSession(flat as Parameters<typeof startSession>[0]);
      router.push(`/e2e/${result.session_id}`);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to rerun session");
    } finally {
      setRerunning(false);
    }
  }

  async function handleStar(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    try {
      await updateSession(session.session_id, { starred: !session.starred });
      onMutate?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to update star");
    }
  }

  async function handleAliasSave() {
    setEditingAlias(false);
    const trimmed = aliasValue.trim();
    if (trimmed !== (session.alias || "")) {
      try {
        await updateSession(session.session_id, { alias: trimmed });
        onMutate?.();
      } catch (err) {
        setAliasValue(session.alias || "");
        alert(err instanceof Error ? err.message : "Failed to update alias");
      }
    }
  }

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Move this session to trash?")) return;
    try {
      await deleteSession(session.session_id);
      onMutate?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete session");
    }
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
      {/* Clickable card body → session detail */}
      <Link href={`/e2e/${session.session_id}`}>
        <div className="p-5 cursor-pointer">
          {/* Top row: star + alias/id + edit + delete */}
          <div className="flex items-center gap-2 mb-1">
            <button
              onClick={handleStar}
              className="cursor-pointer hover:scale-110 transition-transform"
              title={session.starred ? "Unstar" : "Star"}
            >
              <Star
                size={16}
                className={
                  session.starred
                    ? "fill-yellow-400 text-yellow-400"
                    : "text-gray-300 hover:text-yellow-400"
                }
              />
            </button>

            {editingAlias ? (
              <input
                autoFocus
                value={aliasValue}
                onChange={(e) => setAliasValue(e.target.value)}
                onBlur={handleAliasSave}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleAliasSave();
                  if (e.key === "Escape") {
                    setAliasValue(session.alias || "");
                    setEditingAlias(false);
                  }
                }}
                onClick={(e) => e.preventDefault()}
                className="text-sm font-medium text-gray-900 border-b border-blue-400 outline-none bg-transparent px-0 py-0 flex-1"
              />
            ) : (
              <span
                className="text-xs font-mono text-gray-400 cursor-pointer hover:text-gray-600 flex items-center gap-1"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setEditingAlias(true);
                }}
                title="Click to edit alias"
              >
                {session.alias || session.session_id}
                <Pencil size={10} className="text-gray-300" />
              </span>
            )}

            {session.status !== "running" && (
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  onClick={handleRerun}
                  disabled={rerunning}
                  className="cursor-pointer text-gray-300 hover:text-blue-500 transition-colors disabled:opacity-50"
                  title="Rerun with same config"
                >
                  <RotateCcw size={14} className={rerunning ? "animate-spin" : ""} />
                </button>
                <button
                  onClick={handleDelete}
                  className="cursor-pointer text-gray-300 hover:text-red-500 transition-colors"
                  title="Move to trash"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </div>

          {/* Alias subtitle if alias is set */}
          {session.alias && (
            <p className="text-xs font-mono text-gray-400 mb-1">
              {session.session_id}
            </p>
          )}

          {/* Env tag + relative time */}
          <div className="flex items-center gap-2 mb-3">
            {session.env_id && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-700">
                {session.env_id}
              </span>
            )}
            {multiConfig && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
                multi-config
              </span>
            )}
            {session.tags?.length > 0 && session.tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-600"
              >
                {tag}
              </span>
            ))}
            <span className="text-xs text-gray-400" title={session.created_at}>
              {created} · {timeAgo(session.created_at)}
            </span>
          </div>

          {/* Prompt */}
          {session.prompt && (
            <p className="text-sm text-gray-700 mb-3 line-clamp-2">
              &ldquo;{session.prompt}&rdquo;
            </p>
          )}

          {/* Footer: status + score + iteration + stop */}
          <div className="flex items-center gap-3 text-sm">
            <StatusBadge status={session.status} isStale={session.is_stale} />
            <span className="text-gray-500">
              Score{" "}
              <span className="font-mono font-medium text-gray-900">
                {session.best_score.toFixed(2)}
              </span>
            </span>
            <span className="text-gray-400">
              v{session.best_iteration}/{iterCount}
            </span>
            {onStop && session.status === "running" && (
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onStop(session.session_id);
                }}
                className="ml-auto cursor-pointer px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 hover:bg-red-200 transition-colors"
              >
                Stop
              </button>
            )}
          </div>
        </div>
      </Link>

      {/* Iterations toggle */}
      {iterCount > 0 && (
        <div className="border-t border-gray-100">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center gap-2 px-5 py-2.5 text-xs text-gray-500 hover:bg-gray-50 transition-colors cursor-pointer"
          >
            <ChevronRight
              size={12}
              className={`transition-transform ${expanded ? "rotate-90" : ""}`}
            />
            Iterations ({iterCount})
          </button>

          {expanded && (
            <div className="px-5 pb-4 space-y-1.5">
              {session.iterations.map((it) => {
                const iterId = it.iteration_dir ? iterationIdFromDir(it.iteration_dir) : null;
                return (
                  <div
                    key={it.iteration_dir || `it-${it.iteration}`}
                    className="flex items-center gap-3 text-xs rounded-lg bg-gray-50 px-3 py-2"
                  >
                    <span className="font-mono font-medium text-gray-700">
                      v{it.iteration}
                    </span>
                    {iterId && (
                      <span className="font-mono text-gray-400">{iterId}</span>
                    )}
                    {it.intent_score != null && (
                      <span
                        className={`font-mono font-medium ${
                          it.intent_score >= session.pass_threshold
                            ? "text-green-600"
                            : it.intent_score >= Math.max(0, session.pass_threshold - 0.3)
                              ? "text-yellow-600"
                              : "text-red-600"
                        }`}
                      >
                        {it.intent_score.toFixed(2)}
                      </span>
                    )}
                    {it.final_return != null && (
                      <span className="text-gray-500">
                        Return{" "}
                        <span className="font-mono">
                          {it.final_return.toFixed(1)}
                        </span>
                      </span>
                    )}
                    <Link
                      href={`/e2e/${session.session_id}`}
                      onClick={(e) => e.stopPropagation()}
                      className="ml-auto text-blue-500 hover:text-blue-700"
                    >
                      Detail &rarr;
                    </Link>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
