"use client";

import type { JobResponse, JobDiskUsage } from "@/lib/scheduler-api";
import { cancelJob, deleteJob, getJobAllocation, fetchJobDiskUsage } from "@/lib/scheduler-api";
import StatusBadge from "@/components/StatusBadge";
import { Trash2 } from "lucide-react";
import { useState, useEffect } from "react";
import useSWR from "swr";

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

function pct(part: number, total: number): number {
  return total > 0 ? Math.round((part / total) * 100) : 0;
}

function Elapsed({ since }: { since: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const secs = Math.floor((now - new Date(since).getTime()) / 1000);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return (
    <span className="font-mono text-xs text-blue-600">
      {m}:{s.toString().padStart(2, "0")}
    </span>
  );
}

export default function RemoteJobCard({
  job,
  onCancelled,
  onDeleted,
  selectable = false,
  selected = false,
  onToggle,
}: {
  job: JobResponse;
  onCancelled: () => void;
  onDeleted?: () => void;
  selectable?: boolean;
  selected?: boolean;
  onToggle?: () => void;
}) {
  const [cancelling, setCancelling] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const isTerminal = ["completed", "error", "cancelled"].includes(job.status);

  const { data: diskUsage } = useSWR<JobDiskUsage>(
    isTerminal ? ["job-disk", job.job_id] : null,
    () => fetchJobDiskUsage(job.job_id),
    { revalidateOnFocus: false },
  );
  const isRunning = job.status === "running";

  const totalRuns = (job.metadata?.total_runs as number) || job.run_ids.length;
  const alloc = getJobAllocation(job);

  const handleCancel = async () => {
    if (!confirm("Cancel this job?")) return;
    setCancelling(true);
    try {
      await cancelJob(job.job_id);
      onCancelled();
    } finally {
      setCancelling(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm(`Move "${job.job_id}" to trash?`)) return;
    setDeleting(true);
    try {
      await deleteJob(job.job_id);
      onDeleted?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete job");
    } finally {
      setDeleting(false);
    }
  };

  const borderClass = selected
    ? "border-blue-400 ring-1 ring-blue-200"
    : isRunning
      ? "border-blue-200"
      : "border-gray-200";

  return (
    <div className={`rounded-lg border bg-white p-4 shadow-sm ${borderClass}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {selectable && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle?.()}
              className="rounded border-gray-300 cursor-pointer"
            />
          )}
          <a
            href={`/benchmark/job/${job.job_id}`}
            className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
          >
            {job.job_id}
          </a>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
            {job.job_type}
          </span>
          {isRunning && <Elapsed since={job.created_at} />}
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.status} />
          {isTerminal && onDeleted && (
            <button
              onClick={handleDelete}
              disabled={deleting}
              className="p-1 rounded text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-50"
              title="Move to trash"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Run count */}
      {job.run_ids.length > 0 && (
        <p className="text-xs text-gray-500 mb-1">
          {job.run_ids.length} run{job.run_ids.length > 1 ? "s" : ""}
        </p>
      )}

      {/* State breakdown badges */}
      {Object.keys(alloc.stateCounts).length > 0 && (
        <div className="flex flex-wrap gap-1 mb-1">
          {Object.entries(alloc.stateCounts).map(([state, count]) => (
            <span
              key={state}
              className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
                state === "running"
                  ? "bg-blue-50 text-blue-700"
                  : state === "completed"
                    ? "bg-green-50 text-green-700"
                    : state === "error"
                      ? "bg-red-50 text-red-700"
                      : state === "cancelled"
                        ? "bg-gray-100 text-gray-500"
                        : "bg-yellow-50 text-yellow-700"
              }`}
            >
              {count} {state}
            </span>
          ))}
        </div>
      )}

      {/* Node allocation */}
      {Object.keys(alloc.nodeAllocation).length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 mb-1">
          {Object.entries(alloc.nodeAllocation).map(([node, info]) =>
            node === "unassigned" ? (
              <span key={node} className="text-[10px] text-gray-400">
                {info.total} queued
              </span>
            ) : (
              <span key={node} className="text-[10px] text-gray-500">
                {node}: {info.total}/{totalRuns}
              </span>
            ),
          )}
        </div>
      )}

      {/* Session affinity badge */}
      {alloc.sessionAffinity && totalRuns > 1 && (
        <div className="mb-1">
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-50 text-purple-700">
            same-node{alloc.affinityNode ? ` → ${alloc.affinityNode}` : ""}
          </span>
        </div>
      )}

      <div className="text-sm text-gray-600 space-y-1">
        {!isTerminal && totalRuns > 1 && (
          <p className="text-xs text-gray-500">
            {job.run_ids.length} / {totalRuns} runs
          </p>
        )}
        <p className="text-xs text-gray-400">
          Created: {new Date(job.created_at).toLocaleString()}
        </p>
        {job.completed_at && (() => {
          const endLabel = job.status === "cancelled" ? "Cancelled" : job.status === "error" ? "Failed" : "Completed";
          const ms = new Date(job.completed_at).getTime() - new Date(job.created_at).getTime();
          const totalSec = Math.floor(ms / 1000);
          const h = Math.floor(totalSec / 3600);
          const m = Math.floor((totalSec % 3600) / 60);
          const s = totalSec % 60;
          const dur = h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
          return (
            <p className="text-xs text-gray-400">
              {endLabel}: {new Date(job.completed_at).toLocaleString()}
              <span className="ml-2 text-gray-500 font-mono">({dur})</span>
            </p>
          );
        })()}
        {diskUsage && diskUsage.total_bytes > 0 && (
          <p className="text-xs text-gray-400">
            Disk:{" "}
            <span className="font-mono text-gray-500">{formatBytes(diskUsage.total_bytes)}</span>
            <span className="ml-1 text-gray-400">
              · Traj {pct(diskUsage.trajectory_bytes, diskUsage.total_bytes)}%
              · Vid {pct(diskUsage.video_bytes, diskUsage.total_bytes)}%
              · Ckpt {pct(diskUsage.checkpoint_bytes, diskUsage.total_bytes)}%
            </span>
          </p>
        )}
        {job.error && (
          <p className="text-xs text-red-600 bg-red-50 rounded px-2 py-1">
            {job.error}
          </p>
        )}
        {job.metadata?.gate_failed ? (
          <p className="text-xs text-amber-600">
            Gate failed at stage {String(job.metadata.stopped_at_stage)}
          </p>
        ) : null}
      </div>

      {/* Running state: spinner + cancel */}
      {isRunning && (
        <div className="mt-3 flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-500">Running...</span>
          </div>
          <button
            onClick={handleCancel}
            disabled={cancelling}
            className="px-3 py-1 text-xs rounded-md bg-red-50 hover:bg-red-100 text-red-700 disabled:opacity-50"
          >
            {cancelling ? "Cancelling..." : "Cancel"}
          </button>
        </div>
      )}
    </div>
  );
}
