"use client";

import useSWR from "swr";
import { RotateCcw, Trash2 } from "lucide-react";
import {
  fetchTrash,
  restoreSession,
  restoreBenchmark,
  hardDelete,
  hardDeleteAll,
  type TrashItem,
} from "@/lib/api";
import { restoreJob } from "@/lib/scheduler-api";
import { timeAgo } from "@/lib/format";

const entityTypeLabel: Record<string, string> = {
  session: "Session",
  benchmark: "Benchmark",
  job: "Job",
};

const entityTypeColor: Record<string, string> = {
  session: "bg-blue-100 text-blue-700",
  benchmark: "bg-green-100 text-green-700",
  job: "bg-purple-100 text-purple-700",
};

export default function TrashPage() {
  const { data: items, mutate } = useSWR<TrashItem[]>("trash", fetchTrash);

  async function handleRestore(item: TrashItem) {
    try {
      const restoreFn =
        item.entity_type === "session" ? restoreSession
        : item.entity_type === "benchmark" ? restoreBenchmark
        : item.entity_type === "job" ? restoreJob
        : null;
      if (!restoreFn) return;
      await restoreFn(item.entity_id);
      await mutate();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to restore");
    }
  }

  async function handleDeleteAll() {
    if (!items || items.length === 0) return;
    if (
      !confirm(
        `Permanently delete all ${items.length} item(s) in trash?\nThis cannot be undone.`,
      )
    )
      return;
    try {
      await hardDeleteAll();
      await mutate();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete all");
    }
  }

  async function handleHardDelete(item: TrashItem) {
    if (
      !confirm(
        `Permanently delete "${item.alias || item.entity_id}"?\nThis cannot be undone.`,
      )
    )
      return;
    try {
      await hardDelete(item.entity_id);
      await mutate();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete");
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Trash</h1>
      <div className="flex items-center justify-between mb-6">
        <p className="text-sm text-gray-500">
          Deleted sessions and benchmarks. Restore or permanently delete them.
        </p>
        {items && items.length > 0 && (
          <button
            onClick={handleDeleteAll}
            className="cursor-pointer flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-50 text-red-600 hover:bg-red-100 transition-colors"
          >
            <Trash2 size={12} />
            Delete All
          </button>
        )}
      </div>

      {items && items.length === 0 && (
        <p className="text-sm text-gray-400">Trash is empty.</p>
      )}

      {items && items.length > 0 && (
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.entity_id}
              className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 flex items-center gap-4"
            >
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                  entityTypeColor[item.entity_type] ||
                  "bg-gray-100 text-gray-600"
                }`}
              >
                {entityTypeLabel[item.entity_type] || item.entity_type}
              </span>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {item.alias || item.entity_id}
                </p>
                {item.alias && (
                  <p className="text-xs font-mono text-gray-400">
                    {item.entity_id}
                  </p>
                )}
                {item.prompt && (
                  <p className="text-xs text-gray-500 truncate mt-0.5">
                    &ldquo;{item.prompt}&rdquo;
                  </p>
                )}
              </div>

              <span className="text-xs text-gray-400 whitespace-nowrap">
                deleted {timeAgo(item.deleted_at)}
              </span>

              <button
                onClick={() => handleRestore(item)}
                className="cursor-pointer flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors"
                title="Restore"
              >
                <RotateCcw size={12} />
                Restore
              </button>

              <button
                onClick={() => handleHardDelete(item)}
                className="cursor-pointer flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-50 text-red-600 hover:bg-red-100 transition-colors"
                title="Delete permanently"
              >
                <Trash2 size={12} />
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
