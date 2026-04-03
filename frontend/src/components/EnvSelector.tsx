"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import useSWR from "swr";
import { fetchEnvs, type EnvInfo } from "@/lib/api";
import Tooltip from "@/components/Tooltip";

interface EnvSelectorProps {
  value: string;
  onChange: (envId: string, env?: EnvInfo) => void;
}

/** Group label for an env — engine + category. */
function groupKey(env: EnvInfo): string {
  if (env.engine === "mujoco") return "MuJoCo";
  // IsaacLab envs use description as category (set by sync script)
  return `IsaacLab / ${env.description || "Other"}`;
}

/** Preferred group ordering. */
const GROUP_ORDER = [
  "MuJoCo",
  "IsaacLab / Locomotion (Flat)",
  "IsaacLab / Locomotion (Rough)",
  "IsaacLab / Manipulation (Reach)",
  "IsaacLab / Manipulation (Lift/Stack)",
  "IsaacLab / Humanoid",
  "IsaacLab / Classic Control",
  "IsaacLab / Dexterous",
  "IsaacLab / Navigation",
  "IsaacLab / Assembly",
  "IsaacLab / Pick & Place",
  "IsaacLab / Aerial",
  "IsaacLab / Other",
];

function groupSort(a: string, b: string): number {
  const ai = GROUP_ORDER.indexOf(a);
  const bi = GROUP_ORDER.indexOf(b);
  if (ai >= 0 && bi >= 0) return ai - bi;
  if (ai >= 0) return -1;
  if (bi >= 0) return 1;
  return a.localeCompare(b);
}

export default function EnvSelector({ value, onChange }: EnvSelectorProps) {
  const { data: envs } = useSWR<EnvInfo[]>("envs", fetchEnvs);
  const selected = envs?.find((e) => e.env_id === value);
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<{ envId: string; rect: DOMRect } | null>(null);

  const handleEnvHover = useCallback((envId: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setHovered({ envId, rect });
  }, []);

  const handleEnvLeave = useCallback(() => setHovered(null), []);

  // Close dropdown on outside click
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  // Group and filter envs
  const grouped = useMemo(() => {
    if (!envs) return {};
    const q = search.toLowerCase();
    const filtered = q
      ? envs.filter(
          (e) =>
            e.env_id.toLowerCase().includes(q) ||
            e.name.toLowerCase().includes(q) ||
            e.description.toLowerCase().includes(q)
        )
      : envs;

    const groups: Record<string, EnvInfo[]> = {};
    for (const env of filtered) {
      const key = groupKey(env);
      (groups[key] ??= []).push(env);
    }
    return groups;
  }, [envs, search]);

  const sortedGroups = useMemo(
    () => Object.keys(grouped).sort(groupSort),
    [grouped]
  );

  const totalFiltered = useMemo(
    () => Object.values(grouped).reduce((s, g) => s + g.length, 0),
    [grouped]
  );

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Environment
        <Tooltip
          content={
            selected ? (
              <>
                <p className="font-semibold mb-1">{selected.env_id}</p>
                <p className="mb-2">{selected.description}</p>
                <p>
                  obs: {selected.obs_dim}-dim | action: {selected.action_dim}-dim
                </p>
                {selected.engine !== "mujoco" && (
                  <p className="mt-1 text-xs text-blue-400">Engine: {selected.engine}</p>
                )}
                <p className="mt-1">
                  info keys:{" "}
                  {Object.entries(selected.info_keys)
                    .map(([k, v]) => `${k} (${v})`)
                    .join(", ")}
                </p>
              </>
            ) : (
              <p>Select an environment to see details</p>
            )
          }
        />
      </label>

      {/* Selected value button */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-left focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white flex items-center justify-between"
      >
        <span className="truncate">
          {selected ? `${selected.name} — ${selected.description}` : value || "Select..."}
        </span>
        <svg className="w-4 h-4 text-gray-400 ml-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={open ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
        </svg>
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute z-50 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-80 flex flex-col">
          {/* Search input */}
          <div className="p-2 border-b border-gray-100">
            <input
              type="text"
              placeholder="Search environments..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              autoFocus
              className="w-full rounded border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>

          {/* Grouped options */}
          <div className="overflow-y-auto flex-1">
            {sortedGroups.length === 0 ? (
              <div className="px-3 py-4 text-sm text-gray-400 text-center">
                No environments match &quot;{search}&quot;
              </div>
            ) : (
              sortedGroups.map((group) => (
                <div key={group}>
                  <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 bg-gray-50 sticky top-0">
                    {group}
                    <span className="ml-1 text-gray-400">({grouped[group].length})</span>
                  </div>
                  {grouped[group].map((env) => (
                    <button
                      key={env.env_id}
                      type="button"
                      onMouseEnter={(e) => handleEnvHover(env.env_id, e)}
                      onMouseLeave={handleEnvLeave}
                      onClick={() => {
                        onChange(env.env_id, env);
                        setOpen(false);
                        setSearch("");
                      }}
                      className={`w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50 ${
                        env.env_id === value ? "bg-blue-100 font-medium" : ""
                      }`}
                    >
                      {env.name}
                    </button>
                  ))}
                </div>
              ))
            )}
          </div>

          {/* Footer count */}
          <div className="px-3 py-1.5 border-t border-gray-100 text-xs text-gray-400">
            {totalFiltered} of {envs?.length ?? 0} environments
          </div>
        </div>
      )}

      {/* Thumbnail popover — rendered via portal to escape overflow clipping */}
      {open && hovered && typeof document !== "undefined" &&
        createPortal(
          <div
            className="pointer-events-none z-[9999]"
            style={{
              position: "fixed",
              top: hovered.rect.top,
              left: hovered.rect.left - 330,
            }}
          >
            <div className="bg-white border border-gray-200 rounded-lg shadow-xl p-1.5 w-[312px]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/env-thumbnails/${hovered.envId}.png`}
                alt={hovered.envId}
                className="w-full rounded"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
              <p className="text-xs text-gray-500 text-center mt-1 truncate">{hovered.envId}</p>
            </div>
          </div>,
          document.body
        )
      }
    </div>
  );
}
