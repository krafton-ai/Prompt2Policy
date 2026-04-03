"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Lineage, LineageEntry, StructuredLesson } from "@/lib/api";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const NODE_W = 56;
const NODE_H = 42;
const H_GAP = 12;
const V_GAP = 28;
const PAD = 20;

const PASS_THRESHOLD = 0.7;
const WARN_THRESHOLD = Math.max(0, PASS_THRESHOLD - 0.3);

/** Lesson tier badge colors */
const TIER_COLORS: Record<string, string> = {
  HARD: "bg-red-100 text-red-700 border-red-300",
  STRONG: "bg-blue-100 text-blue-700 border-blue-300",
  SOFT: "bg-gray-100 text-gray-500 border-gray-300",
  RETIRED: "bg-gray-50 text-gray-400 border-gray-200 line-through",
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface TreeNode {
  key: string;
  entry: LineageEntry;
  children: TreeNode[];
  x: number;
  y: number;
  depth: number;
}

/** Elapsed time lookup: iteration number -> seconds */
export type ElapsedMap = Record<number, number>;

// ---------------------------------------------------------------------------
// Helpers — colors match LoopTimeline (light backgrounds)
// ---------------------------------------------------------------------------

/** Extract iteration number from lineage key.
 *  "session_.../iter_3"          → 3
 *  "session_.../iter_3/baseline" → 3
 */
function iterNumFromKey(key: string): number | null {
  const m = key.match(/iter_(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

/** Check if a key represents a config-level node (3 parts). */
function isConfigKey(key: string): boolean {
  return key.split("/").length === 3;
}

/** Extract config_id from key like "session_.../iter_3/baseline" → "baseline" */
function configIdFromKey(key: string): string | null {
  const parts = key.split("/");
  return parts.length === 3 ? parts[2] : null;
}

/** Abbreviate config_id for compact display: baseline→b, config_0→c0, config_1→c1 */
function shortCfgId(cfgId: string): string {
  if (cfgId === "baseline") return "b";
  const m = cfgId.match(/^config_(\d+)$/);
  if (m) return `c${m[1]}`;
  return cfgId.length > 4 ? cfgId.slice(0, 3) + "\u2026" : cfgId;
}

/** PPO default values for known hyperparameters. */
const HP_DEFAULTS: Record<string, unknown> = {
  learning_rate: 3e-4,
  gamma: 0.99,
  gae_lambda: 0.95,
  clip_coef: 0.2,
  ent_coef: 0.01,
  vf_coef: 0.5,
  max_grad_norm: 0.5,
  update_epochs: 10,
  num_steps: 128,
};

/** Format HP params as a compact string: "lr=1e-3, ent=0.02" */
function formatHpParams(params: Record<string, unknown>): string {
  const abbrev: Record<string, string> = {
    learning_rate: "lr",
    ent_coef: "ent",
    vf_coef: "vf",
    clip_coef: "clip",
    gamma: "γ",
    gae_lambda: "λ",
    num_steps: "steps",
    update_epochs: "epochs",
    total_timesteps: "timesteps",
    max_grad_norm: "grad_norm",
    net_arch: "arch",
  };
  return Object.entries(params)
    .map(([k, v]) => `${abbrev[k] || k}=${v}`)
    .join(", ");
}

/** Resolve an HP value, falling back to known PPO defaults. */
function hpDisplay(key: string, value: unknown): string {
  if (value !== undefined && value !== null) return String(value);
  const def = HP_DEFAULTS[key];
  return def !== undefined ? String(def) : "default";
}

// Tailwind color equivalents for SVG fills (matching LoopTimeline)
function nodeFill(score: number | undefined): string {
  if (score === undefined) return "#e5e7eb"; // gray-200
  if (score >= PASS_THRESHOLD) return "#dcfce7"; // green-100
  if (score >= WARN_THRESHOLD) return "#fef9c3"; // yellow-100
  return "#fee2e2"; // red-100
}

function nodeStroke(score: number | undefined): string {
  if (score === undefined) return "#d1d5db"; // gray-300
  if (score >= PASS_THRESHOLD) return "#86efac"; // green-300
  if (score >= WARN_THRESHOLD) return "#fde047"; // yellow-300
  return "#fca5a5"; // red-300
}

function scoreTextColor(score: number | undefined): string {
  if (score === undefined) return "#4b5563"; // gray-600
  if (score >= PASS_THRESHOLD) return "#166534"; // green-800
  if (score >= WARN_THRESHOLD) return "#854d0e"; // yellow-800
  return "#991b1b"; // red-800
}

function deltaText(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}`;
}

function deltaColor(val: number): string {
  if (val > 0) return "text-green-400";
  if (val < 0) return "text-red-400";
  return "text-gray-400";
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
}

// ---------------------------------------------------------------------------
// Build tree from flat lineage
// ---------------------------------------------------------------------------

function buildTree(lineage: Lineage): TreeNode[] {
  // Mutable copy — we may inject virtual hub nodes
  const entries: Record<string, LineageEntry> = { ...lineage.iterations };
  const keys = Object.keys(entries);
  if (keys.length === 0) return [];

  // Group config-level keys by iteration prefix (e.g., "session/iter_1")
  const configGroups = new Map<string, string[]>();
  for (const key of keys) {
    if (isConfigKey(key)) {
      const prefix = key.split("/").slice(0, 2).join("/");
      if (!configGroups.has(prefix)) configGroups.set(prefix, []);
      configGroups.get(prefix)!.push(key);
    }
  }

  // Insert virtual hub nodes for config groups
  for (const [prefix, cfgKeys] of configGroups) {
    // Skip if an iteration-level node already exists (legacy data)
    if (prefix in entries) continue;

    // Hub's parent = shared parent of config nodes
    const sharedParent = entries[cfgKeys[0]].parent;

    // Find best config to populate hub with iteration-level summary
    let bestKey = cfgKeys[0];
    let hasStar = false;
    for (const ck of cfgKeys) {
      const e = entries[ck];
      if (e.is_best) bestKey = ck;
      if (e.star) hasStar = true;
    }
    const best = entries[bestKey];

    // Hub carries the best config's iteration-level data (score, lesson, diagnosis)
    entries[prefix] = {
      parent: sharedParent,
      score: best.score,
      final_return: best.final_return,
      lesson: best.lesson,
      diagnosis: best.diagnosis,
      failure_tags: best.failure_tags,
      best_checkpoint: best.best_checkpoint,
      star: hasStar || undefined,
    } as LineageEntry;

    // Reparent config nodes to hub
    for (const ck of cfgKeys) {
      entries[ck] = { ...entries[ck], parent: prefix };
    }
  }

  // Build tree from (possibly augmented) entries
  const allKeys = Object.keys(entries);
  const childrenMap: Record<string, string[]> = {};
  const roots: string[] = [];

  for (const key of allKeys) {
    const parent = entries[key].parent;
    if (parent && parent in entries) {
      if (!childrenMap[parent]) childrenMap[parent] = [];
      childrenMap[parent].push(key);
    } else {
      roots.push(key);
    }
  }

  for (const arr of Object.values(childrenMap)) {
    arr.sort();
  }
  roots.sort();

  let leafX = 0;

  function build(key: string, depth: number): TreeNode {
    const entry = entries[key];
    const childKeys = childrenMap[key] || [];
    const children = childKeys.map((ck) => build(ck, depth + 1));

    let x: number;
    if (children.length === 0) {
      x = leafX;
      leafX += 1;
    } else {
      x = (children[0].x + children[children.length - 1].x) / 2;
    }

    return { key, entry, children, x, y: depth, depth };
  }

  return roots.map((r) => build(r, 0));
}

// ---------------------------------------------------------------------------
// Flatten tree for rendering
// ---------------------------------------------------------------------------

interface FlatNode {
  key: string;
  entry: LineageEntry;
  px: number;
  py: number;
  parentKey: string | null;
}

function flattenTree(roots: TreeNode[]): FlatNode[] {
  const result: FlatNode[] = [];

  function walk(node: TreeNode, parentKey: string | null) {
    result.push({
      key: node.key,
      entry: node.entry,
      px: PAD + node.x * (NODE_W + H_GAP) + NODE_W / 2,
      py: PAD + node.y * (NODE_H + V_GAP),
      parentKey,
    });
    for (const child of node.children) {
      walk(child, node.key);
    }
  }

  for (const root of roots) walk(root, null);
  return result;
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function FloatingTooltip({
  anchor,
  children,
}: {
  anchor: { x: number; y: number } | null;
  children: React.ReactNode;
}) {
  if (!anchor) return null;
  return createPortal(
    <div
      className="fixed z-50 pointer-events-none"
      style={{ top: anchor.y + 8, left: anchor.x, transform: "translateX(-50%)" }}
    >
      <div className="bg-gray-900 text-gray-100 text-xs rounded-xl shadow-2xl p-4 max-w-sm w-max leading-relaxed pointer-events-auto">
        {children}
      </div>
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------
// Pan & zoom constants
// ---------------------------------------------------------------------------

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 6;
const ZOOM_SENSITIVITY = 0.001;

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/** Config metadata for multi-config sessions. */
export interface ConfigInfo {
  config_id: string;
  label: string;
  params: Record<string, unknown>;
}

export default function LineageTree({
  lineage,
  elapsedMap,
  isRunning,
  activeIterationKey,
  onSelectIteration,
  ghostBasedOn,
  configInfos,
}: {
  lineage: Lineage;
  elapsedMap?: ElapsedMap;
  isRunning?: boolean;
  activeIterationKey?: string;
  onSelectIteration?: (key: string) => void;
  /** Iteration number that the ghost node is based on (from revise agent). */
  ghostBasedOn?: number;
  /** Full config metadata for multi-config sessions. */
  configInfos?: ConfigInfo[];
}) {
  const configIds = configInfos?.map((c) => c.config_id);
  const configLookup = useMemo(() => {
    const map = new Map<string, ConfigInfo>();
    for (const c of configInfos ?? []) map.set(c.config_id, c);
    return map;
  }, [configInfos]);

  const [tooltip, setTooltip] = useState<{
    type: "node";
    node: FlatNode;
    pos: { x: number; y: number };
  } | {
    type: "edge";
    from: FlatNode;
    to: FlatNode;
    pos: { x: number; y: number };
  } | {
    type: "ghost";
    cfgInfo: ConfigInfo;
    iterNum: number;
    pos: { x: number; y: number };
  } | null>(null);

  const roots = useMemo(() => buildTree(lineage), [lineage]);
  const nodes = useMemo(() => flattenTree(roots), [roots]);

  // --- Pan & zoom state (must be before early returns) ---
  const svgContainerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const isDragging = useRef(false);
  const [isPanning, setIsPanning] = useState(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Attach wheel listener natively with { passive: false } so preventDefault works
  // (must be before early returns to satisfy rules-of-hooks)
  useEffect(() => {
    const el = svgContainerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      // Zoom toward cursor: remember cursor position in content coords before zoom
      const rect = el.getBoundingClientRect();
      const fx = (e.clientX - rect.left + el.scrollLeft) / el.scrollWidth;
      const fy = (e.clientY - rect.top + el.scrollTop) / el.scrollHeight;
      setZoom((prev) => {
        const factor = Math.exp(-e.deltaY * ZOOM_SENSITIVITY);
        const next = Math.min(Math.max(MIN_ZOOM, prev * factor), MAX_ZOOM);
        // Defer scroll adjustment to after render
        requestAnimationFrame(() => {
          el.scrollLeft = fx * el.scrollWidth - (e.clientX - rect.left);
          el.scrollTop = fy * el.scrollHeight - (e.clientY - rect.top);
        });
        return next;
      });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  if (nodes.length === 0 && !isRunning) {
    return <p className="text-sm text-gray-400">No lineage data yet.</p>;
  }

  // Show standalone ghost(s) when running first iteration (no completed nodes yet)
  if (nodes.length === 0 && isRunning) {
    const ghostCfgs = configInfos && configInfos.length > 1 ? configInfos : null;
    const ghostCount = ghostCfgs ? ghostCfgs.length : 1;
    const totalW = ghostCount * (NODE_W + H_GAP) - H_GAP + PAD * 2;
    // Hub at top row, config children below
    const hubX = PAD + (ghostCount - 1) * (NODE_W + H_GAP) / 2 + NODE_W / 2;
    const hubY = PAD;
    const cfgRowY = hubY + NODE_H + V_GAP;
    const svgH = ghostCfgs ? cfgRowY + NODE_H + PAD : hubY + NODE_H + PAD;
    return (
      <div>
        <div className="overflow-x-auto overflow-y-auto max-h-[500px]">
          <svg width={totalW} height={svgH} className="min-w-full">
            <defs>
              <style>{`
                @keyframes lineage-pulse {
                  0%, 100% { opacity: 1; }
                  50% { opacity: 0.5; }
                }
              `}</style>
              <filter id="blue-glow">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feFlood floodColor="#3b82f6" floodOpacity="0.4" result="color" />
                <feComposite in="color" in2="blur" operator="in" result="glow" />
                <feMerge>
                  <feMergeNode in="glow" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            {ghostCfgs ? (
              <>
                {/* Hub node: v1 */}
                <g
                  className="cursor-pointer"
                  style={{ animation: "lineage-pulse 2s ease-in-out infinite" }}
                  filter="url(#blue-glow)"
                  onClick={() => onSelectIteration?.(`/iter_1`)}
                >
                  <rect
                    x={hubX - NODE_W / 2}
                    y={hubY}
                    width={NODE_W}
                    height={NODE_H}
                    rx={8}
                    ry={8}
                    fill="#eff6ff"
                    stroke="#93c5fd"
                    strokeWidth={1.5}
                  />
                  <text x={hubX} y={hubY + 16} textAnchor="middle" className="text-[10px] font-semibold" fill="#2563eb">
                    v1
                  </text>
                  <text x={hubX} y={hubY + 30} textAnchor="middle" className="text-[9px] font-mono" fill="#2563eb">
                    training...
                  </text>
                </g>
                {/* Config children with edges from hub */}
                {ghostCfgs.map((cfg, idx) => {
                  const gx = PAD + idx * (NODE_W + H_GAP) + NODE_W / 2;
                  const hasParams = Object.keys(cfg.params).length > 0;
                  // Edge from hub to config
                  const x1 = hubX;
                  const y1 = hubY + NODE_H;
                  const x2 = gx;
                  const y2 = cfgRowY;
                  const my = (y1 + y2) / 2;
                  const d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
                  return (
                    <g key={cfg.config_id}>
                      <path d={d} fill="none" stroke="#93c5fd" strokeWidth={1.5} strokeDasharray="4 3" />
                      <g
                        className="cursor-pointer"
                        style={{ animation: "lineage-pulse 2s ease-in-out infinite" }}
                        filter="url(#blue-glow)"
                        onClick={() => onSelectIteration?.(`/iter_1`)}
                        onMouseEnter={(e) => {
                          if (isDragging.current) return;
                          const rect = (e.currentTarget as SVGGElement).getBoundingClientRect();
                          setTooltip({
                            type: "ghost",
                            cfgInfo: cfg,
                            iterNum: 1,
                            pos: { x: rect.left + rect.width / 2, y: rect.bottom },
                          });
                        }}
                        onMouseLeave={() => setTooltip(null)}
                      >
                        <rect
                          x={gx - NODE_W / 2}
                          y={cfgRowY}
                          width={NODE_W}
                          height={NODE_H}
                          rx={8}
                          ry={8}
                          fill={hasParams ? "#f9fafb" : "#eff6ff"}
                          stroke="#93c5fd"
                          strokeWidth={1.5}
                          strokeDasharray={hasParams ? "3 2" : undefined}
                        />
                        <text x={gx} y={cfgRowY + 16} textAnchor="middle" className="text-[10px] font-semibold" fill={hasParams ? "#6b7280" : "#2563eb"}>
                          {shortCfgId(cfg.config_id)}
                        </text>
                        <text x={gx} y={cfgRowY + 30} textAnchor="middle" className="text-[9px] font-mono" fill="#2563eb">
                          training...
                        </text>
                      </g>
                    </g>
                  );
                })}
              </>
            ) : (
              <g
                className="cursor-pointer"
                style={{ animation: "lineage-pulse 2s ease-in-out infinite" }}
                filter="url(#blue-glow)"
                onClick={() => onSelectIteration?.(`/iter_1`)}
              >
                <rect x={PAD} y={PAD} width={NODE_W} height={NODE_H} rx={8} ry={8} fill="#eff6ff" stroke="#93c5fd" strokeWidth={1.5} />
                <text x={PAD + NODE_W / 2} y={PAD + 16} textAnchor="middle" className="text-[10px] font-semibold" fill="#2563eb">v1</text>
                <text x={PAD + NODE_W / 2} y={PAD + 30} textAnchor="middle" className="text-[9px] font-mono" fill="#2563eb">training...</text>
              </g>
            )}
          </svg>
        </div>

        {/* Ghost tooltip */}
        {tooltip?.type === "ghost" && (() => {
          const cfg = tooltip.cfgInfo;
          const baseline = configLookup.get(configInfos?.[0]?.config_id ?? "");
          const isBaseline = baseline?.config_id === cfg.config_id;
          const diffs: { key: string; base: unknown; this_: unknown }[] = [];
          if (!isBaseline && baseline) {
            const allKeys = new Set([...Object.keys(baseline.params), ...Object.keys(cfg.params)]);
            for (const k of allKeys) {
              const bv = baseline.params[k];
              const tv = cfg.params[k];
              if (JSON.stringify(bv) !== JSON.stringify(tv)) {
                diffs.push({ key: k, base: bv, this_: tv });
              }
            }
          }
          return (
            <FloatingTooltip anchor={tooltip.pos}>
              <p className="font-semibold mb-2">v{tooltip.iterNum}/{cfg.config_id}</p>
              <div className="space-y-1">
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Config</span>
                  <span className="font-mono text-gray-300">{cfg.label}</span>
                </div>
                {Object.keys(cfg.params).length > 0 && (
                  <div>
                    <span className="text-gray-400">HP Overrides</span>
                    <div className="mt-0.5 font-mono text-[10px] text-cyan-300">
                      {formatHpParams(cfg.params)}
                    </div>
                  </div>
                )}
                {diffs.length > 0 && (
                  <div className="mt-1 pt-1 border-t border-gray-700">
                    <span className="text-gray-400">vs {baseline!.config_id}</span>
                    <div className="mt-0.5 space-y-0.5">
                      {diffs.map((d) => (
                        <div key={d.key} className="font-mono text-[10px]">
                          <span className="text-gray-500">{d.key}:</span>{" "}
                          <span className="text-cyan-400">{hpDisplay(d.key, d.this_)}</span>
                          <span className="text-gray-600"> vs </span>
                          <span className="text-gray-400">{hpDisplay(d.key, d.base)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {isBaseline && Object.keys(cfg.params).length === 0 && (
                  <p className="text-[10px] text-gray-500">Default hyperparameters</p>
                )}
                <div className="flex justify-between gap-4 mt-1">
                  <span className="text-gray-400">Score</span>
                  <span className="font-mono text-blue-400">training...</span>
                </div>
              </div>
            </FloatingTooltip>
          );
        })()}
      </div>
    );
  }

  const nodeMap = new Map(nodes.map((n) => [n.key, n]));

  // Find the last node (highest iteration) to attach the running ghost node
  const lastNode = nodes.reduce((best, n) => {
    const bestNum = iterNumFromKey(best.key) ?? 0;
    const nNum = iterNumFromKey(n.key) ?? 0;
    return nNum > bestNum ? n : best;
  }, nodes[0]);

  const lastIterNum = iterNumFromKey(lastNode.key) ?? 0;
  const ghostIterNum = lastIterNum + 1;
  const showGhost = isRunning && lastIterNum > 0;

  // Multi-config ghost: hub node + config children below
  const ghostConfigs = showGhost && configIds && configIds.length > 1 ? configIds : null;
  // Ghost hub parent: use based_on from the revise agent when available
  const ghostParent = (() => {
    if (!showGhost) return lastNode;

    // 1. If ghostBasedOn is provided and positive, find the best config node
    //    of that iteration (the revise agent declared "Based On: X")
    if ((ghostBasedOn ?? 0) > 0) {
      const basedOnConfigs = nodes.filter(
        (n) => isConfigKey(n.key) && iterNumFromKey(n.key) === ghostBasedOn,
      );
      if (basedOnConfigs.length > 0) {
        return (
          basedOnConfigs.find((n) => n.entry.is_best) ??
          basedOnConfigs.reduce(
            (a, b) => ((b.entry.score ?? 0) > (a.entry.score ?? 0) ? b : a),
            basedOnConfigs[0],
          )
        );
      }
      // No config-level nodes for that iteration — try hub node
      const basedOnHub = nodes.find(
        (n) => !isConfigKey(n.key) && iterNumFromKey(n.key) === ghostBasedOn,
      );
      if (basedOnHub) return basedOnHub;
    }

    // 2. If ghostBasedOn is 0/undefined, look for a starred node
    const starredNodes = nodes.filter((n) => n.entry.star);
    if (starredNodes.length > 0) {
      // Prefer config-level starred nodes
      const starredConfig = starredNodes.find((n) => isConfigKey(n.key));
      if (starredConfig) return starredConfig;
      // Multiple starred nodes but none at config level — pick highest score
      return starredNodes.reduce((a, b) =>
        (b.entry.score ?? 0) > (a.entry.score ?? 0) ? b : a,
      );
    }

    // 3. Final fallback: best config of last iteration (previous behavior)
    const lastIterConfigs = nodes.filter(
      (n) => isConfigKey(n.key) && iterNumFromKey(n.key) === lastIterNum,
    );
    if (lastIterConfigs.length > 0) {
      return (
        lastIterConfigs.find((n) => n.entry.is_best) ??
        lastIterConfigs.reduce(
          (a, b) => ((b.entry.score ?? 0) > (a.entry.score ?? 0) ? b : a),
          lastIterConfigs[0],
        )
      );
    }
    // Fallback: hub or last node
    return (
      nodes.find(
        (n) => !isConfigKey(n.key) && iterNumFromKey(n.key) === lastIterNum,
      ) ?? lastNode
    );
  })();

  // Hub ghost position: below ALL existing nodes to avoid overlapping siblings.
  // When the parent has many children at depth+1, placing the ghost at
  // parent.py + one row would collide with those children.
  const maxNodePy = Math.max(...nodes.map((n) => n.py), 0);
  const ghostHubPx = ghostParent.px;
  const ghostHubPy = Math.max(
    ghostParent.py + NODE_H + V_GAP,
    maxNodePy + NODE_H + V_GAP,
  );
  // Config ghosts fan from the hub (one row below hub); empty for single-config
  const ghostPositions: { cfgId: string | null; px: number; py: number }[] =
    showGhost && ghostConfigs
      ? ghostConfigs.map((cfgId, idx) => ({
          cfgId,
          px: ghostHubPx + (idx - (ghostConfigs.length - 1) / 2) * (NODE_W + H_GAP),
          py: ghostHubPy + NODE_H + V_GAP,
        }))
      : [];

  // Compute viewport bounds (ghost nodes may extend left of existing nodes)
  const allPxValues = [
    ...nodes.map((n) => n.px),
    ...ghostPositions.map((g) => g.px),
    showGhost ? ghostHubPx : PAD + NODE_W / 2,
  ];
  const minX = Math.min(...allPxValues) - NODE_W / 2 - PAD;
  const maxX =
    Math.max(...allPxValues, 0) +
    NODE_W / 2 +
    PAD;
  const viewX = Math.min(minX, 0);
  const viewW = maxX - viewX;
  const maxY =
    Math.max(
      ...nodes.map((n) => n.py),
      ...ghostPositions.map((g) => g.py),
      showGhost && !ghostConfigs ? ghostHubPy : 0,
      0,
    ) +
    NODE_H +
    PAD;

  const edges: { from: FlatNode; to: FlatNode; dashed: boolean }[] = [];
  for (const node of nodes) {
    if (node.parentKey && nodeMap.has(node.parentKey)) {
      edges.push({ from: nodeMap.get(node.parentKey)!, to: node, dashed: false });
    }
    const alsoFrom = node.entry.also_from;
    if (alsoFrom && nodeMap.has(alsoFrom)) {
      edges.push({ from: nodeMap.get(alsoFrom)!, to: node, dashed: true });
    }
  }

  // --- Pan & zoom handlers ---

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isDragging.current = true;
    setIsPanning(true);
    setTooltip(null);
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const container = svgContainerRef.current;
    if (!container) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    container.scrollLeft -= dx;
    container.scrollTop -= dy;
  };
  const handleMouseUp = () => { isDragging.current = false; setIsPanning(false); };
  const resetView = () => setZoom(1);

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] text-gray-500">{Math.round(zoom * 100)}%</span>
        {zoom !== 1 && (
          <button
            onClick={resetView}
            className="text-[10px] text-blue-500 hover:text-blue-400"
          >
            reset
          </button>
        )}
      </div>
      <div
        ref={svgContainerRef}
        className={`overflow-auto max-h-[500px] select-none ${isPanning ? "cursor-grabbing" : "cursor-grab"}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg width={viewW * zoom} height={maxY * zoom} viewBox={`${viewX} 0 ${viewW} ${maxY}`}>
          {/* Edges (rendered before nodes so they appear behind) */}
          {edges.map((edge, i) => {
            const x1 = edge.from.px;
            const y1 = edge.from.py + NODE_H;
            const x2 = edge.to.px;
            const y2 = edge.to.py;
            const my = (y1 + y2) / 2;
            const d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
            return (
              <g key={`edge-${i}`}>
                {/* Visible edge */}
                <path
                  d={d}
                  fill="none"
                  stroke={edge.dashed ? "#c084fc" : "#d1d5db"}
                  strokeWidth={edge.dashed ? 1 : 1.5}
                  strokeDasharray={edge.dashed ? "4 3" : undefined}
                />
                {/* Wider invisible hit area for hover */}
                <path
                  d={d}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={12}
                  className="cursor-default"
                  onMouseEnter={(e) => {
                    if (isDragging.current) return;
                    const rect = (e.currentTarget as SVGPathElement).getBoundingClientRect();
                    setTooltip({
                      type: "edge",
                      from: edge.from,
                      to: edge.to,
                      pos: { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 },
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
              </g>
            );
          })}
          {/* Ghost edges (also before nodes so they render behind) */}
          {showGhost && (() => {
            const x1 = ghostParent.px;
            const y1 = ghostParent.py + NODE_H;
            const x2 = ghostHubPx;
            const y2 = ghostHubPy;
            const my = (y1 + y2) / 2;
            const d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
            return (
              <g key="ghost-edges">
                {/* Parent → ghost hub edge */}
                <path d={d} fill="none" stroke="#d1d5db" strokeWidth={1.5} />
                {/* Ghost hub → config child edges */}
                {ghostConfigs && ghostPositions.map((ghost) => {
                  const cx1 = ghostHubPx;
                  const cy1 = ghostHubPy + NODE_H;
                  const cx2 = ghost.px;
                  const cy2 = ghost.py;
                  const cmy = (cy1 + cy2) / 2;
                  const cd = `M${cx1},${cy1} C${cx1},${cmy} ${cx2},${cmy} ${cx2},${cy2}`;
                  return (
                    <path key={`ghost-edge-${ghost.cfgId}`} d={cd} fill="none" stroke="#93c5fd" strokeWidth={1.5} strokeDasharray="4 3" />
                  );
                })}
              </g>
            );
          })()}

          {/* Nodes */}
          {nodes.map((node) => {
            const nx = node.px - NODE_W / 2;
            const ny = node.py;
            const score = node.entry.score;
            const fill = nodeFill(score);
            const stroke = nodeStroke(score);
            const textColor = scoreTextColor(score);
            const isStar = node.entry.star;
            const isBest = node.entry.is_best;
            const isConfig = isConfigKey(node.key);
            const isActive = node.key === activeIterationKey;
            const iterNum = iterNumFromKey(node.key);
            const elapsed = iterNum != null && elapsedMap?.[iterNum];
            const cfgId = configIdFromKey(node.key);
            return (
              <g
                key={node.key}
                className="cursor-pointer"
                onClick={() => onSelectIteration?.(node.key)}
                onMouseEnter={(e) => {
                  if (isDragging.current) return;
                  const rect = (e.currentTarget as SVGGElement).getBoundingClientRect();
                  setTooltip({
                    type: "node",
                    node,
                    pos: { x: rect.left + rect.width / 2, y: rect.bottom },
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                {/* Active selection ring */}
                {isActive && (
                  <rect
                    x={nx - 3}
                    y={ny - 3}
                    width={NODE_W + 6}
                    height={NODE_H + 6}
                    rx={11}
                    ry={11}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth={2}
                  />
                )}
                <rect
                  x={nx}
                  y={ny}
                  width={NODE_W}
                  height={NODE_H}
                  rx={8}
                  ry={8}
                  fill={isConfig && !isBest ? "#f9fafb" : fill}
                  stroke={isStar ? "#fbbf24" : isBest ? "#60a5fa" : stroke}
                  strokeWidth={isStar ? 2.5 : isBest ? 2 : 1.5}
                  strokeDasharray={isConfig && !isBest ? "3 2" : undefined}
                />
                {/* Label */}
                <text
                  x={node.px}
                  y={ny + (isConfig ? 17 : 13)}
                  textAnchor="middle"
                  className="text-[10px] font-semibold"
                  fill={isConfig && !isBest ? "#6b7280" : textColor}
                >
                  {isConfig
                    ? shortCfgId(cfgId ?? "")
                    : `v${iterNum ?? "?"}`}
                  {isStar ? " \u2605" : ""}
                </text>
                {/* Score */}
                <text
                  x={node.px}
                  y={ny + (isConfig ? 30 : 26)}
                  textAnchor="middle"
                  className="text-[11px] font-mono font-medium"
                  fill={isConfig && !isBest ? "#9ca3af" : textColor}
                >
                  {score !== undefined ? score.toFixed(2) : "..."}
                </text>
                {/* Elapsed time (iteration nodes only) */}
                {!isConfig && elapsed && (
                  <text
                    x={node.px}
                    y={ny + 37}
                    textAnchor="middle"
                    className="text-[8px]"
                    fill="#6b7280"
                  >
                    {formatElapsed(elapsed)}
                  </text>
                )}
                {/* Star glow */}
                {isStar && (
                  <rect
                    x={nx - 1}
                    y={ny - 1}
                    width={NODE_W + 2}
                    height={NODE_H + 2}
                    rx={9}
                    ry={9}
                    fill="none"
                    stroke="#fbbf24"
                    strokeWidth={1}
                    opacity={0.4}
                  />
                )}
              </g>
            );
          })}

          {/* In-progress ghost node(s) — edges already rendered above */}
          {showGhost && (
            <>
              <defs>
                <style>{`
                  @keyframes lineage-pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                  }
                `}</style>
                <filter id="blue-glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feFlood floodColor="#3b82f6" floodOpacity="0.4" result="color" />
                  <feComposite in="color" in2="blur" operator="in" result="glow" />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              {/* Ghost hub node — no score yet, still training */}
              {(() => {
                const sessionId = lastNode.key.split("/")[0];
                return (
                  <g
                    className="cursor-pointer"
                    opacity={0.6}
                    onClick={() => onSelectIteration?.(`${sessionId}/iter_${ghostIterNum}`)}
                  >
                    <rect
                      x={ghostHubPx - NODE_W / 2}
                      y={ghostHubPy}
                      width={NODE_W}
                      height={NODE_H}
                      rx={8}
                      ry={8}
                      fill="#e5e7eb"
                      stroke="#93c5fd"
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                    />
                    <text x={ghostHubPx} y={ghostHubPy + 16} textAnchor="middle" className="text-[10px] font-semibold" fill="#6b7280">
                      v{ghostIterNum}
                    </text>
                    <text x={ghostHubPx} y={ghostHubPy + 30} textAnchor="middle" className="text-[11px] font-mono font-medium" fill="#3b82f6">
                      ...
                    </text>
                  </g>
                );
              })()}
              {/* Config ghost children (edges already rendered in edge layer) */}
              {ghostConfigs && ghostPositions.map((ghost) => {
                const cfg = ghost.cfgId ? configLookup.get(ghost.cfgId) : null;
                const hasParams = cfg ? Object.keys(cfg.params).length > 0 : false;
                const sessionId = lastNode.key.split("/")[0];
                return (
                  <g key={ghost.cfgId ?? "single"}>
                    <g
                      className="cursor-pointer"
                      style={{ animation: "lineage-pulse 2s ease-in-out infinite" }}
                      filter="url(#blue-glow)"
                      onClick={() => onSelectIteration?.(`${sessionId}/iter_${ghostIterNum}`)}
                      onMouseEnter={(e) => {
                        if (isDragging.current) return;
                        if (cfg) {
                          const rect = (e.currentTarget as SVGGElement).getBoundingClientRect();
                          setTooltip({
                            type: "ghost",
                            cfgInfo: cfg,
                            iterNum: ghostIterNum,
                            pos: { x: rect.left + rect.width / 2, y: rect.bottom },
                          });
                        }
                      }}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <rect
                        x={ghost.px - NODE_W / 2}
                        y={ghost.py}
                        width={NODE_W}
                        height={NODE_H}
                        rx={8}
                        ry={8}
                        fill={hasParams ? "#f9fafb" : "#eff6ff"}
                        stroke="#93c5fd"
                        strokeWidth={1.5}
                        strokeDasharray={hasParams ? "3 2" : undefined}
                      />
                      <text
                        x={ghost.px}
                        y={ghost.py + 16}
                        textAnchor="middle"
                        className="text-[10px] font-semibold"
                        fill={hasParams ? "#6b7280" : "#2563eb"}
                      >
                        {shortCfgId(ghost.cfgId ?? "")}
                      </text>
                      <text
                        x={ghost.px}
                        y={ghost.py + 30}
                        textAnchor="middle"
                        className="text-[10px] font-mono"
                        fill="#2563eb"
                      >
                        training...
                      </text>
                    </g>
                  </g>
                );
              })}
            </>
          )}
        </svg>
      </div>

      {/* Lessons section */}
      {lineage.lessons.length > 0 && (
        <details className="mt-4">
          <summary className="text-xs font-semibold text-gray-600 cursor-pointer hover:text-gray-900">
            Accumulated Lessons ({lineage.lessons.length})
          </summary>
          <ul className="mt-2 space-y-1 text-xs text-gray-600">
            {lineage.lessons.map((lesson, i) => {
              // Handle both string[] (legacy) and StructuredLesson[] formats
              const isStructured = typeof lesson === "object" && lesson !== null && "text" in lesson;
              const sl = isStructured ? (lesson as StructuredLesson) : null;
              const text = sl ? sl.text : (lesson as string);
              const tier = sl?.tier ?? "STRONG";
              const learnedAt = sl?.learned_at;
              const badgeClass = TIER_COLORS[tier] ?? TIER_COLORS.STRONG;
              return (
                <li key={i} className="flex gap-2 items-start">
                  <span className="text-gray-400 font-mono shrink-0">{i + 1}.</span>
                  {sl && (
                    <span className={`shrink-0 px-1 py-0.5 text-[10px] font-semibold border rounded ${badgeClass}`}>
                      {tier}
                    </span>
                  )}
                  <span className={tier === "RETIRED" ? "line-through text-gray-400" : ""}>
                    {text}
                    {learnedAt ? <span className="text-gray-400 ml-1">(iter {learnedAt})</span> : null}
                  </span>
                </li>
              );
            })}
          </ul>
        </details>
      )}

      {/* Node Tooltip */}
      {tooltip?.type === "node" && (() => {
        const tn = tooltip.node;
        const tEntry = tn.entry;
        const isCfg = isConfigKey(tn.key);
        const iterN = iterNumFromKey(tn.key);
        const cfgId = configIdFromKey(tn.key);
        return (
          <FloatingTooltip anchor={tooltip.pos}>
            <p className="font-semibold mb-2">
              {isCfg ? `v${iterN ?? "?"}/${cfgId}` : `v${iterN ?? "?"}`}
              {tEntry.is_best && <span className="text-blue-400 ml-1">(best config)</span>}
            </p>
            <div className="space-y-1">
              {tEntry.score !== undefined && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Score</span>
                  <span className={`font-mono font-medium ${
                    tEntry.score >= PASS_THRESHOLD ? "text-green-400"
                      : tEntry.score >= WARN_THRESHOLD ? "text-yellow-400"
                      : "text-red-400"
                  }`}>
                    {tEntry.score.toFixed(3)}
                    {tEntry.score_std != null ? ` \u00B1${tEntry.score_std.toFixed(3)}` : ""}
                  </span>
                </div>
              )}
              {tEntry.final_return != null && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Final Return</span>
                  <span className="font-mono">
                    {tEntry.final_return.toFixed(1)}
                    {tEntry.return_std != null ? ` \u00B1${tEntry.return_std.toFixed(1)}` : ""}
                  </span>
                </div>
              )}
              {/* HP params for config nodes */}
              {isCfg && tEntry.hp_params && Object.keys(tEntry.hp_params).length > 0 && (
                <div>
                  <span className="text-gray-400">HP Overrides</span>
                  <div className="mt-0.5 font-mono text-[10px] text-cyan-300">
                    {formatHpParams(tEntry.hp_params)}
                  </div>
                </div>
              )}
              {/* vs Best comparison for non-best config siblings */}
              {isCfg && !tEntry.is_best && (() => {
                const iterPrefix = tn.key.split("/").slice(0, 2).join("/") + "/";
                const bestSibling = nodes.find(
                  (n) => n.key.startsWith(iterPrefix) && n.key !== tn.key && n.entry.is_best,
                );
                if (!bestSibling) return null;
                const bestHp = bestSibling.entry.hp_params;
                const thisHp = tEntry.hp_params;
                const diffs: { key: string; best: unknown; this_: unknown }[] = [];
                if (bestHp && thisHp) {
                  const allKeys = new Set([...Object.keys(bestHp), ...Object.keys(thisHp)]);
                  for (const k of allKeys) {
                    const bv = bestHp[k];
                    const tv = thisHp[k];
                    if (JSON.stringify(bv) !== JSON.stringify(tv)) {
                      diffs.push({ key: k, best: bv, this_: tv });
                    }
                  }
                }
                const bestScore = bestSibling.entry.score;
                const thisScore = tEntry.score;
                const sDelta = bestScore != null && thisScore != null ? thisScore - bestScore : null;
                const bestCfgId = configIdFromKey(bestSibling.key);
                return (
                  <div className="mt-1 pt-1 border-t border-gray-700">
                    <span className="text-gray-400">vs Best ({bestCfgId})</span>
                    {sDelta != null && (
                      <div className="flex justify-between gap-4 mt-0.5">
                        <span className="text-gray-500 text-[10px]">Score Delta</span>
                        <span className={`font-mono text-[10px] font-semibold ${deltaColor(sDelta)}`}>
                          {deltaText(sDelta)}
                        </span>
                      </div>
                    )}
                    {diffs.length > 0 && (
                      <div className="mt-0.5 space-y-0.5">
                        {diffs.map((d) => (
                          <div key={d.key} className="font-mono text-[10px]">
                            <span className="text-gray-500">{d.key}:</span>{" "}
                            <span className="text-cyan-400">{String(d.this_ ?? "—")}</span>
                            <span className="text-gray-600"> vs </span>
                            <span className="text-gray-400">{String(d.best ?? "—")}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {diffs.length === 0 && (
                      <p className="text-[10px] text-gray-500 mt-0.5">same HP</p>
                    )}
                  </div>
                );
              })()}
              {tEntry.config_label && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Config</span>
                  <span className="font-mono text-gray-300">{tEntry.config_label}</span>
                </div>
              )}
              {tEntry.best_checkpoint && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Best Checkpoint</span>
                  <span className="font-mono text-gray-300 truncate max-w-[180px]">{tEntry.best_checkpoint}</span>
                </div>
              )}
              {(() => {
                const el = iterN != null && elapsedMap?.[iterN];
                if (!el) return null;
                return (
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-400">Training Time</span>
                    <span className="font-mono">{formatElapsed(el)}</span>
                  </div>
                );
              })()}
              {tEntry.failure_tags && tEntry.failure_tags.length > 0 && (
                <div>
                  <span className="text-gray-400">Failure Tags</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {tEntry.failure_tags.map((tag) => (
                      <span key={tag} className="px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 text-[10px]">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {tEntry.diagnosis && (
                <div className="mt-1">
                  <span className="text-gray-400">Diagnosis</span>
                  <p className="mt-0.5 text-gray-300 line-clamp-3">{tEntry.diagnosis}</p>
                </div>
              )}
              {tEntry.lesson && (
                <div>
                  <span className="text-gray-400">Lesson</span>
                  <p className="mt-0.5 text-amber-300">{tEntry.lesson}</p>
                </div>
              )}
              {tEntry.star && (
                <div className="text-amber-400 font-medium mt-1">{"\u2605"} Best score</div>
              )}
            </div>
          </FloatingTooltip>
        );
      })()}

      {/* Edge Tooltip */}
      {tooltip?.type === "edge" && (() => {
        const isGhostEdge = tooltip.from === tooltip.to;
        const fromIterNum = iterNumFromKey(tooltip.from.key) ?? 0;
        const toIterNum = isGhostEdge ? fromIterNum + 1 : (iterNumFromKey(tooltip.to.key) ?? 0);
        // Hub→config edge (same iteration): show HP diff, not score delta
        const isHubToConfig = !isGhostEdge && !isConfigKey(tooltip.from.key) && isConfigKey(tooltip.to.key) && fromIterNum === toIterNum;

        const fromScore = tooltip.from.entry.score;
        const toScore = tooltip.to.entry.score;
        const scoreDelta = !isGhostEdge && !isHubToConfig && fromScore != null && toScore != null ? toScore - fromScore : null;
        const fromReturn = tooltip.from.entry.final_return;
        const toReturn = tooltip.to.entry.final_return;
        const returnDelta = !isGhostEdge && !isHubToConfig && fromReturn != null && toReturn != null ? toReturn - fromReturn : null;
        const changeSummary = isGhostEdge ? tooltip.from.entry.lesson : isHubToConfig ? undefined : tooltip.to.entry.lesson;
        const diagnosis = isGhostEdge ? tooltip.from.entry.diagnosis : undefined;

        // HP diff: for hub→config, compare config's HP vs baseline (first config sibling)
        const hpDiffs: { key: string; from: unknown; to: unknown }[] = [];
        if (isHubToConfig) {
          // Find baseline sibling (first config with empty params or first sibling)
          const iterPrefix = tooltip.to.key.split("/").slice(0, 2).join("/") + "/";
          const baselineSibling = nodes.find(
            (n) => n.key.startsWith(iterPrefix) && n.key !== tooltip.to.key &&
              (!n.entry.hp_params || Object.keys(n.entry.hp_params).length === 0),
          );
          const baseHp = baselineSibling?.entry.hp_params ?? {};
          const toHp = tooltip.to.entry.hp_params ?? {};
          const allKeys = new Set([...Object.keys(baseHp), ...Object.keys(toHp)]);
          for (const k of allKeys) {
            if (JSON.stringify(baseHp[k]) !== JSON.stringify(toHp[k])) {
              hpDiffs.push({ key: k, from: baseHp[k], to: toHp[k] });
            }
          }
        } else if (!isGhostEdge) {
          // Cross-iteration edge: compare parent HP → child HP
          const fromHp = tooltip.from.entry.hp_params;
          const toHp = tooltip.to.entry.hp_params;
          if (fromHp && toHp) {
            const allKeys = new Set([...Object.keys(fromHp), ...Object.keys(toHp)]);
            for (const k of allKeys) {
              if (JSON.stringify(fromHp[k]) !== JSON.stringify(toHp[k])) {
                hpDiffs.push({ key: k, from: fromHp[k], to: toHp[k] });
              }
            }
          }
        }

        const fromLabel = isConfigKey(tooltip.from.key)
          ? `v${fromIterNum}/${configIdFromKey(tooltip.from.key)}`
          : `v${fromIterNum}`;
        const toLabel = isGhostEdge
          ? `v${toIterNum}`
          : isConfigKey(tooltip.to.key)
            ? `v${toIterNum}/${configIdFromKey(tooltip.to.key)}`
            : `v${toIterNum}`;

        return (
          <FloatingTooltip anchor={tooltip.pos}>
            <p className="font-semibold mb-2">
              {fromLabel} &rarr; {toLabel}
              {isGhostEdge && <span className="text-blue-400 font-normal ml-1">(training)</span>}
            </p>
            <div className="space-y-1">
              {isHubToConfig && tooltip.to.entry.score != null && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Score</span>
                  <span className={`font-mono font-medium ${
                    tooltip.to.entry.score >= PASS_THRESHOLD ? "text-green-400"
                      : tooltip.to.entry.score >= WARN_THRESHOLD ? "text-yellow-400"
                      : "text-red-400"
                  }`}>
                    {tooltip.to.entry.score.toFixed(3)}
                    {tooltip.to.entry.is_best && " (best)"}
                  </span>
                </div>
              )}
              {scoreDelta != null && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Score Delta</span>
                  <span className={`text-sm font-mono font-semibold ${deltaColor(scoreDelta)}`}>
                    {deltaText(scoreDelta)}
                  </span>
                </div>
              )}
              {returnDelta != null && (
                <div className="flex justify-between gap-4">
                  <span className="text-gray-400">Return Delta</span>
                  <span className={`text-sm font-mono font-semibold ${deltaColor(returnDelta)}`}>
                    {deltaText(returnDelta)}
                  </span>
                </div>
              )}
              {/* HP diff */}
              {hpDiffs.length > 0 && (
                <div>
                  <span className="text-gray-400">{isHubToConfig ? "HP Overrides (vs baseline)" : "HP Changes"}</span>
                  <div className="mt-0.5 space-y-0.5">
                    {hpDiffs.map((d) => (
                      <div key={d.key} className="font-mono text-[10px]">
                        <span className="text-gray-500">{d.key}:</span>{" "}
                        <span className="text-red-400">{hpDisplay(d.key, d.from)}</span>
                        {" → "}
                        <span className="text-green-400">{hpDisplay(d.key, d.to)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {isHubToConfig && hpDiffs.length === 0 && (
                <p className="text-[10px] text-gray-500">Default hyperparameters (baseline)</p>
              )}
              {diagnosis && (
                <div className="mt-1">
                  <span className="text-gray-400">Diagnosis</span>
                  <p className="mt-0.5 text-gray-300 line-clamp-3">{diagnosis}</p>
                </div>
              )}
              {changeSummary && (
                <div className="mt-1">
                  <span className="text-gray-400">Change Summary</span>
                  <p className="mt-0.5 text-gray-300 line-clamp-3">{changeSummary}</p>
                </div>
              )}
            </div>
          </FloatingTooltip>
        );
      })()}

      {/* Ghost node Tooltip */}
      {tooltip?.type === "ghost" && (() => {
        const cfg = tooltip.cfgInfo;
        const baseline = configLookup.get(configInfos?.[0]?.config_id ?? "");
        const isBaseline = baseline?.config_id === cfg.config_id;
        const diffs: { key: string; base: unknown; this_: unknown }[] = [];
        if (!isBaseline && baseline) {
          const allKeys = new Set([...Object.keys(baseline.params), ...Object.keys(cfg.params)]);
          for (const k of allKeys) {
            const bv = baseline.params[k];
            const tv = cfg.params[k];
            if (JSON.stringify(bv) !== JSON.stringify(tv)) {
              diffs.push({ key: k, base: bv, this_: tv });
            }
          }
        }
        return (
          <FloatingTooltip anchor={tooltip.pos}>
            <p className="font-semibold mb-2">v{tooltip.iterNum}/{cfg.config_id}</p>
            <div className="space-y-1">
              <div className="flex justify-between gap-4">
                <span className="text-gray-400">Config</span>
                <span className="font-mono text-gray-300">{cfg.label}</span>
              </div>
              {Object.keys(cfg.params).length > 0 && (
                <div>
                  <span className="text-gray-400">HP Overrides</span>
                  <div className="mt-0.5 font-mono text-[10px] text-cyan-300">
                    {formatHpParams(cfg.params)}
                  </div>
                </div>
              )}
              {diffs.length > 0 && (
                <div className="mt-1 pt-1 border-t border-gray-700">
                  <span className="text-gray-400">vs {baseline!.config_id}</span>
                  <div className="mt-0.5 space-y-0.5">
                    {diffs.map((d) => (
                      <div key={d.key} className="font-mono text-[10px]">
                        <span className="text-gray-500">{d.key}:</span>{" "}
                        <span className="text-cyan-400">{hpDisplay(d.key, d.this_)}</span>
                        <span className="text-gray-600"> vs </span>
                        <span className="text-gray-400">{hpDisplay(d.key, d.base)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {isBaseline && Object.keys(cfg.params).length === 0 && (
                <p className="text-[10px] text-gray-500">Default hyperparameters</p>
              )}
              <div className="flex justify-between gap-4 mt-1">
                <span className="text-gray-400">Score</span>
                <span className="font-mono text-blue-400">training...</span>
              </div>
            </div>
          </FloatingTooltip>
        );
      })()}
    </div>
  );
}
