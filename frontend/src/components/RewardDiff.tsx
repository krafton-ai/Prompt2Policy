"use client";

import { useState } from "react";
import { diffLines, type Change } from "diff";

const CONTEXT_LINES = 3;

interface DiffLine {
  type: "added" | "removed" | "unchanged";
  content: string;
  oldNum: number | null;
  newNum: number | null;
}

function computeDiffLines(prev: string, current: string): DiffLine[] {
  const changes: Change[] = diffLines(prev, current);
  const lines: DiffLine[] = [];
  let oldNum = 1;
  let newNum = 1;

  for (const change of changes) {
    const rawLines = change.value.replace(/\n$/, "").split("\n");
    for (const raw of rawLines) {
      if (change.added) {
        lines.push({ type: "added", content: raw, oldNum: null, newNum });
        newNum++;
      } else if (change.removed) {
        lines.push({ type: "removed", content: raw, oldNum, newNum: null });
        oldNum++;
      } else {
        lines.push({ type: "unchanged", content: raw, oldNum, newNum });
        oldNum++;
        newNum++;
      }
    }
  }
  return lines;
}

/** Group consecutive unchanged lines, collapsing middle blocks. */
function groupLines(
  lines: DiffLine[],
): Array<{ kind: "lines"; lines: DiffLine[] } | { kind: "collapsed"; count: number; lines: DiffLine[] }> {
  const groups: Array<
    | { kind: "lines"; lines: DiffLine[] }
    | { kind: "collapsed"; count: number; lines: DiffLine[] }
  > = [];

  let buffer: DiffLine[] = [];
  let bufferKind: "changed" | "unchanged" = "unchanged";

  function flush() {
    if (buffer.length === 0) return;
    if (bufferKind === "changed") {
      groups.push({ kind: "lines", lines: buffer });
    } else {
      // unchanged block — keep first/last CONTEXT_LINES, collapse middle
      if (buffer.length <= CONTEXT_LINES * 2 + 1) {
        groups.push({ kind: "lines", lines: buffer });
      } else {
        const head = buffer.slice(0, CONTEXT_LINES);
        const tail = buffer.slice(-CONTEXT_LINES);
        const middle = buffer.slice(CONTEXT_LINES, -CONTEXT_LINES);
        if (head.length > 0) groups.push({ kind: "lines", lines: head });
        groups.push({ kind: "collapsed", count: middle.length, lines: middle });
        if (tail.length > 0) groups.push({ kind: "lines", lines: tail });
      }
    }
    buffer = [];
  }

  for (const line of lines) {
    const kind = line.type === "unchanged" ? "unchanged" : "changed";
    if (kind !== bufferKind) {
      flush();
      bufferKind = kind;
    }
    buffer.push(line);
  }
  flush();
  return groups;
}

function LineNum({ num }: { num: number | null }) {
  return (
    <span className="inline-block w-10 text-right text-gray-400 select-none pr-2 text-[11px]">
      {num ?? ""}
    </span>
  );
}

function DiffLineRow({ line }: { line: DiffLine }) {
  const bg =
    line.type === "added"
      ? "bg-green-50"
      : line.type === "removed"
        ? "bg-red-50"
        : "";
  const textColor =
    line.type === "added"
      ? "text-green-800"
      : line.type === "removed"
        ? "text-red-800"
        : "text-gray-700";
  const prefix =
    line.type === "added" ? "+" : line.type === "removed" ? "-" : " ";

  return (
    <div className={`flex ${bg} leading-5`}>
      <LineNum num={line.oldNum} />
      <LineNum num={line.newNum} />
      <span className={`flex-1 font-mono text-xs whitespace-pre-wrap ${textColor}`}>
        <span className="inline-block w-4 text-center select-none opacity-60">
          {prefix}
        </span>
        {line.content}
      </span>
    </div>
  );
}

export default function RewardDiff({
  prev,
  current,
  diffSummary,
}: {
  prev: string | null;
  current: string;
  diffSummary?: string;
}) {
  const [expandedCollapsed, setExpandedCollapsed] = useState<Set<number>>(
    new Set(),
  );

  // No prev (iteration 1) — show code only
  if (!prev) {
    return (
      <details className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <summary className="text-sm font-semibold text-gray-900 cursor-pointer">
          Reward Function
        </summary>
        <pre className="mt-3 text-xs font-mono text-gray-600 overflow-x-auto whitespace-pre-wrap">
          {current}
        </pre>
      </details>
    );
  }

  // Same code — no diff needed
  if (prev === current) {
    return (
      <details className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <summary className="text-sm font-semibold text-gray-900 cursor-pointer">
          Reward Function (unchanged)
        </summary>
        <pre className="mt-3 text-xs font-mono text-gray-600 overflow-x-auto whitespace-pre-wrap">
          {current}
        </pre>
      </details>
    );
  }

  const diffLines = computeDiffLines(prev, current);
  const groups = groupLines(diffLines);

  const added = diffLines.filter((l) => l.type === "added").length;
  const removed = diffLines.filter((l) => l.type === "removed").length;

  return (
    <details
      className="bg-white rounded-xl border border-gray-200 shadow-sm p-5"
      open
    >
      <summary className="text-sm font-semibold text-gray-900 cursor-pointer">
        Reward Function Diff{" "}
        <span className="text-xs font-normal text-gray-500">
          (<span className="text-green-600">+{added}</span>{" "}
          <span className="text-red-600">-{removed}</span>)
        </span>
      </summary>

      {diffSummary && (
        <p className="mt-3 text-sm text-gray-700 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2">
          {diffSummary}
        </p>
      )}

      <div className="mt-3 border border-gray-200 rounded-lg overflow-hidden text-xs">
        {groups.map((group, gi) => {
          if (group.kind === "lines") {
            return group.lines.map((line, li) => (
              <DiffLineRow key={`${gi}-${li}`} line={line} />
            ));
          }
          // Collapsed block
          const isExpanded = expandedCollapsed.has(gi);
          if (isExpanded) {
            return group.lines.map((line, li) => (
              <DiffLineRow key={`${gi}-${li}`} line={line} />
            ));
          }
          return (
            <button
              key={gi}
              className="w-full bg-gray-50 text-gray-500 text-center py-1 hover:bg-gray-100 cursor-pointer border-y border-gray-200 text-[11px]"
              onClick={() =>
                setExpandedCollapsed((s) => new Set([...s, gi]))
              }
            >
              ... {group.count} unchanged lines ...
            </button>
          );
        })}
      </div>
    </details>
  );
}
