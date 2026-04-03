"use client";

import { useState } from "react";
import useSWR from "swr";
import {
  fetchEvents,
  fetchEventDetail,
  type EventSummary,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Styling helpers
// ---------------------------------------------------------------------------

const EVENT_COLORS: Record<string, string> = {
  "session.started": "bg-green-500",
  "session.completed": "bg-green-600",
  "session.error": "bg-red-500",
  "iteration.started": "bg-blue-400",
  "iteration.completed": "bg-blue-600",
  "reward.generate.start": "bg-purple-400",
  "reward.generate.end": "bg-purple-600",
  "reward.provided": "bg-purple-500",
  "judge_code.ready": "bg-gray-500",
  "train.start": "bg-amber-400",
  "train.end": "bg-amber-600",
  "judge.start": "bg-cyan-400",
  "judge.end": "bg-cyan-600",
  "revise.start": "bg-indigo-400",
  "revise.end": "bg-indigo-600",
  "llm.call": "bg-violet-500",
  "llm.conversation": "bg-violet-600",
  "vlm.call": "bg-teal-500",
  "guardrail.warning": "bg-red-400",
};

function dotColor(event: string): string {
  return EVENT_COLORS[event] || "bg-gray-400";
}

function formatDuration(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return iso;
  }
}

function truncate(s: unknown, maxLen = 200): string {
  if (typeof s !== "string") return String(s ?? "");
  return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
}

import { getModelPricing, formatCost } from "@/lib/pricing";

interface CostStats { inTot: number; outTot: number; cost: number; count: number }

function computeCost(calls: { data: Record<string, unknown> }[]): CostStats {
  let inTot = 0, outTot = 0, cost = 0;
  for (const e of calls) {
    const inTok = Number(e.data.input_tokens) || 0;
    const outTok = Number(e.data.output_tokens) || 0;
    const p = getModelPricing(String(e.data.model || ""));
    inTot += inTok;
    outTot += outTok;
    cost += (inTok * p.inputPerMTok + outTok * p.outputPerMTok) / 1_000_000;
  }
  return { inTot, outTot, cost, count: calls.length };
}

// Tool call/result types for LLM events
interface ToolCall {
  id: string;
  name: string;
  input: unknown;
}

interface ToolResult {
  tool_use_id: string;
  content: unknown;
}

function formatToolContent(value: unknown): string {
  if (typeof value === "string") {
    // Try to parse as JSON for pretty-printing
    try {
      const parsed = JSON.parse(value);
      return formatObjectReadable(parsed);
    } catch {
      return value;
    }
  }
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return formatObjectReadable(value);
  }
  return JSON.stringify(value, null, 2);
}

/** Render objects so multi-line string values display with real newlines
 *  instead of escaped \n from JSON.stringify. */
function formatObjectReadable(obj: unknown): string {
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    return JSON.stringify(obj, null, 2);
  }
  const entries = Object.entries(obj as Record<string, unknown>);
  const parts: string[] = [];
  for (const [key, val] of entries) {
    if (typeof val === "string" && val.includes("\n")) {
      // Multi-line string: render as labeled block with real newlines
      parts.push(`${key}:\n${val}`);
    } else if (Array.isArray(val) || (typeof val === "object" && val !== null)) {
      parts.push(`${key}: ${JSON.stringify(val, null, 2)}`);
    } else {
      parts.push(`${key}: ${JSON.stringify(val)}`);
    }
  }
  return parts.join("\n\n");
}

// ---------------------------------------------------------------------------
// ExpandableEvent — one row in the timeline
// ---------------------------------------------------------------------------

function ExpandableEvent({
  event,
  sessionId,
  label,
}: {
  event: EventSummary;
  sessionId: string;
  label?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [fullData, setFullData] = useState<Record<string, unknown> | null>(
    null,
  );
  const [loading, setLoading] = useState(false);

  const isLLMCall = event.event === "llm.call";
  const isConversation = event.event === "llm.conversation";
  const data = fullData || event.data;

  async function loadFull() {
    if (fullData) {
      setExpanded(!expanded);
      return;
    }
    setLoading(true);
    try {
      const detail = await fetchEventDetail(sessionId, event.seq);
      setFullData(detail.data);
      setExpanded(true);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }

  // Summary line for the event
  const summaryParts: string[] = [];
  if (isLLMCall) {
    const d = event.data;
    if (d.model) summaryParts.push(String(d.model));
    if (typeof d.thinking === "string" && d.thinking) {
      const effortLabel = d.thinking_effort ? ` (${d.thinking_effort})` : "";
      summaryParts.push(`🧠 thinking${effortLabel}`);
    } else if (d.reasoning_tokens) {
      const effortLabel = d.thinking_effort ? ` (${d.thinking_effort})` : "";
      summaryParts.push(`🧠 reasoning${effortLabel} · ${d.reasoning_tokens} tokens`);
    }
    if (d.input_tokens)
      summaryParts.push(`${d.input_tokens}→${d.output_tokens} tokens`);
    if (d.stop_reason) summaryParts.push(`stop=${String(d.stop_reason)}`);
    if (Array.isArray(d.tool_calls) && d.tool_calls.length > 0)
      summaryParts.push(
        `tools: ${(d.tool_calls as ToolCall[]).map((tc) => tc.name).join(", ")}`,
      );
  } else if (isConversation) {
    const d = event.data;
    if (d.agent) summaryParts.push(String(d.agent));
    if (d.model) summaryParts.push(String(d.model));
    if (d.rounds) summaryParts.push(`${d.rounds} rounds`);
  } else if (event.event === "train.end") {
    const d = event.data;
    if (d.final_return != null)
      summaryParts.push(`return=${Number(d.final_return).toFixed(1)}`);
    if (d.training_time_s != null)
      summaryParts.push(`${Number(d.training_time_s).toFixed(0)}s`);
  } else if (
    event.event === "iteration.completed" ||
    event.event === "judge.end"
  ) {
    const d = event.data;
    if (d.intent_score != null)
      summaryParts.push(`score=${Number(d.intent_score).toFixed(3)}`);
  } else if (event.event === "revise.end") {
    const d = event.data;
    if (d.hp_changes && Object.keys(d.hp_changes as object).length > 0)
      summaryParts.push(`HP: ${JSON.stringify(d.hp_changes)}`);
  } else if (event.event === "guardrail.warning") {
    const d = event.data;
    if (d.type) summaryParts.push(String(d.type));
  }

  return (
    <div className="relative pl-6 pb-4 group">
      {/* Timeline dot */}
      <div
        className={`absolute left-0 top-1.5 w-2.5 h-2.5 rounded-full ${dotColor(event.event)} ring-2 ring-white`}
      />

      {/* Content */}
      <div className="flex items-baseline gap-2 text-sm">
        <span className="text-gray-400 font-mono text-xs shrink-0">
          {formatTime(event.timestamp)}
        </span>
        <span className="font-medium text-gray-800">{event.event}</span>
        {event.iteration != null && (
          <span className="text-xs text-gray-400">iter {event.iteration}</span>
        )}
        {label && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700 font-medium">
            {label}
          </span>
        )}
        {event.duration_ms != null && (
          <span className="text-xs text-amber-600 font-mono">
            {formatDuration(event.duration_ms)}
          </span>
        )}
        {summaryParts.length > 0 && (
          <span className="text-xs text-gray-500">
            {summaryParts.join(" · ")}
          </span>
        )}
        {(event.has_full_content || isConversation) && (
          <button
            onClick={loadFull}
            disabled={loading}
            className="text-xs text-blue-500 hover:text-blue-700 underline disabled:opacity-50"
          >
            {loading ? "Loading..." : expanded ? "Hide" : isConversation ? "Show conversation" : "Show full"}
          </button>
        )}
      </div>

      {/* Expanded content for LLM calls */}
      {expanded && isLLMCall ? (
        <div className="mt-2 space-y-2 text-xs">
          <CollapsibleSection title="System Prompt" color="bg-gray-50">
            <pre className="whitespace-pre-wrap text-gray-600 font-mono">
              {String(data.system_prompt || "(none)")}
            </pre>
          </CollapsibleSection>

          {/* Tool results from previous round (input to this call) */}
          {Array.isArray(data.tool_results_input) &&
            data.tool_results_input.length > 0 && (
              <CollapsibleSection
                title={`Tool Results Input (${(data.tool_results_input as ToolResult[]).length})`}
                color="bg-orange-50"
              >
                <div className="space-y-2">
                  {(data.tool_results_input as ToolResult[]).map((tr, i) => (
                    <div key={i} className="border border-orange-200 rounded p-2">
                      <span className="text-orange-700 font-semibold">
                        tool_use_id: {tr.tool_use_id}
                      </span>
                      <pre className="whitespace-pre-wrap text-gray-600 font-mono mt-1">
                        {formatToolContent(tr.content)}
                      </pre>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

          <CollapsibleSection title="User Prompt" color="bg-blue-50">
            <pre className="whitespace-pre-wrap text-gray-700 font-mono">
              {String(data.user_prompt || "(none)")}
            </pre>
          </CollapsibleSection>

          {typeof data.thinking === "string" && data.thinking && (
            <CollapsibleSection title="Thinking" color="bg-amber-50">
              <pre className="whitespace-pre-wrap text-amber-800 font-mono">
                {data.thinking}
              </pre>
            </CollapsibleSection>
          )}

          {!data.thinking && typeof data.reasoning_tokens === "number" && data.reasoning_tokens > 0 && (
            <CollapsibleSection title={`Reasoning (${data.reasoning_tokens} tokens, hidden by provider)`} color="bg-amber-50">
              <p className="text-sm text-amber-700 italic">
                This model performed internal reasoning ({data.reasoning_tokens} tokens) but the provider does not expose the reasoning text.
              </p>
            </CollapsibleSection>
          )}

          <CollapsibleSection
            title={`Response${data.stop_reason ? ` [${data.stop_reason}]` : ""}`}
            color="bg-green-50"
            defaultOpen
          >
            <pre className="whitespace-pre-wrap text-gray-700 font-mono">
              {String(data.response || "(none)")}
            </pre>
          </CollapsibleSection>

          {/* Tool calls in the response */}
          {Array.isArray(data.tool_calls) && data.tool_calls.length > 0 && (
            <CollapsibleSection
              title={`Tool Calls (${(data.tool_calls as ToolCall[]).length})`}
              color="bg-violet-50"
              defaultOpen
            >
              <div className="space-y-2">
                {(data.tool_calls as ToolCall[]).map((tc, i) => (
                  <div key={i} className="border border-violet-200 rounded p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="bg-violet-200 text-violet-800 px-1.5 py-0.5 rounded font-semibold">
                        {tc.name}
                      </span>
                      <span className="text-gray-400 font-mono text-[10px]">
                        {tc.id}
                      </span>
                    </div>
                    <pre className="whitespace-pre-wrap text-gray-600 font-mono">
                      {formatToolContent(tc.input)}
                    </pre>
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}
        </div>
      ) : null}

      {/* Expanded content for conversation events */}
      {expanded && isConversation ? (
        <ConversationView data={data} />
      ) : null}

      {/* Expanded content for non-LLM events with full content */}
      {expanded && !isLLMCall && !isConversation ? (
        <div className="mt-2">
          <pre className="text-xs font-mono text-gray-600 bg-gray-50 p-3 rounded-lg whitespace-pre-wrap overflow-x-auto max-h-96 overflow-y-auto">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      ) : null}

      {/* Inline data preview for non-LLM events (when not expanded) */}
      {!expanded && !isLLMCall && !isConversation && event.data && Object.keys(event.data).length > 0 ? (
        <div className="mt-0.5 text-xs text-gray-400 font-mono truncate max-w-2xl">
          {Object.entries(event.data)
            .filter(([k]) => !["error"].includes(k))
            .map(([k, v]) => `${k}=${truncate(v, 60)}`)
            .join("  ")}
        </div>
      ) : null}

      {/* Error highlight */}
      {event.data.error ? (
        <div className="mt-1 text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
          {truncate(event.data.error, 300)}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Collapsible section
// ---------------------------------------------------------------------------

function CollapsibleSection({
  title,
  color,
  defaultOpen = false,
  children,
}: {
  title: string;
  color: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`rounded-lg ${color} border border-gray-200`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left px-3 py-1.5 font-medium text-gray-700 flex items-center gap-1"
      >
        <span className="text-gray-400">{open ? "▼" : "▶"}</span>
        {title}
      </button>
      {open && <div className="px-3 pb-2 max-h-96 overflow-y-auto">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ConversationView — renders a full multi-turn conversation
// ---------------------------------------------------------------------------

interface ConversationMessage {
  role: string;
  content: unknown;
  stop_reason?: string;
}

const ROLE_STYLES: Record<string, { bg: string; label: string; border: string }> = {
  system: { bg: "bg-gray-50", label: "System", border: "border-gray-200" },
  user: { bg: "bg-blue-50", label: "User", border: "border-blue-200" },
  assistant: { bg: "bg-green-50", label: "Assistant", border: "border-green-200" },
  tool_results: { bg: "bg-orange-50", label: "Tool Results", border: "border-orange-200" },
};

function ConversationView({ data }: { data: Record<string, unknown> }) {
  const raw = data.conversation;
  const messages = (Array.isArray(raw) ? raw : []) as ConversationMessage[];
  if (messages.length === 0) return null;

  return (
    <div className="mt-2 space-y-2 text-xs">
      {messages.map((msg, i) => {
        const style = ROLE_STYLES[msg.role] ?? ROLE_STYLES.system;
        return (
          <CollapsibleSection
            key={i}
            title={`${style.label}${msg.stop_reason ? ` [${msg.stop_reason}]` : ""}`}
            color={style.bg}
            defaultOpen={i === messages.length - 1}
          >
            <ConversationMessageContent message={msg} />
          </CollapsibleSection>
        );
      })}
    </div>
  );
}

function ConversationMessageContent({ message }: { message: ConversationMessage }) {
  const { role, content } = message;

  // String content (system prompt, user prompt)
  if (typeof content === "string") {
    return (
      <pre className="whitespace-pre-wrap text-gray-700 font-mono">
        {content}
      </pre>
    );
  }

  // Array content — assistant blocks or tool_results
  if (Array.isArray(content)) {
    if (role === "tool_results") {
      return (
        <div className="space-y-2">
          {(content as ToolResult[]).map((tr, i) => (
            <div key={i} className="border border-orange-200 rounded p-2">
              <span className="text-orange-700 font-semibold text-[10px]">
                {tr.tool_use_id}
              </span>
              <pre className="whitespace-pre-wrap text-gray-600 font-mono mt-1">
                {formatToolContent(tr.content)}
              </pre>
            </div>
          ))}
        </div>
      );
    }

    // Assistant content blocks
    return (
      <div className="space-y-2">
        {(content as Array<Record<string, unknown>>).map((block, i) => {
          if (block.type === "text") {
            return (
              <pre key={i} className="whitespace-pre-wrap text-gray-700 font-mono">
                {String(block.text)}
              </pre>
            );
          }
          if (block.type === "tool_use") {
            return (
              <div key={i} className="border border-violet-200 rounded p-2">
                <div className="flex items-center gap-2 mb-1">
                  <span className="bg-violet-200 text-violet-800 px-1.5 py-0.5 rounded font-semibold">
                    {String(block.name)}
                  </span>
                  <span className="text-gray-400 font-mono text-[10px]">
                    {String(block.id)}
                  </span>
                </div>
                <pre className="whitespace-pre-wrap text-gray-600 font-mono">
                  {formatToolContent(block.input)}
                </pre>
              </div>
            );
          }
          return (
            <pre key={i} className="whitespace-pre-wrap text-gray-600 font-mono">
              {JSON.stringify(block, null, 2)}
            </pre>
          );
        })}
      </div>
    );
  }

  // Fallback
  return (
    <pre className="whitespace-pre-wrap text-gray-600 font-mono">
      {JSON.stringify(content, null, 2)}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function EventTimeline({
  sessionId,
  isRunning,
}: {
  sessionId: string;
  isRunning: boolean;
}) {
  const { data: events } = useSWR<EventSummary[]>(
    `events-${sessionId}`,
    () => fetchEvents(sessionId),
    { refreshInterval: isRunning ? 5000 : 0 },
  );
  const [filterIteration, setFilterIteration] = useState<number | "all">("all");
  const [hideStreamingJudge, setHideStreamingJudge] = useState(true);

  if (!events || events.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <h2 className="text-sm font-semibold text-gray-900 mb-2">
          Event Timeline
        </h2>
        <p className="text-sm text-gray-400">No events recorded yet.</p>
      </div>
    );
  }

  // Unique iterations for filter
  const iterations = Array.from(
    new Set(events.filter((e) => e.iteration != null).map((e) => e.iteration!)),
  ).sort((a, b) => a - b);

  // Pre-compute: hide individual llm.call events that are part of llm.conversation
  // (the conversation event already shows the full multi-turn flow)
  const conversationCallSeqs = new Set<number>();
  for (let i = 0; i < events.length; i++) {
    if (events[i].event === "llm.conversation") {
      const rounds = Number(events[i].data.rounds) || 0;
      let found = 0;
      for (let j = i - 1; j >= 0 && found < rounds; j--) {
        if (events[j].event === "llm.call") {
          conversationCallSeqs.add(events[j].seq);
          found++;
        }
      }
    }
  }

  // Label llm.call events by their enclosing span
  const eventLabels = new Map<number, string>();

  // 1) Label generation-phase calls (judge_code.generate, reward.generate).
  //    These spans can overlap (parallel execution). We use a backward-scan
  //    from each span .end to find the nearest preceding llm.call — this is
  //    correct because .end fires in the same thread immediately after the
  //    llm.call returns, so they are always adjacent regardless of interleaving.
  const GEN_SPAN_LABELS: Record<string, string> = {
    "judge_code.generate.end": "Judge Code",
    "reward.generate.end": "Reward Gen",
  };
  const GEN_SPAN_STARTS = new Set(["judge_code.generate.start", "reward.generate.start"]);
  for (let i = 0; i < events.length; i++) {
    const label = GEN_SPAN_LABELS[events[i].event];
    if (!label) continue;
    for (let j = i - 1; j >= 0; j--) {
      if (GEN_SPAN_STARTS.has(events[j].event)) break;
      if (events[j].event === "llm.call" && !conversationCallSeqs.has(events[j].seq) && !eventLabels.has(events[j].seq)) {
        eventLabels.set(events[j].seq, label);
        break;
      }
    }
  }

  // 2) Label revise-phase events: Phase 1, Phase 2, HP Variant
  let inRevise = false;
  let afterConversation = false;
  let phase2Done = false;
  let hpCount = 0;
  for (const e of events) {
    if (e.event === "revise.start") {
      inRevise = true;
      afterConversation = false;
      phase2Done = false;
      hpCount = 0;
    } else if (e.event === "revise.end") {
      inRevise = false;
    } else if (inRevise) {
      if (e.event === "llm.conversation") {
        eventLabels.set(e.seq, "Phase 1");
        afterConversation = true;
      } else if (e.event === "llm.call" && afterConversation && !conversationCallSeqs.has(e.seq)) {
        if (!phase2Done) {
          eventLabels.set(e.seq, "Phase 2");
          phase2Done = true;
        } else {
          hpCount++;
          eventLabels.set(e.seq, `HP Variant ${hpCount}`);
        }
      }
    }
  }

  const filtered = events.filter((e) => {
    if (conversationCallSeqs.has(e.seq)) return false;
    if (hideStreamingJudge) {
      if (e.event === "vlm.call") return false;
      // Synthesis agent events (tagged with agent="synthesis")
      if (e.data?.agent === "synthesis") return false;
      // Streaming judge LLM calls without agent tag (old sessions)
      if (e.event === "llm.call" && (Number(e.data?.input_tokens) || 0) < 2500) return false;
    }
    if (filterIteration !== "all" && e.iteration !== filterIteration && e.iteration !== null) return false;
    return true;
  });

  // Stats — single pass to classify events
  const llmCalls: typeof events = [];
  const vlmCalls: typeof events = [];
  let totalLLMTime = 0;
  for (const e of events) {
    if (e.event === "llm.call") {
      llmCalls.push(e);
      totalLLMTime += e.duration_ms || 0;
    } else if (e.event === "vlm.call") {
      vlmCalls.push(e);
    }
  }
  const llmStats = computeCost(llmCalls);
  const vlmStats = computeCost(vlmCalls);

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      {/* Header with stats */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-900">Event Timeline</h2>
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span>{events.length} events</span>
          <span>
            {llmCalls.length} LLM calls · {llmStats.inTot.toLocaleString()} in / {llmStats.outTot.toLocaleString()} out
            · {formatCost(llmStats.cost)}
            {vlmCalls.length > 0 && (
              <> + VLM {formatCost(vlmStats.cost)}</>
            )}
            {" "}· {formatDuration(totalLLMTime)}
          </span>
          <label className="flex items-center gap-1 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={hideStreamingJudge}
              onChange={(e) => setHideStreamingJudge(e.target.checked)}
              className="rounded border-gray-300 text-teal-600 focus:ring-teal-500"
            />
            Hide Streaming Judge
          </label>
          {iterations.length > 1 && (
            <select
              value={filterIteration}
              onChange={(e) =>
                setFilterIteration(
                  e.target.value === "all" ? "all" : Number(e.target.value),
                )
              }
              className="border border-gray-200 rounded px-2 py-0.5 text-xs"
            >
              <option value="all">All iterations</option>
              {iterations.map((i) => (
                <option key={i} value={i}>
                  Iteration {i}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="border-l-2 border-gray-200 ml-1">
        {filtered.map((event, idx) => (
          <ExpandableEvent
            key={`${event.seq}-${idx}`}
            event={event}
            sessionId={sessionId}
            label={eventLabels.get(event.seq)}
          />
        ))}
      </div>
    </div>
  );
}
