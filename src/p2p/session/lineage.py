"""Experiment lineage: cross-session tree of iterations with lessons.

Persisted as ``lineage.json`` in the runs directory root.  Shared across
all sessions so the revise agent can see the full experiment graph.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from p2p.contracts import StructuredLesson
from p2p.session.iteration_record import _atomic_write, read_json_safe
from p2p.settings import LLM_MODEL

if TYPE_CHECKING:
    import anthropic

    from p2p.contracts import Lineage, LineageEntry

logger = logging.getLogger(__name__)

LINEAGE_FILE = "lineage.json"
MAX_GLOBAL_LESSONS = 25
TIER_ORDER: dict[str, int] = {"HARD": 0, "STRONG": 1, "SOFT": 2, "RETIRED": 3}


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_lineage(session_dir: str | Path) -> Lineage:
    """Load lineage.json from a session directory, creating empty if absent."""
    path = Path(session_dir) / LINEAGE_FILE
    data = read_json_safe(path)
    if data and "iterations" in data:
        return data  # type: ignore[return-value]
    return {"iterations": {}, "lessons": []}


def save_lineage(session_dir: str | Path, lineage: Lineage) -> None:
    """Atomically persist lineage.json into a session directory."""
    path = Path(session_dir) / LINEAGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(lineage, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------


def iteration_key(session_id: str, iteration: int) -> str:
    """Canonical key for an iteration node: ``session_id/iter_N``."""
    return f"{session_id}/iter_{iteration}"


def config_key(session_id: str, iteration: int, config_id: str) -> str:
    """Canonical key for a config node: ``session_id/iter_N/config_id``."""
    return f"{session_id}/iter_{iteration}/{config_id}"


def add_iteration(
    lineage: Lineage,
    *,
    session_id: str,
    iteration: int,
    parent_key: str | None,
    lesson: str = "",
    score: float = 0.0,
    star: bool = False,
    also_from: str | None = None,
    diagnosis: str = "",
    failure_tags: list[str] | None = None,
    final_return: float | None = None,
    best_checkpoint: str = "",
) -> str:
    """Add an iteration node to the lineage tree.  Returns the new key."""
    key = iteration_key(session_id, iteration)
    entry: LineageEntry = {"parent": parent_key}  # type: ignore[typeddict-item]
    if lesson:
        entry["lesson"] = lesson
    if score is not None:
        entry["score"] = score
    if star:
        entry["star"] = True
    if also_from:
        entry["also_from"] = also_from
    if diagnosis:
        entry["diagnosis"] = diagnosis[:300]
    if failure_tags:
        entry["failure_tags"] = failure_tags
    if final_return is not None:
        entry["final_return"] = final_return
    if best_checkpoint:
        entry["best_checkpoint"] = best_checkpoint
    lineage["iterations"][key] = entry
    return key


def is_new_best(lineage: Lineage, score: float) -> bool:
    """Return True if *score* exceeds all existing scores in the lineage."""
    for entry in lineage["iterations"].values():
        if entry.get("score", 0) >= score:
            return False
    return True


def _clear_stars(lineage: Lineage) -> None:
    """Remove the star flag from all iterations."""
    for entry in lineage["iterations"].values():
        entry.pop("star", None)  # type: ignore[typeddict-item]


def _find_starred_iter_num(lineage: Lineage) -> int:
    """Return the iteration number of the starred (best) node, or 0 if none."""
    for key, entry in lineage["iterations"].items():
        if entry.get("star"):
            return _iter_num(key)
    return 0


def find_best_config_key(lineage: Lineage, session_id: str, iteration: int) -> str | None:
    """Find the starred (best) config key for an iteration.

    Searches for ``session_id/iter_N/*`` keys with ``is_best=True`` or
    ``star=True``.  Falls back to the highest-scoring config, then to
    the legacy ``session_id/iter_N`` key.
    """
    prefix = f"{session_id}/iter_{iteration}/"
    config_keys: list[tuple[str, float]] = []
    for key, entry in lineage["iterations"].items():
        if key.startswith(prefix):
            config_keys.append((key, entry.get("score", 0)))
            if entry.get("is_best"):
                return key

    if config_keys:
        # Fall back to highest-scoring config
        config_keys.sort(key=lambda kv: kv[1], reverse=True)
        return config_keys[0][0]

    # Legacy: no config-level keys, try iteration-level key
    legacy = iteration_key(session_id, iteration)
    if legacy in lineage["iterations"]:
        return legacy
    return None


def _auto_lesson_for_config(
    config_id: str,
    config_label: str,
    score: float,
    best_score: float,
    failure_tags: list[str],
    is_best: bool,
) -> str:
    """Generate a short deterministic lesson for a config node."""
    delta = score - best_score
    if is_best:
        outcome = "best config"
    elif abs(delta) < 0.02:
        outcome = "near-best"
    else:
        outcome = f"score delta {delta:+.2f} vs best"

    tags = ", ".join(failure_tags) if failure_tags else "none"
    return f"{config_id} ({config_label}): {score:.2f} ({outcome}). Failures: {tags}."


# ---------------------------------------------------------------------------
# Tree formatting (for prompt injection)
# ---------------------------------------------------------------------------


def _short_key(key: str) -> str:
    """Abbreviate lineage keys for display.

    Strips the session ID prefix (redundant in per-session lineage).

    ``session_.../iter_3``          → ``v3``
    ``session_.../iter_3/baseline`` → ``v3/baseline``
    """
    parts = key.split("/")
    if len(parts) >= 2:
        iter_part = parts[1].replace("iter_", "v")
        if len(parts) == 3:
            return f"{iter_part}/{parts[2]}"
        return iter_part
    return key


def _is_config_node(key: str) -> bool:
    """Return True if *key* is a config-level node (3 path segments)."""
    return len(key.split("/")) == 3


def _iter_prefix(key: str) -> str:
    """Extract the iteration prefix from a key (strips config_id if present).

    ``session/iter_3/baseline`` → ``session/iter_3``
    ``session/iter_3``          → ``session/iter_3``
    """
    parts = key.split("/")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}"
    return key


def _hp_str(hp_params: dict | None) -> str:
    """Format HP params dict for display: ``{gamma: 0.995, ent_coef: 0.005}``."""
    if not hp_params:
        return "{default}"
    parts = [f"{k}: {v}" for k, v in hp_params.items()]
    return "{" + ", ".join(parts) + "}"


def _iter_num(prefix: str) -> int:
    """Extract the integer iteration number from a key prefix."""
    for part in prefix.split("/"):
        if part.startswith("iter_"):
            try:
                return int(part[5:])
            except ValueError:
                pass
    return 0


def format_lineage_tree(lineage: Lineage, *, recent_window: int = 10) -> str:
    """Render the lineage as a flat structured node list for LLM consumption.

    Uses explicit ``parent:`` fields instead of indentation-based ASCII trees.
    Research (Google "Talk like a Graph", ICLR 2024) shows that explicit
    per-node neighbor listing outperforms visual tree formats by 5-60% for
    LLM graph comprehension.  Nodes are ordered most-recent-first to exploit
    recency attention bias.

    All nodes are always shown (no truncation).  Recent iterations (within
    *recent_window*) get full detail: HP, tags, diagnosis, and siblings.
    Older iterations are compressed to a single line: id, parent, score,
    and a short lesson — keeping full history at minimal token cost.

    Example output::

        Best: v3/config_0 (score: 0.72)

        ## Recent (detailed)
        - id: v3/config_0, parent: v2/config_0, score: 0.72*, HP: {gamma: 0.995}
          tags: [none], lesson: "Higher gamma + height gate unlocked stable flips"
          diagnosis: "..."
          siblings: [v3/baseline: 0.65 {default}, v3/config_1: 0.58 {gamma: 0.95}]

        ## Older (compressed)
        - v1/baseline, parent: null, score: 0.20, lesson: "Initial reward"
    """
    if not lineage["iterations"]:
        return "(no lineage data yet)"

    iters = lineage["iterations"]

    # Find overall best (starred) node
    best_key = ""
    best_score = -1.0
    for key, entry in iters.items():
        if entry.get("star"):
            best_key = key
            best_score = entry.get("score", 0)
    if not best_key:
        for key, entry in iters.items():
            s = entry.get("score", 0)
            if s > best_score:
                best_score = s
                best_key = key

    # Group by iteration prefix
    groups: dict[str, list[str]] = {}
    for key in iters:
        prefix = _iter_prefix(key)
        groups.setdefault(prefix, []).append(key)

    # Sort groups by iteration number, most recent first
    sorted_prefixes = sorted(groups.keys(), key=_iter_num, reverse=True)

    lines: list[str] = [f"Best: {_short_key(best_key)} (score: {best_score:.2f})", ""]

    recent_lines: list[str] = []
    older_lines: list[str] = []

    for group_idx, prefix in enumerate(sorted_prefixes):
        keys = groups[prefix]
        is_recent = group_idx < recent_window

        if is_recent:
            recent_lines.extend(_format_group_full(keys, iters))
        else:
            older_lines.extend(_format_group_compact(keys, iters))

    if recent_lines:
        lines.append("## Recent (detailed)")
        lines.extend(recent_lines)

    if older_lines:
        if recent_lines:
            lines.append("")
        lines.append("## Older (compressed)")
        lines.extend(older_lines)

    return "\n".join(lines)


def _format_group_full(keys: list[str], iters: dict) -> list[str]:
    """Format an iteration group with full detail (recent iterations)."""
    lines: list[str] = []

    if len(keys) > 1 or _is_config_node(keys[0]):
        # Multi-config group: find best config
        best_cfg_key = max(keys, key=lambda k: iters[k].get("score", 0))
        for k in keys:
            if iters[k].get("is_best"):
                best_cfg_key = k
                break

        entry = iters[best_cfg_key]
        score = entry.get("score", 0)
        star = "*" if entry.get("star") or entry.get("is_best") else ""
        parent = entry.get("parent")
        parent_str = _short_key(parent) if parent else "null"
        hp = _hp_str(entry.get("hp_params"))
        tags = entry.get("failure_tags", [])
        tag_str = ", ".join(tags) if tags else "none"
        lesson = entry.get("lesson", "")
        lesson_str = f'"{lesson}"' if lesson else "(no lesson)"
        diagnosis = entry.get("diagnosis", "")

        lines.append(
            f"- id: {_short_key(best_cfg_key)}, parent: {parent_str}, "
            f"score: {score:.2f}{star}, HP: {hp}"
        )
        lines.append(f"  tags: [{tag_str}], lesson: {lesson_str}")
        if diagnosis:
            lines.append(f'  diagnosis: "{diagnosis[:120]}"')

        # Siblings
        siblings = [k for k in keys if k != best_cfg_key]
        if siblings:
            sib_parts = []
            for sk in siblings:
                se = iters[sk]
                s_score = se.get("score", 0)
                s_hp = _hp_str(se.get("hp_params"))
                s_tags = se.get("failure_tags", [])
                tag_part = f" tags:[{','.join(s_tags)}]" if s_tags else ""
                sib_parts.append(f"{_short_key(sk)}: {s_score:.2f} {s_hp}{tag_part}")
            lines.append(f"  siblings: [{', '.join(sib_parts)}]")
    else:
        # Single-config format
        key = keys[0]
        entry = iters[key]
        score = entry.get("score", 0)
        star = "*" if entry.get("star") else ""
        parent = entry.get("parent")
        parent_str = _short_key(parent) if parent else "null"
        tags = entry.get("failure_tags", [])
        tag_str = ", ".join(tags) if tags else "none"
        lesson = entry.get("lesson", "")
        lesson_str = f'"{lesson}"' if lesson else "(no lesson)"
        diagnosis = entry.get("diagnosis", "")
        final_return = entry.get("final_return")
        ret_str = f", return: {final_return:.1f}" if final_return is not None else ""

        lines.append(
            f"- id: {_short_key(key)}, parent: {parent_str}, score: {score:.2f}{star}{ret_str}"
        )
        lines.append(f"  tags: [{tag_str}], lesson: {lesson_str}")
        if diagnosis:
            lines.append(f'  diagnosis: "{diagnosis[:120]}"')

    return lines


def _format_group_compact(keys: list[str], iters: dict) -> list[str]:
    """Format an iteration group as a single compressed line (older iterations).

    Shows only the best config: id, parent, score, and a short lesson.
    No HP, tags, diagnosis, or siblings — those can be fetched on demand
    via ``get_iteration_reward_code`` or ``get_iteration_judgment_detail``.
    """
    if len(keys) > 1 or _is_config_node(keys[0]):
        best_cfg_key = max(keys, key=lambda k: iters[k].get("score", 0))
        for k in keys:
            if iters[k].get("is_best"):
                best_cfg_key = k
                break
        entry = iters[best_cfg_key]
    else:
        best_cfg_key = keys[0]
        entry = iters[best_cfg_key]

    score = entry.get("score", 0)
    star = "*" if entry.get("star") or entry.get("is_best") else ""
    parent = entry.get("parent")
    parent_str = _short_key(parent) if parent else "null"
    lesson = entry.get("lesson", "")
    lesson_short = f'"{lesson}"' if lesson else ""

    parts = [f"- {_short_key(best_cfg_key)}, parent: {parent_str}, score: {score:.2f}{star}"]
    if lesson_short:
        parts[0] += f", lesson: {lesson_short}"
    return parts


def format_lessons(lineage: Lineage) -> str:
    """Format global lessons as a tiered numbered list.

    Order: HARD first, then STRONG, then SOFT, then RETIRED.
    Active lessons (HARD/STRONG/SOFT) get ``[TIER]`` prefix and ``(iter N)``
    suffix.  RETIRED lessons are shown under a separator for reference.
    """
    if not lineage["lessons"]:
        return "(no lessons accumulated yet)"

    sorted_lessons = sorted(
        lineage["lessons"],
        key=lambda les: TIER_ORDER.get(les.get("tier", "STRONG"), 1),
    )

    active = [les for les in sorted_lessons if les.get("tier", "STRONG") != "RETIRED"]
    retired = [les for les in sorted_lessons if les.get("tier", "STRONG") == "RETIRED"]

    lines: list[str] = [
        "=== LESSONS (must respect HARD, should respect STRONG, may challenge SOFT) ==="
    ]
    idx = 1
    for lesson in active:
        tier = lesson.get("tier", "STRONG")
        text = lesson.get("text", str(lesson))
        learned_at = lesson.get("learned_at", 0)
        iter_suffix = f"  (iter {learned_at})" if learned_at else ""
        lines.append(f"[{tier}] {idx}. {text}{iter_suffix}")
        idx += 1

    if retired:
        lines.append("--- RETIRED (for reference only) ---")
        for lesson in retired:
            text = lesson.get("text", str(lesson))
            learned_at = lesson.get("learned_at", 0)
            iter_suffix = f"  (iter {learned_at})" if learned_at else ""
            reason = lesson.get("tier_reason", "")
            reason_suffix = f"  [reason: {reason}]" if reason else ""
            lines.append(f"{idx}. {text}{iter_suffix}{reason_suffix}")
            idx += 1

    return "\n".join(lines)


def lesson_tier_counts(lineage: Lineage) -> dict[str, int]:
    """Return counts per tier: {hard: N, strong: N, soft: N, retired: N}."""
    counts = {"hard": 0, "strong": 0, "soft": 0, "retired": 0}
    for lesson in lineage.get("lessons", []):
        tier = lesson.get("tier", "STRONG").lower()
        if tier in counts:
            counts[tier] += 1
    return counts


# ---------------------------------------------------------------------------
# Lineage context for the revise agent
# ---------------------------------------------------------------------------


def format_lineage_context(lineage: Lineage) -> str:
    """Build the full lineage context block for the revise agent prompt."""
    if not lineage["iterations"]:
        return ""

    tree = format_lineage_tree(lineage)
    lessons = format_lessons(lineage)

    return (
        "## Experiment Lineage\n"
        "Flat node list with explicit parent pointers.  Each node is one "
        "experiment with its score, HP config, and lesson learned.\n"
        "DO NOT re-try approaches that already failed.\n"
        "BUILD ON approaches that showed improvement.\n\n"
        "### Nodes\n"
        f"```\n{tree}\n```\n\n"
        "### Accumulated Lessons\n"
        f"{lessons}\n"
    )


# ---------------------------------------------------------------------------
# Lesson generation (LLM)
# ---------------------------------------------------------------------------

_LESSON_PROMPT = """\
You are analyzing the result of an RL experiment iteration.

**What was tried:** {reasoning}
**Score:** {score:.3f} (previous: {prev_score:.3f}, delta: {delta:+.3f})
**Judgment diagnosis:** {diagnosis}
**Failure tags:** {failure_tags}

Write a concise lesson (2-4 sentences) in this format:
<title of finding> — <1 sentence summary of what happened>.
<Why it happened or root cause>. <Actionable takeaway or rule for future>.

Example: "Height-gating rotation prevents ground rolling — adding \
height > 0.3 gate before rotation reward eliminated ground-roll exploit. \
Without the gate the agent finds it easier to spin on the ground than jump. \
Always gate airborne rewards on minimum height."

Reply with ONLY the lesson text, no prefix label or markdown."""

_CONSOLIDATE_PROMPT = """\
Below are the current accumulated lessons from an automated RL experiment campaign.
Each lesson has a tier tag: [HARD], [STRONG], or [SOFT] and an origin tag (iter N).
A new lesson has been learned. Consolidate the list:
- Add the new lesson (as [STRONG] unless merging with a higher-tier lesson)
- Merge duplicates or near-duplicates — merged lessons take the HIGHER tier
- Remove lessons that are superseded by newer, more specific ones
- Keep the list to at most {max_lessons} entries
- Order by importance (most impactful first)
- PRESERVE the [TIER] prefix on every line — do NOT remove or change tiers
  unless merging (merged = higher tier of the two)
- PRESERVE the (iter N) suffix on every line — when merging two lessons,
  keep the EARLIER (lower) iteration number
- Do NOT include RETIRED lessons (they are excluded from this list)

Current lessons:
{current_lessons}

New lesson:
[STRONG] {new_lesson}

Reply with ONLY the consolidated list, one lesson per line, each prefixed with
[HARD], [STRONG], or [SOFT] and suffixed with (iter N). No numbering or bullets."""


def generate_lesson(
    *,
    reasoning: str,
    score: float,
    prev_score: float,
    diagnosis: str,
    failure_tags: list[str],
    client: anthropic.Anthropic,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> str:
    """Ask the LLM to generate a lesson from an iteration's outcome.

    Used by the backfill script for sessions that lack lessons.
    """
    from p2p.inference.llm_client import create_message, extract_response_text

    prompt = _LESSON_PROMPT.format(
        reasoning=reasoning,
        score=score,
        prev_score=prev_score,
        delta=score - prev_score,
        diagnosis=diagnosis,
        failure_tags=", ".join(failure_tags) if failure_tags else "none",
    )

    try:
        resp = create_message(
            client,
            model=model,
            system="You distill RL experiment results into concise, actionable lessons.",
            messages=[{"role": "user", "content": prompt}],
            thinking_effort=thinking_effort,
        )
        return extract_response_text(resp).strip()
    except Exception:
        logger.exception("Failed to generate lesson")
        delta = score - prev_score
        direction = "improved" if delta > 0.02 else "regressed" if delta < -0.02 else "no change"
        return f"{reasoning[:100]} -> {direction} (score {prev_score:.2f} -> {score:.2f})"


def _parse_tier_prefix(line: str) -> tuple[str, str]:
    """Parse ``[TIER] lesson text`` -> (tier, text).  Falls back to STRONG."""
    m = re.match(r"^\[(HARD|STRONG|SOFT|RETIRED)]\s*", line)
    if m:
        return m.group(1), line[m.end() :].strip()
    return "STRONG", line.strip()


def _parse_iter_suffix(text: str) -> tuple[str, int | None]:
    """Extract trailing ``(iter N)`` from lesson text.

    Returns (text_without_suffix, iteration_number_or_None).
    """
    m = re.search(r"\s*\(iter\s+(\d+)\)\s*$", text)
    if m:
        return text[: m.start()].strip(), int(m.group(1))
    return text.strip(), None


def consolidate_lessons(
    lineage: Lineage,
    new_lesson: StructuredLesson,
    *,
    client: anthropic.Anthropic,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> list[StructuredLesson]:
    """Consolidate global lessons list with a new lesson via LLM.

    Tier-aware: passes ``[TIER] text`` to the LLM, parses tiers back out.
    RETIRED lessons are excluded from LLM consolidation and preserved as-is.
    """
    from p2p.inference.llm_client import create_message, extract_response_text

    current = lineage["lessons"]

    # Separate retired from active
    active = [les for les in current if les.get("tier", "STRONG") != "RETIRED"]
    retired = [les for les in current if les.get("tier", "STRONG") == "RETIRED"]

    # If few active lessons, just append without LLM call
    if len(active) < MAX_GLOBAL_LESSONS // 2:
        return active + [new_lesson] + retired

    # Build tier-annotated text for LLM
    current_text = (
        "\n".join(
            f"- [{les.get('tier', 'STRONG')}] {les['text']}"
            f"{f' (iter {la})' if (la := les.get('learned_at', 0)) else ''}"
            for les in active
        )
        if active
        else "(none)"
    )
    new_text = new_lesson["text"]
    new_learned_at = new_lesson.get("learned_at", 0)
    new_iter_tag = f" (iter {new_learned_at})" if new_learned_at else ""
    prompt = _CONSOLIDATE_PROMPT.format(
        max_lessons=MAX_GLOBAL_LESSONS,
        current_lessons=current_text,
        new_lesson=f"{new_text}{new_iter_tag}",
    )

    # Compute fallback learned_at: earliest iteration across all inputs
    all_learned_ats = [la for les in active if (la := les.get("learned_at", 0)) > 0]
    if new_learned_at > 0:
        all_learned_ats.append(new_learned_at)
    min_learned_at = min(all_learned_ats) if all_learned_ats else 0

    try:
        resp = create_message(
            client,
            model=model,
            system="You consolidate experiment lessons into a concise, non-redundant list.",
            messages=[{"role": "user", "content": prompt}],
            thinking_effort=thinking_effort,
        )
        text = extract_response_text(resp).strip()

        # Build a text -> StructuredLesson lookup from input for learned_at inheritance
        input_lookup: dict[str, StructuredLesson] = {}
        for les in active:
            input_lookup[les["text"].strip().lower()] = les
        input_lookup[new_text.strip().lower()] = new_lesson

        consolidated: list[StructuredLesson] = []
        for raw_line in text.splitlines():
            cleaned = raw_line.strip().lstrip("•-0123456789. ")
            if not cleaned:
                continue
            tier, lesson_text = _parse_tier_prefix(cleaned)
            # Extract (iter N) suffix the LLM may have preserved
            lesson_text, iter_from_llm = _parse_iter_suffix(lesson_text)
            # Priority: LLM-preserved iter > exact text match > min of all inputs
            if iter_from_llm is not None:
                learned_at = iter_from_llm
            else:
                match = input_lookup.get(lesson_text.strip().lower())
                learned_at = match.get("learned_at", 0) if match else min_learned_at
            consolidated.append(
                {"text": lesson_text, "tier": tier, "learned_at": learned_at}  # type: ignore[typeddict-item]
            )

        return consolidated[:MAX_GLOBAL_LESSONS] + retired
    except Exception:
        logger.exception("Failed to consolidate lessons")
        return (active + [new_lesson])[:MAX_GLOBAL_LESSONS] + retired


# ---------------------------------------------------------------------------
# High-level: record an iteration into the lineage
# ---------------------------------------------------------------------------


def record_multi_config_iteration(
    session_dir: str | Path,
    *,
    session_id: str,
    iteration: int,
    based_on: int = 0,
    lesson: str,
    config_judgments: dict[str, dict],
    configs: list[dict],
    diagnosis: str,
    client: anthropic.Anthropic,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> None:
    """Record a multi-config iteration as config-level nodes in the lineage.

    Each config becomes a first-class node with its own score, lesson,
    diagnosis, failure tags, and HP params.  The *lesson* (from the revise
    agent) is attached to the best config; other configs get auto-generated
    lessons.  Only the best config's lesson enters the global list.

    The best config is derived from *config_judgments* by highest
    ``mean_intent_score``.
    """
    from p2p.contracts import RunConfigEntry

    lineage = load_lineage(session_dir)

    # Resolve parent: best config from the based_on iteration.
    # When based_on is missing (0), fall back to the starred (best) iteration
    # rather than iteration-1, so fallback/truncated revise outputs still
    # branch from the best known result.
    parent_key: str | None = None
    if iteration > 1:
        if based_on > 0 and based_on != iteration:
            ref_iter = based_on
        else:
            ref_iter = _find_starred_iter_num(lineage) or iteration - 1
        parent_key = find_best_config_key(lineage, session_id, ref_iter)
        # Ultimate fallback: plain iteration key without config suffix
        if parent_key is None:
            parent_key = iteration_key(session_id, ref_iter)
            if parent_key not in lineage["iterations"]:
                parent_key = None

    # Build a lookup: config_id -> RunConfigEntry
    cfg_lookup: dict[str, RunConfigEntry] = {}
    for cfg in configs:
        cfg_lookup[cfg["config_id"]] = cfg  # type: ignore[assignment]

    # Find best config and best score in a single pass
    best_config_id = ""
    best_score = 0.0
    for cid, cj in config_judgments.items():
        s = cj.get("mean_intent_score", 0)
        if s > best_score or not best_config_id:
            best_score = s
            best_config_id = cid

    # Add each config as a node
    for cid, cj in sorted(config_judgments.items()):
        is_best = cid == best_config_id
        cfg_entry = cfg_lookup.get(cid)
        cfg_label = cfg_entry.get("label", cid) if cfg_entry else cid
        hp_params = cfg_entry.get("params", {}) if cfg_entry else {}
        cfg_score = cj.get("mean_intent_score", 0.0)
        cfg_score_std = cj.get("score_std", 0.0)
        cfg_return = cj.get("mean_final_return")
        cfg_return_std = cj.get("return_std", 0.0)
        cfg_tags = cj.get("common_failure_tags", [])

        # Star: check if this config beat ALL existing nodes
        star = is_new_best(lineage, cfg_score)
        if star:
            _clear_stars(lineage)

        # Lesson: revise agent lesson for best, auto for others
        if is_best:
            cfg_lesson = lesson
        else:
            cfg_lesson = _auto_lesson_for_config(
                cid, cfg_label, cfg_score, best_score, cfg_tags, is_best
            )

        # Per-config diagnosis from the judgment
        cfg_diagnosis = ""
        per_seed = cj.get("per_seed", [])
        if per_seed:
            # Use the best seed's diagnosis
            best_seed_entry = max(per_seed, key=lambda s: s.get("intent_score", 0))
            cfg_diagnosis = best_seed_entry.get("diagnosis", "")
        if not cfg_diagnosis:
            cfg_diagnosis = diagnosis[:200] if is_best else ""

        key = config_key(session_id, iteration, cid)
        entry: LineageEntry = {  # type: ignore[typeddict-item]
            "parent": parent_key,
            "config_id": cid,
            "config_label": cfg_label,
            "hp_params": hp_params,
            "score": cfg_score,
            "is_best": is_best,
        }
        if cfg_lesson:
            entry["lesson"] = cfg_lesson
        if star:
            entry["star"] = True
        if cfg_diagnosis:
            entry["diagnosis"] = cfg_diagnosis[:300]
        if cfg_tags:
            entry["failure_tags"] = cfg_tags
        if cfg_return is not None:
            entry["final_return"] = cfg_return
        if cfg_score_std is not None:
            entry["score_std"] = cfg_score_std
        if cfg_return_std is not None:
            entry["return_std"] = cfg_return_std

        lineage["iterations"][key] = entry

    # Consolidate global lessons with the iteration lesson (best config)
    if lesson:
        structured_lesson: StructuredLesson = {
            "text": lesson,
            "tier": "STRONG",
            "learned_at": iteration,
        }
        lineage["lessons"] = consolidate_lessons(
            lineage,
            structured_lesson,
            client=client,
            model=model,
            thinking_effort=thinking_effort,
        )

    save_lineage(session_dir, lineage)
    logger.info(
        "Lineage updated (multi-config): iter %d, %d configs, best=%s, score=%.3f",
        iteration,
        len(config_judgments),
        best_config_id,
        best_score,
    )


def record_iteration(
    session_dir: str | Path,
    *,
    session_id: str,
    iteration: int,
    based_on: int = 0,
    lesson: str,
    score: float,
    diagnosis: str,
    failure_tags: list[str],
    final_return: float | None = None,
    best_checkpoint: str = "",
    client: anthropic.Anthropic,
    model: str = LLM_MODEL,
    thinking_effort: str = "",
) -> None:
    """Record a completed iteration into the session's lineage tree.

    Called from loop.py after each iteration is judged + revised.
    The *lesson* is produced by the revise agent (which has full context),
    NOT by a separate LLM call.  ``client``/``model`` are only used for
    lesson consolidation when the list grows large.

    *based_on* is the iteration number whose reward code the revise agent
    used as its starting point.  When non-zero, the parent edge points to
    that iteration (tree branching) instead of the previous iteration
    (linear chain).
    """
    lineage = load_lineage(session_dir)

    if based_on > 0 and based_on != iteration:
        # Revise agent declared it based the new code on a specific iteration
        parent_key = iteration_key(session_id, based_on)
        # Validate that the parent exists in lineage; fall back to best if not
        if parent_key not in lineage["iterations"]:
            best_iter = _find_starred_iter_num(lineage) or iteration - 1
            parent_key = iteration_key(session_id, best_iter) if iteration > 1 else None
    else:
        # Default: parent is the starred (best) iteration, not iteration-1
        best_iter = _find_starred_iter_num(lineage) or iteration - 1
        parent_key = iteration_key(session_id, best_iter) if iteration > 1 else None
    star = is_new_best(lineage, score)
    if star:
        _clear_stars(lineage)

    add_iteration(
        lineage,
        session_id=session_id,
        iteration=iteration,
        parent_key=parent_key,
        lesson=lesson,
        score=score,
        star=star,
        diagnosis=diagnosis,
        failure_tags=failure_tags,
        final_return=final_return,
        best_checkpoint=best_checkpoint,
    )

    # Update accumulated lessons (consolidate via LLM when list grows large)
    if lesson:
        structured_lesson: StructuredLesson = {
            "text": lesson,
            "tier": "STRONG",
            "learned_at": iteration,
        }
        lineage["lessons"] = consolidate_lessons(
            lineage,
            structured_lesson,
            client=client,
            model=model,
            thinking_effort=thinking_effort,
        )

    save_lineage(session_dir, lineage)
    logger.info(
        "Lineage updated: %s score=%.3f star=%s lesson=%s",
        iteration_key(session_id, iteration),
        score,
        star,
        lesson[:60] if lesson else "(no lesson)",
    )
