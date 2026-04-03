"""ReviseToolDispatch: tool handler dispatch for the revise agent.

Sprouted from ``revise_agent.py`` to reduce module size.
Each ``handle_*`` method corresponds to one tool the LLM can call.
"""

from __future__ import annotations

import difflib
import json as _json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from p2p.analysis.training_dynamics import analyze_training_curves, format_training_dynamics

logger = logging.getLogger(__name__)


class ReviseToolDispatch:
    """Dispatch table for revise-agent tool calls.

    Parameters
    ----------
    iterations:
        List of past iteration dicts / dataclass objects.
    current_judgment:
        The current iteration's judgment dict (may be ``None``).
    lineage:
        Full cross-session experiment lineage (may be ``None``).
    """

    def __init__(
        self,
        iterations: list,
        current_judgment: dict | None = None,
        lineage: dict | None = None,
        session_dir: str | Path | None = None,
    ) -> None:
        self._iterations = iterations
        self._cj = current_judgment or {}
        self._lineage = lineage
        self._session_dir = session_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dispatch(self) -> dict[str, Callable[[dict], dict]]:
        """Return a mapping of tool-name -> handler callable."""
        dispatch: dict[str, Callable[[dict], dict]] = {
            "get_iteration_reward_code": self.handle_get_reward_code,
            "get_iteration_training_dynamics": self.handle_get_training_dynamics,
            "compare_iterations": self.handle_compare_iterations,
            "get_iteration_judgment_detail": self.handle_get_judgment_detail,
            "get_checkpoint_judgments": self.handle_get_checkpoint_judgments,
            "get_rollout_judgment": self.handle_get_rollout_judgment,
            "get_config_comparison": self.handle_get_config_comparison,
            "get_iteration_scores": self.handle_get_iteration_scores,
            "get_strategy_summary": self.handle_get_strategy_summary,
        }
        dispatch["get_experiment_lineage"] = self.handle_get_experiment_lineage
        if self._lineage and self._session_dir:
            dispatch["update_experiment_lessons"] = self.handle_update_experiment_lessons
        return dispatch

    # ------------------------------------------------------------------
    # Iteration accessors
    # ------------------------------------------------------------------

    @staticmethod
    def _find_iteration(iterations: list, iteration: int) -> Any | None:
        for it in iterations:
            num = it.iteration if hasattr(it, "iteration") else it.get("iteration")
            if num == iteration:
                return it
        return None

    @staticmethod
    def _get_reward_code(it: Any) -> str:
        if hasattr(it, "reward_code"):
            return it.reward_code
        return it.get("reward_code", "")

    @staticmethod
    def _get_judgment(it: Any) -> dict:
        if hasattr(it, "judgment"):
            return it.judgment if isinstance(it.judgment, dict) else {}
        return it.get("judgment", {})

    @staticmethod
    def _get_iteration_dir(it: Any) -> str:
        if hasattr(it, "iteration_dir"):
            return it.iteration_dir
        return it.get("iteration_dir", "")

    @staticmethod
    def _get_field(it: Any, field: str, default: Any = "") -> Any:
        if hasattr(it, field):
            return getattr(it, field)
        return it.get(field, default) if isinstance(it, dict) else default

    @staticmethod
    def _resolve_run_scalars(iter_dir: str, run_id: str = "best") -> Path:
        """Resolve scalars.jsonl path for a run within an iteration dir.

        In multi-config mode, reads best_run.json to find the subdir.
        In single-config mode, falls back to iter_dir/metrics/scalars.jsonl.
        """
        base = Path(iter_dir)
        best_run_json = base / "best_run.json"

        if run_id and run_id != "best":
            return base / run_id / "metrics" / "scalars.jsonl"

        if best_run_json.exists():
            try:
                info = _json.loads(best_run_json.read_text())
            except (OSError, _json.JSONDecodeError) as exc:
                logger.warning("Failed to read %s: %s", best_run_json, exc)
            else:
                best = info.get("best_run_id", "")
                if best:
                    return base / best / "metrics" / "scalars.jsonl"

        return base / "metrics" / "scalars.jsonl"

    @staticmethod
    def _resolve_run_judgment(
        current_judgment: dict, run_id: str = "best"
    ) -> tuple[dict, str | None]:
        """Resolve judgment dict for a specific run_id.

        In multi-config mode, current_judgment contains all_run_judgments.
        In single-config mode, current_judgment IS the only judgment.

        Returns (judgment_dict, resolved_run_id).
        resolved_run_id is None in single-config mode.
        """
        if not run_id or run_id == "best":
            best = current_judgment.get("best_run_id")
            return current_judgment, best

        all_runs = current_judgment.get("all_run_judgments", {})
        if run_id in all_runs:
            return all_runs[run_id], run_id

        return current_judgment, None

    @staticmethod
    def _summarize_iteration_judgment(j: dict, label: int) -> dict:
        result: dict = {
            "iteration": label,
            "score": j.get("intent_score", 0),
            "failure_tags": j.get("failure_tags", []),
        }
        config_judgments = j.get("config_judgments")
        if config_judgments:
            result["config_scores"] = {
                cid: {
                    "mean_score": cj.get("mean_intent_score", 0),
                    "score_std": cj.get("score_std", 0),
                    "common_failure_tags": cj.get("common_failure_tags", []),
                }
                for cid, cj in config_judgments.items()
            }
        return result

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def handle_get_reward_code(self, inp: dict) -> dict:
        if "iteration" not in inp:
            return {"error": "'iteration' is a required parameter."}
        it = self._find_iteration(self._iterations, inp["iteration"])
        if it is None:
            return {"error": f"Iteration {inp['iteration']} not found"}
        return {"iteration": inp["iteration"], "reward_code": self._get_reward_code(it)}

    def handle_get_training_dynamics(self, inp: dict) -> dict:
        if "iteration" not in inp:
            return {"error": "'iteration' is a required parameter."}
        it = self._find_iteration(self._iterations, inp["iteration"])
        if it is None:
            return {"error": f"Iteration {inp['iteration']} not found"}
        iter_dir = self._get_iteration_dir(it)
        if not iter_dir:
            return {"error": f"No iteration_dir for iteration {inp['iteration']}"}
        run_id = inp.get("run_id", "best")
        scalars = self._resolve_run_scalars(iter_dir, run_id)
        dynamics = analyze_training_curves(scalars)
        result: dict = {
            "iteration": inp["iteration"],
            "dynamics": format_training_dynamics(dynamics),
        }
        if run_id and run_id != "best":
            result["run_id"] = run_id
        return result

    def handle_compare_iterations(self, inp: dict) -> dict:
        if "iter_a" not in inp or "iter_b" not in inp:
            return {"error": "Both 'iter_a' and 'iter_b' are required parameters."}

        it_a = self._find_iteration(self._iterations, inp["iter_a"])
        it_b = self._find_iteration(self._iterations, inp["iter_b"])
        if it_a is None:
            return {"error": f"Iteration {inp['iter_a']} not found"}
        if it_b is None:
            return {"error": f"Iteration {inp['iter_b']} not found"}

        code_a = self._get_reward_code(it_a)
        code_b = self._get_reward_code(it_b)
        j_a = self._get_judgment(it_a)
        j_b = self._get_judgment(it_b)

        diff = list(
            difflib.unified_diff(
                code_a.splitlines(),
                code_b.splitlines(),
                fromfile=f"iter_{inp['iter_a']}",
                tofile=f"iter_{inp['iter_b']}",
                lineterm="",
            )
        )

        return {
            "iter_a": self._summarize_iteration_judgment(j_a, inp["iter_a"]),
            "iter_b": self._summarize_iteration_judgment(j_b, inp["iter_b"]),
            "reward_code_diff": "\n".join(diff) if diff else "(identical)",
        }

    def handle_get_judgment_detail(self, inp: dict) -> dict:
        if "iteration" not in inp:
            return {"error": "'iteration' is a required parameter."}
        it = self._find_iteration(self._iterations, inp["iteration"])
        if it is None:
            return {"error": f"Iteration {inp['iteration']} not found"}
        j = self._get_judgment(it)

        run_id = inp.get("run_id", "best")
        resolved, resolved_id = self._resolve_run_judgment(j, run_id)

        result: dict = {
            "iteration": inp["iteration"],
            "intent_score": resolved.get("intent_score", 0),
            "diagnosis": resolved.get("diagnosis", ""),
            "failure_tags": resolved.get("failure_tags", []),
        }
        if resolved_id:
            result["resolved_run_id"] = resolved_id

        config_judgments = j.get("config_judgments")
        if config_judgments:
            result["config_scores"] = {
                cid: {
                    "mean_score": cj.get("mean_intent_score", 0),
                    "score_std": cj.get("score_std", 0),
                    "common_failure_tags": cj.get("common_failure_tags", []),
                }
                for cid, cj in config_judgments.items()
            }
            all_runs = j.get("all_run_judgments", {})
            if all_runs:
                result["available_run_ids"] = sorted(all_runs.keys())

        return result

    def handle_get_checkpoint_judgments(self, inp: dict) -> dict:
        """Return rollout judgments + aggregate for a checkpoint step.

        detail levels:
          - 'aggregate': aggregate stats only (default, most token-efficient)
          - 'summary':   aggregate + best and worst rollout diagnoses
          - 'all':       aggregate + every rollout's full diagnosis
        """
        step = inp.get("step")
        if not step:
            return {"error": "'step' is a required parameter."}
        detail = inp.get("detail", "aggregate")
        run_id = inp.get("run_id", "best")

        judgment, resolved_id = self._resolve_run_judgment(self._cj, run_id)
        cp_judgments = judgment.get("checkpoint_judgments", {})
        cp = cp_judgments.get(step)
        if cp is None:
            available = sorted(cp_judgments.keys())
            return {"error": f"Step '{step}' not found. Available: {available}"}

        rollout_judgments = cp.get("rollout_judgments", [])
        aggregate = cp.get("checkpoint_aggregate", {})

        if not rollout_judgments and not aggregate:
            result: dict = {
                "step": step,
                "note": "No per-rollout data available (single-rollout evaluation).",
                "intent_score": cp.get("intent_score", 0),
                "diagnosis": cp.get("diagnosis", ""),
                "failure_tags": cp.get("failure_tags", []),
            }
            if resolved_id:
                result["resolved_run_id"] = resolved_id
            return result

        aggregate_clean = {k: v for k, v in aggregate.items() if k != "rollout_judgments"}
        result: dict = {"step": step, "aggregate": aggregate_clean}
        if resolved_id:
            result["resolved_run_id"] = resolved_id

        if detail == "all":
            result["rollout_judgments"] = rollout_judgments
        elif detail == "summary" and rollout_judgments:
            sorted_rjs = sorted(rollout_judgments, key=lambda r: r.get("intent_score", 0))
            result["worst_rollout"] = sorted_rjs[0]
            result["median_rollout"] = sorted_rjs[len(sorted_rjs) // 2]
            result["best_rollout"] = sorted_rjs[-1]
            result["num_rollouts"] = len(rollout_judgments)

        return result

    def handle_get_rollout_judgment(self, inp: dict) -> dict:
        """Return a single rollout judgment by step + rollout_label or episode_idx."""
        step = inp.get("step")
        rollout_label = inp.get("rollout_label")
        episode_idx = inp.get("episode_idx")
        if not step:
            return {"error": "'step' is a required parameter."}
        if rollout_label is None and episode_idx is None:
            return {
                "error": (
                    "Either 'rollout_label' or 'episode_idx' is required. "
                    "Use rollout_label='p10'/'median'/'p90' for parallel eval, "
                    "or episode_idx (integer) for sequential eval."
                )
            }

        run_id = inp.get("run_id", "best")
        judgment, resolved_id = self._resolve_run_judgment(self._cj, run_id)
        cp_judgments = judgment.get("checkpoint_judgments", {})
        cp = cp_judgments.get(step)
        if cp is None:
            available = sorted(cp_judgments.keys())
            return {"error": f"Step '{step}' not found. Available: {available}"}

        rollout_judgments = cp.get("rollout_judgments", [])
        if not rollout_judgments:
            return {
                "error": "No per-rollout data available for this checkpoint.",
                "step": step,
            }

        # Match by rollout_label first (parallel eval), then episode_idx (sequential)
        if rollout_label is not None:
            for rj in rollout_judgments:
                if rj.get("rollout_label") == rollout_label:
                    result = {"step": step, **rj}
                    if resolved_id:
                        result["resolved_run_id"] = resolved_id
                    return result
        else:
            for rj in rollout_judgments:
                if rj.get("episode_idx") == episode_idx:
                    result = {"step": step, **rj}
                    if resolved_id:
                        result["resolved_run_id"] = resolved_id
                    return result

        # Not found -- list available rollout labels and episode indices
        available_labels = [
            rj.get("rollout_label") for rj in rollout_judgments if rj.get("rollout_label")
        ]
        available_eps = [
            rj.get("episode_idx") for rj in rollout_judgments if rj.get("episode_idx") is not None
        ]
        lookup_key = (
            f"rollout_label='{rollout_label}'"
            if rollout_label is not None
            else f"episode_idx={episode_idx}"
        )
        parts = [f"Rollout {lookup_key} not found."]
        if available_labels:
            parts.append(f"Available rollout_labels: {available_labels}")
        if available_eps:
            parts.append(f"Available episode_idx: {available_eps}")
        return {"error": " ".join(parts)}

    def handle_get_config_comparison(self, inp: dict) -> dict:
        """Return cross-config comparison for multi-config training."""
        config_judgments = self._cj.get("config_judgments")
        if not config_judgments:
            return {
                "error": (
                    "No multi-config data available. "
                    "This tool is only for multi-config training. "
                    "Use get_checkpoint_judgments for single-config."
                )
            }

        detail = inp.get("detail", "aggregate")

        all_runs = self._cj.get("all_run_judgments", {})
        result: dict = {
            "num_configs": len(config_judgments),
            "available_run_ids": sorted(all_runs.keys()) if all_runs else [],
            "configs": {},
        }
        for config_id, cj in config_judgments.items():
            entry: dict = {
                "config_id": config_id,
                "num_seeds": cj.get("num_seeds", 0),
                "mean_intent_score": cj.get("mean_intent_score", 0),
                "score_std": cj.get("score_std", 0),
                "mean_final_return": cj.get("mean_final_return", 0),
                "return_std": cj.get("return_std", 0),
                "common_failure_tags": cj.get("common_failure_tags", []),
            }

            per_seed = cj.get("per_seed", [])
            if detail == "all" and per_seed:
                entry["per_seed"] = per_seed
            elif detail == "summary" and per_seed:
                sorted_seeds = sorted(per_seed, key=lambda s: s.get("intent_score", 0))
                entry["worst_seed"] = sorted_seeds[0]
                entry["best_seed"] = sorted_seeds[-1]

            result["configs"][config_id] = entry

        return result

    def handle_get_iteration_scores(self, inp: dict) -> dict:
        """Return compact score timeline with trend analysis and best iteration context."""
        if not self._iterations:
            return {"error": "No past iterations available."}

        entries: list[dict] = []
        for it in self._iterations:
            num = it.iteration if hasattr(it, "iteration") else it.get("iteration", 0)
            j = self._get_judgment(it)
            score = j.get("intent_score", 0) if isinstance(j, dict) else 0
            failure_tags = j.get("failure_tags", []) if isinstance(j, dict) else []
            entry: dict = {"iteration": num, "score": score, "failure_tags": failure_tags}

            config_judgments = j.get("config_judgments") if isinstance(j, dict) else None
            if config_judgments:
                entry["config_scores"] = {
                    cid: round(cj.get("mean_intent_score", 0), 3)
                    for cid, cj in config_judgments.items()
                }
            entries.append(entry)

        entries.sort(key=lambda e: e["iteration"])
        scores = [e["score"] for e in entries]

        if len(scores) >= 2:
            improving = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
            declining = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            if improving:
                trend = "improving"
            elif declining:
                trend = "declining"
            elif scores[-1] >= max(scores[:-1]):
                trend = "new_best"
            elif abs(max(scores) - min(scores)) < 0.05:
                trend = "plateau"
            else:
                trend = "erratic"
        else:
            trend = "insufficient_data"

        best_entry = max(entries, key=lambda e: e["score"])
        best_num = best_entry["iteration"]

        best_it = self._find_iteration(self._iterations, best_num)
        best_code = self._get_reward_code(best_it) if best_it else ""

        top_k = inp.get("top_k")
        if top_k and isinstance(top_k, int) and top_k > 0:
            display_entries = sorted(entries, key=lambda e: e["score"], reverse=True)[:top_k]
        else:
            display_entries = entries

        return {
            "timeline": display_entries,
            "trend": trend,
            "best_iteration": best_num,
            "best_score": best_entry["score"],
            "best_reward_code": best_code,
            "total_iterations": len(entries),
            "score_range": {
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
            },
        }

    def handle_get_strategy_summary(self, _inp: dict) -> dict:
        """Build a strategic 'tree of thoughts' summary of the full iteration history."""
        if not self._iterations:
            return {"error": "No iterations yet."}

        # 1. Extract per-iteration data
        timeline: list[dict] = []
        for it in self._iterations:
            num = it.iteration if hasattr(it, "iteration") else it.get("iteration", 0)
            j = self._get_judgment(it)
            score = j.get("intent_score", 0) if isinstance(j, dict) else 0
            tags = j.get("failure_tags", []) if isinstance(j, dict) else []
            diag = j.get("diagnosis", "") if isinstance(j, dict) else ""
            hp_changes = self._get_field(it, "hp_changes", {}) or {}
            reasoning = self._get_field(it, "reward_reasoning", "")
            hp_reason = self._get_field(it, "hp_reasoning", "")

            change = reasoning[:100] if reasoning else ""
            if hp_reason and not change:
                change = hp_reason[:100]
            if not change:
                change = "initial" if num == 1 else "(no reasoning recorded)"

            timeline.append(
                {
                    "iter": num,
                    "score": round(score, 3),
                    "tags": tags,
                    "diag_short": diag[:80] if diag else "",
                    "change": change,
                    "hp": hp_changes if hp_changes else None,
                }
            )

        timeline.sort(key=lambda e: e["iter"])

        # 2. Compute deltas and classify trends
        for i, entry in enumerate(timeline):
            if i == 0:
                entry["delta"] = None
                entry["dir"] = "initial"
            else:
                prev_score = timeline[i - 1]["score"]
                d = entry["score"] - prev_score
                entry["delta"] = round(d, 3)
                if d > 0.02:
                    entry["dir"] = "up"
                elif d < -0.02:
                    entry["dir"] = "down"
                else:
                    entry["dir"] = "flat"

        # 3. Group into phases by consecutive trend direction
        phases: list[dict] = []
        current_phase: dict | None = None
        for entry in timeline:
            d = entry["dir"]
            if current_phase is None or current_phase["trend"] != d:
                current_phase = {
                    "trend": d,
                    "iters": [entry["iter"], entry["iter"]],
                    "score_range": [entry["score"], entry["score"]],
                    "count": 1,
                    "entries": [entry],
                }
                phases.append(current_phase)
            else:
                current_phase["iters"][1] = entry["iter"]
                sr = current_phase["score_range"]
                sr[0] = min(sr[0], entry["score"])
                sr[1] = max(sr[1], entry["score"])
                current_phase["count"] += 1
                current_phase["entries"].append(entry)

        phase_summary = []
        for p in phases:
            ps: dict = {
                "trend": p["trend"],
                "iters": p["iters"],
                "score_range": p["score_range"],
                "count": p["count"],
            }
            entries = p["entries"]
            if len(entries) == 1:
                ps["changes"] = [entries[0]["change"]]
            else:
                ps["changes"] = [entries[0]["change"], entries[-1]["change"]]
            phase_summary.append(ps)

        # 4. Regression log
        regressions = []
        for entry in timeline:
            if entry["delta"] is not None and entry["delta"] < -0.02:
                regressions.append(
                    {
                        "iter": entry["iter"],
                        "delta": entry["delta"],
                        "score": entry["score"],
                        "change": entry["change"],
                        "hp": entry["hp"],
                    }
                )

        # 5. Persistent failure patterns (>= 30% of iterations, min 2)
        tag_counter: Counter[str] = Counter()
        tag_first: dict[str, int] = {}
        tag_last: dict[str, int] = {}
        for entry in timeline:
            for tag in entry["tags"]:
                tag_counter[tag] += 1
                if tag not in tag_first:
                    tag_first[tag] = entry["iter"]
                tag_last[tag] = entry["iter"]

        total = len(timeline)
        threshold = max(2, int(total * 0.3))
        persistent = []
        for tag, count in tag_counter.most_common():
            if count < threshold:
                continue
            persistent.append(
                {
                    "tag": tag,
                    "count": count,
                    "pct": round(count / total * 100),
                    "first": tag_first[tag],
                    "last": tag_last[tag],
                }
            )

        # 6. HP oscillation detection
        hp_history: dict[str, list[tuple[int, Any]]] = {}
        for entry in timeline:
            if entry["hp"]:
                for k, v in entry["hp"].items():
                    hp_history.setdefault(k, []).append((entry["iter"], v))

        hp_oscillations = []
        for param, history in hp_history.items():
            if len(history) < 3:
                continue
            values = [v for _, v in history]
            for i in range(len(values) - 2):
                if values[i] == values[i + 2] and values[i] != values[i + 1]:
                    iters = [history[i][0], history[i + 1][0], history[i + 2][0]]
                    hp_oscillations.append(
                        {
                            "param": param,
                            "pattern": [values[i], values[i + 1], values[i + 2]],
                            "iters": iters,
                        }
                    )
                    break

        # 7. Reward oscillation
        reward_oscillations = []
        codes = [
            (
                e["iter"],
                self._get_reward_code(it)
                if (it := self._find_iteration(self._iterations, e["iter"]))
                else "",
            )
            for e in timeline
        ]
        for i in range(2, len(codes)):
            if not codes[i][1] or not codes[i - 1][1] or not codes[i - 2][1]:
                continue
            sim_prev = difflib.SequenceMatcher(None, codes[i][1], codes[i - 1][1]).ratio()
            sim_prev2 = difflib.SequenceMatcher(None, codes[i][1], codes[i - 2][1]).ratio()
            if sim_prev2 > 0.85 and sim_prev2 > sim_prev + 0.1:
                reward_oscillations.append(
                    {
                        "iter": codes[i][0],
                        "similar_to": codes[i - 2][0],
                        "similarity": round(sim_prev2, 2),
                    }
                )

        # 8. Tried changes ledger — what was attempted and its outcome
        tried_changes: list[dict] = []
        for i, entry in enumerate(timeline):
            if (
                entry["iter"] <= 1
                or not entry.get("change")
                or entry["change"] == "(no reasoning recorded)"
            ):
                continue
            prev_score = timeline[i - 1]["score"] if i > 0 else 0
            delta = entry["score"] - prev_score
            if delta > 0.02:
                outcome = "improved"
            elif delta < -0.02:
                outcome = "regressed"
            else:
                outcome = "no_effect"
            tried_changes.append(
                {
                    "iter": entry["iter"],
                    "change": entry["change"],
                    "outcome": outcome,
                    "delta": round(delta, 3),
                }
            )

        # 9. Strategic notes (rule-based)
        notes: list[str] = []
        best_entry = max(timeline, key=lambda e: e["score"])
        current = timeline[-1]

        iters_since_best = len(timeline) - best_entry["iter"]
        if iters_since_best >= 3:
            notes.append(
                f"STAGNATION: {iters_since_best} iterations since best score "
                f"(iter {best_entry['iter']}, {best_entry['score']:.2f}). "
                "Coefficient tuning is exhausted — try a structural change "
                "or use NO_CHANGE to keep the best iteration's code."
            )

        if len(phases) > 0 and phases[-1]["trend"] == "flat" and phases[-1]["count"] >= 3:
            notes.append(
                f"Plateau for {phases[-1]['count']} iterations — "
                "consider a structural reward redesign or larger HP change."
            )
        if regressions:
            notes.append(
                f"{len(regressions)} regression(s) — check regression_log to avoid repeating."
            )
        if persistent:
            top_tag = persistent[0]
            notes.append(
                f"'{top_tag['tag']}' persists across {top_tag['count']}/{total} iterations "
                "— may need a fundamentally different approach."
            )
        if hp_oscillations:
            params = [o["param"] for o in hp_oscillations]
            notes.append(f"HP oscillation on {', '.join(params)} — commit to a direction.")
        if reward_oscillations:
            notes.append(
                f"Reward code reverted toward earlier versions {len(reward_oscillations)} time(s) "
                "— avoid undoing previous improvements."
            )
        if current["score"] < best_entry["score"] - 0.05:
            notes.append(
                f"Current score ({current['score']:.2f}) is below best "
                f"(iter {best_entry['iter']}, {best_entry['score']:.2f}) — "
                "consider building on the best iteration's reward."
            )

        # Count failed changes to detect exhaustion
        failed_changes = [c for c in tried_changes if c["outcome"] in ("no_effect", "regressed")]
        if len(failed_changes) >= 5:
            notes.append(
                f"{len(failed_changes)} of {len(tried_changes)} past changes had no effect or "
                "regressed. Most easy modifications are exhausted. Try something structurally "
                "different or use NO_CHANGE."
            )

        return {
            "total_iterations": total,
            "best": {"iter": best_entry["iter"], "score": best_entry["score"]},
            "current": {"iter": current["iter"], "score": current["score"]},
            "phases": phase_summary,
            "regression_log": regressions,
            "persistent_failures": persistent,
            "oscillations": {
                "hp": hp_oscillations,
                "reward": reward_oscillations,
            },
            "tried_changes": tried_changes,
            "strategic_notes": notes[:6],
        }

    def handle_get_experiment_lineage(self, inp: dict) -> dict:
        """Return the session experiment lineage tree and accumulated lessons."""
        if not self._lineage:
            return {"error": "No lineage data available."}

        from p2p.session.lineage import format_lessons, format_lineage_tree, lesson_tier_counts

        tree = format_lineage_tree(self._lineage)
        lessons = format_lessons(self._lineage)
        total = len(self._lineage.get("iterations", {}))
        num_lessons = len(self._lineage.get("lessons", []))

        # Find starred (best) iterations
        starred = [
            {"key": k, "score": v.get("score", 0), "lesson": v.get("lesson", "")[:80]}
            for k, v in self._lineage.get("iterations", {}).items()
            if v.get("star")
        ]

        return {
            "total_iterations": total,
            "num_lessons": num_lessons,
            "lesson_counts": lesson_tier_counts(self._lineage),
            "tree": tree,
            "lessons": lessons,
            "starred_iterations": starred,
        }

    def handle_update_experiment_lessons(self, inp: dict) -> dict:
        """Change one lesson's tier with a mandatory reason for audit trail."""
        if not self._lineage or not self._session_dir:
            return {"error": "No lineage data or session directory available."}

        from p2p.session.lineage import TIER_ORDER, save_lineage

        index = inp.get("index")
        tier = inp.get("tier")
        reason = inp.get("reason", "")
        if index is None or tier is None:
            return {"error": "'index' (1-based) and 'tier' are required."}
        if not reason:
            return {"error": "'reason' is required for audit trail."}
        if tier not in TIER_ORDER:
            return {"error": f"'tier' must be one of {list(TIER_ORDER)}."}
        lessons = self._lineage.get("lessons", [])
        if not isinstance(index, int) or index < 1 or index > len(lessons):
            return {"error": f"'index' must be 1-{len(lessons)}."}
        # format_lessons() sorts by TIER_ORDER, so the 1-based display index
        # maps to the sorted view, not the raw list.  sorted() preserves
        # object references, so mutating the target also updates lineage.
        sorted_lessons = sorted(
            lessons,
            key=lambda les: TIER_ORDER.get(les.get("tier", "STRONG"), 1),
        )
        target = sorted_lessons[index - 1]
        old_tier = target.get("tier", "STRONG")
        target["tier"] = tier
        target["tier_reason"] = reason
        save_lineage(self._session_dir, self._lineage)
        return {
            "status": "ok",
            "index": index,
            "old_tier": old_tier,
            "new_tier": tier,
            "reason": reason,
            "text": target.get("text", "")[:80],
        }
