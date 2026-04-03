"""Background enrichment — diff summaries for session history."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from p2p.inference.llm_client import create_message, extract_response_text, get_client
from p2p.session.iteration_record import SessionRecord
from p2p.settings import LLM_MODEL_LIGHT, resolve_session_dir

logger = logging.getLogger(__name__)


def _get_session_record(session_id: str) -> SessionRecord:
    return SessionRecord(resolve_session_dir(session_id))


# ---------------------------------------------------------------------------
# Lazy enrichment helpers
# ---------------------------------------------------------------------------


def _summarize_diff(prev_code: str, current_code: str) -> str:
    """Generate a one-line summary of reward function changes."""
    response = create_message(
        get_client(),
        model=LLM_MODEL_LIGHT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Below are two versions of a reward function for RL training.\n\n"
                    f"--- PREVIOUS ---\n{prev_code}\n\n"
                    f"--- CURRENT ---\n{current_code}\n\n"
                    "Summarize what changed in ONE concise sentence. "
                    "Focus on the functional difference, not line-by-line edits. "
                    "Return ONLY the summary sentence."
                ),
            }
        ],
    )
    if not response.content:
        logger.warning("Empty API response in _summarize_diff(), falling back to empty string")
        return ""
    return extract_response_text(response).strip()


def _ensure_diff_summaries(history: dict, session_id: str) -> bool:
    """Lazily generate diff summaries for iterations and cache in loop_history.json.

    Returns True if any summaries were added (file was updated).
    """
    iterations = history.get("iterations", [])
    dirty = False

    for i, it in enumerate(iterations):
        # Skip iteration 1 (no previous) or already summarized
        if it.get("iteration", 0) <= 1:
            continue
        if it.get("reward_diff_summary"):
            continue

        # Find previous iteration's reward code
        prev_it = iterations[i - 1] if i > 0 else None
        if prev_it is None:
            continue

        prev_code = prev_it.get("reward_code", "")
        current_code = it.get("reward_code", "")

        # Skip if same or either is empty
        if not prev_code or not current_code or prev_code == current_code:
            continue

        try:
            summary = _summarize_diff(prev_code, current_code)
            if summary:
                it["reward_diff_summary"] = summary
                dirty = True
        except Exception:
            logger.exception(
                "Diff summary failed for session %s iter %d",
                session_id,
                it.get("iteration", 0),
            )

    if dirty:
        _get_session_record(session_id).save_history(history)

    return dirty


# ---------------------------------------------------------------------------
# Background enrichment scheduler
# ---------------------------------------------------------------------------

_enriching_sessions: set[str] = set()
_enriching_lock = threading.Lock()
_enrichment_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="enrich")


def _enrich_session_background(session_id: str) -> None:
    """Run diff summaries in a background thread."""
    try:
        sr = _get_session_record(session_id)
        history = sr.read_history()
        if history is None:
            return
        _ensure_diff_summaries(history, session_id)
    except Exception:
        logger.exception("Background enrichment failed for session %s", session_id)
    finally:
        with _enriching_lock:
            _enriching_sessions.discard(session_id)


def schedule_enrichment(session_id: str) -> None:
    """Fire-and-forget background enrichment, deduped by session_id."""
    with _enriching_lock:
        if session_id in _enriching_sessions:
            return
        _enriching_sessions.add(session_id)
    _enrichment_pool.submit(_enrich_session_background, session_id)


def shutdown_pool() -> None:
    """Shut down the enrichment thread pool.

    Waits for running tasks to finish but cancels queued ones.
    """
    _enrichment_pool.shutdown(wait=True, cancel_futures=True)
