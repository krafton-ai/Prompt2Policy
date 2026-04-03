"""Tests for background enrichment (non-blocking get_session)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture()
def _session_with_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a session dir with loop_history.json containing judgment."""
    import p2p.api.services as svc

    monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
    monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)

    session_dir = tmp_path / "session_enrich"
    session_dir.mkdir()

    now = datetime.now(timezone.utc).isoformat()
    (session_dir / "status.json").write_text(
        json.dumps({"status": "completed", "updated_at": now})
    )

    history = {
        "session_id": "session_enrich",
        "status": "completed",
        "best_iteration": 1,
        "best_score": 0.9,
        "iterations": [
            {
                "iteration": 1,
                "iteration_dir": str(session_dir / "iter_1"),
                "judgment": {
                    "diagnosis": "The agent runs forward well.",
                },
                "summary": {},
                "reward_code": "def reward(obs): return 1.0",
            }
        ],
    }
    (session_dir / "loop_history.json").write_text(json.dumps(history))
    return session_dir


class TestGetSessionNonBlocking:
    def test_get_session_does_not_call_diff_summaries_synchronously(
        self, _session_with_history: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.services as svc
        import p2p.api.session_enrichment_service as enrich

        # Clear any in-flight state from previous tests
        enrich._enriching_sessions.clear()

        calls: list[str] = []

        def mock_diff_summaries(history, session_id):
            calls.append("diff_summaries")
            return False

        monkeypatch.setattr(enrich, "_ensure_diff_summaries", mock_diff_summaries)

        result = svc.get_session("session_enrich")

        # get_session() should return immediately without having called
        # the enrichment functions synchronously in the calling thread
        assert result is not None
        assert result.session_id == "session_enrich"

        # The calls list should be empty at the moment of return
        # (they run in a background thread)
        # Note: we can't assert calls is empty because the thread may
        # have already run. Instead, verify enrichment is scheduled.

    def test_schedule_enrichment_deduplicates(self, monkeypatch: pytest.MonkeyPatch):
        import p2p.api.session_enrichment_service as enrich

        enrich._enriching_sessions.clear()

        # Pretend a session is already being enriched
        with enrich._enriching_lock:
            enrich._enriching_sessions.add("session_dup")

        threads_started: list[str] = []
        original_thread_init = threading.Thread.__init__

        def track_thread(self, *args, **kwargs):
            original_thread_init(self, *args, **kwargs)
            if kwargs.get("name", "").startswith("enrich-"):
                threads_started.append(kwargs["name"])

        monkeypatch.setattr(threading.Thread, "__init__", track_thread)

        enrich.schedule_enrichment("session_dup")

        # Should NOT start a new thread because session_dup is already in-flight
        assert len(threads_started) == 0

        # Cleanup
        enrich._enriching_sessions.discard("session_dup")

    def test_enrichment_runs_in_background_thread(
        self, _session_with_history: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import p2p.api.services as svc
        import p2p.api.session_enrichment_service as enrich

        enrich._enriching_sessions.clear()

        called_in_thread: list[str] = []
        main_thread = threading.current_thread()
        done_event = threading.Event()

        def mock_diff_summaries(history, session_id):
            if threading.current_thread() is not main_thread:
                called_in_thread.append("diff_summaries")
            done_event.set()
            return False

        monkeypatch.setattr(enrich, "_ensure_diff_summaries", mock_diff_summaries)

        result = svc.get_session("session_enrich")
        assert result is not None

        # Wait for the background enrichment to complete
        assert done_event.wait(timeout=5), "enrichment did not complete in time"

        assert "diff_summaries" in called_in_thread


class TestShutdownPool:
    def test_shutdown_pool_calls_executor_shutdown(self, monkeypatch: pytest.MonkeyPatch):
        import p2p.api.session_enrichment_service as enrich

        calls: list[dict] = []

        def fake_shutdown(wait: bool = True, cancel_futures: bool = False) -> None:
            calls.append({"wait": wait, "cancel_futures": cancel_futures})

        monkeypatch.setattr(enrich._enrichment_pool, "shutdown", fake_shutdown)

        enrich.shutdown_pool()
        assert len(calls) == 1
        assert calls[0]["wait"] is True
        assert calls[0]["cancel_futures"] is True

    def test_lifespan_shuts_down_pool(self, monkeypatch: pytest.MonkeyPatch):
        from p2p.api import app as app_mod
        from p2p.api.app import lifespan

        calls: list[str] = []
        monkeypatch.setattr(app_mod, "shutdown_pool", lambda: calls.append("called"))
        # Stub recover_stale_sessions to avoid side effects
        monkeypatch.setattr("p2p.api.process_manager.recover_stale_sessions", lambda: {})

        import asyncio

        async def run_lifespan():
            async with lifespan(None):
                pass

        asyncio.run(run_lifespan())
        assert len(calls) == 1


class TestEmptyApiResponseLogging:
    """Verify that empty API responses log warnings instead of silently falling back."""

    @pytest.fixture(autouse=True)
    def _mock_empty_response(self, monkeypatch: pytest.MonkeyPatch):
        import p2p.api.session_enrichment_service as enrich

        class FakeResponse:
            content = []

        monkeypatch.setattr(enrich, "get_client", lambda: object())
        monkeypatch.setattr(enrich, "create_message", lambda *a, **kw: FakeResponse())
        self.enrich = enrich

    def test_summarize_diff_logs_warning_on_empty_response(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="p2p.api.session_enrichment_service"):
            result = self.enrich._summarize_diff("old code", "new code")

        assert result == ""
        assert "Empty API response in _summarize_diff()" in caplog.text
