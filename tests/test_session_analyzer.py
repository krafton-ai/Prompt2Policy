"""Tests for session_analyzer tool functions and agentic loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from p2p.agents.session_analyzer import (
    _compare_reward,
    _extract_json,
    _read_detail,
    _read_metrics,
    _read_overview,
    analyze_session,
)
from p2p.session.iteration_record import SessionRecord


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a minimal session directory with 2 iterations."""
    sd = tmp_path / "session_test123"
    sd.mkdir()

    # status.json
    (sd / "status.json").write_text(json.dumps({"status": "passed"}))

    # loop_history.json
    history = {
        "session_id": "session_test123",
        "prompt": "make it walk fast",
        "status": "passed",
        "best_iteration": 1,
        "best_score": 0.75,
        "iterations": [
            {
                "iteration": 0,
                "iteration_dir": str(sd / "iter_0"),
                "reward_code": "def reward_fn(o, a, n, i):\n    return o[0], {}\n",
                "summary": {
                    "final_episodic_return": 100.0,
                    "total_timesteps": 50000,
                    "training_time_s": 30.0,
                    "total_episodes": 100,
                },
                "judgment": {
                    "intent_score": 0.3,
                    "passed": False,
                    "diagnosis": "Agent barely moves",
                    "failure_tags": ["low_velocity", "no_progress"],
                },
            },
            {
                "iteration": 1,
                "iteration_dir": str(sd / "iter_1"),
                "reward_code": ("def reward_fn(o, a, n, i):\n    return o[0] + o[8], {}\n"),
                "summary": {
                    "final_episodic_return": 500.0,
                    "total_timesteps": 50000,
                    "training_time_s": 28.0,
                    "total_episodes": 120,
                },
                "judgment": {
                    "intent_score": 0.75,
                    "passed": True,
                    "diagnosis": "Good locomotion achieved",
                    "failure_tags": [],
                },
            },
        ],
    }
    (sd / "loop_history.json").write_text(json.dumps(history))

    # iter_0
    i0 = sd / "iter_0"
    i0.mkdir()
    config = {"env_id": "HalfCheetah-v5", "total_timesteps": 50000}
    (i0 / "config.json").write_text(json.dumps(config))
    (i0 / "reward_fn.py").write_text("def reward_fn(o, a, n, i):\n    return o[0], {}\n")
    (i0 / "summary.json").write_text(json.dumps(history["iterations"][0]["summary"]))
    (i0 / "judgment.json").write_text(json.dumps(history["iterations"][0]["judgment"]))
    metrics_dir = i0 / "metrics"
    metrics_dir.mkdir()
    scalars = [
        {
            "global_step": 1000,
            "iteration": 1,
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "entropy": 0.6,
            "sps": 1000,
        },
        {
            "global_step": 50000,
            "iteration": 50,
            "policy_loss": 0.1,
            "value_loss": 0.1,
            "entropy": 0.3,
            "sps": 1200,
            "episodic_return": 100.0,
        },
    ]
    (metrics_dir / "scalars.jsonl").write_text("\n".join(json.dumps(s) for s in scalars))

    # iter_1
    i1 = sd / "iter_1"
    i1.mkdir()
    (i1 / "config.json").write_text(json.dumps(config))
    (i1 / "reward_fn.py").write_text("def reward_fn(o, a, n, i):\n    return o[0] + o[8], {}\n")
    (i1 / "summary.json").write_text(json.dumps(history["iterations"][1]["summary"]))
    (i1 / "judgment.json").write_text(json.dumps(history["iterations"][1]["judgment"]))

    return sd


class TestToolFunctions:
    def test_read_overview(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _read_overview(session)
        assert result["session_id"] == "session_test123"
        assert result["total_iterations"] == 2
        assert result["best_score"] == 0.75
        assert len(result["iterations"]) == 2
        assert result["iterations"][0]["intent_score"] == 0.3
        assert result["iterations"][1]["passed"] is True

    def test_read_detail(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _read_detail(session, 0)
        assert result["iteration"] == 0
        assert result["judgment"]["intent_score"] == 0.3
        assert result["judgment"]["failure_tags"] == ["low_velocity", "no_progress"]
        assert "reward_fn" in result["reward_code"]

    def test_read_detail_not_found(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _read_detail(session, 99)
        assert "error" in result

    def test_read_metrics(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _read_metrics(session, 0)
        assert result["iteration"] == 0
        assert result["training_summary"]["total_entries"] == 2
        assert result["training_summary"]["last_step"] == 50000
        assert result["training_summary"]["final_episodic_return"] == 100.0

    def test_read_metrics_not_found(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _read_metrics(session, 99)
        assert "error" in result

    def test_compare_reward(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _compare_reward(session, 0, 1)
        assert result["iter_a"] == 0
        assert result["iter_b"] == 1
        assert "o[8]" in result["diff"]

    def test_compare_reward_identical(self, session_dir: Path):
        session = SessionRecord(session_dir)
        result = _compare_reward(session, 0, 0)
        assert result["diff"] == "(identical)"


class TestExtractJson:
    def test_plain_json(self):
        text = '{"analysis_en": "hello", "key_findings": []}'
        result = _extract_json(text)
        assert result["analysis_en"] == "hello"

    def test_markdown_fenced(self):
        text = 'Here is my analysis:\n```json\n{"analysis_en": "hello"}\n```'
        result = _extract_json(text)
        assert result["analysis_en"] == "hello"

    def test_json_in_text(self):
        text = 'Some text before {"analysis_en": "hello"} some text after'
        result = _extract_json(text)
        assert result["analysis_en"] == "hello"

    def test_no_json(self):
        result = _extract_json("just some plain text")
        assert result == {}


class TestAnalyzeSession:
    def test_single_round_end_turn(self, session_dir: Path):
        """Test that analyze_session works when LLM returns end_turn immediately."""
        client = MagicMock()

        # Mock response with end_turn
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = json.dumps(
            {
                "analysis_en": "The session improved well.",
                "key_findings": ["Score improved from 0.3 to 0.75"],
                "recommendations": ["Try longer training"],
            }
        )

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 200

        client.messages.create.return_value = mock_response

        result = analyze_session(
            "session_test123",
            client=client,
            model="claude-sonnet-4-6",
            runs_dir=session_dir.parent,
        )

        assert result["session_id"] == "session_test123"
        assert result["analysis_en"] == "The session improved well."
        assert len(result["key_findings"]) == 1
        assert result["tool_calls_used"] == 0

    def test_tool_use_then_end_turn(self, session_dir: Path):
        """Test that analyze_session handles one round of tool use then end_turn."""
        client = MagicMock()

        # First response: tool_use
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_session_overview"
        mock_tool_block.input = {}
        mock_tool_block.id = "tool_1"

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_block]
        first_response.usage.input_tokens = 100
        first_response.usage.output_tokens = 50

        # Second response: end_turn with analysis
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = json.dumps(
            {
                "analysis_en": "Session went well.",
                "key_findings": ["Good improvement"],
                "recommendations": ["Continue training"],
            }
        )

        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [mock_text_block]
        second_response.usage.input_tokens = 200
        second_response.usage.output_tokens = 300

        client.messages.create.side_effect = [first_response, second_response]

        status_messages = []
        result = analyze_session(
            "session_test123",
            client=client,
            model="claude-sonnet-4-6",
            runs_dir=session_dir.parent,
            on_status=lambda msg: status_messages.append(msg),
        )

        assert result["session_id"] == "session_test123"
        assert result["tool_calls_used"] == 1
        assert len(status_messages) > 0
        assert client.messages.create.call_count == 2

    def test_session_not_found(self, tmp_path: Path):
        client = MagicMock()
        with pytest.raises(FileNotFoundError, match="Session not found"):
            analyze_session("nonexistent", client=client, runs_dir=tmp_path)


def _valid_analysis_data(**overrides: object) -> dict:
    """Return a minimal valid SessionAnalysis dict."""
    base: dict = {
        "session_id": "test",
        "analysis_en": "hello",
        "key_findings": ["f1"],
        "recommendations": ["r1"],
        "tool_calls_used": 3,
        "model": "claude-test",
        "created_at": "2025-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestSessionRecordAnalysisCache:
    def test_save_and_read_analysis(self, session_dir: Path):
        session = SessionRecord(session_dir)
        data = _valid_analysis_data()
        session.save_analysis(data)
        result = session.read_analysis()
        assert result is not None
        assert result["analysis_en"] == "hello"

    def test_read_analysis_missing(self, tmp_path: Path):
        session = SessionRecord(tmp_path / "nonexistent")
        assert session.read_analysis() is None

    def test_read_analysis_missing_required_keys(self, session_dir: Path):
        """Legacy analysis.json missing required keys returns None."""
        session = SessionRecord(session_dir)
        session.save_analysis({"session_id": "test", "analysis_en": "partial"})
        assert session.read_analysis() is None

    def test_read_analysis_single_missing_key(self, session_dir: Path):
        """Dropping exactly one required key still rejects."""
        session = SessionRecord(session_dir)
        data = _valid_analysis_data()
        del data["created_at"]
        session.save_analysis(data)
        assert session.read_analysis() is None

    def test_read_analysis_extra_keys_accepted(self, session_dir: Path):
        """Unknown keys must not cause rejection (forward compatibility)."""
        session = SessionRecord(session_dir)
        data = _valid_analysis_data(extra_future_field="ok")
        session.save_analysis(data)
        result = session.read_analysis()
        assert result is not None
