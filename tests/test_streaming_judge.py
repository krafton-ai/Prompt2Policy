"""Tests for StreamingJudge — file watching and result reuse."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from unittest.mock import patch

from p2p.agents.judge_agent import StreamingJudge, judge_all_checkpoints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_png(width: int = 64, height: int = 64, color: int = 128) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), (color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_checkpoint(iteration_dir: Path, step: str) -> None:
    """Create eval video for one checkpoint."""
    videos_dir = iteration_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / f"eval_{step}.mp4").write_bytes(b"\x00" * 100)


def _write_trajectory(iteration_dir: Path, step: str, n_steps: int = 20) -> None:
    import numpy as np

    traj_path = iteration_dir / f"trajectory_{step}.npz"
    term_names = ["forward", "alive"]
    reward_terms = np.column_stack([np.full(n_steps, 0.8), np.full(n_steps, 0.2)])
    np.savez_compressed(
        traj_path,
        step=np.arange(n_steps),
        timestamp=np.arange(n_steps) * 0.02,
        reward=np.ones(n_steps),
        reward_terms=reward_terms,
        _reward_term_names=np.array(term_names),
    )


def _mock_vlm_response(score: float = 0.7) -> str:
    return json.dumps(
        {
            "intent_score": score,
            "diagnosis": "ok",
            "failure_tags": [],
        }
    )


# ---------------------------------------------------------------------------
# StreamingJudge — start/stop lifecycle
# ---------------------------------------------------------------------------


def test_streaming_judge_starts_and_stops(tmp_path):
    iteration_dir = tmp_path / "run_001"
    iteration_dir.mkdir()

    sj = StreamingJudge(iteration_dir, "run forward", env_name="HalfCheetah", poll_interval=0.1)
    sj.start()

    assert sj._thread is not None
    assert sj._thread.is_alive()

    sj.stop()

    assert not sj._thread.is_alive()


def test_streaming_judge_results_empty_initially(tmp_path):
    iteration_dir = tmp_path / "run_001"
    iteration_dir.mkdir()

    sj = StreamingJudge(iteration_dir, "run forward", env_name="HalfCheetah", poll_interval=0.1)
    assert sj.results == {}


# ---------------------------------------------------------------------------
# StreamingJudge — detects and judges new checkpoints
# ---------------------------------------------------------------------------


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.8)),
)
def test_streaming_judge_detects_new_checkpoint(mock_vlm, mock_avail, tmp_path):
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "100000")

    sj = StreamingJudge(
        iteration_dir,
        "run forward",
        env_name="HalfCheetah",
        poll_interval=0.1,
        vlm_model="qwen3.5:27b",
    )
    sj.start()

    # Wait for the judge to process
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not sj.results:
        time.sleep(0.1)

    sj.stop()

    assert "100000" in sj.results
    assert sj.results["100000"]["intent_score"] > 0


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.6)),
)
def test_streaming_judge_skips_already_judged(mock_vlm, mock_avail, tmp_path):
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "100000")

    sj = StreamingJudge(
        iteration_dir,
        "run forward",
        env_name="HalfCheetah",
        poll_interval=0.1,
        vlm_model="qwen3.5:27b",
    )
    sj.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not sj.results:
        time.sleep(0.1)

    # Add same step's frame again — should not re-judge
    initial_call_count = mock_vlm.call_count
    time.sleep(0.3)

    sj.stop()

    assert mock_vlm.call_count == initial_call_count


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.75)),
)
def test_streaming_judge_detects_multiple_checkpoints(mock_vlm, mock_avail, tmp_path):
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _create_checkpoint(iteration_dir, "200000")
    _write_trajectory(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "200000")

    sj = StreamingJudge(
        iteration_dir,
        "run forward",
        env_name="HalfCheetah",
        poll_interval=0.1,
        vlm_model="qwen3.5:27b",
    )
    sj.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(sj.results) < 2:
        time.sleep(0.1)

    sj.stop()

    assert "100000" in sj.results
    assert "200000" in sj.results


# ---------------------------------------------------------------------------
# StreamingJudge — results are thread-safe
# ---------------------------------------------------------------------------


def test_streaming_judge_results_returns_copy(tmp_path):
    iteration_dir = tmp_path / "run_001"
    iteration_dir.mkdir()

    sj = StreamingJudge(iteration_dir, "run forward", env_name="HalfCheetah")
    # Manually inject a result
    sj._results["test"] = {"intent_score": 0.5}

    results = sj.results
    results["mutated"] = True

    assert "mutated" not in sj._results


# ---------------------------------------------------------------------------
# judge_all_checkpoints — streaming result reuse
# ---------------------------------------------------------------------------


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.9)),
)
def test_judge_all_checkpoints_reuses_streaming_results(mock_vlm, mock_avail, tmp_path):
    """Checkpoints already judged by StreamingJudge are not re-judged."""
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _create_checkpoint(iteration_dir, "200000")
    _write_trajectory(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "200000")

    # Pre-computed streaming result for step 100000
    streaming_results = {
        "100000": {
            "intent_score": 0.85,
            "passed": True,
            "diagnosis": "pre-computed",
            "failure_tags": [],
            "evidence": [],
            "reward_term_analysis": {},
            "vlm_score": 0.85,
            "scoring_method": "vlm",
        },
    }

    result = judge_all_checkpoints(
        "run forward",
        iteration_dir,
        {},
        vlm_model="qwen3.5:27b",
        streaming_results=streaming_results,
        env_name="HalfCheetah",
    )

    # Only step 200000 should be judged (100000 reused from streaming)
    # call_vlm_auto called once for 200000
    assert mock_vlm.call_count == 1
    assert "checkpoint_judgments" in result
    assert "100000" in result["checkpoint_judgments"]
    assert "200000" in result["checkpoint_judgments"]
    # The reused result preserves the original diagnosis
    assert result["checkpoint_judgments"]["100000"]["diagnosis"] == "pre-computed"


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.5)),
)
def test_judge_all_checkpoints_picks_best_across_streaming_and_new(mock_vlm, mock_avail, tmp_path):
    """Best checkpoint is picked correctly when mixing streaming and new results."""
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _create_checkpoint(iteration_dir, "200000")
    _write_trajectory(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "200000")

    # Streaming gave step 100000 a high score
    streaming_results = {
        "100000": {
            "intent_score": 0.95,
            "passed": True,
            "diagnosis": "excellent",
            "failure_tags": [],
            "evidence": [],
            "reward_term_analysis": {},
            "vlm_score": 0.95,
            "scoring_method": "vlm",
        },
    }

    result = judge_all_checkpoints(
        "run forward",
        iteration_dir,
        {},
        vlm_model="qwen3.5:27b",
        streaming_results=streaming_results,
        judgment_select="best",
        env_name="HalfCheetah",
    )

    # step 100000 (0.95) should be picked as best over 200000 (low score from mock)
    assert result["best_checkpoint"] == "100000"
    assert result["intent_score"] == 0.95


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.3)),
)
def test_judge_all_checkpoints_picks_last_over_best(mock_vlm, mock_avail, tmp_path):
    """judgment_select='last' picks the final checkpoint even when an earlier one scores higher."""
    iteration_dir = tmp_path / "run_001"
    _create_checkpoint(iteration_dir, "100000")
    _create_checkpoint(iteration_dir, "200000")
    _write_trajectory(iteration_dir, "100000")
    _write_trajectory(iteration_dir, "200000")

    # Streaming gave step 100000 a high score
    streaming_results = {
        "100000": {
            "intent_score": 0.95,
            "passed": True,
            "diagnosis": "excellent",
            "failure_tags": [],
            "evidence": [],
            "reward_term_analysis": {},
            "vlm_score": 0.95,
            "scoring_method": "vlm",
        },
    }

    result = judge_all_checkpoints(
        "run forward",
        iteration_dir,
        {},
        vlm_model="qwen3.5:27b",
        streaming_results=streaming_results,
        judgment_select="last",
        env_name="HalfCheetah",
    )

    # step 200000 should be picked (last) despite 100000 having higher score (0.95 vs 0.3)
    assert result["best_checkpoint"] == "200000"
    assert result["intent_score"] == 0.3
