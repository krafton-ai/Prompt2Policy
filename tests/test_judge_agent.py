"""Tests for behavior judge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from p2p.analysis.trajectory_metrics import analyze_trajectory
from p2p.inference.vlm import extract_json

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_valid():
    text = '{"intent_score": 0.8, "diagnosis": "ok", "failure_tags": []}'
    result = extract_json(text)
    assert result["intent_score"] == 0.8


def test_extract_json_fenced():
    text = '```json\n{"intent_score": 0.5, "diagnosis": "meh"}\n```'
    result = extract_json(text)
    assert result["intent_score"] == 0.5


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json("not json at all")


# ---------------------------------------------------------------------------
# reward term analysis
# ---------------------------------------------------------------------------


def _make_trajectory(n_steps: int = 100) -> list[dict]:
    """Create a synthetic trajectory with reward terms for testing."""
    trajectory = []
    for i in range(n_steps):
        trajectory.append(
            {
                "step": i,
                "timestamp": i * 0.02,
                "reward": 1.0 + 0.01 * i,
                "reward_terms": {"forward": 0.8, "energy": -0.2, "alive": 0.5},
            }
        )
    return trajectory


def test_analyze_trajectory_basic():
    traj = _make_trajectory(100)
    result = analyze_trajectory(traj)

    assert "forward" in result
    assert "energy" in result
    assert result["forward"]["fraction_of_total"] > 0


def test_analyze_trajectory_empty():
    result = analyze_trajectory([])
    assert result == {}


def test_judge_all_checkpoints_no_videos(tmp_path):
    """No eval videos returns zero-score result with no_eval_videos tag."""
    from p2p.agents.judge_agent import judge_all_checkpoints

    iteration_dir = tmp_path / "run_empty"
    (iteration_dir / "videos").mkdir(parents=True)

    result = judge_all_checkpoints("run fast", iteration_dir, {}, env_name="HalfCheetah")

    assert result["passed"] is False
    assert result["intent_score"] == 0.0
    assert result["scoring_method"] == "no_judge"
    assert result["failure_tags"] == ["no_eval_videos"]
    assert "No eval videos" in result["diagnosis"]


def test_synthesize_passes_traj_analysis_directly():
    """_synthesize stores traj_analysis dict as reward_term_analysis."""
    from p2p.agents.judge_agent import _synthesize

    traj_analysis = {
        "forward": {"mean": 1.5, "std": 0.2, "trend": "increasing", "fraction_of_total": 0.8},
    }
    vlm_result = {"intent_score": 0.7, "diagnosis": "ok", "failure_tags": []}

    result = _synthesize(traj_analysis, vlm_result, pass_threshold=0.5)

    assert result["reward_term_analysis"] is traj_analysis
    assert result["reward_term_analysis"]["forward"]["mean"] == 1.5


def _create_test_video(path: Path) -> None:
    """Create a minimal fake MP4 video with imageio."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=30)
    for i in range(10):
        frame = np.full((64, 64, 3), 50 + i * 20, dtype=np.uint8)
        writer.append_data(frame)
    writer.close()


def _mock_vlm_response(intent_score: float = 0.7, diagnosis: str = "ok") -> str:
    return json.dumps(
        {
            "intent_score": intent_score,
            "diagnosis": diagnosis,
            "failure_tags": [],
        }
    )


# ---------------------------------------------------------------------------
# Video path passing tests
# ---------------------------------------------------------------------------


def _create_iteration_with_video(tmp_path: Path, step: str = "204800") -> Path:
    """Create a fake run directory with video file and trajectory."""
    iteration_dir = tmp_path / "run_vid"
    videos_dir = iteration_dir / "videos"
    videos_dir.mkdir(parents=True)

    _create_test_video(videos_dir / f"eval_{step}.mp4")

    # Trajectory
    traj = _make_trajectory(50)
    traj_path = iteration_dir / "trajectory.npz"
    np.savez_compressed(
        traj_path,
        step=np.array([t["step"] for t in traj]),
        x_velocity=np.array([t.get("x_velocity", 0.0) for t in traj]),
        reward=np.array([t.get("reward", 0.0) for t in traj]),
        z_height=np.array([t.get("z_height", 0.0) for t in traj]),
    )

    return iteration_dir


@patch("p2p.agents.judge_agent._vllm_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.85)),
)
def test_judge_checkpoint_passes_video_path_for_vllm(mock_vlm, mock_vllm_avail, tmp_path):
    """When using vLLM model, video_path is passed to call_vlm_two_turn."""
    from p2p.agents.judge_agent import judge_all_checkpoints

    iteration_dir = _create_iteration_with_video(tmp_path)

    judge_all_checkpoints(
        "backflip",
        iteration_dir,
        {},
        vlm_model="vllm-Qwen/Qwen3.5-27B",
        env_name="HalfCheetah",
    )

    mock_vlm.assert_called_once()
    call_kwargs = mock_vlm.call_args.kwargs
    assert call_kwargs.get("video_path") is not None
    assert str(call_kwargs["video_path"]).endswith(".mp4")


@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.8)),
)
def test_judge_checkpoint_passes_video_path_for_gemini(mock_vlm, tmp_path):
    """When using Gemini model, video_path is passed to call_vlm_two_turn."""
    from p2p.agents.judge_agent import judge_all_checkpoints

    iteration_dir = _create_iteration_with_video(tmp_path)

    judge_all_checkpoints(
        "backflip",
        iteration_dir,
        {},
        vlm_model="gemini-3.1-pro-preview",
        env_name="HalfCheetah",
    )

    mock_vlm.assert_called_once()
    call_kwargs = mock_vlm.call_args.kwargs
    assert call_kwargs.get("video_path") is not None


@patch("p2p.agents.judge_agent._ollama_available", return_value=True)
@patch(
    "p2p.agents.judge_agent.call_vlm_two_turn",
    return_value=("mock criteria", _mock_vlm_response(0.7)),
)
def test_judge_checkpoint_no_video_for_ollama(mock_vlm, mock_avail, tmp_path):
    """Ollama provider does NOT receive video_path (image-only)."""
    from p2p.agents.judge_agent import judge_all_checkpoints

    iteration_dir = _create_iteration_with_video(tmp_path)

    judge_all_checkpoints(
        "backflip",
        iteration_dir,
        {},
        vlm_model="qwen3.5:27b",
        env_name="HalfCheetah",
    )

    mock_vlm.assert_called_once()
    call_kwargs = mock_vlm.call_args.kwargs
    assert call_kwargs.get("video_path") is None
