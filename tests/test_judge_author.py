"""Tests for code-based judge generation and execution."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from p2p.agents.judge_author import execute_judge_code, generate_judge_code
from p2p.prompts.judge_author import DECOMPOSE_REVIEW_MSG, build_judge_system_prompt
from p2p.training.env_spec import get_env_spec

# ---------------------------------------------------------------------------
# build_judge_system_prompt
# ---------------------------------------------------------------------------


def testbuild_judge_system_prompt_includes_env_details():
    env = get_env_spec("HalfCheetah-v5")
    prompt = build_judge_system_prompt("run forward fast", env)

    assert "HalfCheetah-v5" in prompt
    assert "run forward fast" in prompt
    assert "obs_dim" in prompt or "17" in prompt
    assert "action_dim" in prompt or "6" in prompt


def testbuild_judge_system_prompt_auto_detects_mujoco():
    """MuJoCo envs automatically get body layout and extended schema."""
    env = get_env_spec("HalfCheetah-v5")
    prompt = build_judge_system_prompt("backflip", env)

    assert "MuJoCo robotics evaluator" in prompt
    assert "xpos" in prompt
    assert "xquat" in prompt
    assert "Body Layout" in prompt


def testbuild_judge_system_prompt_includes_intent():
    env = get_env_spec("Ant-v5")
    prompt = build_judge_system_prompt("walk in a circle", env)

    assert "walk in a circle" in prompt
    assert "Ant-v5" in prompt


def testbuild_judge_system_prompt_includes_max_episode_steps():
    """max_episode_steps appears in env info and function contract."""
    env = get_env_spec("HalfCheetah-v5")
    prompt = build_judge_system_prompt("run forward", env, max_episode_steps=300)

    assert "max_episode_steps: 300" in prompt
    # The old vague "200-1000" should not appear
    assert "200-1000" not in prompt
    # Survival fraction example should use the actual value
    assert "len(trajectory) / 300" in prompt


def test_decompose_review_checks_survival():
    """Decomposition review prompt includes a check for naive step-count survival."""
    msg = DECOMPOSE_REVIEW_MSG.format(intent="run forward", remaining=3)
    assert "No naive step-count survival" in msg
    assert "episode length" in msg


def testbuild_judge_system_prompt_includes_no_survival_rule():
    """System prompt forbids naive step-count survival but allows physics-based criteria."""
    env = get_env_spec("HalfCheetah-v5")
    prompt = build_judge_system_prompt("do a backflip", env)

    assert "No naive step-count survival" in prompt
    assert "Physics-based survival" in prompt


def testbuild_judge_system_prompt_handles_unknown_env():
    """Non-MuJoCo envs produce a valid prompt without body layout."""
    from p2p.training.env_spec import EnvSpec

    fake_env = EnvSpec(
        env_id="FakeEnv-v1",
        name="FakeEnv",
        obs_dim=10,
        action_dim=3,
        info_keys={"x_velocity": "speed"},
        description="A fake environment",
    )
    prompt = build_judge_system_prompt("do something", fake_env)

    assert "FakeEnv-v1" in prompt
    assert "do something" in prompt
    # No MuJoCo body layout section — auto-detected as non-MuJoCo
    assert "Body Layout" not in prompt
    assert "RL trajectory evaluator" in prompt


# ---------------------------------------------------------------------------
# generate_judge_code (mocked Claude client)
# ---------------------------------------------------------------------------

_MOCK_JUDGE_CODE = """\
def judge_fn(trajectory, summary):
    if not trajectory:
        return {
            "intent_score": 0.0, "diagnosis": "empty",
            "failure_tags": ["no_data"],
        }
    avg_speed = sum(t.get("x_velocity", 0) for t in trajectory) / len(trajectory)
    score = min(1.0, avg_speed / 10.0)
    return {
        "intent_score": score,
        "diagnosis": f"Average speed: {avg_speed:.2f}",
        "failure_tags": [] if score > 0.5 else ["too_slow"],
    }
"""

_MOCK_DECOMPOSITION = """\
```
Structure:
1 -> 2

Legend:
1. Forward Speed
2. Stability

Scoring Criteria:
  1 Forward Speed
      metric: average x_velocity
      threshold: >= 5.0 m/s
      partial: linear from 0 to threshold
  2 Stability
      metric: std of z_height
      threshold: < 0.1m
      partial: linear decay from threshold to 2x threshold
```
"""


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock LLM response with the given text content."""
    response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 100
    response.usage.output_tokens = 200
    return response


def _make_mock_client(code: str = _MOCK_JUDGE_CODE) -> MagicMock:
    """Create a mock Anthropic client that returns decomposition -> LGTM -> code -> LGTM."""
    client = MagicMock()
    client.messages.create.side_effect = [
        _make_mock_response(_MOCK_DECOMPOSITION),
        _make_mock_response("LGTM"),
        _make_mock_response(f"```python\n{code}\n```"),
        _make_mock_response("LGTM"),
    ]
    return client


def test_generate_judge_code_returns_valid_code():
    client = _make_mock_client()
    code = generate_judge_code(
        "run forward", client=client, model="claude-opus-4-6", env=get_env_spec("HalfCheetah-v5")
    )

    assert "def judge_fn" in code
    assert client.messages.create.call_count == 4


def test_generate_judge_code_uses_env():
    client = _make_mock_client()
    env = get_env_spec("Ant-v5")
    generate_judge_code("walk", client=client, model="claude-opus-4-6", env=env)

    call_kwargs = client.messages.create.call_args_list[0].kwargs
    assert "Ant-v5" in call_kwargs["system"]


# ---------------------------------------------------------------------------
# execute_judge_code
# ---------------------------------------------------------------------------


def test_execute_judge_code_valid():
    """Valid judge code returns correct shape."""
    code = """\
def judge_fn(trajectory, summary):
    score = 0.75
    return {
        "intent_score": score,
        "diagnosis": "looks good",
        "failure_tags": [],
    }
"""
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.75
    assert result["diagnosis"] == "looks good"
    assert isinstance(result["failure_tags"], list)


def test_execute_judge_code_uses_trajectory_data():
    """Judge code can process trajectory data."""
    code = """\
def judge_fn(trajectory, summary):
    if not trajectory:
        return {"intent_score": 0.0, "diagnosis": "empty", "failure_tags": []}
    avg_vel = sum(t.get("x_velocity", 0) for t in trajectory) / len(trajectory)
    return {
        "intent_score": min(1.0, avg_vel / 10.0),
        "diagnosis": f"avg velocity {avg_vel:.1f}",
        "failure_tags": [],
    }
"""
    traj = [{"x_velocity": 5.0, "step": i} for i in range(10)]
    result = execute_judge_code(code, traj, {})

    assert result["intent_score"] == pytest.approx(0.5)


def test_execute_judge_code_clamps_score_above_1():
    code = """\
def judge_fn(trajectory, summary):
    return {"intent_score": 5.0, "diagnosis": "too high", "failure_tags": []}
"""
    result = execute_judge_code(code, [], {})
    assert result["intent_score"] == 1.0


def test_execute_judge_code_clamps_score_below_0():
    code = """\
def judge_fn(trajectory, summary):
    return {"intent_score": -2.0, "diagnosis": "negative", "failure_tags": []}
"""
    result = execute_judge_code(code, [], {})
    assert result["intent_score"] == 0.0


def test_execute_judge_code_handles_syntax_error():
    code = "def judge_fn(trajectory, summary:\n  return {}"
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.0
    assert "compilation" in result["diagnosis"].lower() or "error" in result["diagnosis"].lower()


def test_execute_judge_code_handles_runtime_error():
    code = """\
def judge_fn(trajectory, summary):
    raise ValueError("intentional error")
"""
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.0
    assert "runtime" in result["diagnosis"].lower() or "error" in result["diagnosis"].lower()


def test_execute_judge_code_handles_missing_fn():
    code = "x = 42"
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.0
    assert "missing" in result["diagnosis"].lower()


def test_execute_judge_code_handles_non_dict_return():
    code = """\
def judge_fn(trajectory, summary):
    return 0.5
"""
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.0
    assert "dict" in result["diagnosis"].lower()


def test_execute_judge_code_uses_numpy():
    """Judge code can use numpy."""
    code = """\
import numpy as np
def judge_fn(trajectory, summary):
    arr = np.array([0.1, 0.2, 0.3])
    return {
        "intent_score": float(np.mean(arr)),
        "diagnosis": "ok", "failure_tags": [],
    }
"""
    result = execute_judge_code(code, [], {})
    assert result["intent_score"] == pytest.approx(0.2)


def test_execute_judge_code_defaults_missing_keys():
    """Missing keys in the result get sensible defaults."""
    code = """\
def judge_fn(trajectory, summary):
    return {"intent_score": 0.6}
"""
    result = execute_judge_code(code, [], {})

    assert result["intent_score"] == 0.6
    assert result["diagnosis"] == ""
    assert result["failure_tags"] == []


# ---------------------------------------------------------------------------
# Integration: judge_agent.py code-judge routing
# ---------------------------------------------------------------------------


def test_judge_checkpoint_code_judge_routing(tmp_path):
    """_judge_single_checkpoint routes to code judge when vlm_model='code-judge'."""
    from p2p.agents.judge_agent import _judge_single_checkpoint

    # Create minimal iteration structure
    iteration_dir = tmp_path / "run_001"
    frames_dir = iteration_dir / "videos" / "frames"
    frames_dir.mkdir(parents=True)

    # Create a trajectory file
    traj = [{"step": i, "x_velocity": 8.0, "reward": 1.0, "z_height": 0.5} for i in range(50)]
    traj_path = iteration_dir / "trajectory.npz"
    np.savez_compressed(
        traj_path,
        step=np.array([t["step"] for t in traj]),
        x_velocity=np.array([t["x_velocity"] for t in traj]),
        reward=np.array([t["reward"] for t in traj]),
        z_height=np.array([t["z_height"] for t in traj]),
    )

    # Create a dummy video file
    videos_dir = iteration_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / "eval_204800.mp4"
    video_path.write_bytes(b"dummy")

    judge_code = """\
def judge_fn(trajectory, summary):
    score = 0.85 if len(trajectory) > 10 else 0.1
    return {
        "intent_score": score, "diagnosis": "code judge",
        "failure_tags": [],
    }
"""

    result = _judge_single_checkpoint(
        prompt="run fast",
        iteration_dir=iteration_dir,
        video_path=video_path,
        step_label="204800",
        pass_threshold=0.7,
        env_name="HalfCheetah",
        vlm_model="code-judge",
        judge_code=judge_code,
    )

    # The code judge should have been used (vlm_score should come from code judge)
    assert result["intent_score"] > 0
    assert "scoring_method" in result
