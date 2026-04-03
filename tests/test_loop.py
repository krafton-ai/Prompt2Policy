"""Tests for the orchestrator loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic

from p2p.config import LoopConfig
from p2p.contracts import ReviseResult
from p2p.session.iteration_record import IterationData
from p2p.session.loop import run_loop

_DUMMY_CODE = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
_DUMMY_REVISE: ReviseResult = {
    "reward_code": _DUMMY_CODE,
    "diagnosis": "Agent is static, needs rotation signal.",
    "reward_reasoning": "Added rotation term.",
    "hp_reasoning": "Entropy healthy, no HP changes.",
    "hp_changes": {},
    "training_dynamics": "## Training Dynamics\nstable",
}
_DUMMY_SUMMARY = {
    "final_episodic_return": 500.0,
    "total_timesteps": 100000,
    "training_time_s": 30.0,
    "total_episodes": 50,
}


def _make_judgment(intent_score: float, passed: bool) -> dict:
    return {
        "intent_score": intent_score,
        "passed": passed,
        "diagnosis": "ok",
        "failure_tags": [],
        "evidence": [],
    }


def _setup_mocks(judgments: list[dict], tmp_path: Path):
    """Set up all mocks for the loop, returning (patches, client)."""
    client = MagicMock()
    judgment_iter = iter(judgments)

    def fake_run_iteration(**kwargs):
        iteration = kwargs["iteration"]
        iteration_dir = kwargs["iteration_dir"]
        reward_code = kwargs["reward_code"]
        judgment = next(judgment_iter)

        record = IterationData(
            iteration=iteration,
            iteration_dir=str(iteration_dir),
            reward_code=reward_code,
            summary=_DUMMY_SUMMARY,
            judgment=judgment,
            is_multi_config=True,
        )
        return record, judgment

    patches = {
        "generate": patch("p2p.session.loop.generate", return_value=_DUMMY_CODE),
        "revise_multi": patch("p2p.session.loop.revise_multi", return_value=[_DUMMY_REVISE]),
        "run_iteration": patch("p2p.session.loop.run_iteration", side_effect=fake_run_iteration),
    }
    return patches, client


def test_loop_passes_on_first_try(tmp_path):
    judgments = [_make_judgment(0.9, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"] as mock_revise,
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=5, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    assert result["status"] == "passed"
    assert len(result["iterations"]) == 1
    assert result["best_score"] == 0.9
    mock_revise.assert_not_called()


def test_loop_reaches_max_iterations(tmp_path):
    judgments = [_make_judgment(0.3, False)] * 3
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"] as mock_revise,
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=3, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    assert result["status"] == "max_iterations"
    assert len(result["iterations"]) == 3
    # Revise called after each failed iteration except the last
    assert mock_revise.call_count == 2


def test_loop_revision_flow(tmp_path):
    """Pass on second iteration after revision."""
    judgments = [_make_judgment(0.4, False), _make_judgment(0.8, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"] as mock_revise,
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=5, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    assert result["status"] == "passed"
    assert len(result["iterations"]) == 2
    assert result["best_iteration"] == 2
    assert result["best_score"] == 0.8
    mock_revise.assert_called_once()


def test_loop_saves_history(tmp_path):
    judgments = [_make_judgment(0.9, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"],
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=5, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    history_path = tmp_path / result["session_id"] / "loop_history.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert history["status"] == "passed"
    assert len(history["iterations"]) == 1


def test_loop_tracks_best_across_iterations(tmp_path):
    """Best score should be tracked even if later iterations score lower."""
    judgments = [
        _make_judgment(0.5, False),
        _make_judgment(0.6, False),
        _make_judgment(0.4, False),
    ]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"],
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=3, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    assert result["best_iteration"] == 2
    assert result["best_score"] == 0.6


def test_loop_persists_revise_fields(tmp_path):
    """Revise diagnosis and reasoning fields should be wired to history."""
    judgments = [_make_judgment(0.4, False), _make_judgment(0.8, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"],
        patches["run_iteration"],
    ):
        result = run_loop(
            "run fast",
            LoopConfig(max_iterations=5, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )

    # Check in-memory record (iterations are serialized dicts)
    it1 = result["iterations"][0]
    assert it1["revise_diagnosis"] == "Agent is static, needs rotation signal."
    assert it1["reward_reasoning"] == "Added rotation term."
    assert it1["hp_reasoning"] == "Entropy healthy, no HP changes."
    assert it1["training_dynamics"] != ""

    # Check persisted loop_history.json
    history_path = tmp_path / result["session_id"] / "loop_history.json"
    history = json.loads(history_path.read_text())
    saved_it1 = history["iterations"][0]
    assert saved_it1["revise_diagnosis"] == "Agent is static, needs rotation signal."
    assert saved_it1["reward_reasoning"] == "Added rotation term."
    assert saved_it1["hp_reasoning"] == "Entropy healthy, no HP changes."


# ---------------------------------------------------------------------------
# Exception handling tests (#13)
# ---------------------------------------------------------------------------


def _run_loop_with_generate_error(tmp_path, error):
    """Helper: run loop where generate() raises the given error."""
    client = MagicMock()
    with (
        patch("p2p.session.loop.generate", side_effect=error),
        patch("p2p.session.loop.revise_multi", return_value=[_DUMMY_REVISE]),
        patch("p2p.session.loop.run_iteration"),
    ):
        return run_loop(
            "run fast",
            LoopConfig(max_iterations=3, runs_dir=tmp_path, pass_threshold=0.7),
            client=client,
        )


def test_loop_rate_limit_error_sets_status(tmp_path):
    """RateLimitError should set status to 'rate_limited'."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    error = anthropic.RateLimitError(
        message="rate limited",
        response=mock_response,
        body=None,
    )

    result = _run_loop_with_generate_error(tmp_path, error)

    assert result["status"] == "rate_limited"
    assert "rate limit" in result["error"].lower()


def test_loop_auth_error_sets_status(tmp_path):
    """AuthenticationError should set status to 'auth_error'."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    error = anthropic.AuthenticationError(
        message="invalid api key",
        response=mock_response,
        body=None,
    )

    result = _run_loop_with_generate_error(tmp_path, error)

    assert result["status"] == "auth_error"
    assert "authentication" in result["error"].lower()


def test_loop_syntax_error_sets_status(tmp_path):
    """SyntaxError from generated code should set status to 'invalid_code'."""
    error = SyntaxError("unexpected EOF while parsing")

    result = _run_loop_with_generate_error(tmp_path, error)

    assert result["status"] == "invalid_code"
    assert "syntax error" in result["error"].lower() or "Generated code" in result["error"]


def test_loop_value_error_falls_through_to_generic_error(tmp_path):
    """ValueError is too broad for invalid_code — should fall to generic error."""
    error = ValueError("No valid Python code found in LLM response")

    result = _run_loop_with_generate_error(tmp_path, error)

    assert result["status"] == "error"
    assert "ValueError" in result["error"]
    assert "No valid Python code" in result["error"]


def test_loop_unexpected_error_preserves_traceback(tmp_path):
    """Unexpected exceptions should preserve traceback in error field."""
    error = RuntimeError("something went very wrong")

    result = _run_loop_with_generate_error(tmp_path, error)

    assert result["status"] == "error"
    assert "RuntimeError" in result["error"]
    assert "something went very wrong" in result["error"]
    # Traceback should be included
    assert "Traceback" in result["error"]


def test_loop_error_persists_to_history(tmp_path):
    """Error status and message should be persisted to loop_history.json."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    error = anthropic.AuthenticationError(
        message="bad key",
        response=mock_response,
        body=None,
    )

    result = _run_loop_with_generate_error(tmp_path, error)

    history_path = tmp_path / result["session_id"] / "loop_history.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert history["status"] == "auth_error"
    assert history["error"] is not None


# ---------------------------------------------------------------------------
# Trainer integration tests (#220)
# ---------------------------------------------------------------------------


def test_loop_constructs_default_local_trainer(tmp_path):
    """run_loop() should construct LocalTrainer from LoopConfig when no trainer given."""
    judgments = [_make_judgment(0.9, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    with (
        patches["generate"],
        patches["revise_multi"],
        patches["run_iteration"] as mock_run_iter,
    ):
        run_loop(
            "run fast",
            LoopConfig(
                max_iterations=1,
                runs_dir=tmp_path,
                cores_per_run=4,
                max_parallel=2,
                cores_pool=[0, 1, 2, 3],
            ),
            client=client,
        )

    # Verify trainer was passed to run_iteration
    call_kwargs = mock_run_iter.call_args.kwargs
    trainer = call_kwargs["trainer"]
    from p2p.training.trainer import LocalTrainer

    assert isinstance(trainer, LocalTrainer)
    assert trainer._cores_per_run == 4
    assert trainer._max_parallel == 2
    assert trainer._cores_pool == [0, 1, 2, 3]


def test_loop_uses_injected_trainer(tmp_path):
    """run_loop() should forward an injected trainer to run_iteration."""
    judgments = [_make_judgment(0.9, True)]
    patches, client = _setup_mocks(judgments, tmp_path)

    mock_trainer = MagicMock()

    with (
        patches["generate"],
        patches["revise_multi"],
        patches["run_iteration"] as mock_run_iter,
    ):
        run_loop(
            "run fast",
            LoopConfig(max_iterations=1, runs_dir=tmp_path),
            client=client,
            trainer=mock_trainer,
        )

    call_kwargs = mock_run_iter.call_args.kwargs
    assert call_kwargs["trainer"] is mock_trainer
