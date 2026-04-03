"""Tests for the revise agent module."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from p2p.agents.revise_agent import (
    _parse_phase1_response,
    _parse_revise_response,
    _resolve_base_code,
    _run_phase2,
    revise,
    validate_hp_changes,
)
from p2p.agents.revise_tool_dispatch import ReviseToolDispatch
from p2p.analysis.training_dynamics import (
    _empty_dynamics,
    analyze_training_curves,
    format_current_config,
    format_iteration_history,
    format_training_dynamics,
)
from p2p.config import TrainConfig
from p2p.training.env_spec import get_env_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_scalars(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _make_scalars_entries(n: int = 20) -> list[dict]:
    """Generate synthetic scalars.jsonl entries with plausible PPO metrics."""
    entries = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        entries.append(
            {
                "global_step": i * 2048,
                "iteration": i,
                "entropy": -2.0 * (1 - 0.5 * frac),  # decays from -2.0 to -1.0
                "value_loss": 10.0 * (1 - 0.6 * frac),  # decreases
                "policy_loss": -0.02 * (1 + 0.5 * frac),
                "approx_kl": 0.005 + 0.001 * frac,
                "clip_fraction": 0.1 - 0.02 * frac,
                "explained_variance": 0.3 + 0.4 * frac,  # 0.3 -> 0.7
                "episodic_return": 100 + 200 * frac,  # 100 -> 300
                "sps": 5000,
                "learning_rate": 3e-4,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# test_analyze_training_curves
# ---------------------------------------------------------------------------


class TestAnalyzeTrainingCurves:
    def test_basic_analysis(self, tmp_path: Path):
        entries = _make_scalars_entries(20)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        dyn = analyze_training_curves(scalars_path)

        assert dyn["num_entries"] == 20
        assert dyn["entropy_initial"] == entries[0]["entropy"]
        assert dyn["entropy_final"] == entries[-1]["entropy"]
        assert dyn["episodic_return_final"] == pytest.approx(entries[-1]["episodic_return"])
        assert dyn["sps_mean"] == pytest.approx(5000.0)

    def test_skips_eval_entries(self, tmp_path: Path):
        entries = _make_scalars_entries(5)
        entries.append({"global_step": 99999, "type": "eval", "total_reward": 500})
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        dyn = analyze_training_curves(scalars_path)
        assert dyn["num_entries"] == 5  # eval entry skipped

    def test_empty_file(self, tmp_path: Path):
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        scalars_path.parent.mkdir(parents=True, exist_ok=True)
        scalars_path.write_text("")

        dyn = analyze_training_curves(scalars_path)
        assert dyn["num_entries"] == 0

    def test_missing_file(self, tmp_path: Path):
        dyn = analyze_training_curves(tmp_path / "missing.jsonl")
        assert dyn["num_entries"] == 0

    def test_entropy_too_fast(self, tmp_path: Path):
        """Entropy decaying >90% should flag entropy_too_fast."""
        entries = []
        for i in range(10):
            frac = i / 9
            entries.append(
                {
                    "global_step": i * 2048,
                    "iteration": i,
                    "entropy": -5.0 * (1 - 0.95 * frac),  # 95% decay
                    "episodic_return": 100,
                    "sps": 5000,
                }
            )
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        dyn = analyze_training_curves(scalars_path)
        assert dyn["entropy_too_fast"]

    def test_explained_variance_good(self, tmp_path: Path):
        entries = _make_scalars_entries(20)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        dyn = analyze_training_curves(scalars_path)
        # Last entry has explained_variance = 0.7 > 0.5
        assert dyn["explained_variance_good"]


# ---------------------------------------------------------------------------
# test_format_training_dynamics
# ---------------------------------------------------------------------------


class TestFormatTrainingDynamics:
    def test_has_expected_sections(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["entropy_initial"] = -2.0
        dyn["entropy_final"] = -1.5
        dyn["entropy_trend"] = "decreasing"
        dyn["entropy_decay_rate"] = 0.25
        dyn["value_loss_initial"] = 10.0
        dyn["value_loss_final"] = 4.0
        dyn["value_loss_trend"] = "decreasing"
        dyn["value_loss_stability"] = 0.1
        dyn["episodic_return_final"] = 300.0
        dyn["episodic_return_max"] = 300.0
        dyn["episodic_return_trend"] = "increasing"
        dyn["sps_mean"] = 5000.0
        text = format_training_dynamics(dyn)
        assert "Training Dynamics Analysis" in text
        assert "Entropy" in text
        assert "Value loss" in text
        assert "Episodic return" in text
        assert "Throughput" in text

    def test_empty_dynamics(self):
        text = format_training_dynamics(_empty_dynamics())
        assert "No training dynamics data" in text

    def test_warnings_appear(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["entropy_too_fast"] = True
        dyn["value_loss_diverging"] = True
        dyn["explained_variance_final"] = 0.2
        dyn["explained_variance_good"] = False
        dyn["approx_kl_spike_count"] = 10
        text = format_training_dynamics(dyn)
        assert "WARNING" in text
        assert "collapsed" in text.lower() or "entropy" in text.lower()


# ---------------------------------------------------------------------------
# test_parse_revise_response
# ---------------------------------------------------------------------------


class TestParseReviseResponse:
    _VALID_RESPONSE = textwrap.dedent("""\
        ## Reward Reasoning
        The previous reward overweights velocity. We need more rotation emphasis.

        ## Revised Reward Function
        ```python
        def reward_fn(obs, action, next_obs, info):
            \"\"\"LaTeX: r = v + w\"\"\"
            return 1.0, {"speed": 1.0}
        ```

        ## HP Reasoning
        Entropy is collapsing, increase ent_coef.

        ## HP Changes
        ```json
        {"ent_coef": 0.05, "learning_rate": 1e-4}
        ```
    """)

    _VALID_RESPONSE_WITH_DIAGNOSIS = textwrap.dedent("""\
        ## Diagnosis
        1. Behavioral: Agent runs forward but does not rotate backward.
        2. Reward root-cause: velocity term (70% of total) dominates; rotation term is too small.
        3. Training dynamics: explained_variance=0.65, entropy decay=45% — optimizer is healthy.
        4. Cross-iteration: Iter 1 had no rotation term. Iter 2 added it at 10% weight.
        5. Proposal: Triple rotation weight to 50% of total, keep velocity as secondary.

        ## Reward Reasoning
        The previous reward overweights velocity. We need more rotation emphasis.

        ## Revised Reward Function
        ```python
        def reward_fn(obs, action, next_obs, info):
            \"\"\"LaTeX: r = v + 3w\"\"\"
            return 4.0, {"speed": 1.0, "rotation": 3.0}
        ```

        ## HP Reasoning
        Training dynamics are healthy — no HP changes.

        ## HP Changes
        ```json
        {}
        ```
    """)

    def test_valid_response(self):
        result = _parse_revise_response(self._VALID_RESPONSE)
        assert "def reward_fn" in result["reward_code"]
        assert result["reward_reasoning"] != ""
        assert result["hp_reasoning"] != ""
        assert result["hp_changes"] == {"ent_coef": 0.05, "learning_rate": 1e-4}
        # Old format without Diagnosis — should be empty string
        assert result["diagnosis"] == ""

    def test_diagnosis_parsed(self):
        result = _parse_revise_response(self._VALID_RESPONSE_WITH_DIAGNOSIS)
        assert "def reward_fn" in result["reward_code"]
        assert result["diagnosis"] != ""
        assert "Behavioral" in result["diagnosis"]
        assert "rotation" in result["diagnosis"].lower()
        assert result["reward_reasoning"] != ""
        assert result["hp_changes"] == {}

    def test_missing_code_raises(self):
        text = textwrap.dedent("""\
            ## Reward Reasoning
            Something

            ## HP Reasoning
            Nothing

            ## HP Changes
            ```json
            {}
            ```
        """)
        with pytest.raises(ValueError, match="No reward function code"):
            _parse_revise_response(text)

    def test_missing_hp_section(self):
        text = textwrap.dedent("""\
            ## Reward Reasoning
            Change weights.

            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                return 0.0, {}
            ```
        """)
        result = _parse_revise_response(text)
        assert result["hp_changes"] == {}
        assert result["reward_code"] != ""

    def test_invalid_hp_json_ignored(self):
        text = textwrap.dedent("""\
            ## Reward Reasoning
            Fix it.

            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                return 0.0, {}
            ```

            ## HP Reasoning
            Something.

            ## HP Changes
            ```json
            {not valid json}
            ```
        """)
        result = _parse_revise_response(text)
        assert result["hp_changes"] == {}

    def test_stateful_closure_parsed(self):
        """Stateful _make_reward closure should be parsed and get assignment appended."""
        text = textwrap.dedent("""\
            ## Reward Reasoning
            Task requires phased reward for jump-then-rotate.

            ## Revised Reward Function
            ```python
            def _make_reward():
                state = {"step": 0, "jumped": False}
                def reward_fn(obs, action, next_obs, info):
                    state["step"] += 1
                    return 1.0, {"step": float(state["step"])}
                return reward_fn
            ```

            ## HP Reasoning
            No changes.

            ## HP Changes
            ```json
            {}
            ```
        """)
        result = _parse_revise_response(text)
        assert "_make_reward" in result["reward_code"]
        assert "reward_fn = _make_reward()" in result["reward_code"]

    def test_stateful_closure_no_duplicate_assignment(self):
        """If LLM already includes module-level assignment, don't duplicate."""
        text = textwrap.dedent("""\
            ## Reward Reasoning
            Phased reward.

            ## Revised Reward Function
            ```python
            def _make_reward():
                state = {"step": 0}
                def reward_fn(obs, action, next_obs, info):
                    return 0.0, {}
                return reward_fn
            reward_fn = _make_reward()
            ```

            ## HP Reasoning
            No changes.

            ## HP Changes
            ```json
            {}
            ```
        """)
        result = _parse_revise_response(text)
        assert result["reward_code"].count("reward_fn = _make_reward()") == 1


# ---------------------------------------------------------------------------
# test_validate_hp_changes
# ---------------------------------------------------------------------------


class TestValidateHpChanges:
    def test_unsafe_params_stripped(self):
        changes = {
            "learning_rate": 1e-4,
            "env_id": "Ant-v5",  # unsafe
            "seed": 999,  # unsafe
            "device": "cuda",  # unsafe
        }
        cleaned = validate_hp_changes(changes)
        assert "env_id" not in cleaned
        assert "seed" not in cleaned
        assert "device" not in cleaned
        assert cleaned["learning_rate"] == 1e-4

    def test_bounds_clamped(self):
        changes = {
            "learning_rate": 1.0,  # too high, max 3e-3
            "ent_coef": -0.5,  # too low, min 0.0001
            "total_timesteps": 5,  # too low, min 500_000
        }
        cleaned = validate_hp_changes(changes)
        assert cleaned["learning_rate"] <= 3e-3
        assert cleaned["ent_coef"] >= 0.0001
        assert cleaned["total_timesteps"] >= 500_000

    def test_int_types_preserved(self):
        changes = {
            "num_steps": 512.0,
            "update_epochs": 5.0,
        }
        cleaned = validate_hp_changes(changes)
        assert isinstance(cleaned["num_steps"], int)
        assert isinstance(cleaned["update_epochs"], int)

    def test_empty_dict(self):
        assert validate_hp_changes({}) == {}

    def test_bool_params_pass_through(self):
        changes = {"normalize_obs": False, "normalize_reward": True}
        cleaned = validate_hp_changes(changes)
        assert cleaned["normalize_obs"] is False
        assert cleaned["normalize_reward"] is True

    def test_non_bool_values_stripped_for_bool_params(self):
        changes = {"normalize_obs": 1, "normalize_reward": "yes", "learning_rate": 1e-4}
        cleaned = validate_hp_changes(changes)
        assert "normalize_obs" not in cleaned
        assert "normalize_reward" not in cleaned
        assert cleaned["learning_rate"] == 1e-4


# ---------------------------------------------------------------------------
# test_config_apply_updates
# ---------------------------------------------------------------------------


class TestConfigApplyUpdates:
    def test_basic_update(self):
        cfg = TrainConfig(learning_rate=3e-4, ent_coef=0.01)
        new_cfg = cfg.apply_updates({"learning_rate": 1e-4, "ent_coef": 0.05})
        assert new_cfg.learning_rate == 1e-4
        assert new_cfg.ent_coef == 0.05
        # Original unchanged
        assert cfg.learning_rate == 3e-4

    def test_derived_fields_recomputed(self):
        cfg = TrainConfig(num_envs=4, num_steps=512, total_timesteps=100_000)
        new_cfg = cfg.apply_updates({"num_steps": 1024})
        assert new_cfg.num_steps == 1024
        assert new_cfg.batch_size == 4 * 1024
        assert new_cfg.num_iterations == 100_000 // (4 * 1024)

    def test_unsafe_params_ignored(self):
        cfg = TrainConfig(env_id="HalfCheetah-v5", seed=1)
        new_cfg = cfg.apply_updates({"env_id": "Ant-v5", "seed": 999, "learning_rate": 1e-4})
        assert new_cfg.env_id == "HalfCheetah-v5"
        assert new_cfg.seed == 1
        assert new_cfg.learning_rate == 1e-4

    def test_empty_updates(self):
        cfg = TrainConfig()
        new_cfg = cfg.apply_updates({})
        assert new_cfg.learning_rate == cfg.learning_rate

    def test_values_within_bounds_kept(self):
        cfg = TrainConfig()
        new_cfg = cfg.apply_updates({"learning_rate": 1e-3, "ent_coef": 0.05})
        assert new_cfg.learning_rate == 1e-3
        assert new_cfg.ent_coef == 0.05

    def test_values_above_max_clamped(self):
        cfg = TrainConfig()
        # HP_BOUNDS["learning_rate"] = (1e-6, 3e-3)
        new_cfg = cfg.apply_updates({"learning_rate": 100.0})
        assert new_cfg.learning_rate == 3e-3

    def test_values_below_min_clamped(self):
        cfg = TrainConfig()
        # HP_BOUNDS["ent_coef"] = (0.0001, 0.1)
        new_cfg = cfg.apply_updates({"ent_coef": -5.0})
        assert new_cfg.ent_coef == 0.0001

    def test_int_type_preserved_when_clamped(self):
        cfg = TrainConfig()
        # HP_BOUNDS["num_steps"] = (64, 8192); num_steps is int
        new_cfg = cfg.apply_updates({"num_steps": 99999})
        assert new_cfg.num_steps == 8192
        assert isinstance(new_cfg.num_steps, int)

    def test_none_bound_skipped(self):
        """When HP_BOUNDS has None for a bound, clamping should skip that side."""
        from p2p.config import HP_BOUNDS

        original = HP_BOUNDS.get("learning_rate")
        HP_BOUNDS["learning_rate"] = (1e-6, None)  # no upper bound
        try:
            cfg = TrainConfig()
            updated = cfg.apply_updates({"learning_rate": 999.0})
            assert updated.learning_rate == 999.0
        finally:
            if original is not None:
                HP_BOUNDS["learning_rate"] = original

    def test_non_numeric_keys_not_affected(self):
        cfg = TrainConfig()
        new_cfg = cfg.apply_updates({"net_arch": [128, 128]})
        assert new_cfg.net_arch == [128, 128]


# ---------------------------------------------------------------------------
# test_format helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_iteration_history_empty(self):
        history_text, best_section = format_iteration_history([])
        assert "No previous" in history_text
        assert best_section == ""

    def test_format_iteration_history_with_data(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {
                    "intent_score": 0.3,
                    "failure_tags": ["static_behavior"],
                    "diagnosis": "stuck",
                },
                "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 0.0, {}",
                "reward_reasoning": "Initial attempt",
                "hp_reasoning": "",
                "hp_changes": {},
            },
        ]
        history_text, best_section = format_iteration_history(iterations)
        # All iterations in thin block format
        assert "Iter 1" in history_text
        assert "0.30" in history_text
        assert "static_behavior" in history_text
        assert "Score trend" in history_text

    def test_format_iteration_history_best_marker(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {"intent_score": 0.3, "failure_tags": [], "diagnosis": "bad"},
                "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 0.0, {}",
            },
            {
                "iteration": 2,
                "judgment": {"intent_score": 0.8, "failure_tags": [], "diagnosis": "good"},
                "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}",
            },
        ]
        history_text, best_section = format_iteration_history(
            iterations, best_iteration=2, best_score=0.8
        )
        assert "Best: iter 2" in history_text
        assert "(best)" in history_text  # best marker in block
        assert "Best Iteration Reference" in best_section

    def test_format_current_config(self):
        cfg = TrainConfig(learning_rate=1e-3, ent_coef=0.02)
        text = format_current_config(cfg)
        assert "learning_rate" in text
        assert "ent_coef" in text
        assert "Hyperparameters" in text


# ---------------------------------------------------------------------------
# test_revise (mocked LLM)
# ---------------------------------------------------------------------------


class TestRunReviseMocked:
    def test_end_to_end(self, tmp_path: Path):
        # Set up scalars
        entries = _make_scalars_entries(10)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        # Mock Anthropic client
        mock_response_text = textwrap.dedent("""\
            ## Diagnosis
            1. Behavioral: Agent is static, not attempting any movement.
            2. Reward root-cause: reward is constant 1.0 with no task signal.
            3. Training dynamics: Stable but uninformative.
            4. Cross-iteration: First iteration, no history.
            5. Proposal: Add rotation signal as primary reward term.

            ## Reward Reasoning
            Need more rotation signal.

            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                return 2.0, {"rotation": 2.0}
            ```

            ## HP Reasoning
            Training looks stable, no changes needed.

            ## HP Changes
            ```json
            {}
            ```
        """)
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = mock_response_text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 200
        client = MagicMock()
        client.messages.create.return_value = mock_response

        from p2p.training.env_spec import get_env_spec

        env = get_env_spec("HalfCheetah-v5")

        result = revise(
            prompt="Do a backflip",
            reward_code="def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}",
            judgment={
                "intent_score": 0.3,
                "diagnosis": "static",
                "failure_tags": [],
            },
            summary={
                "final_episodic_return": 100,
                "total_timesteps": 50000,
                "training_time_s": 10,
            },
            config=TrainConfig(),
            iterations=[],
            scalars_path=scalars_path,
            client=client,
            model="claude-opus-4-6",
            env=env,
        )

        assert "def reward_fn" in result["reward_code"]
        assert result["hp_changes"] == {}
        assert result["reward_reasoning"] != ""
        assert result["training_dynamics"] != ""
        assert result["diagnosis"] != ""
        assert "rotation" in result["diagnosis"].lower()
        client.messages.create.assert_called_once()

    def test_with_hp_changes(self, tmp_path: Path):
        entries = _make_scalars_entries(5)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        mock_response_text = textwrap.dedent("""\
            ## Diagnosis
            1. Behavioral: Agent barely moves, very low return.
            2. Reward root-cause: Uniform reward provides no gradient signal.
            3. Training dynamics: Entropy collapsed, policy stuck.
            4. Cross-iteration: No prior data.
            5. Proposal: Add rotation term, increase ent_coef to restore exploration.

            ## Reward Reasoning
            Increase rotation weight.

            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                return 3.0, {"rot": 3.0}
            ```

            ## HP Reasoning
            Entropy collapsed, bump ent_coef.

            ## HP Changes
            ```json
            {"ent_coef": 0.05}
            ```
        """)
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = mock_response_text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 200
        client = MagicMock()
        client.messages.create.return_value = mock_response

        result = revise(
            prompt="backflip",
            reward_code="def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}",
            judgment={
                "intent_score": 0.2,
                "diagnosis": "bad",
                "failure_tags": [],
            },
            summary={
                "final_episodic_return": 50,
                "total_timesteps": 10000,
                "training_time_s": 5,
            },
            config=TrainConfig(),
            iterations=[],
            scalars_path=scalars_path,
            client=client,
            model="claude-opus-4-6",
            env=get_env_spec("HalfCheetah-v5"),
        )

        assert result["hp_changes"] == {"ent_coef": 0.05}
        assert "Training Dynamics" in result["training_dynamics"]
        assert result["diagnosis"] != ""

    def test_tool_use_loop(self, tmp_path: Path):
        """Verify that tool_use stop_reason triggers the dispatch loop."""
        entries = _make_scalars_entries(5)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        final_text = textwrap.dedent("""\
            ## Diagnosis
            1. Agent static. 2. No signal. 3. Stable. 4. No history. 5. Add rotation.

            ## Reward Reasoning
            Add rotation.

            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                return 2.0, {"r": 2.0}
            ```

            ## HP Reasoning
            No changes.

            ## HP Changes
            ```json
            {}
            ```
        """)

        # First response: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_iteration_reward_code"
        tool_block.input = {"iteration": 1}
        tool_block.id = "tool_123"

        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"
        tool_response.usage.input_tokens = 100
        tool_response.usage.output_tokens = 50

        # Second response: end_turn with text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = final_text

        final_response = MagicMock()
        final_response.content = [text_block]
        final_response.stop_reason = "end_turn"
        final_response.usage.input_tokens = 200
        final_response.usage.output_tokens = 300

        client = MagicMock()
        client.messages.create.side_effect = [tool_response, final_response]

        iterations = [
            {
                "iteration": 1,
                "judgment": {"intent_score": 0.5, "failure_tags": [], "diagnosis": "ok"},
                "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}",
                "iteration_dir": str(tmp_path),
            },
        ]

        result = revise(
            prompt="backflip",
            reward_code="def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}",
            judgment={
                "intent_score": 0.5,
                "diagnosis": "ok",
                "failure_tags": [],
            },
            summary={"final_episodic_return": 50, "total_timesteps": 10000, "training_time_s": 5},
            config=TrainConfig(),
            iterations=iterations,
            scalars_path=scalars_path,
            client=client,
            model="claude-opus-4-6",
            env=get_env_spec("HalfCheetah-v5"),
            best_iteration=1,
            best_score=0.5,
        )

        assert "def reward_fn" in result["reward_code"]
        # Two LLM calls: first with tools, second after tool results
        assert client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# test_tool_handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    _ITERATIONS = [
        {
            "iteration": 1,
            "judgment": {
                "intent_score": 0.3,
                "failure_tags": ["static"],
                "diagnosis": "Not moving",
            },
            "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 0.0, {}",
            "iteration_dir": "/tmp/nonexistent_iter_1",
        },
        {
            "iteration": 2,
            "judgment": {
                "intent_score": 0.7,
                "failure_tags": [],
                "diagnosis": "Good rotation",
            },
            "reward_code": (
                "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {'rot': 1.0}"
            ),
            "iteration_dir": "/tmp/nonexistent_iter_2",
        },
    ]

    def _dispatch(self, iterations=None):
        return ReviseToolDispatch(self._ITERATIONS if iterations is None else iterations)

    def test_get_reward_code(self):
        result = self._dispatch().handle_get_reward_code({"iteration": 1})
        assert result["iteration"] == 1
        assert "def reward_fn" in result["reward_code"]

    def test_get_reward_code_missing(self):
        result = self._dispatch().handle_get_reward_code({"iteration": 99})
        assert "error" in result

    def test_get_judgment_detail(self):
        result = self._dispatch().handle_get_judgment_detail({"iteration": 2})
        assert result["intent_score"] == 0.7
        assert result["diagnosis"] == "Good rotation"
        assert result["failure_tags"] == []

    def test_get_judgment_detail_missing(self):
        result = self._dispatch().handle_get_judgment_detail({"iteration": 99})
        assert "error" in result

    def test_compare_iterations(self):
        result = self._dispatch().handle_compare_iterations({"iter_a": 1, "iter_b": 2})
        assert result["iter_a"]["score"] == 0.3
        assert result["iter_b"]["score"] == 0.7
        assert "reward_code_diff" in result

    def test_compare_iterations_missing(self):
        result = self._dispatch().handle_compare_iterations({"iter_a": 1, "iter_b": 99})
        assert "error" in result

    def test_get_training_dynamics_no_dir(self):
        result = self._dispatch().handle_get_training_dynamics({"iteration": 1})
        # iteration_dir doesn't exist, should still return dynamics (empty)
        assert "dynamics" in result

    def test_get_training_dynamics_with_scalars(self, tmp_path: Path):
        entries = _make_scalars_entries(5)
        scalars_path = tmp_path / "metrics" / "scalars.jsonl"
        _write_scalars(scalars_path, entries)

        iterations = [
            {
                "iteration": 1,
                "judgment": {"intent_score": 0.5},
                "reward_code": "def reward_fn(obs, action, next_obs, info):\n    return 0.0, {}",
                "iteration_dir": str(tmp_path),
            }
        ]
        result = self._dispatch(iterations).handle_get_training_dynamics({"iteration": 1})
        assert "Training Dynamics" in result["dynamics"]


class TestResolveRunScalars:
    """Tests for ReviseToolDispatch._resolve_run_scalars error handling."""

    def test_malformed_json_falls_back(self, tmp_path):
        """Malformed best_run.json falls back to default metrics path."""
        (tmp_path / "best_run.json").write_text("{bad json")

        result = ReviseToolDispatch._resolve_run_scalars(str(tmp_path))

        assert result == tmp_path / "metrics" / "scalars.jsonl"

    def test_permission_error_falls_back(self, tmp_path):
        """Unreadable best_run.json falls back to default metrics path."""
        best_run = tmp_path / "best_run.json"
        best_run.write_text('{"best_run_id": "run_1"}')
        best_run.chmod(0o000)

        try:
            result = ReviseToolDispatch._resolve_run_scalars(str(tmp_path))
        finally:
            best_run.chmod(0o644)
        assert result == tmp_path / "metrics" / "scalars.jsonl"

    def test_valid_best_run_resolves(self, tmp_path):
        """Valid best_run.json resolves to the best run's scalars path."""
        (tmp_path / "best_run.json").write_text('{"best_run_id": "run_42"}')

        result = ReviseToolDispatch._resolve_run_scalars(str(tmp_path))

        assert result == tmp_path / "run_42" / "metrics" / "scalars.jsonl"

    def test_explicit_run_id_bypasses_best_run(self, tmp_path):
        """Explicit run_id skips best_run.json entirely."""
        result = ReviseToolDispatch._resolve_run_scalars(str(tmp_path), run_id="run_7")

        assert result == tmp_path / "run_7" / "metrics" / "scalars.jsonl"


# ---------------------------------------------------------------------------
# Two-phase revision tests
# ---------------------------------------------------------------------------


class TestTwoPhaseRevision:
    """Tests for the two-phase revision pipeline helpers."""

    _VALID_PHASE1 = textwrap.dedent("""\
        ## Diagnosis
        1. Agent runs forward but does not rotate.
        2. Rotation term weight is too low relative to velocity.
        3. Training dynamics healthy: explained_var=0.65.
        4. Cross-iteration: iter 2 had highest score.
        5. Proposal: triple rotation weight.
        6. Coordinated: no HP changes needed.

        ## Lesson
        Rotation weight must dominate velocity for backflip tasks.

        ## Based On
        2

        ## Planned Changes
        Triple the rotation reward coefficient from 1.0 to 3.0.
        Keep forward velocity term unchanged at 1.0.

        ## HP Reasoning
        Training dynamics are healthy, no HP changes.

        ## HP Changes
        ```json
        {}
        ```
    """)

    def test_parse_phase1_response_valid(self):
        """Full Phase 1 text with all sections parses correctly."""
        result = _parse_phase1_response(self._VALID_PHASE1)

        assert result["based_on"] == 2
        assert "Triple the rotation" in result["planned_changes"]
        assert "rotation" in result["diagnosis"].lower()
        assert result["lesson"] != ""
        assert result["hp_changes"] == {}
        assert result["hp_reasoning"] != ""

    def test_parse_phase1_response_missing_planned_changes(self):
        """Phase 1 text without Planned Changes raises ValueError."""
        text = textwrap.dedent("""\
            ## Diagnosis
            Agent is static.

            ## Lesson
            Need rotation signal.

            ## Based On
            1

            ## Reward Reasoning
            This uses the old format, not Planned Changes.

            ## HP Reasoning
            No changes.

            ## HP Changes
            ```json
            {}
            ```
        """)
        with pytest.raises(ValueError, match="Planned Changes"):
            _parse_phase1_response(text)

    def test_resolve_base_code_found(self):
        """Finds matching iteration's reward code."""
        iterations = [
            {"iteration": 1, "reward_code": "def reward_fn(o,a,n,i): return 0.0, {}"},
            {"iteration": 2, "reward_code": "def reward_fn(o,a,n,i): return 1.0, {}"},
        ]
        code = _resolve_base_code(2, iterations, "fallback code")
        assert "return 1.0" in code

    def test_resolve_base_code_not_found(self):
        """Falls back to current_code when no match."""
        iterations = [
            {"iteration": 1, "reward_code": "def reward_fn(o,a,n,i): return 0.0, {}"},
        ]
        code = _resolve_base_code(99, iterations, "current code")
        assert code == "current code"

    def test_resolve_base_code_zero(self):
        """based_on=0 returns current_code."""
        code = _resolve_base_code(0, [], "current code")
        assert code == "current code"

    def test_resolve_base_code_dataclass_like(self):
        """Works with object attributes, not just dicts."""

        class FakeIter:
            def __init__(self, iteration, reward_code):
                self.iteration = iteration
                self.reward_code = reward_code

        iterations = [FakeIter(3, "def reward_fn(o,a,n,i): return 3.0, {}")]
        code = _resolve_base_code(3, iterations, "fallback")
        assert "return 3.0" in code

    def test_run_phase2_success(self):
        """Mocked LLM returns valid code in Phase 2."""
        phase2_output = textwrap.dedent("""\
            ## Revised Reward Function
            ```python
            def reward_fn(obs, action, next_obs, info):
                rotation = 3.0 * obs[0]
                return rotation, {"rotation": rotation}
            ```
        """)

        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = phase2_output
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 200
        client = MagicMock()
        client.messages.create.return_value = mock_response

        from p2p.training.env_spec import get_env_spec

        env = get_env_spec("HalfCheetah-v5")

        code = _run_phase2(
            base_code="def reward_fn(o,a,n,i): return 1.0, {}",
            planned_changes="Triple the rotation coefficient.",
            env=env,
            side_info=False,
            client=client,
            model="test-model",
        )

        assert "def reward_fn" in code
        assert "rotation" in code
        client.messages.create.assert_called_once()
