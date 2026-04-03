"""Tests for training_dynamics module.

Covers: analyze_training_curves, format_training_dynamics,
format_iteration_history, format_current_config, and internal helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from p2p.analysis.training_dynamics import (
    _compute_trend,
    _empty_dynamics,
    _mean,
    _std,
    analyze_training_curves,
    format_current_config,
    format_iteration_history,
    format_training_dynamics,
)
from p2p.config import TrainConfig

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestMean:
    def test_normal_values(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_list_returns_zero(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([5.0]) == 5.0


class TestStd:
    def test_normal_values(self):
        # population std of [2, 4, 4, 4, 5, 5, 7, 9]
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert _std(vals) == pytest.approx(2.0, abs=0.01)

    def test_single_value_returns_zero(self):
        assert _std([3.0]) == 0.0

    def test_empty_list_returns_zero(self):
        assert _std([]) == 0.0

    def test_identical_values(self):
        assert _std([5.0, 5.0, 5.0]) == 0.0


class TestComputeTrend:
    def test_increasing(self):
        assert _compute_trend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == "increasing"

    def test_decreasing(self):
        assert _compute_trend([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) == "decreasing"

    def test_flat(self):
        assert _compute_trend([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) == "flat"

    def test_too_few_values_returns_flat(self):
        assert _compute_trend([1.0, 2.0, 3.0]) == "flat"

    def test_empty_returns_flat(self):
        assert _compute_trend([]) == "flat"


# ---------------------------------------------------------------------------
# analyze_training_curves
# ---------------------------------------------------------------------------


def _make_scalars_entry(**overrides: float) -> dict:
    """Build a single scalars.jsonl entry with sensible defaults."""
    base = {
        "entropy": -2.0,
        "value_loss": 10.0,
        "policy_loss": -0.01,
        "approx_kl": 0.005,
        "clip_fraction": 0.1,
        "explained_variance": 0.3,
        "episodic_return": 100.0,
        "sps": 5000.0,
    }
    base.update(overrides)
    return base


def _write_scalars(path: Path, entries: list[dict]) -> Path:
    scalars = path / "scalars.jsonl"
    scalars.write_text("\n".join(json.dumps(e) for e in entries))
    return scalars


class TestAnalyzeTrainingCurves:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        dyn = analyze_training_curves(tmp_path / "nonexistent.jsonl")
        assert dyn["num_entries"] == 0

    def test_empty_file_returns_empty(self, tmp_path: Path):
        scalars = tmp_path / "scalars.jsonl"
        scalars.write_text("")
        dyn = analyze_training_curves(scalars)
        assert dyn["num_entries"] == 0

    def test_normal_data(self, tmp_path: Path):
        entries = [
            _make_scalars_entry(entropy=-2.0, episodic_return=50.0),
            _make_scalars_entry(entropy=-1.8, episodic_return=100.0),
            _make_scalars_entry(entropy=-1.6, episodic_return=150.0),
            _make_scalars_entry(entropy=-1.4, episodic_return=200.0),
            _make_scalars_entry(entropy=-1.2, episodic_return=250.0),
            _make_scalars_entry(entropy=-1.0, episodic_return=300.0),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)

        assert dyn["num_entries"] == 6
        assert dyn["entropy_initial"] == -2.0
        assert dyn["entropy_final"] == -1.0
        assert dyn["episodic_return_final"] == 300.0
        assert dyn["episodic_return_max"] == 300.0
        assert dyn["sps_mean"] == 5000.0

    def test_eval_entries_are_skipped(self, tmp_path: Path):
        entries = [
            _make_scalars_entry(),
            {"type": "eval", "episodic_return": 999.0},
            _make_scalars_entry(),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["num_entries"] == 2

    def test_entropy_too_fast_detection(self, tmp_path: Path):
        # Entropy decays from -10.0 to -0.5 => decay_rate = 1 - 0.5/10 = 0.95 > 0.9
        entries = [
            _make_scalars_entry(entropy=-10.0),
            _make_scalars_entry(entropy=-8.0),
            _make_scalars_entry(entropy=-3.0),
            _make_scalars_entry(entropy=-0.5),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["entropy_too_fast"] is True

    def test_entropy_too_high_detection(self, tmp_path: Path):
        # Entropy barely decays: -10.0 to -9.0 => decay_rate = 1 - 9/10 = 0.1 < 0.2
        entries = [
            _make_scalars_entry(entropy=-10.0),
            _make_scalars_entry(entropy=-9.5),
            _make_scalars_entry(entropy=-9.2),
            _make_scalars_entry(entropy=-9.0),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["entropy_too_high"] is True

    def test_value_loss_diverging(self, tmp_path: Path):
        # value_loss goes from 1.0 to 100.0 (>5x) and trend is increasing
        entries = [
            _make_scalars_entry(value_loss=1.0),
            _make_scalars_entry(value_loss=10.0),
            _make_scalars_entry(value_loss=50.0),
            _make_scalars_entry(value_loss=100.0),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["value_loss_diverging"] is True

    def test_kl_spike_count(self, tmp_path: Path):
        entries = [
            _make_scalars_entry(approx_kl=0.001),
            _make_scalars_entry(approx_kl=0.03),  # spike
            _make_scalars_entry(approx_kl=0.05),  # spike
            _make_scalars_entry(approx_kl=0.01),
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["approx_kl_spike_count"] == 2
        assert dyn["approx_kl_max"] == pytest.approx(0.05)

    def test_episodic_return_converged(self, tmp_path: Path):
        # All returns at 100 => converged (tail within 5% of max)
        entries = [_make_scalars_entry(episodic_return=100.0) for _ in range(10)]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert dyn["episodic_return_converged"] is True

    def test_nan_inf_in_data(self, tmp_path: Path):
        entries = [
            _make_scalars_entry(entropy=float("nan"), value_loss=float("inf")),
            _make_scalars_entry(entropy=-1.0, value_loss=5.0),
        ]
        scalars = _write_scalars(tmp_path, entries)
        # Should not raise — NaN/Inf propagate through math but don't crash
        dyn = analyze_training_curves(scalars)
        assert dyn["num_entries"] == 2
        assert math.isnan(dyn["entropy_initial"])
        assert math.isinf(dyn["value_loss_initial"])

    def test_reward_term_stats(self, tmp_path: Path):
        entries = [
            {**_make_scalars_entry(), "reward_term/speed": 1.0, "reward_term/height": 0.5},
            {**_make_scalars_entry(), "reward_term/speed": 2.0, "reward_term/height": 0.8},
            {**_make_scalars_entry(), "reward_term/speed": 3.0, "reward_term/height": 1.0},
            {**_make_scalars_entry(), "reward_term/speed": 4.0, "reward_term/height": 1.2},
        ]
        scalars = _write_scalars(tmp_path, entries)
        dyn = analyze_training_curves(scalars)
        assert "reward_term/speed" in dyn["reward_term_stats"]
        assert "reward_term/height" in dyn["reward_term_stats"]
        speed_stats = dyn["reward_term_stats"]["reward_term/speed"]
        assert speed_stats["final"] == 4.0
        assert speed_stats["trend"] == "increasing"


# ---------------------------------------------------------------------------
# format_training_dynamics (golden snapshot)
# ---------------------------------------------------------------------------


class TestFormatTrainingDynamics:
    def test_empty_dynamics_message(self):
        text = format_training_dynamics(_empty_dynamics())
        assert text == "No training dynamics data available."

    def test_output_contains_all_sections(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        text = format_training_dynamics(dyn)
        for section in [
            "Training Dynamics Analysis",
            "Entropy",
            "Value loss",
            "Policy loss",
            "Approx KL",
            "Clip fraction",
            "Explained variance",
            "Episodic return",
            "Throughput",
        ]:
            assert section in text, f"Missing section: {section}"

    def test_entropy_collapse_warning(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["entropy_too_fast"] = True
        text = format_training_dynamics(dyn)
        assert "WARNING" in text
        assert "collapsed too fast" in text

    def test_entropy_too_high_warning(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["entropy_too_high"] = True
        text = format_training_dynamics(dyn)
        assert "barely decreased" in text

    def test_value_loss_diverging_warning(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["value_loss_diverging"] = True
        text = format_training_dynamics(dyn)
        assert "diverging" in text

    def test_kl_spike_warning(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["approx_kl_spike_count"] = 10
        text = format_training_dynamics(dyn)
        assert "Frequent KL spikes" in text

    def test_low_explained_variance_warning(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["explained_variance_good"] = False
        text = format_training_dynamics(dyn)
        assert "Low explained variance" in text

    def test_convergence_note(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["episodic_return_converged"] = True
        text = format_training_dynamics(dyn)
        assert "Converged" in text

    def test_reward_term_stats_section(self):
        dyn = _empty_dynamics()
        dyn["num_entries"] = 10
        dyn["reward_term_stats"] = {
            "reward_term/speed": {
                "mean_first_n": 1.0,
                "mean_last_n": 3.0,
                "final": 4.0,
                "trend": "increasing",
            }
        }
        text = format_training_dynamics(dyn)
        assert "Per-Term Reward Means" in text
        assert "reward_term/speed" in text


# ---------------------------------------------------------------------------
# format_iteration_history
# ---------------------------------------------------------------------------


class TestFormatIterationHistory:
    def test_empty_iterations(self):
        text, best_code = format_iteration_history([])
        assert text == "No previous iterations."
        assert best_code == ""

    def test_single_iteration_dict(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {
                    "intent_score": 3.5,
                    "diagnosis": "Needs work",
                    "failure_tags": ["slow"],
                },
            }
        ]
        text, _ = format_iteration_history(iterations)
        assert "Iteration History" in text
        assert "Iter 1" in text
        assert "3.50" in text
        assert "slow" in text

    def test_multiple_iterations_with_best_marker(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {"intent_score": 2.0, "diagnosis": "", "failure_tags": []},
            },
            {
                "iteration": 2,
                "judgment": {"intent_score": 4.0, "diagnosis": "", "failure_tags": []},
                "reward_code": "def reward_fn(): pass",
            },
        ]
        text, best_code = format_iteration_history(iterations, best_iteration=2, best_score=4.0)
        assert "Best: iter 2" in text
        assert "Iter 2 (best)" in text
        assert "Iter 1" in text
        # Best code section should be returned
        assert "Best Iteration" in best_code

    def test_multi_config_breakdown(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {
                    "intent_score": 3.0,
                    "diagnosis": "ok",
                    "failure_tags": [],
                    "config_judgments": {
                        "cfg_a": {
                            "mean_intent_score": 3.0,
                            "score_std": 0.5,
                            "mean_final_return": 200.0,
                            "common_failure_tags": ["wobble"],
                        }
                    },
                },
            }
        ]
        text, _ = format_iteration_history(iterations, best_iteration=1, best_score=3.0)
        assert "cfg_a" in text
        assert "wobble" in text
        assert "get_config_comparison" in text

    def test_hp_changes_displayed(self):
        iterations = [
            {
                "iteration": 1,
                "judgment": {"intent_score": 3.0, "diagnosis": "", "failure_tags": []},
                "hp_changes": {"learning_rate": 0.001},
            }
        ]
        text, _ = format_iteration_history(iterations)
        assert "learning_rate=0.001" in text

    def test_missing_judgment_fields(self):
        iterations = [{"iteration": 1, "judgment": {}}]
        text, _ = format_iteration_history(iterations)
        assert "Iter 1" in text


# ---------------------------------------------------------------------------
# format_current_config
# ---------------------------------------------------------------------------


class TestFormatCurrentConfig:
    def test_output_contains_tunable_keys(self):
        config = TrainConfig(env_id="HalfCheetah-v5")
        text = format_current_config(config)
        assert "Current Hyperparameters" in text
        assert "learning_rate" in text
        assert "ent_coef" in text
        assert "gamma" in text
