"""Characterization tests for benchmark service functions."""

from __future__ import annotations

import json

import pytest

from p2p.api.benchmark_service import _build_group_stats
from p2p.api.schemas import BenchmarkTestCaseResult
from p2p.benchmark.benchmark_helpers import (
    best_streaming_score as _best_streaming_score,
)
from p2p.benchmark.benchmark_helpers import (
    build_default_stages as _build_default_stages,
)
from p2p.benchmark.benchmark_helpers import (
    evaluate_gate as _evaluate_gate,
)
from p2p.config import LoopConfig, TrainConfig

# ---------------------------------------------------------------------------
# _build_default_stages
# ---------------------------------------------------------------------------


def _make_test_cases(n: int = 10) -> list[dict]:
    """Generate fake test cases with varying difficulty and env."""
    envs = ["HalfCheetah-v5", "Ant-v5", "Hopper-v5"]
    diffs = ["easy", "medium", "hard"]
    return [
        {
            "env_id": envs[i % len(envs)],
            "instruction": f"case {i}",
            "category": "locomotion",
            "difficulty": diffs[i % len(diffs)],
        }
        for i in range(n)
    ]


class TestBuildDefaultStages:
    def test_all_cases_assigned_exactly_once(self):
        cases = _make_test_cases(30)
        stages = _build_default_stages(cases, num_stages=5)
        all_indices = []
        for s in stages:
            all_indices.extend(s["case_indices"])
        assert sorted(all_indices) == list(range(30))

    def test_stage_count_does_not_exceed_num_stages(self):
        cases = _make_test_cases(10)
        stages = _build_default_stages(cases, num_stages=25)
        assert len(stages) <= 25

    def test_last_stage_gate_threshold_is_zero(self):
        cases = _make_test_cases(20)
        stages = _build_default_stages(cases, num_stages=5, gate_threshold=0.7)
        last = stages[-1]
        assert last["gate_threshold"] == 0.0

    def test_non_last_stages_have_gate_threshold(self):
        cases = _make_test_cases(20)
        stages = _build_default_stages(cases, num_stages=5, gate_threshold=0.7)
        for s in stages[:-1]:
            assert s["gate_threshold"] == 0.7

    def test_empty_cases_returns_empty(self):
        stages = _build_default_stages([], num_stages=5)
        assert stages == []


# ---------------------------------------------------------------------------
# _evaluate_gate
# ---------------------------------------------------------------------------


class TestEvaluateGate:
    def test_all_passed(self, monkeypatch):
        monkeypatch.setattr(
            "p2p.benchmark.benchmark_helpers.lightweight_session_info",
            lambda sid: {"best_score": 0.9, "status": "passed"},
        )
        entries = [{"session_id": f"s{i}"} for i in range(3)]
        result = _evaluate_gate(entries, [0, 1, 2], pass_threshold=0.7, gate_threshold=0.7)
        assert result["passed"] is True
        assert result["avg_score"] == pytest.approx(0.9)
        assert result["success_rate"] == pytest.approx(1.0)

    def test_below_gate(self, monkeypatch):
        monkeypatch.setattr(
            "p2p.benchmark.benchmark_helpers.lightweight_session_info",
            lambda sid: {"best_score": 0.3, "status": "failed"},
        )
        entries = [{"session_id": f"s{i}"} for i in range(3)]
        result = _evaluate_gate(entries, [0, 1, 2], pass_threshold=0.7, gate_threshold=0.7)
        assert result["passed"] is False

    def test_missing_session_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "p2p.benchmark.benchmark_helpers.lightweight_session_info",
            lambda sid: None,
        )
        entries = [{"session_id": ""}, {"session_id": "s1"}]
        result = _evaluate_gate(entries, [0, 1], pass_threshold=0.7, gate_threshold=0.5)
        # entry 0 has no session_id so skipped, entry 1 returns None so skipped
        assert result["completed"] == 0


# ---------------------------------------------------------------------------
# _build_group_stats
# ---------------------------------------------------------------------------


class TestBuildGroupStats:
    def test_basic_stats(self):
        results = [
            BenchmarkTestCaseResult(
                index=0,
                env_id="e",
                instruction="i",
                category="c",
                difficulty="easy",
                session_id="s0",
                session_status="completed",
                best_score=0.8,
                passed=True,
                iterations_completed=3,
                video_urls=[],
            ),
            BenchmarkTestCaseResult(
                index=1,
                env_id="e",
                instruction="i",
                category="c",
                difficulty="easy",
                session_id="s1",
                session_status="completed",
                best_score=0.4,
                passed=False,
                iterations_completed=3,
                video_urls=[],
            ),
        ]
        stats = _build_group_stats(results)
        assert stats.total == 2
        assert stats.completed == 2
        assert stats.passed == 1
        assert stats.success_rate == pytest.approx(0.5)
        assert stats.average_score == pytest.approx(0.6)

    def test_running_not_counted_as_completed(self):
        results = [
            BenchmarkTestCaseResult(
                index=0,
                env_id="e",
                instruction="i",
                category="c",
                difficulty="easy",
                session_id="s0",
                session_status="running",
                best_score=0.0,
                passed=False,
                iterations_completed=0,
                video_urls=[],
            ),
        ]
        stats = _build_group_stats(results)
        assert stats.total == 1
        assert stats.completed == 0

    def test_empty_results(self):
        stats = _build_group_stats([])
        assert stats.total == 0
        assert stats.average_score == 0.0


# ---------------------------------------------------------------------------
# _best_streaming_score
# ---------------------------------------------------------------------------


class TestBestStreamingScore:
    def test_null_intent_score_does_not_raise(self, tmp_path):
        """intent_score: null in JSON should be handled gracefully (issue #160)."""
        sj_dir = tmp_path / "iter_0" / "streaming_judgments"
        sj_dir.mkdir(parents=True)
        (sj_dir / "step_1000.json").write_text(json.dumps({"intent_score": None}))

        assert _best_streaming_score(tmp_path) == 0.0

    def test_valid_scores_returns_best(self, tmp_path):
        sj_dir = tmp_path / "iter_0" / "streaming_judgments"
        sj_dir.mkdir(parents=True)
        (sj_dir / "step_1000.json").write_text(json.dumps({"intent_score": 0.5}))
        (sj_dir / "step_2000.json").write_text(json.dumps({"intent_score": 0.8}))

        assert _best_streaming_score(tmp_path) == pytest.approx(0.8)

    def test_missing_intent_score_defaults_to_zero(self, tmp_path):
        sj_dir = tmp_path / "iter_0" / "streaming_judgments"
        sj_dir.mkdir(parents=True)
        (sj_dir / "step_1000.json").write_text(json.dumps({"other_field": 42}))

        assert _best_streaming_score(tmp_path) == 0.0


# ---------------------------------------------------------------------------
# Manifest LoopConfig round-trip
# ---------------------------------------------------------------------------


class TestManifestLoopConfigRoundTrip:
    """LoopConfig stored in manifest survives serialize → deserialize."""

    def test_round_trip_preserves_all_fields(self) -> None:
        original = LoopConfig(
            train=TrainConfig(
                total_timesteps=500_000,
                seed=42,
                num_envs=4,
                side_info=True,
                device="cpu",
            ),
            seeds=[1, 2, 3],
            max_iterations=10,
            pass_threshold=0.8,
            vlm_model="gemini-2.0-flash",
            thinking_effort="high",
            cores_per_run=4,
            max_parallel=8,
            hp_tuning=True,
            use_code_judge=True,
            review_reward=False,
            review_judge=False,
            use_zoo_preset=False,
        )
        json_str = original.to_json()
        restored = LoopConfig.from_json(json_str)

        assert restored.train.total_timesteps == 500_000
        assert restored.train.seed == 42
        assert restored.train.device == "cpu"
        assert restored.seeds == [1, 2, 3]
        assert restored.max_iterations == 10
        assert restored.vlm_model == "gemini-2.0-flash"
        assert restored.thinking_effort == "high"
        assert restored.hp_tuning is True
        assert restored.use_code_judge is True
        assert restored.review_reward is False
        assert restored.use_zoo_preset is False
