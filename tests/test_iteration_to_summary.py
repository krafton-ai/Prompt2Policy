"""Characterization tests for _iteration_to_summary().

Golden-snapshot style: record what the function currently does so future
refactors have a safety net.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2p.api.services import _iteration_to_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_DIR = "/nonexistent/session_abc/iter_001"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_iteration_dir(tmp_path: Path, name: str = "iter_001") -> Path:
    d = tmp_path / "session_abc" / name
    d.mkdir(parents=True)
    return d


def _make_it(it_dir: str | Path = _FAKE_DIR, iteration: int = 1, **overrides: object) -> dict:
    base: dict = {
        "iteration": iteration,
        "iteration_dir": str(it_dir),
        "judgment": {"diagnosis": "", "failure_tags": []},
        "summary": {},
        "reward_code": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Normal iteration with judgment + reward_spec
# ---------------------------------------------------------------------------


class TestNormalIteration:
    def test_basic_fields(self):
        it = _make_it(
            judgment={
                "intent_score": 0.72,
                "best_checkpoint": "step_5000",
                "diagnosis": "Agent moves forward but wobbles.",
                "failure_tags": ["wobble"],
                "checkpoint_judgments": {},
            },
            summary={"final_episodic_return": 1234.5},
            reward_code="def compute(obs): return 1.0",
        )
        result = _iteration_to_summary(it)

        assert result.iteration == 1
        assert result.intent_score == 0.72
        assert result.best_checkpoint == "step_5000"
        assert result.diagnosis == "Agent moves forward but wobbles."
        assert result.failure_tags == ["wobble"]
        assert result.reward_code == "def compute(obs): return 1.0"
        assert result.final_return == 1234.5
        assert result.is_multi_config is False

    def test_checkpoint_judgments_extraction(self):
        it = _make_it(
            iteration=2,
            judgment={
                "intent_score": 0.5,
                "diagnosis": "",
                "failure_tags": [],
                "checkpoint_judgments": {
                    "5000": {
                        "intent_score": 0.4,
                        "diagnosis": "early wobble",
                        "rollout_judgments": [
                            {"episode_idx": 0, "intent_score": 0.3, "diagnosis": "fell"},
                            {"episode_idx": 1, "intent_score": 0.5, "diagnosis": "ok"},
                        ],
                    },
                    "10000": {
                        "intent_score": 0.6,
                        "diagnosis": "improved",
                        "rollout_judgments": [],
                    },
                },
            },
        )
        result = _iteration_to_summary(it)

        assert result.checkpoint_scores == {"5000": 0.4, "10000": 0.6}
        assert result.checkpoint_diagnoses == {"5000": "early wobble", "10000": "improved"}
        assert result.rollout_scores == {"5000_ep0": 0.3, "5000_ep1": 0.5}
        assert result.rollout_diagnoses == {"5000_ep0": "fell", "5000_ep1": "ok"}

    def test_revise_agent_fields(self):
        it = _make_it(
            iteration=3,
            reward_reasoning="Added forward velocity bonus.",
            hp_reasoning="Increased learning rate.",
            hp_changes={"learning_rate": 0.001},
            training_dynamics="Stable after step 2k.",
            revise_diagnosis="Reward hacking detected.",
        )
        result = _iteration_to_summary(it)

        assert result.reward_reasoning == "Added forward velocity bonus."
        assert result.hp_reasoning == "Increased learning rate."
        assert result.hp_changes == {"learning_rate": 0.001}
        assert result.training_dynamics == "Stable after step 2k."
        assert result.revise_diagnosis == "Reward hacking detected."


# ---------------------------------------------------------------------------
# 2. Judgment missing
# ---------------------------------------------------------------------------


class TestJudgmentMissing:
    def test_no_judgment_key(self):
        it = {
            "iteration": 1,
            "iteration_dir": _FAKE_DIR,
            "summary": {"final_episodic_return": 100.0},
            "reward_code": "pass",
        }
        result = _iteration_to_summary(it)

        assert result.intent_score is None
        assert result.best_checkpoint == ""
        assert result.diagnosis == ""
        assert result.failure_tags == []
        assert result.checkpoint_scores == {}
        assert result.final_return == 100.0

    def test_empty_judgment(self):
        it = _make_it(judgment={})
        result = _iteration_to_summary(it)

        assert result.intent_score is None
        assert result.diagnosis == ""
        assert result.failure_tags == []


# ---------------------------------------------------------------------------
# 3. Multi-config iteration (aggregation.json exists)
# ---------------------------------------------------------------------------


class TestMultiConfig:
    def test_with_aggregation_json(self, tmp_path: Path):
        it_dir = _make_iteration_dir(tmp_path)
        agg = {
            "best_config_id": "fast",
            "best_run_id": "fast_seed_42",
            "configs": {
                "fast": {
                    "mean_best_score": 200.0,
                    "std_best_score": 10.0,
                    "mean_final_return": 180.0,
                    "std_final_return": 5.0,
                    "per_seed": [{"seed": 42, "best_score": 200.0, "final_return": 180.0}],
                },
            },
        }
        _write_json(it_dir / "aggregation.json", agg)
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.is_multi_config is True
        assert result.aggregation == agg["configs"]
        assert result.best_config_id == "fast"
        assert result.best_run_id == "fast_seed_42"

    def test_best_run_json_overrides_aggregation(self, tmp_path: Path):
        """best_run.json takes precedence over aggregation.json top-level fields."""
        it_dir = _make_iteration_dir(tmp_path)
        _write_json(
            it_dir / "aggregation.json",
            {"best_config_id": "slow", "best_run_id": "slow_seed_0", "configs": {"a": {}}},
        )
        _write_json(
            it_dir / "best_run.json",
            {"best_config_id": "a", "best_run_id": "a_seed_0"},
        )
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.best_config_id == "a"
        assert result.best_run_id == "a_seed_0"

    def test_aggregation_json_without_configs_key(self, tmp_path: Path):
        """When aggregation.json has no 'configs' key, the raw dict is used as-is."""
        it_dir = _make_iteration_dir(tmp_path)
        flat_agg = {
            "fast": {"mean_best_score": 150.0, "mean_final_return": 140.0},
            "slow": {"mean_best_score": 80.0, "mean_final_return": 70.0},
        }
        _write_json(it_dir / "aggregation.json", flat_agg)
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.is_multi_config is True
        assert result.aggregation == flat_agg

    def test_live_aggregation_from_seed_dirs(self, tmp_path: Path):
        """In-progress multi-config: no aggregation.json yet, but seed dirs exist."""
        it_dir = _make_iteration_dir(tmp_path)
        for seed, best, final in [(0, 100.0, 90.0), (1, 120.0, 110.0)]:
            sub = it_dir / f"alpha_seed_{seed}"
            sub.mkdir()
            _write_json(
                sub / "summary.json",
                {"best_episodic_return": best, "final_episodic_return": final},
            )
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.is_multi_config is True
        assert result.aggregation is not None
        assert "alpha" in result.aggregation
        alpha = result.aggregation["alpha"]
        assert alpha["mean_best_score"] == pytest.approx(110.0)
        assert alpha["mean_final_return"] == pytest.approx(100.0)
        assert len(alpha["per_seed"]) == 2

    def test_live_best_config_derived_from_aggregation(self, tmp_path: Path):
        """When best_run.json is absent, best_config_id is derived from live aggregation."""
        it_dir = _make_iteration_dir(tmp_path)
        for cfg, seed, final in [("slow", 0, 50.0), ("fast", 0, 200.0)]:
            sub = it_dir / f"{cfg}_seed_{seed}"
            sub.mkdir()
            _write_json(
                sub / "summary.json",
                {"best_episodic_return": final, "final_episodic_return": final},
            )
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.best_config_id == "fast"
        assert result.best_run_id == "fast_seed_0"


# ---------------------------------------------------------------------------
# 4. In-progress iteration (scalars only, no judgment)
# ---------------------------------------------------------------------------


class TestInProgressIteration:
    def test_scalars_only_no_summary(self, tmp_path: Path):
        it_dir = _make_iteration_dir(tmp_path)
        metrics_dir = it_dir / "metrics"
        metrics_dir.mkdir()
        lines = [
            json.dumps({"elapsed_time": 10.0}),
            json.dumps({"elapsed_time": 25.0}),
        ]
        (metrics_dir / "scalars.jsonl").write_text("\n".join(lines))
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.final_return is None
        assert result.intent_score is None
        assert result.is_multi_config is False
        assert result.elapsed_time_s == 25.0

    def test_streaming_judgments_fallback(self, tmp_path: Path):
        """When checkpoint_judgments is empty, fall back to streaming_judgments dir."""
        it_dir = _make_iteration_dir(tmp_path)
        sj_dir = it_dir / "streaming_judgments"
        sj_dir.mkdir()
        _write_json(
            sj_dir / "3000.json",
            {
                "intent_score": 0.55,
                "diagnosis": "getting better",
                "rollout_judgments": [
                    {"episode_idx": 0, "intent_score": 0.5, "diagnosis": "slow"},
                ],
            },
        )
        it = _make_it(
            it_dir,
            judgment={
                "diagnosis": "",
                "failure_tags": [],
                "checkpoint_judgments": {},
            },
        )
        result = _iteration_to_summary(it)

        assert result.checkpoint_scores == {"3000": 0.55}
        assert result.checkpoint_diagnoses == {"3000": "getting better"}
        assert result.rollout_scores == {"3000_ep0": 0.5}

    def test_streaming_judgments_from_sub_run(self, tmp_path: Path):
        """Multi-config: streaming judgments in sub-run dir take priority over iteration-level."""
        it_dir = _make_iteration_dir(tmp_path)
        # Create sub-run with videos (sets video_source_run_id)
        sub = it_dir / "cfg_seed_0"
        sub.mkdir()
        vd = sub / "videos"
        vd.mkdir()
        (vd / "eval.mp4").touch()
        # Sub-run streaming judgments (should be found first)
        sub_sj = sub / "streaming_judgments"
        sub_sj.mkdir()
        _write_json(
            sub_sj / "5000.json",
            {
                "intent_score": 0.8,
                "diagnosis": "sub-run score",
                "rollout_judgments": [],
            },
        )
        # Iteration-level streaming judgments (should be ignored)
        iter_sj = it_dir / "streaming_judgments"
        iter_sj.mkdir()
        _write_json(
            iter_sj / "5000.json",
            {
                "intent_score": 0.2,
                "diagnosis": "iteration-level score",
                "rollout_judgments": [],
            },
        )
        it = _make_it(
            it_dir,
            judgment={
                "diagnosis": "",
                "failure_tags": [],
                "checkpoint_judgments": {},
            },
        )
        result = _iteration_to_summary(it)

        assert result.checkpoint_scores == {"5000": 0.8}
        assert result.checkpoint_diagnoses == {"5000": "sub-run score"}


# ---------------------------------------------------------------------------
# 5. Empty iteration directory
# ---------------------------------------------------------------------------


class TestEmptyIteration:
    def test_empty_dir(self, tmp_path: Path):
        it_dir = _make_iteration_dir(tmp_path)
        result = _iteration_to_summary(_make_it(it_dir, iteration=0))

        assert result.iteration == 0
        assert result.is_multi_config is False
        assert result.video_urls == []
        assert result.aggregation is None
        assert result.final_return is None
        assert result.elapsed_time_s is None

    def test_nonexistent_dir(self):
        result = _iteration_to_summary(_make_it(iteration=0))

        assert result.iteration == 0
        assert result.is_multi_config is False
        assert result.video_urls == []

    def test_no_iteration_dir(self):
        it = {
            "iteration": 0,
            "judgment": {"diagnosis": "", "failure_tags": []},
            "summary": {},
            "reward_code": "",
        }
        result = _iteration_to_summary(it)

        assert result.iteration_dir == ""
        assert result.video_urls == []


# ---------------------------------------------------------------------------
# 6. Video URL construction
# ---------------------------------------------------------------------------


class TestVideoUrls:
    def test_iteration_level_videos(self, tmp_path: Path):
        it_dir = _make_iteration_dir(tmp_path)
        videos_dir = it_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "eval_step5000.mp4").touch()
        (videos_dir / "eval_step10000.mp4").touch()
        result = _iteration_to_summary(_make_it(it_dir))

        assert len(result.video_urls) == 2
        assert all("/videos/" in url for url in result.video_urls)
        # video_filenames() returns sorted order: 10000 < 5000 lexicographically
        assert "eval_step10000.mp4" in result.video_urls[0]
        assert "eval_step5000.mp4" in result.video_urls[1]

    def test_sub_run_video_fallback(self, tmp_path: Path):
        """When no iteration-level videos, fall back to sub-run videos."""
        it_dir = _make_iteration_dir(tmp_path)
        sub = it_dir / "fast_seed_0"
        sub.mkdir()
        videos_dir = sub / "videos"
        videos_dir.mkdir()
        (videos_dir / "eval.mp4").touch()
        _write_json(sub / "summary.json", {"final_episodic_return": 300.0})
        result = _iteration_to_summary(_make_it(it_dir))

        assert len(result.video_urls) == 1
        assert "fast_seed_0" in result.video_urls[0]
        assert result.video_source_run_id == "fast_seed_0"
        assert result.video_source_return == 300.0

    def test_best_run_video_preferred(self, tmp_path: Path):
        """When best_run_id is set, prefer that sub-run for videos."""
        it_dir = _make_iteration_dir(tmp_path)
        _write_json(
            it_dir / "aggregation.json",
            {
                "best_config_id": "beta",
                "best_run_id": "beta_seed_1",
                "configs": {"beta": {}},
            },
        )
        for name in ["beta_seed_0", "beta_seed_1"]:
            sub = it_dir / name
            sub.mkdir()
            vd = sub / "videos"
            vd.mkdir()
            (vd / "eval.mp4").touch()
        result = _iteration_to_summary(_make_it(it_dir))

        assert result.video_source_run_id == "beta_seed_1"


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_reward_diff_summary(self):
        it = _make_it(reward_diff_summary="Changed velocity weight from 1.0 to 2.0")
        result = _iteration_to_summary(it)
        assert result.reward_diff_summary == "Changed velocity weight from 1.0 to 2.0"

    def test_run_dir_fallback(self):
        """iteration_dir absent but run_dir present (backward compat)."""
        it = {
            "iteration": 1,
            "run_dir": _FAKE_DIR,
            "judgment": {"diagnosis": "", "failure_tags": []},
            "summary": {},
            "reward_code": "",
        }
        result = _iteration_to_summary(it)
        assert result.iteration_dir == _FAKE_DIR

    def test_minimal_dict(self):
        """Absolute minimum input: empty dict."""
        result = _iteration_to_summary({})
        assert result.iteration == 0
        assert result.iteration_dir == ""
        assert result.intent_score is None
        assert result.diagnosis == ""
        assert result.failure_tags == []
        assert result.reward_code == ""
        assert result.final_return is None
