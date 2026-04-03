"""Tests for iteration_runner: unified run_iteration and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from p2p.session.iteration_runner import aggregate_judgments, judge_run, run_iteration

_DUMMY_CODE = "def reward_fn(obs, action, next_obs, info):\n    return 1.0, {}"
_DUMMY_SUMMARY = {
    "final_episodic_return": 500.0,
    "total_timesteps": 100000,
    "training_time_s": 30.0,
    "total_episodes": 50,
}
_DUMMY_JUDGMENT = {
    "intent_score": 0.85,
    "passed": True,
    "diagnosis": "good behavior",
    "failure_tags": [],
    "evidence": [],
}
_DEFAULT_CONFIGS = [{"config_id": "default", "label": "default", "params": {}}]
_DEFAULT_SEEDS = [1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_aggregation(configs, seeds):
    """Build a minimal IterationAggregation for testing."""
    config_stats = {}
    best_config_id = configs[0]["config_id"]
    best_run_id = f"{best_config_id}_seed_{seeds[0]}"
    for cfg in configs:
        cid = cfg["config_id"]
        per_seed = [{"seed": float(s), "best_score": 0.0, "final_return": 500.0} for s in seeds]
        config_stats[cid] = {
            "mean_best_score": 0.0,
            "std_best_score": 0.0,
            "mean_final_return": 500.0,
            "std_final_return": 0.0,
            "per_seed": per_seed,
        }
    return {
        "best_config_id": best_config_id,
        "best_run_id": best_run_id,
        "configs": config_stats,
    }


def _setup_run_iteration_mocks(
    iteration_dir: Path,
    configs: list[dict],
    seeds: list[int],
    aggregation: dict | None = None,
):
    """Create run directories with summaries and return a mock trainer + judge patch."""
    for cfg in configs:
        for seed in seeds:
            run_id = f"{cfg['config_id']}_seed_{seed}"
            run_dir = iteration_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(_DUMMY_SUMMARY))
            videos_dir = run_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            (videos_dir / "eval_100000.mp4").write_bytes(b"\x00" * 100)

    agg = aggregation or _fake_aggregation(configs, seeds)
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = agg

    return {
        "trainer": mock_trainer,
        "judge": patch(
            "p2p.session.iteration_runner.judge_all_checkpoints",
            return_value=_DUMMY_JUDGMENT,
        ),
    }


# ---------------------------------------------------------------------------
# run_iteration
# ---------------------------------------------------------------------------


def test_run_iteration_returns_iteration_data_and_judgment(tmp_path):
    """run_iteration should return (IterationData, judgment dict)."""
    from p2p.config import TrainConfig
    from p2p.session.iteration_record import SessionRecord

    session_dir = tmp_path / "session_1"
    session_dir.mkdir()
    session = SessionRecord(session_dir)
    config = TrainConfig()
    iteration_dir = session_dir / "iter_1"
    iteration_dir.mkdir()

    mocks = _setup_run_iteration_mocks(iteration_dir, _DEFAULT_CONFIGS, _DEFAULT_SEEDS)

    with mocks["judge"]:
        record, judgment = run_iteration(
            iteration=1,
            iteration_dir=iteration_dir,
            reward_code=_DUMMY_CODE,
            config=config,
            configs=_DEFAULT_CONFIGS,
            seeds=_DEFAULT_SEEDS,
            env_id="HalfCheetah-v5",
            env_name="HalfCheetah",
            prompt="run fast",
            pass_threshold=0.7,
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            trainer=mocks["trainer"],
            session=session,
        )

    assert record.iteration == 1
    assert record.reward_code == _DUMMY_CODE
    assert record.summary == _DUMMY_SUMMARY
    assert record.is_multi_config is True
    assert judgment["intent_score"] == 0.85
    assert judgment["passed"] is True


def test_run_iteration_saves_prompt_file(tmp_path):
    """run_iteration should save prompt.txt in the iteration directory."""
    from p2p.config import TrainConfig
    from p2p.session.iteration_record import SessionRecord

    session_dir = tmp_path / "session_2"
    session_dir.mkdir()
    session = SessionRecord(session_dir)
    config = TrainConfig()
    iteration_dir = session_dir / "iter_1"
    iteration_dir.mkdir()

    mocks = _setup_run_iteration_mocks(iteration_dir, _DEFAULT_CONFIGS, _DEFAULT_SEEDS)

    with mocks["judge"]:
        run_iteration(
            iteration=1,
            iteration_dir=iteration_dir,
            reward_code=_DUMMY_CODE,
            config=config,
            configs=_DEFAULT_CONFIGS,
            seeds=_DEFAULT_SEEDS,
            env_id="HalfCheetah-v5",
            env_name="HalfCheetah",
            prompt="run fast and turn",
            pass_threshold=0.7,
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            trainer=mocks["trainer"],
            session=session,
        )

    assert (iteration_dir / "prompt.txt").read_text() == "run fast and turn"


def test_run_iteration_saves_judgment(tmp_path):
    """run_iteration should persist judgment.json in the iteration directory."""
    from p2p.config import TrainConfig
    from p2p.session.iteration_record import SessionRecord

    session_dir = tmp_path / "session_3"
    session_dir.mkdir()
    session = SessionRecord(session_dir)
    config = TrainConfig()
    iteration_dir = session_dir / "iter_1"
    iteration_dir.mkdir()

    mocks = _setup_run_iteration_mocks(iteration_dir, _DEFAULT_CONFIGS, _DEFAULT_SEEDS)

    with mocks["judge"]:
        run_iteration(
            iteration=1,
            iteration_dir=iteration_dir,
            reward_code=_DUMMY_CODE,
            config=config,
            configs=_DEFAULT_CONFIGS,
            seeds=_DEFAULT_SEEDS,
            env_id="HalfCheetah-v5",
            env_name="HalfCheetah",
            prompt="run fast",
            pass_threshold=0.7,
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            trainer=mocks["trainer"],
            session=session,
        )

    judgment_path = iteration_dir / "judgment.json"
    assert judgment_path.exists()
    saved = json.loads(judgment_path.read_text())
    assert saved["intent_score"] == 0.85


def test_run_iteration_saves_best_run_json(tmp_path):
    """run_iteration should save best_run.json with config and run IDs."""
    from p2p.config import TrainConfig
    from p2p.session.iteration_record import SessionRecord

    session_dir = tmp_path / "session_4"
    session_dir.mkdir()
    session = SessionRecord(session_dir)
    config = TrainConfig()
    iteration_dir = session_dir / "iter_1"
    iteration_dir.mkdir()

    mocks = _setup_run_iteration_mocks(iteration_dir, _DEFAULT_CONFIGS, _DEFAULT_SEEDS)

    with mocks["judge"]:
        run_iteration(
            iteration=1,
            iteration_dir=iteration_dir,
            reward_code=_DUMMY_CODE,
            config=config,
            configs=_DEFAULT_CONFIGS,
            seeds=_DEFAULT_SEEDS,
            env_id="HalfCheetah-v5",
            env_name="HalfCheetah",
            prompt="run fast",
            pass_threshold=0.7,
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            trainer=mocks["trainer"],
            session=session,
        )

    best_run_path = iteration_dir / "best_run.json"
    assert best_run_path.exists()
    best_info = json.loads(best_run_path.read_text())
    assert best_info["best_config_id"] == "default"
    assert best_info["best_run_id"] == "default_seed_1"


def test_run_iteration_multi_config_reselects_best_by_intent_score(tmp_path):
    """With 2 configs, best config should be selected by intent_score, not return."""
    from p2p.config import TrainConfig
    from p2p.session.iteration_record import SessionRecord

    session_dir = tmp_path / "session_mc"
    session_dir.mkdir()
    session = SessionRecord(session_dir)
    config = TrainConfig()
    iteration_dir = session_dir / "iter_1"
    iteration_dir.mkdir()

    configs = [
        {"config_id": "cfg_a", "label": "a", "params": {}},
        {"config_id": "cfg_b", "label": "b", "params": {}},
    ]
    seeds = [1, 2]

    # cfg_a has higher return but lower intent_score
    agg = {
        "best_config_id": "cfg_a",  # parallel_trainer selects by return
        "best_run_id": "cfg_a_seed_1",
        "configs": {
            "cfg_a": {
                "mean_best_score": 0.0,
                "std_best_score": 0.0,
                "mean_final_return": 1000.0,
                "std_final_return": 0.0,
                "per_seed": [
                    {"seed": 1.0, "best_score": 0.0, "final_return": 1000.0},
                    {"seed": 2.0, "best_score": 0.0, "final_return": 900.0},
                ],
            },
            "cfg_b": {
                "mean_best_score": 0.0,
                "std_best_score": 0.0,
                "mean_final_return": 500.0,
                "std_final_return": 0.0,
                "per_seed": [
                    {"seed": 1.0, "best_score": 0.0, "final_return": 500.0},
                    {"seed": 2.0, "best_score": 0.0, "final_return": 400.0},
                ],
            },
        },
    }

    mocks = _setup_run_iteration_mocks(iteration_dir, configs, seeds, aggregation=agg)

    # cfg_b gets higher intent scores than cfg_a
    judgment_scores = {
        "cfg_a_seed_1": 0.5,
        "cfg_a_seed_2": 0.4,
        "cfg_b_seed_1": 0.9,
        "cfg_b_seed_2": 0.8,
    }

    def fake_judge(prompt, run_dir, summary, **kwargs):
        run_id = run_dir.name
        score = judgment_scores.get(run_id, 0.5)
        return {
            **_DUMMY_JUDGMENT,
            "intent_score": score,
            "passed": score >= 0.7,
        }

    with patch("p2p.session.iteration_runner.judge_all_checkpoints", side_effect=fake_judge):
        record, judgment = run_iteration(
            iteration=1,
            iteration_dir=iteration_dir,
            reward_code=_DUMMY_CODE,
            config=config,
            configs=configs,
            seeds=seeds,
            env_id="HalfCheetah-v5",
            env_name="HalfCheetah",
            prompt="run fast",
            pass_threshold=0.7,
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            trainer=mocks["trainer"],
            session=session,
        )

    # Best should be cfg_b (higher intent_score), not cfg_a (higher return)
    best_info = json.loads((iteration_dir / "best_run.json").read_text())
    assert best_info["best_config_id"] == "cfg_b"
    assert best_info["best_run_id"] == "cfg_b_seed_1"
    assert judgment["intent_score"] == 0.9
    assert "config_judgments" in judgment
    assert len(judgment["config_judgments"]) == 2


# ---------------------------------------------------------------------------
# judge_run
# ---------------------------------------------------------------------------


def test_judge_run_returns_run_id_and_judgment(tmp_path):
    """judge_run should return (run_id, judgment) and save judgment.json."""
    run_dir = tmp_path / "cfg_a_seed_42"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(_DUMMY_SUMMARY))

    with patch("p2p.session.iteration_runner.judge_all_checkpoints", return_value=_DUMMY_JUDGMENT):
        run_id, judgment = judge_run(
            run_id="cfg_a_seed_42",
            run_dir=run_dir,
            prompt="run fast",
            reward_code=_DUMMY_CODE,
            pass_threshold=0.7,
            env_name="HalfCheetah",
            vlm_model="test-model",
            client=MagicMock(),
            model="test-model",
            judge_code="",
        )

    assert run_id == "cfg_a_seed_42"
    assert judgment["intent_score"] == 0.85
    assert (run_dir / "judgment.json").exists()


# ---------------------------------------------------------------------------
# aggregate_judgments
# ---------------------------------------------------------------------------


def test_aggregate_judgments_single_config_single_seed():
    """Aggregate with 1 config x 1 seed should return exact values."""
    configs = [{"config_id": "cfg_a"}]
    seeds = [42]
    run_judgments = {
        "cfg_a_seed_42": {
            "intent_score": 0.8,
            "diagnosis": "ok",
            "failure_tags": ["wobble"],
            "best_checkpoint": "ckpt_1",
        },
    }
    aggregation = {
        "configs": {
            "cfg_a": {
                "per_seed": [{"seed": 42, "final_return": 1200.0}],
            },
        },
    }

    result = aggregate_judgments(configs, seeds, run_judgments, aggregation)

    assert "cfg_a" in result
    agg = result["cfg_a"]
    assert agg["num_seeds"] == 1
    assert agg["mean_intent_score"] == 0.8
    assert agg["score_std"] == 0.0
    assert agg["mean_final_return"] == 1200.0
    assert agg["best_seed"] == 42
    assert agg["worst_seed"] == 42


def test_aggregate_judgments_multi_config_multi_seed():
    """Aggregate with 2 configs x 2 seeds should compute correct stats."""
    configs = [{"config_id": "cfg_a"}, {"config_id": "cfg_b"}]
    seeds = [1, 2]
    run_judgments = {
        "cfg_a_seed_1": {
            "intent_score": 0.6,
            "diagnosis": "ok",
            "failure_tags": ["wobble"],
            "best_checkpoint": "c1",
        },
        "cfg_a_seed_2": {
            "intent_score": 0.8,
            "diagnosis": "ok",
            "failure_tags": [],
            "best_checkpoint": "c2",
        },
        "cfg_b_seed_1": {
            "intent_score": 0.9,
            "diagnosis": "ok",
            "failure_tags": [],
            "best_checkpoint": "c3",
        },
        "cfg_b_seed_2": {
            "intent_score": 0.7,
            "diagnosis": "ok",
            "failure_tags": ["flip"],
            "best_checkpoint": "c4",
        },
    }
    aggregation = {
        "configs": {
            "cfg_a": {
                "per_seed": [
                    {"seed": 1, "final_return": 100.0},
                    {"seed": 2, "final_return": 200.0},
                ]
            },
            "cfg_b": {
                "per_seed": [
                    {"seed": 1, "final_return": 300.0},
                    {"seed": 2, "final_return": 400.0},
                ]
            },
        },
    }

    result = aggregate_judgments(configs, seeds, run_judgments, aggregation)

    assert result["cfg_a"]["num_seeds"] == 2
    assert result["cfg_a"]["mean_intent_score"] == 0.7  # (0.6+0.8)/2
    assert result["cfg_a"]["best_seed"] == 2  # seed_2 scored 0.8
    assert result["cfg_b"]["num_seeds"] == 2
    assert result["cfg_b"]["mean_intent_score"] == 0.8  # (0.9+0.7)/2
    assert result["cfg_b"]["best_seed"] == 1  # seed_1 scored 0.9


def test_aggregate_judgments_common_failure_tags():
    """Tags appearing in >50% of seeds should be marked as common."""
    configs = [{"config_id": "cfg_a"}]
    seeds = [1, 2, 3]
    run_judgments = {
        "cfg_a_seed_1": {
            "intent_score": 0.5,
            "diagnosis": "",
            "failure_tags": ["wobble", "flip"],
            "best_checkpoint": "",
        },
        "cfg_a_seed_2": {
            "intent_score": 0.5,
            "diagnosis": "",
            "failure_tags": ["wobble"],
            "best_checkpoint": "",
        },
        "cfg_a_seed_3": {
            "intent_score": 0.5,
            "diagnosis": "",
            "failure_tags": ["flip"],
            "best_checkpoint": "",
        },
    }
    aggregation = {"configs": {"cfg_a": {"per_seed": []}}}

    result = aggregate_judgments(configs, seeds, run_judgments, aggregation)

    # wobble: 2/3 > 50%, flip: 2/3 > 50%
    assert "wobble" in result["cfg_a"]["common_failure_tags"]
    assert "flip" in result["cfg_a"]["common_failure_tags"]


def test_aggregate_judgments_missing_run_skipped():
    """Runs not present in run_judgments should be skipped gracefully."""
    configs = [{"config_id": "cfg_a"}]
    seeds = [1, 2]
    run_judgments = {
        "cfg_a_seed_1": {
            "intent_score": 0.7,
            "diagnosis": "",
            "failure_tags": [],
            "best_checkpoint": "",
        },
        # cfg_a_seed_2 is missing
    }
    aggregation = {"configs": {"cfg_a": {"per_seed": [{"seed": 1, "final_return": 100.0}]}}}

    result = aggregate_judgments(configs, seeds, run_judgments, aggregation)

    assert result["cfg_a"]["num_seeds"] == 1
    assert result["cfg_a"]["mean_intent_score"] == 0.7
