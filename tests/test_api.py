import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from p2p.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared fixture: fake session with iterations inside
# ---------------------------------------------------------------------------

_FAKE_ITERATION_ID = "iter_1"
_FAKE_SESSION_ID = "session_fixture_001"


def _create_fake_iteration(session_dir: Path, iteration_id: str) -> Path:
    """Create a minimal completed iteration inside a session directory."""
    iteration_dir = session_dir / iteration_id
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "config.json").write_text(
        json.dumps({"env_id": "HalfCheetah-v5", "total_timesteps": 100_000})
    )
    (iteration_dir / "reward_fn.py").write_text(
        'def reward_fn(o, a, n, i):\n    return 0.0, {"x": 0.0}\n'
    )
    (iteration_dir / "reward_spec.json").write_text(
        json.dumps({"latex": "r=0", "terms": {}, "description": "test"})
    )
    (iteration_dir / "summary.json").write_text(
        json.dumps(
            {
                "final_episodic_return": 500.0,
                "total_timesteps": 100_000,
                "training_time_s": 30.0,
            }
        )
    )
    (iteration_dir / "status.json").write_text(json.dumps({"status": "completed"}))
    metrics = iteration_dir / "metrics"
    metrics.mkdir()
    train_entry = {
        "global_step": 50_000,
        "iteration": 1,
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 0.6,
        "approx_kl": 0.01,
        "clip_fraction": 0.1,
        "explained_variance": 0.5,
        "learning_rate": 0.0003,
        "sps": 1000,
    }
    train_entry2 = {**train_entry, "global_step": 100_000, "iteration": 2}
    lines = [
        json.dumps(train_entry),
        json.dumps(train_entry2),
        json.dumps(
            {
                "global_step": 10_000,
                "type": "eval",
                "total_reward": 100,
                "episode_length": 200,
                "reward_terms": {"forward": 80, "energy": -10},
            }
        ),
        json.dumps(
            {
                "global_step": 50_000,
                "type": "eval",
                "total_reward": 300,
                "episode_length": 400,
                "reward_terms": {"forward": 250, "energy": -15},
            }
        ),
        json.dumps(
            {
                "global_step": 100_000,
                "type": "eval",
                "total_reward": 500,
                "episode_length": 500,
                "reward_terms": {"forward": 450, "energy": -20},
            }
        ),
    ]
    (metrics / "scalars.jsonl").write_text("\n".join(lines) + "\n")
    vdir = iteration_dir / "videos"
    vdir.mkdir()
    frames = vdir / "frames"
    frames.mkdir()
    (frames / "100000_0000.png").write_bytes(b"fake")
    return iteration_dir


@pytest.fixture()
def _runs_fixture(monkeypatch, tmp_path):
    """Create a fake session with an iteration inside, point RUNS_DIR to tmp_path."""
    import p2p.api.services as svc

    session_dir = tmp_path / _FAKE_SESSION_ID
    session_dir.mkdir()
    (session_dir / "status.json").write_text(json.dumps({"status": "completed"}))
    _create_fake_iteration(session_dir, _FAKE_ITERATION_ID)
    monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
    monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)


def _get_first_iteration_id() -> str:
    resp = client.get("/api/iterations")
    iterations = resp.json()
    assert len(iterations) > 0, "No iterations found in fixture"
    return iterations[0]["iteration_id"]


# ---------------------------------------------------------------------------
# Iteration endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_runs_fixture")
def test_list_iterations_returns_200():
    resp = client.get("/api/iterations")
    assert resp.status_code == 200
    iterations = resp.json()
    assert isinstance(iterations, list)
    assert len(iterations) >= 1


@pytest.mark.usefixtures("_runs_fixture")
def test_list_iterations_has_required_fields():
    resp = client.get("/api/iterations")
    iteration = resp.json()[0]
    required = {
        "iteration_id",
        "session_id",
        "env_id",
        "status",
        "created_at",
        "total_timesteps",
        "final_episodic_return",
        "reward_latex",
        "reward_description",
        "video_urls",
        "progress",
    }
    assert required.issubset(iteration.keys())


@pytest.mark.usefixtures("_runs_fixture")
def test_list_iterations_field_types():
    resp = client.get("/api/iterations")
    it = resp.json()[0]
    assert isinstance(it["iteration_id"], str)
    assert isinstance(it["session_id"], str)
    assert isinstance(it["env_id"], str)
    assert isinstance(it["status"], str)
    assert isinstance(it["created_at"], str)
    assert isinstance(it["total_timesteps"], int)
    ret = it["final_episodic_return"]
    assert ret is None or isinstance(ret, (int, float))
    assert isinstance(it["reward_latex"], str)
    assert isinstance(it["reward_description"], str)
    assert isinstance(it["video_urls"], list)
    assert it["progress"] is None or isinstance(it["progress"], (int, float))


@pytest.mark.usefixtures("_runs_fixture")
def test_list_iterations_includes_session_id():
    resp = client.get("/api/iterations")
    iteration = resp.json()[0]
    assert iteration["session_id"] == _FAKE_SESSION_ID


@pytest.mark.usefixtures("_runs_fixture")
def test_get_iteration_detail():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["iteration_id"] == _FAKE_ITERATION_ID
    assert detail["session_id"] == _FAKE_SESSION_ID
    assert "config" in detail
    assert "reward_spec" in detail
    assert "reward_source" in detail
    assert "eval_results" in detail
    assert len(detail["eval_results"]) == 3


@pytest.mark.usefixtures("_runs_fixture")
def test_get_iteration_detail_field_types():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}")
    d = resp.json()
    # Inherited IterationSummary fields
    assert isinstance(d["iteration_id"], str)
    assert isinstance(d["session_id"], str)
    assert isinstance(d["env_id"], str)
    assert isinstance(d["status"], str)
    assert isinstance(d["total_timesteps"], int)
    assert isinstance(d["video_urls"], list)
    # IterationDetail-specific fields
    assert isinstance(d["config"], dict)
    assert isinstance(d["reward_spec"], dict)
    assert isinstance(d["reward_source"], str)
    assert d["summary"] is None or isinstance(d["summary"], dict)
    assert isinstance(d["eval_results"], list)
    assert d["judgment"] is None or isinstance(d["judgment"], dict)
    assert isinstance(d["training"], list)


@pytest.mark.usefixtures("_runs_fixture")
def test_get_iteration_detail_eval_entry_shape():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}")
    entry = resp.json()["eval_results"][0]
    assert isinstance(entry["global_step"], int)
    assert entry["type"] == "eval"
    assert isinstance(entry["total_reward"], (int, float))
    assert isinstance(entry["episode_length"], (int, float))
    assert isinstance(entry["reward_terms"], dict)


@pytest.mark.usefixtures("_runs_fixture")
def test_get_iteration_detail_404():
    resp = client.get("/api/iterations/nonexistent_iteration")
    assert resp.status_code == 404


@pytest.mark.usefixtures("_runs_fixture")
def test_get_metrics():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "training" in data
    assert "evaluation" in data
    assert len(data["training"]) > 0
    assert len(data["evaluation"]) == 3


@pytest.mark.usefixtures("_runs_fixture")
def test_metrics_training_entry_has_required_fields():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}/metrics")
    entry = resp.json()["training"][0]
    required = {
        "global_step",
        "iteration",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "sps",
    }
    assert required.issubset(entry.keys())


@pytest.mark.usefixtures("_runs_fixture")
def test_metrics_training_entry_field_types():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}/metrics")
    entry = resp.json()["training"][0]
    assert isinstance(entry["global_step"], int)
    assert isinstance(entry["iteration"], int)
    assert isinstance(entry["policy_loss"], (int, float))
    assert isinstance(entry["value_loss"], (int, float))
    assert isinstance(entry["entropy"], (int, float))
    assert isinstance(entry["approx_kl"], (int, float))
    assert isinstance(entry["clip_fraction"], (int, float))
    assert isinstance(entry["explained_variance"], (int, float))
    assert isinstance(entry["learning_rate"], (int, float))
    assert isinstance(entry["sps"], (int, float))


@pytest.mark.usefixtures("_runs_fixture")
def test_metrics_eval_entry_has_reward_terms():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}/metrics")
    entry = resp.json()["evaluation"][0]
    assert entry["type"] == "eval"
    assert "reward_terms" in entry
    assert len(entry["reward_terms"]) > 0


@pytest.mark.usefixtures("_runs_fixture")
def test_metrics_eval_entry_field_types():
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}/metrics")
    entry = resp.json()["evaluation"][0]
    assert isinstance(entry["global_step"], int)
    assert isinstance(entry["type"], str)
    assert isinstance(entry["total_reward"], (int, float))
    assert isinstance(entry["episode_length"], (int, float))
    assert isinstance(entry["reward_terms"], dict)
    for term_name, term_value in entry["reward_terms"].items():
        assert isinstance(term_name, str)
        assert isinstance(term_value, (int, float))


@pytest.mark.usefixtures("_runs_fixture")
def test_static_urls_include_session_id():
    """Static URLs should include session_id in the path."""
    resp = client.get(f"/api/iterations/{_FAKE_ITERATION_ID}")
    detail = resp.json()
    video_urls = detail["video_urls"]
    for url in video_urls:
        assert _FAKE_SESSION_ID in url


# ---------------------------------------------------------------------------
# Session endpoint tests
# ---------------------------------------------------------------------------

_LOOP_HISTORY = {
    "session_id": "session_fixture_001",
    "prompt": "run forward fast",
    "status": "passed",
    "best_iteration": 2,
    "best_score": 0.85,
    "error": None,
    "iterations": [
        {
            "iteration": 1,
            "iteration_dir": "runs/session_fixture_001/iter_1",
            "reward_code": "def reward_fn(o,a,n,i):\n    return 1.0, {}",
            "summary": {"final_episodic_return": 200.0},
            "judgment": {
                "intent_score": 0.4,
                "diagnosis": "moving slowly",
                "failure_tags": ["not_moving"],
                "passed": False,
            },
        },
        {
            "iteration": 2,
            "iteration_dir": "runs/session_fixture_001/iter_2",
            "reward_code": "def reward_fn(o,a,n,i):\n    return 2.0, {}",
            "summary": {"final_episodic_return": 800.0},
            "judgment": {
                "intent_score": 0.85,
                "diagnosis": "good forward motion",
                "failure_tags": [],
                "passed": True,
            },
        },
    ],
}


@pytest.fixture(scope="module")
def _session_env(tmp_path_factory):
    """Create a fake session directory once for all session tests.

    Module-scoped to avoid copying the 869MB runs/ dir per test.
    """
    import p2p.api.services as svc
    import p2p.settings as settings

    tmp_path = tmp_path_factory.mktemp("api_sessions")

    session_dir = tmp_path / "session_fixture_001"
    session_dir.mkdir()
    (session_dir / "loop_history.json").write_text(json.dumps(_LOOP_HISTORY))
    (session_dir / "status.json").write_text(json.dumps({"status": "passed"}))

    original_svc = svc.RUNS_DIR
    original_settings = settings.RUNS_DIR
    svc.RUNS_DIR = tmp_path
    settings.RUNS_DIR = tmp_path
    yield tmp_path
    svc.RUNS_DIR = original_svc
    settings.RUNS_DIR = original_settings


@pytest.mark.usefixtures("_session_env")
def test_list_sessions():
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert isinstance(sessions, list)
    assert any(s["session_id"] == "session_fixture_001" for s in sessions)


@pytest.mark.usefixtures("_session_env")
def test_list_sessions_entry_has_required_fields():
    resp = client.get("/api/sessions")
    session = next(s for s in resp.json() if s["session_id"] == "session_fixture_001")
    required = {
        "session_id",
        "prompt",
        "status",
        "best_iteration",
        "best_score",
        "iterations",
        "error",
        "env_id",
        "created_at",
        "total_timesteps",
        "pass_threshold",
        "alias",
        "starred",
        "tags",
    }
    assert required.issubset(session.keys())


@pytest.mark.usefixtures("_session_env")
def test_get_session_detail():
    resp = client.get("/api/sessions/session_fixture_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "session_fixture_001"
    assert data["status"] == "passed"
    assert data["best_score"] == 0.85
    assert len(data["iterations"]) == 2


@pytest.mark.usefixtures("_session_env")
def test_get_session_detail_field_types():
    resp = client.get("/api/sessions/session_fixture_001")
    d = resp.json()
    assert isinstance(d["session_id"], str)
    assert isinstance(d["prompt"], str)
    assert isinstance(d["status"], str)
    assert d["best_iteration"] is None or isinstance(d["best_iteration"], int)
    assert isinstance(d["best_score"], (int, float))
    assert isinstance(d["iterations"], list)
    assert d["error"] is None or isinstance(d["error"], str)
    assert isinstance(d["env_id"], str)
    assert isinstance(d["created_at"], str)
    assert isinstance(d["total_timesteps"], int)
    assert isinstance(d["pass_threshold"], (int, float))
    assert isinstance(d["alias"], str)
    assert isinstance(d["starred"], bool)
    assert isinstance(d["tags"], list)


@pytest.mark.usefixtures("_session_env")
def test_get_session_loop_iterations():
    resp = client.get("/api/sessions/session_fixture_001/loop-iterations")
    assert resp.status_code == 200
    iterations = resp.json()
    assert len(iterations) == 2
    assert iterations[0]["iteration"] == 1
    assert iterations[1]["intent_score"] == 0.85


@pytest.mark.usefixtures("_session_env")
def test_get_session_404():
    resp = client.get("/api/sessions/nonexistent_session")
    assert resp.status_code == 404


@pytest.mark.usefixtures("_session_env")
def test_session_iteration_has_required_fields():
    resp = client.get("/api/sessions/session_fixture_001/loop-iterations")
    it = resp.json()[0]
    required = {
        "iteration",
        "iteration_dir",
        "intent_score",
        "best_checkpoint",
        "checkpoint_scores",
        "checkpoint_diagnoses",
        "diagnosis",
        "failure_tags",
        "reward_code",
        "final_return",
        "video_urls",
        "is_multi_config",
    }
    assert required.issubset(it.keys())


@pytest.mark.usefixtures("_session_env")
def test_session_iteration_field_types():
    resp = client.get("/api/sessions/session_fixture_001/loop-iterations")
    it = resp.json()[0]
    assert isinstance(it["iteration"], int)
    assert isinstance(it["iteration_dir"], str)
    assert it["intent_score"] is None or isinstance(it["intent_score"], (int, float))
    assert isinstance(it["best_checkpoint"], str)
    assert isinstance(it["checkpoint_scores"], dict)
    assert isinstance(it["checkpoint_diagnoses"], dict)
    assert isinstance(it["diagnosis"], str)
    assert isinstance(it["failure_tags"], list)
    assert isinstance(it["reward_code"], str)
    assert it["final_return"] is None or isinstance(it["final_return"], (int, float))
    assert isinstance(it["video_urls"], list)
    assert isinstance(it["is_multi_config"], bool)


@pytest.mark.usefixtures("_runs_fixture")
def test_list_iterations_excludes_non_session_dirs(tmp_path):
    """Non-session directories should not appear in the iterations list."""
    # The _runs_fixture already creates a session dir; add a non-session dir
    non_session = tmp_path / "random_dir"
    non_session.mkdir()
    (non_session / "config.json").write_text(json.dumps({"env_id": "test"}))

    resp = client.get("/api/iterations")
    iterations = resp.json()
    iteration_ids = [r["iteration_id"] for r in iterations]
    # Non-session dir contents should not leak into iterations list
    assert "random_dir" not in iteration_ids
