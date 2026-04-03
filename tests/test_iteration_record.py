"""Tests for IterationRecord typed iteration directory accessor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2p.session.iteration_record import IterationRecord


@pytest.fixture()
def iteration_dir(tmp_path: Path) -> Path:
    """Create a minimal valid iteration directory."""
    d = tmp_path / "20260303_120000_abcd1234"
    d.mkdir()
    config = {"env_id": "HalfCheetah-v4", "total_timesteps": 100_000}
    (d / "config.json").write_text(json.dumps(config))
    (d / "reward_fn.py").write_text('def reward_fn(o, a, n, i): return 0.0, {"x": 0.0}\n')
    (d / "reward_spec.json").write_text(json.dumps({"latex": "r=0", "terms": {}}))
    return d


class TestIterationRecordPaths:
    def test_iteration_id(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.iteration_id == iteration_dir.name

    def test_all_paths_are_under_iteration_dir(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        paths = [
            rec.config_path,
            rec.reward_fn_path,
            rec.reward_spec_path,
            rec.summary_path,
            rec.scalars_path,
            rec.trajectory_path,
            rec.prompt_path,
            rec.videos_dir,
            rec.judgment_path,
            rec.revised_reward_path,
        ]
        for p in paths:
            assert str(p).startswith(str(iteration_dir))


class TestIterationRecordRead:
    def test_read_config(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        cfg = rec.read_config()
        assert cfg is not None
        assert cfg["env_id"] == "HalfCheetah-v4"

    def test_read_summary_missing(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.read_summary() is None

    def test_read_summary_present(self, iteration_dir: Path):
        (iteration_dir / "summary.json").write_text(json.dumps({"final_episodic_return": 500.0}))
        rec = IterationRecord(iteration_dir)
        s = rec.read_summary()
        assert s is not None
        assert s["final_episodic_return"] == 500.0

    def test_read_reward_source(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        src = rec.read_reward_source()
        assert "reward_fn" in src

    def test_read_reward_spec(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        spec = rec.read_reward_spec()
        assert spec["latex"] == "r=0"


class TestIterationRecordStatus:
    def test_status_pending(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "pending"

    def test_status_running(self, iteration_dir: Path):
        metrics = iteration_dir / "metrics"
        metrics.mkdir()
        (metrics / "scalars.jsonl").write_text(json.dumps({"global_step": 50_000}) + "\n")
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "running"

    def test_status_completed(self, iteration_dir: Path):
        (iteration_dir / "summary.json").write_text(json.dumps({"final_episodic_return": 100}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "completed"

    def test_status_unknown(self, tmp_path: Path):
        empty = tmp_path / "empty_iteration"
        empty.mkdir()
        rec = IterationRecord(empty)
        assert rec.derive_status() == "unknown"


class TestIterationRecordStatusJson:
    """Tests for explicit status.json based derive_status."""

    def test_status_json_running(self, iteration_dir: Path):
        (iteration_dir / "status.json").write_text(json.dumps({"status": "running"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "running"

    def test_status_json_completed(self, iteration_dir: Path):
        (iteration_dir / "status.json").write_text(json.dumps({"status": "completed"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "completed"

    def test_status_json_error(self, iteration_dir: Path):
        data = {"status": "error", "error": "CUDA OOM"}
        (iteration_dir / "status.json").write_text(json.dumps(data))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "error"

    def test_status_json_cancelled(self, iteration_dir: Path):
        (iteration_dir / "status.json").write_text(json.dumps({"status": "cancelled"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "cancelled"

    def test_status_json_overrides_file_inference(self, iteration_dir: Path):
        """status.json should take priority over summary.json presence."""
        (iteration_dir / "summary.json").write_text(json.dumps({"final_episodic_return": 100}))
        (iteration_dir / "status.json").write_text(json.dumps({"status": "error"}))
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "error"

    def test_fallback_when_no_status_json(self, iteration_dir: Path):
        """Without status.json, should fall back to file inference."""
        rec = IterationRecord(iteration_dir)
        assert rec.derive_status() == "pending"  # config.json exists


class TestIterationRecordProgress:
    def test_progress_no_config(self, tmp_path: Path):
        rec = IterationRecord(tmp_path / "no_config")
        rec.path.mkdir()
        assert rec.compute_progress() is None

    def test_progress_no_scalars(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.compute_progress() == 0.0

    def test_progress_partial(self, iteration_dir: Path):
        metrics = iteration_dir / "metrics"
        metrics.mkdir()
        (metrics / "scalars.jsonl").write_text(json.dumps({"global_step": 50_000}) + "\n")
        rec = IterationRecord(iteration_dir)
        assert rec.compute_progress() == pytest.approx(0.5)


class TestIterationRecordValidate:
    def test_valid_iteration(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.validate() == []

    def test_missing_dir(self, tmp_path: Path):
        rec = IterationRecord(tmp_path / "nonexistent")
        issues = rec.validate()
        assert len(issues) == 1
        assert "does not exist" in issues[0]

    def test_missing_config(self, tmp_path: Path):
        d = tmp_path / "bad_iteration"
        d.mkdir()
        (d / "reward_fn.py").write_text("x = 1")
        rec = IterationRecord(d)
        assert any("config.json" in i for i in rec.validate())


class TestIterationRecordScalars:
    def test_parse_scalars_empty(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        training, evaluation = rec.parse_scalars()
        assert training == []
        assert evaluation == []

    def test_parse_scalars_mixed(self, iteration_dir: Path):
        metrics = iteration_dir / "metrics"
        metrics.mkdir()
        lines = [
            json.dumps({"global_step": 1000, "policy_loss": 0.5}),
            json.dumps({"global_step": 2000, "type": "eval", "total_reward": 100}),
        ]
        (metrics / "scalars.jsonl").write_text("\n".join(lines) + "\n")
        rec = IterationRecord(iteration_dir)
        training, evaluation = rec.parse_scalars()
        assert len(training) == 1
        assert len(evaluation) == 1
        assert evaluation[0]["total_reward"] == 100


class TestReadJsonCorruptedFiles:
    """Regression tests for #54: _read_json must tolerate partial/corrupt files."""

    def test_read_json_empty_file(self, iteration_dir: Path):
        (iteration_dir / "status.json").write_text("")
        rec = IterationRecord(iteration_dir)
        assert rec._read_json(iteration_dir / "status.json") is None

    def test_read_json_invalid_json(self, iteration_dir: Path):
        (iteration_dir / "status.json").write_text('{"status": "running"')
        rec = IterationRecord(iteration_dir)
        assert rec._read_json(iteration_dir / "status.json") is None

    def test_read_json_missing_file(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec._read_json(iteration_dir / "nonexistent.json") is None

    def test_read_json_valid_file(self, iteration_dir: Path):
        data = {"status": "running"}
        (iteration_dir / "status.json").write_text(json.dumps(data))
        rec = IterationRecord(iteration_dir)
        assert rec._read_json(iteration_dir / "status.json") == data


class TestSessionRecordReadJsonCorrupted:
    """Regression tests for #54: SessionRecord._read_json must tolerate partial writes."""

    def test_session_read_history_corrupt(self, tmp_path: Path):
        from p2p.session.iteration_record import SessionRecord

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        (session_dir / "loop_history.json").write_text('{"session_id": "abc"')
        sr = SessionRecord(session_dir)
        assert sr.read_history() is None

    def test_session_read_status_corrupt(self, tmp_path: Path):
        from p2p.session.iteration_record import SessionRecord

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        (session_dir / "status.json").write_text("")
        sr = SessionRecord(session_dir)
        assert sr.read_status() is None

    def test_session_read_history_valid(self, tmp_path: Path):
        from p2p.session.iteration_record import SessionRecord

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        data = {"session_id": "abc", "status": "running"}
        (session_dir / "loop_history.json").write_text(json.dumps(data))
        sr = SessionRecord(session_dir)
        assert sr.read_history() == data


class TestAtomicWrite:
    """Tests for _atomic_write ensuring no partial reads."""

    def test_atomic_write_creates_file(self, tmp_path: Path):
        from p2p.session.iteration_record import _atomic_write

        target = tmp_path / "test.json"
        _atomic_write(target, '{"key": "value"}')
        assert json.loads(target.read_text()) == {"key": "value"}

    def test_atomic_write_overwrites_file(self, tmp_path: Path):
        from p2p.session.iteration_record import _atomic_write

        target = tmp_path / "test.json"
        target.write_text('{"old": true}')
        _atomic_write(target, '{"new": true}')
        assert json.loads(target.read_text()) == {"new": True}

    def test_atomic_write_no_temp_file_left(self, tmp_path: Path):
        from p2p.session.iteration_record import _atomic_write

        target = tmp_path / "test.json"
        _atomic_write(target, '{"key": "value"}')
        files = list(tmp_path.iterdir())
        assert files == [target]


class TestIterationRecordFiles:
    def test_video_filenames_empty(self, iteration_dir: Path):
        rec = IterationRecord(iteration_dir)
        assert rec.video_filenames() == []

    def test_video_filenames(self, iteration_dir: Path):
        vdir = iteration_dir / "videos"
        vdir.mkdir()
        (vdir / "eval_100000.mp4").write_bytes(b"fake")
        (vdir / "eval_500000.mp4").write_bytes(b"fake")
        rec = IterationRecord(iteration_dir)
        assert rec.video_filenames() == ["eval_100000.mp4", "eval_500000.mp4"]
