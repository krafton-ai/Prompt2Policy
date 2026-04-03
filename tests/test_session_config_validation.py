"""Tests for runtime validation in get_session_config()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2p.api import services


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sess_001"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _patch_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("p2p.settings.RUNS_DIR", tmp_path)
    monkeypatch.setattr(services, "RUNS_DIR", tmp_path)


def _write_config(session_dir: Path, data: dict) -> None:
    (session_dir / "session_config.json").write_text(json.dumps(data))


class TestGetSessionConfig:
    def test_valid_config(self, session_dir: Path):
        _write_config(
            session_dir,
            {
                "prompt": "run fast",
                "train": {"total_timesteps": 100_000, "seed": 42},
            },
        )
        result = services.get_session_config("sess_001")
        assert result is not None
        assert result["prompt"] == "run fast"
        assert result["train"]["seed"] == 42

    def test_missing_required_key_returns_none(self, session_dir: Path):
        _write_config(session_dir, {"prompt": "run fast"})
        assert services.get_session_config("sess_001") is None

    def test_empty_json_returns_none(self, session_dir: Path):
        _write_config(session_dir, {})
        assert services.get_session_config("sess_001") is None

    def test_no_file_returns_none(self, tmp_path: Path):
        (tmp_path / "sess_002").mkdir()
        assert services.get_session_config("sess_002") is None

    def test_corrupt_json_returns_none(self, session_dir: Path):
        (session_dir / "session_config.json").write_text("{bad json")
        assert services.get_session_config("sess_001") is None

    def test_extra_keys_accepted(self, session_dir: Path):
        """Optional/unknown keys must not cause rejection."""
        _write_config(
            session_dir,
            {
                "prompt": "run fast",
                "train": {"total_timesteps": 100_000, "seed": 42},
                "env_id": "HalfCheetah-v5",
                "unknown_future_field": True,
            },
        )
        result = services.get_session_config("sess_001")
        assert result is not None

    def test_new_nested_format(self, session_dir: Path):
        """New format with LoopConfig-derived train dict."""
        _write_config(
            session_dir,
            {
                "prompt": "run fast",
                "train": {
                    "env_id": "HalfCheetah-v5",
                    "total_timesteps": 100_000,
                    "seed": 42,
                },
                "max_iterations": 5,
            },
        )
        result = services.get_session_config("sess_001")
        assert result is not None
        assert result["prompt"] == "run fast"
        assert result["train"]["total_timesteps"] == 100_000

    def test_new_format_missing_prompt_returns_none(self, session_dir: Path):
        _write_config(
            session_dir,
            {"train": {"total_timesteps": 100_000, "seed": 42}},
        )
        assert services.get_session_config("sess_001") is None

    def test_missing_train_key_returns_none(self, session_dir: Path):
        """Missing required 'train' key → None."""
        _write_config(session_dir, {"prompt": "run fast"})
        assert services.get_session_config("sess_001") is None
