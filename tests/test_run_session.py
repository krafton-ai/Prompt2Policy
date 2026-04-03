"""Tests for run_session.py CLI entrypoint."""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from p2p.config import LoopConfig, TrainConfig


def test_import():
    """run_session module imports without error."""
    import p2p.session.run_session  # noqa: F401


def _make_loop_config_json(tmp_path: Path, **overrides) -> str:
    """Build a LoopConfig JSON string for CLI testing."""
    train = TrainConfig(total_timesteps=50000, seed=42)
    lc = LoopConfig(train=train, runs_dir=tmp_path, **overrides)
    return lc.to_json()


class TestArgParsing:
    """Test argument parsing by calling main() with sys.argv patched."""

    def _run_main(self, args: list[str]):
        """Call main() with mocked dependencies, return call_args."""
        mock_result = {
            "session_id": "test-sess",
            "prompt": "test",
            "status": "passed",
            "iterations": [],
            "best_iteration": 0,
            "best_score": 0.8,
            "pass_threshold": 0.7,
        }

        with (
            patch("sys.argv", ["run_session", *args]),
            patch("p2p.inference.llm_client.get_client", return_value=MagicMock()),
            patch("p2p.session.loop.run_loop", return_value=mock_result) as mock_run_loop,
            patch("p2p.session.iteration_record.SessionRecord"),
        ):
            from p2p.session.run_session import main

            main()
            return mock_run_loop.call_args

    def test_loop_config_deserialized(self, tmp_path: Path):
        lc_json = _make_loop_config_json(tmp_path, max_iterations=10, pass_threshold=0.9)
        call = self._run_main(
            [
                "--session-id",
                "sess-1",
                "--prompt",
                "make it run fast",
                "--loop-config",
                lc_json,
            ]
        )

        assert call.args[0] == "make it run fast"
        assert call.kwargs["session_id"] == "sess-1"
        loop_config = call.args[1]
        assert loop_config.max_iterations == 10
        assert loop_config.pass_threshold == 0.9
        assert loop_config.train.total_timesteps == 50000
        assert loop_config.train.seed == 42

    def test_defaults_from_loop_config(self, tmp_path: Path):
        lc_json = _make_loop_config_json(tmp_path)
        call = self._run_main(
            [
                "--session-id",
                "sess-1",
                "--prompt",
                "test",
                "--loop-config",
                lc_json,
            ]
        )

        loop_config = call.args[1]
        assert loop_config.max_iterations == 5
        assert loop_config.pass_threshold == 0.7


class TestMainSmoke:
    def test_main_calls_run_loop(self, tmp_path: Path):
        """main() wires args correctly and calls run_loop."""
        mock_result = {
            "session_id": "test-sess",
            "prompt": "run fast",
            "status": "passed",
            "iterations": [],
            "best_iteration": 0,
            "best_score": 0.8,
            "pass_threshold": 0.7,
        }

        lc_json = _make_loop_config_json(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "run_session",
                    "--session-id",
                    "test-sess",
                    "--prompt",
                    "run fast",
                    "--loop-config",
                    lc_json,
                ],
            ),
            patch("p2p.inference.llm_client.get_client", return_value=MagicMock()),
            patch("p2p.session.loop.run_loop", return_value=mock_result) as mock_run_loop,
            patch("p2p.session.iteration_record.SessionRecord"),
        ):
            from p2p.session.run_session import main

            main()

            mock_run_loop.assert_called_once()
            kw = mock_run_loop.call_args.kwargs
            loop_config = mock_run_loop.call_args.args[1]
            assert kw["session_id"] == "test-sess"
            assert loop_config.runs_dir == tmp_path

    def test_sigterm_handler_installed(self, tmp_path: Path):
        """main() registers a SIGTERM handler using signal.signal()."""
        mock_result = {
            "session_id": "test-sess",
            "prompt": "run fast",
            "status": "passed",
            "iterations": [],
            "best_iteration": 0,
            "best_score": 0.8,
            "pass_threshold": 0.7,
        }

        lc_json = _make_loop_config_json(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "run_session",
                    "--session-id",
                    "test-sess",
                    "--prompt",
                    "run fast",
                    "--loop-config",
                    lc_json,
                ],
            ),
            patch("p2p.inference.llm_client.get_client", return_value=MagicMock()),
            patch("p2p.session.loop.run_loop", return_value=mock_result),
            patch("p2p.session.iteration_record.SessionRecord"),
            patch("signal.signal") as mock_signal,
        ):
            from p2p.session.run_session import main

            main()

            mock_signal.assert_any_call(signal.SIGTERM, ANY)
