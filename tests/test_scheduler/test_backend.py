"""Tests for backend utility functions and Backend implementations."""

from unittest.mock import MagicMock, patch

import pytest

from p2p.scheduler import node_store
from p2p.scheduler.backend import (
    SSHBackend,
    check_node,
)
from p2p.scheduler.ssh_utils import params_to_cli_args as _params_to_cli_args
from p2p.scheduler.types import RunSpec

# ---------------------------------------------------------------------------
# _params_to_cli_args
# ---------------------------------------------------------------------------


class TestParamsToCli:
    def test_string_param(self) -> None:
        args = _params_to_cli_args({"prompt": "hello world"})
        assert args == ["--prompt", "hello world"]

    def test_int_param(self) -> None:
        args = _params_to_cli_args({"seed": 42})
        assert args == ["--seed", "42"]

    def test_bool_true(self) -> None:
        args = _params_to_cli_args({"side_info": True})
        assert args == ["--side-info"]

    def test_bool_false_omitted(self) -> None:
        args = _params_to_cli_args({"side_info": False})
        assert args == []

    def test_none_omitted(self) -> None:
        args = _params_to_cli_args({"vlm_model": None})
        assert args == []

    def test_list_param(self) -> None:
        args = _params_to_cli_args({"seeds": [1, 2, 3]})
        assert args == ["--seeds", "1,2,3"]

    def test_empty_list_omitted(self) -> None:
        args = _params_to_cli_args({"seeds": []})
        assert args == []

    def test_dict_param_json(self) -> None:
        args = _params_to_cli_args({"config_overrides": {"lr": 0.001}})
        assert args[0] == "--config-overrides"
        assert '"lr"' in args[1]

    def test_underscore_to_dash(self) -> None:
        args = _params_to_cli_args({"total_timesteps": 1000})
        assert args[0] == "--total-timesteps"

    def test_multiple_params(self) -> None:
        args = _params_to_cli_args(
            {
                "prompt": "test",
                "seed": 1,
                "side_info": True,
                "vlm_model": None,
            }
        )
        assert "--prompt" in args
        assert "--seed" in args
        assert "--side-info" in args
        assert "--vlm-model" not in args


# ---------------------------------------------------------------------------
# check_node
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path):
    node_store.set_store_path(tmp_path / "nodes.json")


class TestCheckNode:
    def test_check_node_not_found(self) -> None:
        result = check_node("nonexistent")
        assert result["online"] is False
        assert "not found" in (result.get("error") or "")

    def test_check_node_success(self) -> None:
        node_store.add_node(
            {
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "user",
                "port": 22,
                "max_cores": 60,
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="ok\n/usr/local/bin/uv\n",
                stderr="",
            )
            result = check_node("n1")

        assert result["online"] is True
        assert result["uv_available"] is True

    def test_check_node_timeout(self) -> None:
        import subprocess

        node_store.add_node(
            {
                "node_id": "n1",
                "host": "10.0.0.1",
                "user": "user",
                "port": 22,
                "max_cores": 60,
            }
        )
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 5)):
            result = check_node("n1")

        assert result["online"] is False
        assert "timed out" in (result.get("error") or "")


# ---------------------------------------------------------------------------
# SSHBackend
# ---------------------------------------------------------------------------

_NODE = {
    "node_id": "gpu-1",
    "host": "10.0.0.1",
    "user": "user",
    "port": 22,
    "max_cores": 60,
}


def _make_spec(run_id: str = "run_abc") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        entry_point="p2p.session.run_session",
        parameters={"session_id": run_id, "prompt": "test"},
        cpu_cores=4,
    )


class TestSSHBackend:
    def test_submit_success(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(12345, None)),
        ):
            status = backend.submit(_make_spec())

        assert status["state"] == "running"
        assert status["pid"] == 12345
        assert status["node_id"] == "gpu-1"

    def test_submit_sync_failure(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=False),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
        ):
            status = backend.submit(_make_spec())

        assert status["state"] == "error"
        assert "sync" in (status.get("error") or "").lower()

    def test_submit_ssh_error(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch(
                "p2p.scheduler.ssh_utils.submit_ssh",
                return_value=(None, "Connection refused"),
            ),
        ):
            status = backend.submit(_make_spec())

        assert status["state"] == "error"
        assert "Connection refused" in (status.get("error") or "")

    def test_submit_with_allocated_cores(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(999, None)) as mock_submit,
        ):
            backend.submit(_make_spec(), allocated_cores=[0, 1, 2, 3])

        assert mock_submit.call_args.kwargs["cpu_cores"] == [0, 1, 2, 3]

    def test_status_alive(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(100, None)),
        ):
            backend.submit(_make_spec("run_1"))

        with patch("p2p.scheduler.ssh_utils.check_ssh_alive", return_value=(True, "")):
            status = backend.status("run_1")

        assert status["state"] == "running"

    def test_status_dead_completed(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(100, None)),
        ):
            backend.submit(_make_spec("run_2"))

        with patch("p2p.scheduler.ssh_utils.check_ssh_alive", return_value=(False, "completed")):
            status = backend.status("run_2")

        assert status["state"] == "completed"

    def test_status_dead_error(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(100, None)),
        ):
            backend.submit(_make_spec("run_3"))

        with patch("p2p.scheduler.ssh_utils.check_ssh_alive", return_value=(False, "error")):
            status = backend.status("run_3")

        assert status["state"] == "error"

    def test_status_unknown_run(self) -> None:
        backend = SSHBackend(_NODE)
        status = backend.status("nonexistent")
        assert status["state"] == "error"
        assert "Unknown" in (status.get("error") or "")

    def test_cancel(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(100, None)),
        ):
            backend.submit(_make_spec("run_c"))

        with patch("p2p.scheduler.ssh_utils.kill_ssh_process", return_value=True):
            assert backend.cancel("run_c") is True

        with patch("p2p.scheduler.ssh_utils.check_ssh_alive", return_value=(False, "")):
            status = backend.status("run_c")
        assert status["state"] == "cancelled"

    def test_cancel_unknown_run(self) -> None:
        backend = SSHBackend(_NODE)
        assert backend.cancel("nonexistent") is False

    def test_sync_results(self) -> None:
        backend = SSHBackend(_NODE)
        with (
            patch("p2p.scheduler.ssh_utils.sync_code", return_value=True),
            patch("p2p.scheduler.ssh_utils.remote_work_dir", return_value="/tmp/p2p-abc"),
            patch("p2p.scheduler.ssh_utils.submit_ssh", return_value=(100, None)),
        ):
            backend.submit(_make_spec("run_s"))

        with patch("p2p.scheduler.ssh_utils.sync_full_results", return_value=True):
            assert backend.sync_results("run_s") is True

    def test_sync_results_unknown_run(self) -> None:
        backend = SSHBackend(_NODE)
        assert backend.sync_results("nonexistent") is False
