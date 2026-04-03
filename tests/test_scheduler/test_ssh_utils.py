"""Tests for p2p.scheduler.ssh_utils."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from p2p.scheduler import ssh_utils
from p2p.scheduler.ssh_utils import (
    check_ssh_alive,
    cleanup_remote_session,
    find_node,
    kill_ssh_process,
    load_nodes,
    params_to_cli_args,
    remote_work_dir,
    resolve_node,
    rsync_pull_cmd,
    ssh_base_cmd,
    submit_ssh,
    sync_code,
    sync_full_results,
    sync_lite_results,
    sync_running_status,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(
    node_id: str = "n1",
    host: str = "10.0.0.1",
    port: int = 22,
    user: str = "testuser",
    max_cores: int = 8,
    **kwargs,
) -> dict:
    return {
        "node_id": node_id,
        "host": host,
        "port": port,
        "user": user,
        "max_cores": max_cores,
        "base_dir": "/tmp/p2p",
        **kwargs,
    }


@pytest.fixture()
def node():
    return _node()


@pytest.fixture(autouse=True)
def _reset_failure_counter():
    """Reset the module-level consecutive failure counter between tests."""
    ssh_utils._ssh_consecutive_failures.clear()
    yield
    ssh_utils._ssh_consecutive_failures.clear()


# ---------------------------------------------------------------------------
# ssh_base_cmd
# ---------------------------------------------------------------------------


class TestSSHBaseCmd:
    def test_basic_command_structure(self, node):
        cmd = ssh_base_cmd(node)
        assert cmd[0] == "ssh"
        assert "-p" in cmd
        assert "22" in cmd
        assert "testuser@10.0.0.1" in cmd

    def test_includes_standard_options(self, node):
        cmd = ssh_base_cmd(node)
        joined = " ".join(cmd)
        assert "StrictHostKeyChecking=accept-new" in joined
        assert "BatchMode=yes" in joined
        assert "ConnectTimeout=10" in joined

    def test_includes_keepalive_options(self, node):
        """ServerAliveInterval and ServerAliveCountMax prevent stale hangs (#388)."""
        cmd = ssh_base_cmd(node)
        joined = " ".join(cmd)
        assert "ServerAliveInterval=15" in joined
        assert "ServerAliveCountMax=2" in joined

    def test_custom_port(self):
        node = _node(port=2222)
        cmd = ssh_base_cmd(node)
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "2222"


# ---------------------------------------------------------------------------
# rsync_pull_cmd
# ---------------------------------------------------------------------------


class TestRsyncPullCmd:
    def test_basic_structure(self, node):
        cmd = rsync_pull_cmd(node, "/remote/path/", "/local/path/")
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        assert "--timeout=30" in cmd
        assert f"{node['user']}@{node['host']}:/remote/path/" in cmd
        assert "/local/path/" in cmd

    def test_ssh_options_in_command(self, node):
        cmd = rsync_pull_cmd(node, "/r", "/l")
        ssh_flag_idx = cmd.index("-e")
        ssh_opts = cmd[ssh_flag_idx + 1]
        assert f"-p {node['port']}" in ssh_opts
        assert "BatchMode=yes" in ssh_opts


# ---------------------------------------------------------------------------
# params_to_cli_args
# ---------------------------------------------------------------------------


class TestParamsToCliArgs:
    def test_empty_params(self):
        assert params_to_cli_args({}) == []

    def test_none_value_skipped(self):
        assert params_to_cli_args({"key": None}) == []

    def test_false_value_skipped(self):
        assert params_to_cli_args({"verbose": False}) == []

    def test_true_value_becomes_flag(self):
        assert params_to_cli_args({"verbose": True}) == ["--verbose"]

    def test_string_value(self):
        assert params_to_cli_args({"name": "hello"}) == ["--name", "hello"]

    def test_int_value(self):
        assert params_to_cli_args({"count": 42}) == ["--count", "42"]

    def test_float_value(self):
        assert params_to_cli_args({"lr": 0.001}) == ["--lr", "0.001"]

    def test_list_value_joined(self):
        result = params_to_cli_args({"seeds": [1, 2, 3]})
        assert result == ["--seeds", "1,2,3"]

    def test_empty_list_skipped(self):
        assert params_to_cli_args({"seeds": []}) == []

    def test_dict_value_as_json(self):
        d = {"key": "val"}
        result = params_to_cli_args({"config": d})
        assert result == ["--config", json.dumps(d)]

    def test_underscore_to_hyphen(self):
        result = params_to_cli_args({"loop_config": "data"})
        assert result[0] == "--loop-config"

    def test_mixed_params(self):
        params = {
            "verbose": True,
            "quiet": False,
            "name": "test",
            "seeds": [1, 2],
            "extra": None,
            "config": {"a": 1},
        }
        result = params_to_cli_args(params)
        assert "--verbose" in result
        assert "--quiet" not in result
        assert "--extra" not in result
        assert "--name" in result
        assert "--seeds" in result
        assert "--config" in result


# ---------------------------------------------------------------------------
# resolve_node
# ---------------------------------------------------------------------------


class TestResolveNode:
    @pytest.fixture(autouse=True)
    def _mock_load_nodes(self):
        nodes: list[dict] = []
        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=nodes):
            self._nodes = nodes
            yield

    def test_empty_nodes_returns_none(self):
        assert resolve_node(None, {}) is None

    def test_explicit_node_id_found(self):
        self._nodes.append(_node("n1"))
        result = resolve_node("n1", {})
        assert result is not None
        assert result["node_id"] == "n1"

    def test_explicit_node_id_not_found(self):
        self._nodes.append(_node("n1"))
        assert resolve_node("n2", {}) is None

    def test_disabled_node_explicit_returns_none(self):
        self._nodes.append(_node("n1", enabled=False))
        assert resolve_node("n1", {}) is None

    def test_disabled_node_excluded_from_auto(self):
        self._nodes.append(_node("n1", max_cores=8, enabled=False))
        self._nodes.append(_node("n2", max_cores=4))
        result = resolve_node(None, {})
        assert result is not None
        assert result["node_id"] == "n2"

    def test_auto_assign_picks_most_free_cores(self):
        self._nodes.append(_node("n1", max_cores=4))
        self._nodes.append(_node("n2", max_cores=8))
        result = resolve_node(None, {})
        assert result is not None
        assert result["node_id"] == "n2"

    def test_auto_assign_accounts_for_used_cores(self):
        self._nodes.append(_node("n1", max_cores=8))
        self._nodes.append(_node("n2", max_cores=8))
        used = {"n2": 6}  # n2 has only 2 free
        result = resolve_node(None, used, required_cores=4)
        assert result is not None
        assert result["node_id"] == "n1"

    def test_insufficient_cores_returns_none(self):
        self._nodes.append(_node("n1", max_cores=4))
        used = {"n1": 3}  # only 1 free
        assert resolve_node(None, used, required_cores=2) is None

    def test_skip_nodes_excludes(self):
        self._nodes.append(_node("n1", max_cores=8))
        self._nodes.append(_node("n2", max_cores=4))
        result = resolve_node(None, {}, skip_nodes={"n1"})
        assert result is not None
        assert result["node_id"] == "n2"

    def test_skip_nodes_ignored_for_explicit(self):
        self._nodes.append(_node("n1", max_cores=8))
        result = resolve_node("n1", {}, skip_nodes={"n1"})
        assert result is not None
        assert result["node_id"] == "n1"

    def test_enabled_defaults_to_true(self):
        n = _node("n1")
        n.pop("enabled", None)  # no explicit 'enabled' key
        self._nodes.append(n)
        result = resolve_node(None, {})
        assert result is not None


# ---------------------------------------------------------------------------
# submit_ssh
# ---------------------------------------------------------------------------


class TestSubmitSSH:
    def _base_kwargs(self, node):
        return {
            "node": node,
            "remote_dir": "/tmp/p2p-abc123",
            "entry_point": "p2p.session.run_session",
            "parameters": {"session_id": "s1"},
            "run_id": "r1",
        }

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success_returns_pid(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="12345\n", stderr="")
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid == 12345
        assert err is None

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure_returns_error(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Permission denied")
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid is None
        assert "Permission denied" in err

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure_empty_stderr_fallback(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid is None
        assert err == "SSH command failed"

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout_returns_error(self, mock_run, node):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=60)
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid is None
        assert err is not None

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_invalid_pid_returns_error(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="not-a-number\n", stderr="")
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid is None
        assert err is not None  # ValueError message

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_os_error_returns_error(self, mock_run, node):
        mock_run.side_effect = OSError("Connection refused")
        pid, err = submit_ssh(**self._base_kwargs(node))
        assert pid is None
        assert "Connection refused" in err

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_loop_config_json_injection(self, mock_run, node):
        """Verify loop_config gets remote runs_dir and cores_pool injected."""
        lc = json.dumps({"train": {}, "runs_dir": "/local/runs"})
        kwargs = self._base_kwargs(node)
        kwargs["parameters"] = {"session_id": "s1", "loop_config": lc}
        kwargs["cpu_cores"] = [0, 1, 2]
        mock_run.return_value = MagicMock(returncode=0, stdout="999\n", stderr="")

        submit_ssh(**kwargs)

        call_args = mock_run.call_args[0][0]
        # The remote command is the last element in the SSH command
        remote_cmd = call_args[-1]
        assert "p2p.session.run_session" in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_localhost_preserves_original_runs_dir(self, mock_run):
        """localhost nodes should NOT override runs_dir in LoopConfig."""
        node = _node(host="127.0.0.1")
        kwargs = self._base_kwargs(node)
        original_lc = {"train": {}, "runs_dir": "/local/runs"}
        kwargs["parameters"] = {"session_id": "s1", "loop_config": json.dumps(original_lc)}
        mock_run.return_value = MagicMock(returncode=0, stdout="100\n", stderr="")
        submit_ssh(**kwargs)

        remote_cmd = mock_run.call_args[0][0][-1]
        assert "/local/runs" in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_cpu_cores_taskset(self, mock_run, node):
        """CPU cores should produce a taskset prefix."""
        kwargs = self._base_kwargs(node)
        kwargs["cpu_cores"] = [0, 1]
        mock_run.return_value = MagicMock(returncode=0, stdout="555\n", stderr="")
        submit_ssh(**kwargs)

        call_args = mock_run.call_args[0][0]
        remote_cmd = call_args[-1]
        assert "taskset -c 0,1" in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_no_cpu_cores_no_taskset(self, mock_run, node):
        kwargs = self._base_kwargs(node)
        mock_run.return_value = MagicMock(returncode=0, stdout="555\n", stderr="")
        submit_ssh(**kwargs)

        call_args = mock_run.call_args[0][0]
        remote_cmd = call_args[-1]
        assert "taskset" not in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_shlex_quote_in_args(self, mock_run, node):
        """Parameters with special chars should be shell-quoted."""
        import shlex

        dangerous = "hello world; rm -rf /"
        kwargs = self._base_kwargs(node)
        kwargs["parameters"] = {"session_id": "s1", "prompt": dangerous}
        mock_run.return_value = MagicMock(returncode=0, stdout="123\n", stderr="")
        submit_ssh(**kwargs)

        call_args = mock_run.call_args[0][0]
        remote_cmd = call_args[-1]
        # The dangerous string must appear shell-quoted, not bare
        assert shlex.quote(dangerous) in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_mujoco_gl_for_remote_node(self, mock_run):
        """Remote (non-localhost) nodes should export MUJOCO_GL=egl."""
        node = _node(host="10.0.0.5")
        mock_run.return_value = MagicMock(returncode=0, stdout="100\n", stderr="")
        submit_ssh(**self._base_kwargs(node))

        remote_cmd = mock_run.call_args[0][0][-1]
        assert "MUJOCO_GL=egl" in remote_cmd

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_mujoco_gl_for_localhost(self, mock_run):
        """Localhost nodes should also export MUJOCO_GL=egl (headless Linux)."""
        node = _node(host="localhost")
        mock_run.return_value = MagicMock(returncode=0, stdout="100\n", stderr="")
        submit_ssh(**self._base_kwargs(node))

        remote_cmd = mock_run.call_args[0][0][-1]
        assert "MUJOCO_GL=egl" in remote_cmd


# ---------------------------------------------------------------------------
# check_ssh_alive
# ---------------------------------------------------------------------------


class TestCheckSSHAlive:
    def _base_kwargs(self, node):
        return {
            "pid": 12345,
            "node": node,
            "remote_dir": "/tmp/p2p",
            "session_id": "s1",
        }

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_alive_response(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="alive\n", stderr="")
        alive, status = check_ssh_alive(**self._base_kwargs(node))
        assert alive is True
        assert status == ""

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_dead_with_valid_status_json(self, mock_run, node):
        status_json = json.dumps({"status": "completed"})
        mock_run.return_value = MagicMock(returncode=0, stdout=f"dead\n{status_json}", stderr="")
        alive, status = check_ssh_alive(**self._base_kwargs(node))
        assert alive is False
        assert status == "completed"

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_dead_with_malformed_json(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="dead\n{bad json", stderr="")
        alive, status = check_ssh_alive(**self._base_kwargs(node))
        assert alive is False
        assert status == ""

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_dead_with_empty_status(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="dead\n{}", stderr="")
        alive, status = check_ssh_alive(**self._base_kwargs(node))
        assert alive is False
        assert status == ""

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_dead_no_second_line(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0, stdout="dead", stderr="")
        alive, status = check_ssh_alive(**self._base_kwargs(node))
        assert alive is False
        assert status == ""

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout_counted_as_failure(self, mock_run, node):
        """SSH timeout is transient — first failure returns alive (threshold=3)."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        kwargs = self._base_kwargs(node)

        alive, _ = check_ssh_alive(**kwargs)
        assert alive is True  # first failure, still alive

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout_consecutive_failures_at_threshold(self, mock_run, node):
        """After 3 consecutive timeouts, process is declared dead."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        kwargs = self._base_kwargs(node)

        check_ssh_alive(**kwargs)  # failure 1
        check_ssh_alive(**kwargs)  # failure 2
        alive, status = check_ssh_alive(**kwargs)  # failure 3
        assert alive is False
        assert status == "error"

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_os_error_counted_as_failure(self, mock_run, node):
        """OSError is transient — first failure returns alive (threshold=3)."""
        mock_run.side_effect = OSError("Network unreachable")
        kwargs = self._base_kwargs(node)

        alive, _ = check_ssh_alive(**kwargs)
        assert alive is True  # first failure, still alive

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_os_error_consecutive_failures_at_threshold(self, mock_run, node):
        """After 3 consecutive OSErrors, process is declared dead."""
        mock_run.side_effect = OSError("Network unreachable")
        kwargs = self._base_kwargs(node)

        check_ssh_alive(**kwargs)  # failure 1
        check_ssh_alive(**kwargs)  # failure 2
        alive, status = check_ssh_alive(**kwargs)  # failure 3 — threshold

        assert alive is False
        assert status == "error"

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success_resets_os_error_counter(self, mock_run, node):
        """A successful check resets the consecutive OSError counter."""
        kwargs = self._base_kwargs(node)

        # 2 OSError failures
        mock_run.side_effect = OSError("Network unreachable")
        check_ssh_alive(**kwargs)
        check_ssh_alive(**kwargs)

        # 1 success — resets counter
        mock_run.side_effect = None
        mock_run.return_value = MagicMock(returncode=0, stdout="alive\n", stderr="")
        alive, _ = check_ssh_alive(**kwargs)
        assert alive is True

        # 2 more OSErrors — should NOT hit threshold (counter was reset)
        mock_run.side_effect = OSError("Network unreachable")
        alive3, _ = check_ssh_alive(**kwargs)
        alive4, _ = check_ssh_alive(**kwargs)
        assert alive3 is True
        assert alive4 is True

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_os_error_threshold_clears_counter_key(self, mock_run, node):
        """After OSError threshold, the counter key should be cleaned up."""
        mock_run.side_effect = OSError("Network unreachable")
        kwargs = self._base_kwargs(node)

        check_ssh_alive(**kwargs)  # failure 1
        check_ssh_alive(**kwargs)  # failure 2
        check_ssh_alive(**kwargs)  # failure 3 — threshold reached

        fail_key = f"{node['node_id']}:{kwargs['pid']}"
        assert fail_key not in ssh_utils._ssh_consecutive_failures


# ---------------------------------------------------------------------------
# sync_code
# ---------------------------------------------------------------------------


class TestSyncCode:
    @pytest.fixture(autouse=True)
    def _patch_sync_deps(self, tmp_path):
        with (
            patch("p2p.scheduler.ssh_utils.get_project_root", return_value=tmp_path),
            patch("p2p.scheduler.ssh_utils.subprocess.run") as mock_run,
        ):
            self.mock_run = mock_run
            self.project_root = tmp_path
            yield

    def test_full_success(self, node):
        self.mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = sync_code(node, "/remote/dir")
        assert result is True
        assert self.mock_run.call_count == 3  # mkdir + rsync + uv setup

    def test_mkdir_failure(self, node):
        self.mock_run.return_value = MagicMock(returncode=1, stderr="Permission denied")
        result = sync_code(node, "/remote/dir")
        assert result is False
        assert self.mock_run.call_count == 1  # only mkdir attempted

    def test_rsync_failure(self, node):
        self.mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="rsync error"),
        ]
        result = sync_code(node, "/remote/dir")
        assert result is False
        assert self.mock_run.call_count == 2

    def test_uv_setup_failure(self, node):
        self.mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="uv not found"),
        ]
        result = sync_code(node, "/remote/dir")
        assert result is False

    def test_cache_hit_skips_sync(self, node):
        cache: set[str] = {f"{node['node_id']}:/remote/dir"}
        result = sync_code(node, "/remote/dir", synced_cache=cache)
        assert result is True
        self.mock_run.assert_not_called()

    def test_cache_added_on_success(self, node):
        self.mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        cache: set[str] = set()
        sync_code(node, "/remote/dir", synced_cache=cache)
        assert f"{node['node_id']}:/remote/dir" in cache

    def test_cache_not_added_on_failure(self, node):
        self.mock_run.return_value = MagicMock(returncode=1, stderr="fail")
        cache: set[str] = set()
        sync_code(node, "/remote/dir", synced_cache=cache)
        assert len(cache) == 0

    def test_mkdir_timeout(self, node):
        self.mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)
        result = sync_code(node, "/remote/dir")
        assert result is False

    def test_rsync_timeout(self, node):
        self.mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),  # mkdir OK
            subprocess.TimeoutExpired(cmd="rsync", timeout=120),
        ]
        result = sync_code(node, "/remote/dir")
        assert result is False

    def test_env_file_synced_when_exists(self, node):
        (self.project_root / ".env").write_text("SECRET=abc")
        self.mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = sync_code(node, "/remote/dir")
        assert result is True
        assert self.mock_run.call_count == 4  # mkdir + rsync + .env rsync + uv

    def test_env_sync_failure(self, node):
        (self.project_root / ".env").write_text("SECRET=abc")
        self.mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),  # mkdir
            MagicMock(returncode=0, stderr=""),  # rsync
            MagicMock(returncode=1, stderr="env sync fail"),  # .env
        ]
        result = sync_code(node, "/remote/dir")
        assert result is False


# ---------------------------------------------------------------------------
# kill_ssh_process
# ---------------------------------------------------------------------------


class TestKillSSHProcess:
    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0)
        assert kill_ssh_process(pid=123, node=node) is True

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=1)
        assert kill_ssh_process(pid=123, node=node) is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout(self, mock_run, node):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)
        assert kill_ssh_process(pid=123, node=node) is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_os_error(self, mock_run, node):
        mock_run.side_effect = OSError("No route to host")
        assert kill_ssh_process(pid=123, node=node) is False


# ---------------------------------------------------------------------------
# sync_full_results
# ---------------------------------------------------------------------------


class TestSyncFullResults:
    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = sync_full_results(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is True
        assert (tmp_path / "s1").is_dir()

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_rsync_failure(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="rsync error")
        result = sync_full_results(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout(self, mock_run, node, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=120)
        result = sync_full_results(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_uses_base_dir_when_remote_dir_empty(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        sync_full_results(session_id="s1", node=node, remote_dir="", runs_dir=tmp_path)
        call_args = mock_run.call_args[0][0]
        # Should use node's base_dir
        assert any("/tmp/p2p/runs/s1/" in str(a) for a in call_args)


# ---------------------------------------------------------------------------
# cleanup_remote_session
# ---------------------------------------------------------------------------


class TestCleanupRemoteSession:
    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=0)
        result = cleanup_remote_session(session_id="s1", node=node, remote_dir="/tmp/p2p")
        assert result is True

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure(self, mock_run, node):
        mock_run.return_value = MagicMock(returncode=1, stderr="permission denied")
        result = cleanup_remote_session(session_id="s1", node=node, remote_dir="/tmp/p2p")
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout(self, mock_run, node):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        result = cleanup_remote_session(session_id="s1", node=node, remote_dir="/tmp/p2p")
        assert result is False


# ---------------------------------------------------------------------------
# sync_lite_results
# ---------------------------------------------------------------------------


class TestSyncLiteResults:
    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = sync_lite_results(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is True

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="fail")
        result = sync_lite_results(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_excludes_videos_and_trajectories(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        sync_lite_results(session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path)
        call_args = mock_run.call_args[0][0]
        assert "--exclude=videos/" in call_args
        assert "--exclude=trajectory_*.jsonl" in call_args
        assert "--exclude=trajectory_*.npz" in call_args


# ---------------------------------------------------------------------------
# sync_running_status
# ---------------------------------------------------------------------------


class TestSyncRunningStatus:
    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_success(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = sync_running_status(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is True

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_failure(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=1)
        result = sync_running_status(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_timeout(self, mock_run, node, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=15)
        result = sync_running_status(
            session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path
        )
        assert result is False

    @patch("p2p.scheduler.ssh_utils.subprocess.run")
    def test_includes_status_files(self, mock_run, node, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        sync_running_status(session_id="s1", node=node, remote_dir="/tmp/p2p", runs_dir=tmp_path)
        call_args = mock_run.call_args[0][0]
        assert "--include=status.json" in call_args
        assert "--include=loop_history.json" in call_args
        assert "--exclude=*" in call_args


# ---------------------------------------------------------------------------
# load_nodes / find_node
# ---------------------------------------------------------------------------


class TestLoadNodes:
    def test_file_not_found_returns_empty(self, tmp_path):
        with patch("p2p.settings.RUNS_DIR", tmp_path):
            result = load_nodes()
        assert result == []

    def test_valid_json(self, tmp_path):
        nodes_dir = tmp_path / "scheduler"
        nodes_dir.mkdir()
        nodes_file = nodes_dir / "nodes.json"
        nodes_file.write_text(json.dumps([_node("n1")]))

        with patch("p2p.settings.RUNS_DIR", tmp_path):
            result = load_nodes()
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"

    def test_invalid_json_returns_empty(self, tmp_path):
        nodes_dir = tmp_path / "scheduler"
        nodes_dir.mkdir()
        (nodes_dir / "nodes.json").write_text("{bad json")

        with patch("p2p.settings.RUNS_DIR", tmp_path):
            result = load_nodes()
        assert result == []


class TestFindNode:
    def test_found(self):
        nodes = [_node("n1"), _node("n2")]
        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=nodes):
            result = find_node("n2")
        assert result is not None
        assert result["node_id"] == "n2"

    def test_not_found(self):
        nodes = [_node("n1")]
        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=nodes):
            result = find_node("nonexistent")
        assert result is None

    def test_empty_nodes(self):
        with patch("p2p.scheduler.ssh_utils.load_nodes", return_value=[]):
            assert find_node("n1") is None


# ---------------------------------------------------------------------------
# remote_work_dir
# ---------------------------------------------------------------------------


class TestRemoteWorkDir:
    def test_uses_base_dir_when_present(self):
        node = _node(base_dir="/opt/p2p")
        assert remote_work_dir(node) == "/opt/p2p"

    @patch("p2p.scheduler.ssh_utils.get_git_sha", return_value="abc123def456")
    def test_fallback_to_tmp_with_sha(self, _mock_sha):
        node = _node()
        node["base_dir"] = ""
        assert remote_work_dir(node) == "/tmp/p2p-abc123def456"


# ---------------------------------------------------------------------------
# rewrite_remote_paths (#389)
# ---------------------------------------------------------------------------


class TestRewriteRemotePaths:
    def test_rewrites_loop_history(self, tmp_path):
        """Remote paths in loop_history.json are replaced with local paths."""
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        lh = session_dir / "loop_history.json"
        lh.write_text(
            json.dumps(
                {
                    "runs_dir": "/NHNHOME/user/p2p/runs/session_abc",
                    "video": "/NHNHOME/user/p2p/runs/session_abc/iter_0/video.mp4",
                }
            )
        )

        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        rewrite_remote_paths(session_dir, "/NHNHOME/user/p2p", local_runs)

        data = json.loads(lh.read_text())
        assert str(local_runs) in data["runs_dir"]
        assert "/NHNHOME/" not in data["runs_dir"]
        assert str(local_runs) in data["video"]

    def test_rewrites_lineage_json(self, tmp_path):
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        lj = session_dir / "lineage.json"
        lj.write_text(
            json.dumps(
                {
                    "path": "/remote/base/runs/session_abc/iter_0",
                }
            )
        )

        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        rewrite_remote_paths(session_dir, "/remote/base", local_runs)

        data = json.loads(lj.read_text())
        assert str(local_runs) in data["path"]

    def test_rewrites_session_config(self, tmp_path):
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        sc = session_dir / "session_config.json"
        sc.write_text(
            json.dumps(
                {
                    "runs_dir": "/remote/base/runs/session_abc",
                }
            )
        )

        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        rewrite_remote_paths(session_dir, "/remote/base", local_runs)

        data = json.loads(sc.read_text())
        assert str(local_runs) in data["runs_dir"]

    def test_rewrites_iter_subdirectories(self, tmp_path):
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        iter_dir = session_dir / "iter_0"
        iter_dir.mkdir(parents=True)
        sc = iter_dir / "session_config.json"
        sc.write_text(
            json.dumps(
                {
                    "path": "/remote/base/runs/session_abc/iter_0",
                }
            )
        )

        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        rewrite_remote_paths(session_dir, "/remote/base", local_runs)

        data = json.loads(sc.read_text())
        assert str(local_runs) in data["path"]

    def test_no_op_when_paths_match(self, tmp_path):
        """When remote and local prefixes are identical, no rewriting occurs."""
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        lh = session_dir / "loop_history.json"
        original = json.dumps({"path": f"{local_runs}/session_abc"})
        lh.write_text(original)

        # remote_base_dir == local parent of runs_dir => same prefix => no-op
        rewrite_remote_paths(session_dir, str(tmp_path), local_runs)
        assert lh.read_text() == original

    def test_skips_missing_files(self, tmp_path):
        """No error when target JSON files do not exist."""
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        # No JSON files exist — should not raise
        rewrite_remote_paths(session_dir, "/remote/base", tmp_path / "local_runs")

    def test_skips_file_without_remote_prefix(self, tmp_path):
        """Files not containing the remote prefix are left untouched."""
        from p2p.scheduler.ssh_utils import rewrite_remote_paths

        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        lh = session_dir / "loop_history.json"
        original = '{"path": "/other/prefix/runs/session_abc"}'
        lh.write_text(original)

        local_runs = tmp_path / "local_runs"
        local_runs.mkdir()
        rewrite_remote_paths(session_dir, "/remote/base", local_runs)

        assert lh.read_text() == original
