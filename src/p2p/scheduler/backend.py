"""Scheduler backends and shared utility functions.

Provides ``Backend`` protocol and concrete implementations
(``LocalBackend``, ``SSHBackend``) used by ``job_scheduler.py``.
Also contains SSH/rsync helpers used by ``routes.py``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Protocol, cast

from p2p.scheduler import ssh_utils
from p2p.scheduler.ssh_utils import params_to_cli_args
from p2p.scheduler.types import (
    SUCCESS_STATES,
    TERMINAL_STATES,
    NodeConfig,
    RunSpec,
    RunState,
    RunStatus,
    now_iso,
)
from p2p.settings import RUNS_DIR, resolve_session_dir
from p2p.utils.process_safety import safe_killpg
from p2p.utils.subprocess_utils import python_cmd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Execution backend protocol. Backends only see RunSpec.

    This is the pluggable execution seam for #220 (Trainer abstraction).
    New backends (K8s, SLURM, Ray) implement this protocol — no changes
    to ``job_scheduler.py`` or controllers required.
    """

    def submit(
        self,
        spec: RunSpec,
        *,
        allocated_cores: list[int] | None = None,
        allocated_gpus: list[int] | None = None,
    ) -> RunStatus: ...
    def status(self, run_id: str) -> RunStatus: ...
    def cancel(self, run_id: str) -> bool: ...
    def sync_results(self, run_id: str) -> bool: ...
    def sync_running(self, run_id: str) -> bool: ...
    def cleanup(self, run_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class LocalBackend:
    """Run jobs as local subprocesses."""

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._statuses: dict[str, RunStatus] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        spec: RunSpec,
        *,
        allocated_cores: list[int] | None = None,
        allocated_gpus: list[int] | None = None,
    ) -> RunStatus:
        run_id = spec["run_id"]
        entry_point = spec["entry_point"]
        params = dict(spec["parameters"])

        if (allocated_cores or allocated_gpus) and "loop_config" in params:
            lc = json.loads(params["loop_config"])
            if allocated_cores:
                lc["cores_pool"] = allocated_cores
            if allocated_gpus:
                lc["gpu_pool"] = allocated_gpus
            params["loop_config"] = json.dumps(lc)

        cmd = [*python_cmd(), "-m", entry_point, *params_to_cli_args(params)]

        # Pin to specific CPU cores via taskset
        if allocated_cores:
            core_list = ",".join(str(c) for c in allocated_cores)
            cmd = ["taskset", "-c", core_list, *cmd]

        env = {**os.environ}
        if "env" in spec:
            env.update(spec["env"])

        if allocated_gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in allocated_gpus)

        session_id = params.get("session_id", run_id)
        log_dir = resolve_session_dir(session_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "subprocess.log"

        logger.info("LocalBackend launching: %s", " ".join(cmd))
        log_file = open(log_path, "w", buffering=1)  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        # Write PID file so process_manager.recover_sessions() can re-attach
        # after a backend restart.
        (log_dir / "pid").write_text(str(proc.pid))

        status: RunStatus = {
            "run_id": run_id,
            "state": "running",
            "pid": proc.pid,
            "node_id": "local",
            "started_at": now_iso(),
        }
        with self._lock:
            self._procs[run_id] = proc
            self._statuses[run_id] = status

        def _watch() -> None:
            try:
                proc.wait()
            finally:
                log_file.close()
            state: RunState = "completed" if proc.returncode == 0 else "error"
            with self._lock:
                self._procs.pop(run_id, None)
                if run_id in self._statuses:
                    self._statuses[run_id]["state"] = state
                    self._statuses[run_id]["exit_code"] = proc.returncode
                    self._statuses[run_id]["completed_at"] = now_iso()

        threading.Thread(target=_watch, daemon=True, name=f"watch-{run_id}").start()
        return status

    def status(self, run_id: str) -> RunStatus:
        with self._lock:
            if run_id in self._statuses:
                return dict(self._statuses[run_id])  # type: ignore[return-value]
        return {"run_id": run_id, "state": "error", "error": "Unknown run_id"}

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(run_id)
        if proc is None:
            return False
        safe_killpg(proc.pid, expected_cmdline=run_id)
        with self._lock:
            if run_id in self._statuses:
                self._statuses[run_id]["state"] = "cancelled"
                self._statuses[run_id]["completed_at"] = now_iso()
        return True

    def sync_results(self, run_id: str) -> bool:
        return True  # local, no sync needed

    def sync_running(self, run_id: str) -> bool:
        return True  # local, files already visible

    def cleanup(self, run_id: str) -> bool:
        return True  # local, nothing to clean up


# ---------------------------------------------------------------------------
# SSHBackend
# ---------------------------------------------------------------------------


class _SSHRunInfo:
    """Internal tracking state for one SSH run."""

    __slots__ = ("remote_dir", "pid", "session_id", "status")

    def __init__(
        self,
        remote_dir: str,
        pid: int,
        session_id: str,
        status: RunStatus,
    ) -> None:
        self.remote_dir = remote_dir
        self.pid = pid
        self.session_id = session_id
        self.status = status


class SSHBackend:
    """Execute runs on a remote SSH node.

    Each instance is bound to a single node. The ``Scheduler`` (or caller)
    is responsible for selecting the right backend instance per node.

    All heavy lifting is delegated to ``ssh_utils`` — this class only
    provides the ``Backend`` protocol surface and tracks per-run state.
    """

    def __init__(self, node: NodeConfig) -> None:
        self._node = node
        self._runs: dict[str, _SSHRunInfo] = {}
        self._synced_cache: set[str] = set()
        self._lock = threading.Lock()

    @property
    def node_id(self) -> str:
        return self._node["node_id"]

    def submit(
        self,
        spec: RunSpec,
        *,
        allocated_cores: list[int] | None = None,
        allocated_gpus: list[int] | None = None,
    ) -> RunStatus:
        run_id = spec["run_id"]
        rdir = ssh_utils.remote_work_dir(self._node)

        if not ssh_utils.sync_code(self._node, rdir, synced_cache=self._synced_cache):
            return {
                "run_id": run_id,
                "state": "error",
                "node_id": self.node_id,
                "error": "Failed to sync code to remote node",
            }

        remote_pid, error = ssh_utils.submit_ssh(
            node=self._node,
            remote_dir=rdir,
            entry_point=spec["entry_point"],
            parameters=spec["parameters"],
            run_id=run_id,
            cpu_cores=allocated_cores,
            gpu_ids=allocated_gpus,
        )

        if error is not None:
            return {
                "run_id": run_id,
                "state": "error",
                "node_id": self.node_id,
                "error": error,
            }

        if remote_pid is None:
            msg = (
                f"submit_ssh returned pid=None without error for {run_id} on node={self.node_id!r}"
            )
            raise RuntimeError(msg)
        session_id = spec["parameters"].get("session_id", run_id)
        status: RunStatus = {
            "run_id": run_id,
            "state": "running",
            "pid": remote_pid,
            "node_id": self.node_id,
            "remote_dir": rdir,
            "started_at": now_iso(),
        }
        with self._lock:
            self._runs[run_id] = _SSHRunInfo(
                remote_dir=rdir,
                pid=remote_pid,
                session_id=session_id,
                status=status,
            )
        return status

    def status(self, run_id: str) -> RunStatus:
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return {"run_id": run_id, "state": "error", "error": "Unknown run_id"}

        # Already terminal — no need to re-check liveness
        if info.status["state"] in TERMINAL_STATES:
            return cast(RunStatus, dict(info.status))

        alive, remote_status = ssh_utils.check_ssh_alive(
            pid=info.pid,
            node=self._node,
            remote_dir=info.remote_dir,
            session_id=info.session_id,
        )

        if alive:
            return cast(RunStatus, dict(info.status))

        state: RunState = "completed" if remote_status in SUCCESS_STATES else "error"
        with self._lock:
            # Re-check: cancel() may have transitioned state while we
            # were doing the SSH liveness check without the lock held.
            if info.status["state"] in TERMINAL_STATES:
                return cast(RunStatus, dict(info.status))
            info.status["state"] = state
            info.status["completed_at"] = now_iso()
            if state == "error" and remote_status:
                info.status["error"] = f"Remote session ended: {remote_status}"
            return cast(RunStatus, dict(info.status))

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return False

        ok = ssh_utils.kill_ssh_process(pid=info.pid, node=self._node)
        if ok:
            with self._lock:
                info.status["state"] = "cancelled"
                info.status["completed_at"] = now_iso()
        return ok

    def sync_results(self, run_id: str) -> bool:
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return False

        return ssh_utils.sync_full_results(
            session_id=info.session_id,
            node=self._node,
            remote_dir=info.remote_dir,
            runs_dir=RUNS_DIR,
        )

    def sync_running(self, run_id: str) -> bool:
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return False

        return ssh_utils.sync_running_status(
            session_id=info.session_id,
            node=self._node,
            remote_dir=info.remote_dir,
            runs_dir=RUNS_DIR,
        )

    def cleanup(self, run_id: str) -> bool:
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return False

        return ssh_utils.cleanup_remote_session(
            session_id=info.session_id,
            node=self._node,
            remote_dir=info.remote_dir,
        )


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------


def _ssh_base_cmd(node: NodeConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-p",
        str(node["port"]),
        f"{node['user']}@{node['host']}",
    ]


def _rsync_cmd(node: NodeConfig, remote_path: str, local_path: str) -> list[str]:
    return [
        "rsync",
        "-az",
        "--timeout=30",
        "-e",
        f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes",
        f"{node['user']}@{node['host']}:{remote_path}",
        local_path,
    ]


def _get_project_root() -> Path:
    """Find the local project root (where pyproject.toml lives)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        pass
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _get_git_sha() -> str:
    """Get current git short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Node connectivity check (used by routes.py)
# ---------------------------------------------------------------------------


def check_node(node_id: str) -> dict:
    """Test SSH connectivity to a node."""
    from p2p.scheduler import node_store

    node = node_store.get_node(node_id)
    if node is None:
        return {
            "online": False,
            "uv_available": False,
            "gpu": None,
            "mps_active": False,
            "error": "Node not found",
        }

    check_script = (
        "export PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH;"
        " echo ok; which uv 2>/dev/null || echo nouv;"
        " timeout 5 nvidia-smi --query-gpu=name,memory.total"
        " --format=csv,noheader,nounits 2>/dev/null || echo nogpu;"
        " pgrep -x nvidia-cuda-mps >/dev/null 2>&1 && echo mps_active || echo mps_inactive"
    )

    if ssh_utils.is_localhost(node):
        cmd: list[str] = ["bash", "-c", check_script]
    else:
        cmd = [*_ssh_base_cmd(node), check_script]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {
                "online": False,
                "uv_available": False,
                "gpu": None,
                "mps_active": False,
                "error": result.stderr.strip(),
            }
        lines = result.stdout.strip().split("\n")
        online = lines[0] == "ok" if lines else False
        uv_line = lines[1] if len(lines) > 1 else ""
        # Last line is always MPS status; GPU info is between uv and MPS
        mps_active = lines[-1].strip() == "mps_active" if lines else False
        gpu_lines = lines[2:-1] if len(lines) > 2 else []
        gpu = _parse_gpu_info(gpu_lines)

        # Persist GPU info only when it changed (avoid unconditional disk writes).
        # Skip gpu_memory_mb when 0 — likely a transient nvidia-smi failure, not
        # a real hardware change. Preserves the valid value from a prior check.
        valid_gpu_lines = [ln for ln in gpu_lines if ln.strip() and ln.strip() != "nogpu"]
        if valid_gpu_lines:
            num_gpus = len(valid_gpu_lines)
            gpu_mem = _parse_first_gpu_mem(valid_gpu_lines)
            updates: dict = {}
            if node.get("num_gpus") != num_gpus:
                updates["num_gpus"] = num_gpus
                if gpu_mem == 0:
                    logger.warning(
                        "Node %s: num_gpus changed to %d but gpu_memory_mb "
                        "unreadable; keeping previous value",
                        node_id,
                        num_gpus,
                    )
            if gpu_mem > 0 and node.get("gpu_memory_mb") != gpu_mem:
                updates["gpu_memory_mb"] = gpu_mem
            if updates:
                try:
                    node_store.update_node(node_id, updates)
                except (KeyError, OSError):
                    pass

        return {
            "online": online,
            "uv_available": uv_line != "nouv" and "uv" in uv_line,
            "gpu": gpu,
            "mps_active": mps_active,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "online": False,
            "uv_available": False,
            "gpu": None,
            "mps_active": False,
            "error": "Connection timed out",
        }


def setup_node(node_id: str) -> dict:
    """Install uv (if missing), create base_dir, sync code, and run uv sync.

    Returns a progress dict with step results.
    """
    from p2p.scheduler import node_store
    from p2p.scheduler.ssh_utils import remote_work_dir

    node = node_store.get_node(node_id)
    if node is None:
        return {"ok": False, "error": "Node not found"}

    steps: dict[str, str] = {}

    # 1. Install uv if not available
    install_script = (
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH";'
        " which uv >/dev/null 2>&1 && echo exists"
        " || (curl -LsSf https://astral.sh/uv/install.sh | sh && echo installed)"
    )
    if ssh_utils.is_localhost(node):
        install_cmd: list[str] = ["bash", "-c", install_script]
    else:
        install_cmd = [*_ssh_base_cmd(node), install_script]
    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=60)
        out = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
        if result.returncode != 0:
            steps["uv_install"] = f"failed: {result.stderr.strip()[:200]}"
            return {"ok": False, "steps": steps, "error": "uv installation failed"}
        steps["uv_install"] = "skipped (already exists)" if out == "exists" else "installed"
    except subprocess.TimeoutExpired:
        return {"ok": False, "steps": steps, "error": "uv install timed out"}

    # 2. Create base_dir
    base_dir = remote_work_dir(node)
    if ssh_utils.is_localhost(node):
        mkdir_cmd: list[str] = ["mkdir", "-p", base_dir]
    else:
        mkdir_cmd = [*_ssh_base_cmd(node), f"mkdir -p {base_dir}"]
    try:
        subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=10)
        steps["mkdir"] = "ok"
    except (subprocess.TimeoutExpired, OSError):
        steps["mkdir"] = "failed"

    # 3. Sync code
    from p2p.scheduler.ssh_utils import sync_code

    if sync_code(node, base_dir):
        steps["code_sync"] = "ok"
    else:
        steps["code_sync"] = "failed"
        return {"ok": False, "steps": steps, "error": "Code sync failed"}

    return {"ok": True, "steps": steps}


def _parse_gpu_info(lines: list[str]) -> str | None:
    """Parse nvidia-smi CSV output into a human-readable GPU summary.

    Example input lines: ["NVIDIA RTX A5500, 24564"]
    Example output: "RTX A5500 (24 GB)"
    """
    if not lines or lines[0].strip() == "nogpu":
        return None
    gpus: list[str] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0].replace("NVIDIA ", "")
            try:
                mem_gb = round(int(float(parts[1])) / 1024)
                gpus.append(f"{name} ({mem_gb} GB)")
            except (ValueError, OverflowError):
                gpus.append(name)
        elif parts[0] and parts[0] != "nogpu":
            gpus.append(parts[0].replace("NVIDIA ", ""))
    if not gpus:
        return None
    counts = Counter(gpus)
    result = []
    for gpu, count in counts.items():
        result.append(f"{count}x {gpu}" if count > 1 else gpu)
    return ", ".join(result)


def _parse_first_gpu_mem(lines: list[str]) -> int:
    """Extract the VRAM in MB from the first nvidia-smi CSV line.

    Input format: ``"NVIDIA A100-SXM4-40GB, 40536"`` → 40536.
    """
    if not lines:
        return 0
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) >= 2:
        try:
            return int(float(parts[1]))
        except (ValueError, OverflowError):
            pass
    return 0
