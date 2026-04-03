"""Shared SSH/rsync utilities for scheduler modules.

Used by ``job_scheduler.py`` for both session and benchmark scheduling.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

from p2p.settings import resolve_session_dir, resolve_session_subpath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSH / rsync primitives
# ---------------------------------------------------------------------------


def ssh_base_cmd(node: dict) -> list[str]:
    """Build base SSH command with standard options.

    Includes ``ServerAliveInterval`` / ``ServerAliveCountMax`` to detect
    stale connections promptly and avoid indefinite hangs (#388).
    """
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


def rsync_pull_cmd(node: dict, remote_path: str, local_path: str) -> list[str]:
    """Build rsync command to pull remote_path → local_path."""
    return [
        "rsync",
        "-az",
        "--timeout=30",
        "-e",
        f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes",
        f"{node['user']}@{node['host']}:{remote_path}",
        local_path,
    ]


# ---------------------------------------------------------------------------
# Project / git helpers
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Return the project root directory."""
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


def get_git_sha() -> str:
    """Return the short git SHA of the current HEAD."""
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
# CLI helpers
# ---------------------------------------------------------------------------


def params_to_cli_args(parameters: dict) -> list[str]:
    """Convert a parameters dict to CLI arguments."""
    args: list[str] = []
    for key, value in parameters.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None or value is False:
            continue
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            if value:
                args.extend([flag, ",".join(str(v) for v in value)])
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value)])
        else:
            args.extend([flag, str(value)])
    return args


# ---------------------------------------------------------------------------
# Code sync
# ---------------------------------------------------------------------------

RSYNC_EXCLUDE = [
    ".venv",
    "runs",
    "node_modules",
    ".next",
    "__pycache__",
    ".git",
    "*.pyc",
    ".pytest_cache",
    ".ruff_cache",
]


def is_localhost(node: dict) -> bool:
    """Check if the node refers to the local machine.

    Detects ``127.0.0.1``, ``localhost``, ``::1``, and ``0.0.0.0``.
    Does **not** resolve the machine's real hostname or network interfaces.
    """
    return node["host"] in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def remote_work_dir(node: dict) -> str:
    """Determine the remote working directory for a node.

    For localhost nodes without an explicit ``base_dir``, returns the local
    project root so that no rsync is needed and no ``/tmp`` space is consumed.
    """
    if node.get("base_dir"):
        return node["base_dir"]
    if is_localhost(node):
        return str(get_project_root())
    return f"/tmp/p2p-{get_git_sha()}"


def sync_code(node: dict, remote_dir: str, *, synced_cache: set[str] | None = None) -> bool:
    """Sync project code to remote node. Skips if already synced.

    For localhost nodes whose ``remote_dir`` resolves to the project root,
    the sync is skipped entirely (the code is already in place).

    Supports two sync modes (``node["sync_mode"]``):

    - ``"rsync"`` (default): rsync code + .env, then ``uv sync`` to
      install/update the venv.  Safe for fresh nodes but resets the venv
      (wipes manually-installed packages like IsaacLab).
    - ``"git"``: ``git pull`` on the remote repo + rsync .env only.
      Preserves the venv — use for nodes with manual dependencies.
    """
    if synced_cache is None:
        synced_cache = set()
    cache_key = f"{node['node_id']}:{remote_dir}"
    if cache_key in synced_cache:
        return True

    project_root = get_project_root()

    # Localhost pointing at the project root — nothing to sync.
    if is_localhost(node) and Path(remote_dir).resolve() == project_root.resolve():
        logger.info("Skipping code sync for localhost (already at project root)")
        synced_cache.add(cache_key)
        return True

    mode = node.get("sync_mode", "rsync")
    if mode == "git":
        ok = _sync_code_git(node, remote_dir, project_root)
    else:
        ok = _sync_code_rsync(node, remote_dir, project_root)
    if ok:
        synced_cache.add(cache_key)
    return ok


def _sync_code_git(node: dict, remote_dir: str, project_root: Path) -> bool:
    """Sync via git pull + .env rsync. Preserves the remote venv."""
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    branch = branch_result.stdout.strip() or "HEAD"

    git_cmd = [
        *ssh_base_cmd(node),
        f"cd {remote_dir} && git fetch origin {branch} --quiet"
        f" && git checkout {branch} --quiet"
        f" && git reset --hard origin/{branch} --quiet"
        f" && echo git_ok",
    ]
    logger.info("Git sync to %s:%s (branch=%s)", node["node_id"], remote_dir, branch)
    try:
        result = subprocess.run(git_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or "git_ok" not in result.stdout:
            logger.warning("Git sync failed on %s: %s", node["node_id"], result.stderr.strip())
            return False
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Git sync failed on %s: %s", node["node_id"], e)
        return False

    # Sync .env (not in git)
    return _sync_env_file(node, remote_dir, project_root)


def _sync_env_file(node: dict, remote_dir: str, project_root: Path) -> bool:
    """Rsync .env file to remote node."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return True
    ssh_opts = f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    env_cmd = [
        "rsync",
        "-az",
        "-e",
        ssh_opts,
        str(env_file),
        f"{node['user']}@{node['host']}:{remote_dir}/.env",
    ]
    try:
        env_result = subprocess.run(env_cmd, capture_output=True, text=True, timeout=30)
        if env_result.returncode != 0:
            logger.warning(
                "Failed to sync .env to %s: %s",
                node["node_id"],
                env_result.stderr.strip(),
            )
            return False
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("Failed to sync .env to %s", node["node_id"])
        return False
    return True


def _sync_code_rsync(node: dict, remote_dir: str, project_root: Path) -> bool:
    """Sync via rsync + uv sync. Resets venv to pyproject.toml state."""
    # Ensure remote directory exists
    mkdir_cmd = [*ssh_base_cmd(node), f"mkdir -p {remote_dir}"]
    try:
        result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(
                "Failed to create remote dir %s on %s: %s",
                remote_dir,
                node["node_id"],
                result.stderr.strip(),
            )
            return False
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("Failed to create remote dir %s on %s", remote_dir, node["node_id"])
        return False

    exclude_args: list[str] = []
    for ex in RSYNC_EXCLUDE:
        exclude_args.extend(["--exclude", ex])

    sync_cmd = [
        "rsync",
        "-az",
        "--timeout=60",
        *exclude_args,
        "-e",
        f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes",
        f"{project_root}/",
        f"{node['user']}@{node['host']}:{remote_dir}/",
    ]
    logger.info("Syncing code to %s:%s", node["node_id"], remote_dir)
    try:
        result = subprocess.run(sync_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Code sync failed: %s", result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Code sync timed out for %s", node["node_id"])
        return False

    if not _sync_env_file(node, remote_dir, project_root):
        return False

    # Install uv if missing, then setup venv
    setup_cmd = [
        *ssh_base_cmd(node),
        f'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"; '
        f"which uv >/dev/null 2>&1"
        f" || (curl -LsSf https://astral.sh/uv/install.sh | sh"
        f' && export PATH="$HOME/.local/bin:$PATH"); '
        f"cd {remote_dir} && uv sync --quiet 2>&1 | tail -1",
    ]
    try:
        setup_result = subprocess.run(setup_cmd, capture_output=True, text=True, timeout=180)
        if setup_result.returncode != 0:
            logger.warning(
                "uv setup failed on %s: %s",
                node["node_id"],
                setup_result.stderr.strip(),
            )
            return False
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("uv setup failed on %s", node["node_id"])
        return False

    return True


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------


def load_nodes() -> list[dict]:
    """Load nodes from the node store JSON file."""
    from p2p.settings import RUNS_DIR

    nodes_path = RUNS_DIR / "scheduler" / "nodes.json"
    try:
        return json.loads(nodes_path.read_text())  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def find_node(node_id: str) -> dict | None:
    """Find a node by ID. Returns None if not found."""
    for n in load_nodes():
        if n["node_id"] == node_id:
            return n
    return None


def resolve_node(
    node_id: str | None,
    used_cores_per_node: dict[str, int],
    required_cores: int = 1,
    *,
    skip_nodes: set[str] | None = None,
    allowed_nodes: set[str] | None = None,
) -> dict | None:
    """Resolve target node. Auto-assign (most-free-cores) if node_id is None.

    Only considers enabled nodes (``enabled`` defaults to ``True``).

    Args:
        node_id: Explicit node hint (None = auto-assign).
        used_cores_per_node: Mapping of node_id → total CPU cores currently in use.
        required_cores: CPU cores the new run needs.
        skip_nodes: Node IDs to exclude from auto-assignment (e.g. nodes that
            failed earlier in the same submission round).  Ignored when
            *node_id* is explicitly provided (affinity hint).
        allowed_nodes: When set, only these node IDs are eligible for
            auto-assignment.  Ignored when *node_id* is explicitly provided.
    """
    nodes = load_nodes()
    if node_id is not None:
        for n in nodes:
            if n["node_id"] == node_id and n.get("enabled", True):
                return n
        return None

    if not nodes:
        return None

    candidates = []
    for n in nodes:
        if not n.get("enabled", True):
            continue
        if allowed_nodes and n["node_id"] not in allowed_nodes:
            continue
        if skip_nodes and n["node_id"] in skip_nodes:
            continue
        used = used_cores_per_node.get(n["node_id"], 0)
        free = n["max_cores"] - used
        if free >= required_cores:
            candidates.append((free, n))

    if not candidates:
        return None
    # Pick the node with the most free cores (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# SSH session launch
# ---------------------------------------------------------------------------


def submit_ssh(
    *,
    node: dict,
    remote_dir: str,
    entry_point: str,
    parameters: dict,
    run_id: str,
    cpu_cores: list[int] | None = None,
    gpu_ids: list[int] | None = None,
) -> tuple[int | None, str | None]:
    """Launch a process on a remote node via SSH.

    Returns ``(remote_pid, None)`` on success, ``(None, error_msg)`` on failure.
    """
    params = dict(parameters)
    remote_runs_dir = f"{remote_dir}/runs"
    is_local = is_localhost(node)
    if "loop_config" in params:
        lc = json.loads(params["loop_config"])
        if not is_local:
            lc["runs_dir"] = remote_runs_dir
        if cpu_cores is not None:
            lc["cores_pool"] = cpu_cores
        if gpu_ids is not None:
            lc["gpu_pool"] = gpu_ids
        params["loop_config"] = json.dumps(lc)

    cli_args = params_to_cli_args(params)
    args_str = " ".join(shlex.quote(a) for a in cli_args)

    env_exports = (
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"'
        "; export MUJOCO_GL=egl"
    )
    if gpu_ids:
        cvd = ",".join(str(g) for g in gpu_ids)
        env_exports += f"; export CUDA_VISIBLE_DEVICES={cvd}"

    session_id = params.get("session_id", run_id)
    if is_local:
        log_dir = str(resolve_session_dir(session_id))
    else:
        log_dir = f"{remote_runs_dir}/{resolve_session_subpath(session_id)}"
    log_file = f"{log_dir}/subprocess.log"

    q_dir = shlex.quote(remote_dir)
    venv_python = shlex.quote(f"{remote_dir}/.venv/bin/python")
    q_log_dir = shlex.quote(log_dir)
    q_log_file = shlex.quote(log_file)
    pid_file = f"{log_dir}/pid"
    q_pid_file = shlex.quote(pid_file)

    # Wrap with taskset for CPU affinity pinning
    taskset_prefix = ""
    if cpu_cores:
        core_list = ",".join(str(c) for c in cpu_cores)
        taskset_prefix = f"taskset -c {core_list} "

    remote_cmd = (
        f"mkdir -p {q_log_dir} && "
        f"{env_exports}; "
        f"cd {q_dir} && "
        f"(nohup setsid {taskset_prefix}{venv_python} -m {entry_point} {args_str} "
        f"> {q_log_file} 2>&1 & echo $! > {q_pid_file}) && "
        f"sleep 0.1 && cat {q_pid_file}"
    )

    cmd = [*ssh_base_cmd(node), remote_cmd]
    logger.info("SSH launch on %s: %s", node["node_id"], run_id)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            error = result.stderr.strip() or "SSH command failed"
            return None, error
        remote_pid = int(result.stdout.strip())
        return remote_pid, None
    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# SSH liveness check
# ---------------------------------------------------------------------------

_SSH_FAIL_THRESHOLD = 3
_ssh_consecutive_failures: dict[str, int] = {}


def check_ssh_alive(
    *,
    pid: int,
    node: dict,
    remote_dir: str,
    session_id: str,
) -> tuple[bool, str]:
    """Check whether a remote process is alive.

    Returns ``(True, "")`` if alive, ``(False, remote_status)`` if dead.
    ``remote_status`` is the value from the remote ``status.json`` or ``""``
    if unreadable.

    On transient SSH failure, uses a consecutive-failure counter.  After
    ``_SSH_FAIL_THRESHOLD`` consecutive failures the process is declared dead
    so it does not permanently occupy a ``max_parallel`` slot.
    """
    fail_key = f"{node['node_id']}:{pid}"
    base = remote_dir or node.get("base_dir", "")
    status_path = f"{base}/runs/{resolve_session_subpath(session_id)}/status.json"
    check_cmd = [
        *ssh_base_cmd(node),
        f"kill -0 {pid} 2>/dev/null && echo alive || "
        f"(echo dead; cat {status_path} 2>/dev/null || echo '{{}}')",
    ]
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        _ssh_consecutive_failures[fail_key] = _ssh_consecutive_failures.get(fail_key, 0) + 1
        count = _ssh_consecutive_failures[fail_key]
        logger.warning(
            "SSH liveness check failed for pid %d (%d/%d): %s",
            pid,
            count,
            _SSH_FAIL_THRESHOLD,
            e,
        )
        if count >= _SSH_FAIL_THRESHOLD:
            _ssh_consecutive_failures.pop(fail_key, None)
            return False, "error"
        return True, ""

    # Successful SSH connection — reset counter
    _ssh_consecutive_failures.pop(fail_key, None)

    if output.startswith("dead"):
        lines = output.split("\n", 1)
        remote_status = ""
        if len(lines) > 1:
            try:
                status_data = json.loads(lines[1])
                remote_status = status_data.get("status", "")
            except (ValueError, TypeError):
                pass
        return False, remote_status

    return True, ""


# ---------------------------------------------------------------------------
# SSH process kill
# ---------------------------------------------------------------------------


def kill_ssh_process(*, pid: int, node: dict) -> bool:
    """Kill a remote process via SSH.  Returns True if the kill succeeded."""
    kill_script = (
        f"children=$(pgrep -P {pid} 2>/dev/null); "
        f'gchildren=""; '
        f"for c in $children; do "
        f'gchildren="$gchildren $(pgrep -P $c 2>/dev/null)"; '
        f"done; "
        f'all="{pid} $children $gchildren"; '
        f"kill $all 2>/dev/null; "
        f"sleep 0.5; "
        f"kill -9 $all 2>/dev/null; "
        f"echo ok"
    )
    cmd = [*ssh_base_cmd(node), kill_script]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Failed to kill remote pid %d on %s: %s", pid, node["node_id"], e)
        return False


# ---------------------------------------------------------------------------
# Remote path rewriting (#389)
# ---------------------------------------------------------------------------

# JSON files that may contain remote filesystem paths after rsync.
_PATH_REWRITE_FILES = ("loop_history.json", "lineage.json", "session_config.json")


def rewrite_remote_paths(local_session_dir: Path, remote_base_dir: str, runs_dir: Path) -> None:
    """Replace remote filesystem paths with local RUNS_DIR paths in synced JSON files.

    After rsyncing results from a remote SSH node, JSON artifacts may contain
    paths like ``/NHNHOME/.../runs/...`` that the dashboard cannot resolve.
    This rewrites them to the local ``runs_dir`` equivalent (#389).

    Only touches the specific files listed in ``_PATH_REWRITE_FILES``.
    Walks one level into ``iter_*/`` subdirectories as well.
    """
    remote_runs_prefix = f"{remote_base_dir}/runs/"
    local_runs_prefix = str(runs_dir) + "/"

    if remote_runs_prefix == local_runs_prefix:
        return

    # Collect candidate files: session root + iter_* subdirectories
    candidates: list[Path] = []
    for fname in _PATH_REWRITE_FILES:
        p = local_session_dir / fname
        if p.is_file():
            candidates.append(p)
    for child in local_session_dir.iterdir():
        if child.is_dir() and child.name.startswith("iter_"):
            for fname in _PATH_REWRITE_FILES:
                p = child / fname
                if p.is_file():
                    candidates.append(p)

    for fpath in candidates:
        try:
            text = fpath.read_text(encoding="utf-8")
            if remote_runs_prefix not in text:
                continue
            rewritten = text.replace(remote_runs_prefix, local_runs_prefix)
            fpath.write_text(rewritten, encoding="utf-8")
            logger.debug("Rewrote remote paths in %s", fpath)
        except OSError as e:
            logger.warning("Failed to rewrite paths in %s: %s", fpath, e)


# ---------------------------------------------------------------------------
# Result sync
# ---------------------------------------------------------------------------


def sync_full_results(
    *,
    session_id: str,
    node: dict,
    remote_dir: str,
    runs_dir: Path,
) -> bool:
    """Rsync full session results from remote to local runs directory."""
    base = remote_dir or node.get("base_dir", "")

    # Localhost writing to local runs dir — results are already in place.
    if is_localhost(node) and Path(base).resolve() == get_project_root().resolve():
        return True

    subpath = resolve_session_subpath(session_id)
    remote_path = f"{base}/runs/{subpath}/"
    local_path = str(runs_dir / subpath) + "/"

    Path(local_path).mkdir(parents=True, exist_ok=True)
    cmd = rsync_pull_cmd(node, remote_path, local_path)
    logger.info("Syncing full results: %s -> %s", remote_path, local_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("rsync failed: %s", result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.warning("rsync timed out for session %s", session_id)
        return False

    # Rewrite remote paths in synced JSON files so the dashboard can
    # resolve them locally (#389).
    rewrite_remote_paths(Path(local_path), base, runs_dir)
    return True


def cleanup_remote_session(
    *,
    session_id: str,
    node: dict,
    remote_dir: str,
) -> bool:
    """Remove a session's results directory from the remote node.

    Called after sync completes (success or error) to free /tmp space.
    """
    base = remote_dir or node.get("base_dir", "")

    # Localhost using project root — never delete local runs.
    if is_localhost(node) and Path(base).resolve() == get_project_root().resolve():
        return True

    remote_path = f"{base}/runs/{resolve_session_subpath(session_id)}"
    cmd = [*ssh_base_cmd(node), f"rm -rf {remote_path}"]
    logger.info("Cleaning remote session: %s:%s", node["node_id"], remote_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("Remote cleanup failed: %s", result.stderr.strip())
            return False
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Remote cleanup failed for %s on %s: %s", session_id, node["node_id"], e)
        return False
    return True


def sync_lite_results(
    *,
    session_id: str,
    node: dict,
    remote_dir: str,
    runs_dir: Path,
) -> bool:
    """Rsync session results excluding videos and trajectories.

    Includes status, loop_history, events, scalars, summaries, judgments,
    reward functions, and configs — everything needed for charts and review.
    Typically ~100KB per iteration vs ~100MB+ for full sync.
    """
    base = remote_dir or node.get("base_dir", "")

    # Localhost using project root — results are already in place.
    if is_localhost(node) and Path(base).resolve() == get_project_root().resolve():
        return True

    subpath = resolve_session_subpath(session_id)
    remote_base = f"{base}/runs/{subpath}/"
    local_base = str(runs_dir / subpath) + "/"

    Path(local_base).mkdir(parents=True, exist_ok=True)

    ssh_opts = f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    cmd = [
        "rsync",
        "-az",
        "--timeout=60",
        "--exclude=videos/",
        "--exclude=trajectory_*.jsonl",
        "--exclude=trajectory_*.npz",
        "-e",
        ssh_opts,
        f"{node['user']}@{node['host']}:{remote_base}",
        local_base,
    ]
    logger.info("Lite sync: %s -> %s", remote_base, local_base)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            logger.warning("Lite sync failed: %s", result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Lite sync timed out for %s", session_id)
        return False

    # Rewrite remote paths in synced JSON files (#389).
    rewrite_remote_paths(Path(local_base), base, runs_dir)
    return True


def sync_running_status(
    *,
    session_id: str,
    node: dict,
    remote_dir: str,
    runs_dir: Path,
) -> bool:
    """Rsync lightweight status files from a running remote session.

    Only syncs status.json, loop_history.json, and streaming_judgments/
    to keep bandwidth low (~10KB per session).
    """
    base = remote_dir or node.get("base_dir", "")

    # Localhost using project root — files are already local.
    if is_localhost(node) and Path(base).resolve() == get_project_root().resolve():
        return True

    subpath = resolve_session_subpath(session_id)
    remote_base = f"{base}/runs/{subpath}/"
    local_base = str(runs_dir / subpath) + "/"

    Path(local_base).mkdir(parents=True, exist_ok=True)

    ssh_opts = f"ssh -p {node['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    cmd = [
        "rsync",
        "-az",
        "--timeout=10",
        "--include=status.json",
        "--include=loop_history.json",
        "--include=iter_*/",
        "--include=iter_*/scalars.csv",
        "--include=iter_*/eval_results.json",
        "--include=iter_*/config.json",
        "--include=iter_*/reward_fn.py",
        "--include=iter_*/reward_spec.json",
        "--include=iter_*/streaming_judgments/",
        "--include=iter_*/streaming_judgments/**",
        # Multi-config sub-runs (e.g. baseline_seed_1/, config_1_seed_1/)
        "--include=iter_*/*/",
        "--include=iter_*/*/status.json",
        "--include=iter_*/*/config.json",
        "--include=iter_*/*/scalars.csv",
        "--include=iter_*/*/eval_results.json",
        "--include=iter_*/*/reward_fn.py",
        "--include=iter_*/*/reward_spec.json",
        "--include=iter_*/*/metrics/",
        "--include=iter_*/*/metrics/scalars.jsonl",
        "--exclude=*",
        "-e",
        ssh_opts,
        f"{node['user']}@{node['host']}:{remote_base}",
        local_base,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return False
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("Status sync failed for %s on %s", session_id, node["node_id"])
        return False

    # Rewrite remote paths in synced JSON files (#389).
    rewrite_remote_paths(Path(local_base), base, runs_dir)
    return True
