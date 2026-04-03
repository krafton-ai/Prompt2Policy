"""Subprocess lifecycle management for training sessions.

Sprouted from ``services.py`` — owns all state and logic related to
launching, tracking, recovering, and stopping background training
subprocesses.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from p2p.config import LoopConfig
from p2p.session.iteration_record import SessionRecord
from p2p.session.session_id import generate_session_id
from p2p.settings import (
    ANTHROPIC_API_KEY,
    RUNS_DIR,
    resolve_session_dir,
    resolve_session_subpath,
)
from p2p.utils.process_safety import (
    force_kill_pids,
    get_descendant_pids,
    is_pid_alive,
    safe_killpg,
    verify_pid_ownership,
)
from p2p.utils.subprocess_utils import python_cmd
from p2p.utils.utils import read_log_tail

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory process tracking
# ---------------------------------------------------------------------------

_active_procs: dict[str, subprocess.Popen | int] = {}
_active_procs_lock = threading.Lock()
# Track CPU core allocations made by the API server for each session.
# Maps session_id -> alloc_id used with CPUManager.allocate().
_session_cpu_allocs: dict[str, str] = {}
_session_cpu_allocs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _release_session_cores(session_id: str) -> None:
    """Release CPU cores allocated by the API server for *session_id*."""
    with _session_cpu_allocs_lock:
        alloc_id = _session_cpu_allocs.pop(session_id, None)
    if alloc_id:
        from p2p.training.cpu_manager import get_cpu_manager

        get_cpu_manager().release(alloc_id)


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill a subprocess and all its children (entire process group).

    Relies on ``start_new_session=True`` at Popen time so that
    ``os.killpg`` targets only this tree, not the API server.

    Also walks the full child tree via /proc to catch orphaned
    subprocesses (e.g. Isaac Sim's Omniverse Kit workers) that
    may have escaped the process group.
    """
    # Collect all descendant PIDs before killing (they may reparent to init)
    children = get_descendant_pids(proc.pid)

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()

    # Kill any orphaned children that survived the process group kill
    force_kill_pids(children)


def _kill_pid_tree(pid: int, *, session_id: str | None = None) -> None:
    """Kill a process tree by PID (used for recovered sessions without Popen handle).

    Verifies PID ownership before sending signals to prevent killing
    unrelated processes if the PID was recycled by the OS.
    """
    children = get_descendant_pids(pid)
    killed = safe_killpg(pid, expected_cmdline=session_id or None)
    if killed:
        force_kill_pids(children)


def _kill_session_by_name(session_id: str) -> bool:
    """Find and kill processes whose cmdline contains the session_id.

    Used as fallback when _active_procs lost the handle (server reload).
    """
    killed = False
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return False

    my_pid = os.getpid()
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if session_id not in cmdline:
            continue
        if "p2p.session.run_session" not in cmdline and "p2p.executor" not in cmdline:
            continue
        # Kill the matched process and its entire descendant tree
        children = get_descendant_pids(pid)
        try:
            os.kill(pid, signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError):
            pass
        force_kill_pids(children)

    return killed


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------

_STALE_MINUTES = 5


def _is_scheduler_managed(session_id: str) -> bool:
    """Check if a session is managed by the job scheduler (remote SSH run).

    If the session is tracked as 'running' in any job manifest,
    the scheduler owns its liveness — local PID checks are meaningless.
    """
    jobs_dir = RUNS_DIR / "scheduler" / "jobs"
    if not jobs_dir.exists():
        return False
    for manifest_path in jobs_dir.glob("*/manifest.json"):
        try:
            import json

            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") not in ("running", "pending"):
                continue
            for run in manifest.get("runs", []):
                sid = run.get("spec", {}).get("parameters", {}).get("session_id", "")
                if sid == session_id and run.get("state") == "running":
                    return True
        except (OSError, ValueError, KeyError):
            continue
    return False


def is_stale(status_data: dict | None, session_id: str | None = None) -> bool:
    """Return True if status is 'running' but the process is no longer alive."""
    if status_data is None:
        return False
    if status_data.get("status") != "running":
        return False
    updated_at = status_data.get("updated_at")
    if updated_at is None:
        return False
    try:
        ts = datetime.fromisoformat(updated_at)
        if datetime.now(timezone.utc) - ts <= timedelta(minutes=_STALE_MINUTES):
            return False
    except (ValueError, TypeError):
        return False

    # If the scheduler is managing this session (remote SSH), trust it
    if session_id and _is_scheduler_managed(session_id):
        return False

    if session_id:
        pid_path = resolve_session_dir(session_id) / "pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                if verify_pid_ownership(pid, expected_cmdline=session_id):
                    return False
            except (ValueError, OSError):
                pass

    return True


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def recover_stale_sessions() -> dict[str, str]:
    """Recover session states after backend restart.

    Scans RUNS_DIR for sessions that claim to be "running" but whose
    subprocess is no longer alive.  Marks them as "error" so the UI
    reflects the truth.

    For sessions whose subprocess *is* still alive (survived the
    backend restart thanks to ``start_new_session=True``), re-register
    them in ``_active_procs`` so stop/cancel still works.

    Returns a dict of {session_id: action} for logging.
    """
    if not RUNS_DIR.exists():
        return {}

    actions: dict[str, str] = {}

    for session_dir in RUNS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        # Only look at session-like directories
        if not session_dir.name.startswith("session_"):
            continue

        session = SessionRecord(session_dir)
        status_data = session.read_status()
        if not status_data:
            continue
        if status_data.get("status") != "running":
            continue

        pid_path = session_dir / "pid"
        if not pid_path.exists():
            # No PID file — mark as error
            session.set_status("error", error="Backend restarted, no PID file to recover")
            actions[session.session_id] = "marked_error_no_pid"
            continue

        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, OSError):
            session.set_status("error", error="Backend restarted, corrupt PID file")
            actions[session.session_id] = "marked_error_bad_pid"
            continue

        if verify_pid_ownership(pid, expected_cmdline=session.session_id):
            # Process survived — re-attach watchdog
            _reattach_session(session, pid)
            # Re-allocate CPU cores so ResourceBar reflects usage
            # Guard: skip if already tracked (defensive against double-call)
            with _session_cpu_allocs_lock:
                already_tracked = session.session_id in _session_cpu_allocs
            alloc_file = session_dir / "cpu_alloc_count"
            if not already_tracked and alloc_file.exists():
                try:
                    from p2p.training.cpu_manager import get_cpu_manager

                    n_cores = int(alloc_file.read_text().strip())
                    alloc_id = f"session_{session.session_id}"
                    if get_cpu_manager().allocate(alloc_id, n_cores) is not None:
                        with _session_cpu_allocs_lock:
                            _session_cpu_allocs[session.session_id] = alloc_id
                except (ValueError, OSError):
                    pass
            actions[session.session_id] = "reattached"
        else:
            # Process died while backend was down
            error_msg = "Process terminated while backend was down"
            tail = read_log_tail(session_dir / "subprocess.log")
            if tail:
                error_msg = f"{error_msg}\n{tail}"
            session.set_status("error", error=error_msg)
            actions[session.session_id] = "marked_error_dead"

    return actions


def _reattach_session(session: SessionRecord, pid: int) -> None:
    """Re-attach a watchdog thread to a still-running subprocess by PID."""
    session_id = session.session_id

    def _watch_pid() -> None:
        # Poll until the process exits (we don't have a Popen handle)
        # Max 24h timeout to avoid infinite polling on zombie processes
        max_polls = 24 * 3600 // 2  # 24 hours at 2s intervals
        for _ in range(max_polls):
            if not is_pid_alive(pid):
                break
            time.sleep(2)

        # Process exited — clean up tracking state
        with _active_procs_lock:
            _active_procs.pop(session_id, None)
        _release_session_cores(session_id)

        error_msg = "Subprocess exited (recovered after backend restart)"
        tail = read_log_tail(session.path / "subprocess.log")
        if tail:
            error_msg = f"{error_msg}\n{tail}"
        if not session.set_status_if(
            "error",
            only_if=("running", "pending"),
            error=error_msg,
        ):
            logger.debug("Skipped error status for %s (already terminal)", session_id)

    # We can't get a real Popen handle, but store pid for stop_session
    # Use a sentinel to indicate "reattached by PID"
    with _active_procs_lock:
        _active_procs[session_id] = pid

    threading.Thread(target=_watch_pid, daemon=True, name=f"recover-{session_id}").start()


# ---------------------------------------------------------------------------
# Start session
# ---------------------------------------------------------------------------


def start_session(
    prompt: str,
    loop_config: LoopConfig,
    *,
    cpu_cores: list[int] | None = None,
    session_id: str | None = None,
    runs_dir: str | Path | None = None,
) -> str:
    """Start a loop session in a background subprocess.

    Parameters
    ----------
    prompt:
        Natural language behavior description.
    loop_config:
        Complete loop configuration (training, LLM, execution settings).
    cpu_cores:
        If given, pin the subprocess to these CPU cores via ``taskset``.
    session_id:
        Optional explicit session ID (used by run orchestrator).
    runs_dir:
        Override the default RUNS_DIR (used by run orchestrator
        to place sessions inside the run directory).

    Returns the session_id for polling.
    """
    if session_id is None:
        session_id = generate_session_id()

    target_runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR

    # Copy with overridden runs_dir — never mutate the caller's object
    loop_config = dataclasses.replace(loop_config, runs_dir=target_runs_dir)

    # Create session dir + status.json immediately so polling can start
    session = SessionRecord(target_runs_dir / resolve_session_subpath(session_id))
    session.set_status("running")

    # Persist the full start config so the session can be re-run later.
    # Derived from LoopConfig via asdict() so new fields auto-propagate.
    start_config = dataclasses.asdict(loop_config)
    start_config["runs_dir"] = str(start_config["runs_dir"])  # Path → str
    start_config["prompt"] = prompt
    try:
        session.config_path.write_text(json.dumps(start_config, indent=2))
    except OSError:
        logger.warning("Failed to save session_config.json for %s", session_id)

    # Allocate CPU cores in the API server's CPUManager so ResourceBar
    # reflects actual usage.  The subprocess manages its own internal
    # scheduling but this reservation prevents over-subscription and
    # keeps the UI accurate.
    cores_pool = loop_config.cores_pool
    cores_per_run = loop_config.cores_per_run
    max_parallel = loop_config.max_parallel
    if not cpu_cores and not cores_pool and cores_per_run > 0:
        from p2p.training.cpu_manager import get_cpu_manager

        cpu_mgr = get_cpu_manager()
        effective_cores_per = cores_per_run or 2
        configs = loop_config.configs
        seeds = loop_config.seeds
        num_configs_count = len(configs) if configs else 1
        num_seeds_count = len(seeds) if seeds else 1
        total_runs = num_configs_count * num_seeds_count
        if max_parallel > 0:
            concurrent = min(max_parallel, total_runs)
        else:
            concurrent = min(
                max(1, cpu_mgr.available_count() // effective_cores_per),
                total_runs,
            )
        total_needed = concurrent * effective_cores_per
        alloc_id = f"session_{session_id}"
        allocated = cpu_mgr.allocate(alloc_id, total_needed)
        if allocated is not None:
            with _session_cpu_allocs_lock:
                _session_cpu_allocs[session_id] = alloc_id
            # Persist so recover_stale_sessions can re-allocate after restart
            (session.path / "cpu_alloc_count").write_text(str(total_needed))
            # Pass allocated cores to subprocess for actual CPU pinning
            cpu_cores = allocated
            loop_config = dataclasses.replace(loop_config, cores_pool=allocated)
        else:
            logger.warning(
                "CPU allocation failed for session %s (requested %d, available %d)",
                session_id,
                total_needed,
                cpu_mgr.available_count(),
            )

    # Launch as subprocess so MuJoCo GLFW can use the process main thread
    env = {**os.environ, "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY}
    cmd = [
        *python_cmd(),
        "-m",
        "p2p.session.run_session",
        "--session-id",
        session_id,
        "--prompt",
        prompt,
        "--loop-config",
        loop_config.to_json(),
    ]
    # Pin to specific CPU cores if requested
    pin_cores = cores_pool or cpu_cores
    if pin_cores:
        core_list = ",".join(str(c) for c in pin_cores)
        cmd = ["taskset", "-c", core_list, *cmd]

    logger.info("Launching: %s", " ".join(cmd))

    # Redirect stdout/stderr to a log file so subprocess errors are visible
    log_path = session.path / "subprocess.log"
    log_file = open(log_path, "w", buffering=1)  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Save PID so we can re-attach after backend restart
    (session.path / "pid").write_text(str(proc.pid))
    with _active_procs_lock:
        _active_procs[session_id] = proc

    # Watchdog: if subprocess crashes early, mark session as error
    def _watch_subprocess() -> None:
        try:
            proc.wait()
        finally:
            log_file.close()
            with _active_procs_lock:
                _active_procs.pop(session_id, None)
            _release_session_cores(session_id)
        if proc.returncode != 0:
            error_tail = read_log_tail(log_path)
            if not session.set_status_if(
                "error",
                only_if=("running", "pending"),
                error=f"Subprocess exited with code {proc.returncode}\n{error_tail}",
            ):
                logger.debug("Skipped error status for %s (already terminal)", session_id)

    threading.Thread(target=_watch_subprocess, daemon=True, name=f"watch-{session_id}").start()

    return session_id


# ---------------------------------------------------------------------------
# Stop / Cancel
# ---------------------------------------------------------------------------


def stop_session(session_id: str, *, runs_dir: Path | None = None) -> bool:
    """Terminate a running session subprocess.

    Parameters
    ----------
    runs_dir:
        Override the default RUNS_DIR.  Sessions started with a custom
        ``runs_dir`` (e.g. by the run orchestrator) must pass the same
        path here so the correct status file is updated.

    Returns True if the process was stopped or marked cancelled.
    Falls back to OS process search when the in-memory tracker has lost
    the handle (e.g. after server reload).
    """
    runs_dir = runs_dir or RUNS_DIR
    killed = False

    # Try in-memory handle first (poll + kill under lock to avoid TOCTOU)
    with _active_procs_lock:
        proc_or_pid = _active_procs.pop(session_id, None)
        if isinstance(proc_or_pid, int):
            # Reattached by PID after recovery — kill by PID directly
            _kill_pid_tree(proc_or_pid, session_id=session_id)
            killed = True
        elif proc_or_pid is not None and proc_or_pid.poll() is None:
            _kill_proc_tree(proc_or_pid)
            killed = True

    # Fallback: find session processes via /proc cmdline scan
    if not killed:
        killed = _kill_session_by_name(session_id)

    # Release CPU cores reserved for this session
    _release_session_cores(session_id)

    # Atomically mark as cancelled if status is still running/pending
    sr = SessionRecord(runs_dir / resolve_session_subpath(session_id))
    if sr.set_status_if("cancelled", only_if=("running", "pending")):
        history = sr.read_history()
        if history is not None:
            history["status"] = "cancelled"
            sr.save_history(history)
        return True

    return killed
