"""PID verification and safe process group kill utilities.

Guards against PID reuse: if a process exits and the OS recycles its PID,
blind ``os.killpg()`` would kill an unrelated process group.  All kill
paths should use ``safe_killpg()`` instead of calling ``os.killpg()``
directly.  See issue #380.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def verify_pid_ownership(pid: int, *, expected_cmdline: str | None = None) -> bool:
    """Verify a PID still belongs to the expected process.

    Guards against PID reuse: if a process exits and the OS recycles its
    PID, blind ``os.killpg()`` would kill an unrelated process group.

    Parameters
    ----------
    pid:
        The PID to verify.
    expected_cmdline:
        Substring expected in ``/proc/{pid}/cmdline`` (e.g. session_id,
        ``"p2p.session.run_session"``, ``"p2p.scheduler.job_scheduler"``).
        If ``None``, falls back to checking for any ``"p2p"`` marker.

    Returns True only if the PID is alive AND its cmdline matches.
    """
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = proc_cmdline.read_bytes().decode(errors="replace")
    except OSError:
        return False
    if not cmdline:
        return False
    if "p2p" not in cmdline:
        return False
    if expected_cmdline:
        return expected_cmdline in cmdline
    return True


def verify_pgid_ownership(pid: int) -> int | None:
    """Get the PGID for a PID, but only if PGID == PID.

    Processes spawned with ``start_new_session=True`` become their own
    process group leaders (PGID == PID).  If PGID != PID the process was
    either not started with ``start_new_session=True`` or the PID was
    recycled into a different process group — either way,
    ``os.killpg()`` is unsafe.

    Returns the PGID if safe, ``None`` otherwise.
    """
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None
    if pgid != pid:
        logger.warning(
            "PGID mismatch for PID %d: PGID=%d (expected %d). "
            "PID may have been recycled — refusing to killpg.",
            pid,
            pgid,
            pid,
        )
        return None
    return pgid


def get_descendant_pids(root_pid: int) -> list[int]:
    """Walk /proc to find all descendant PIDs of *root_pid*.

    Finds children via two methods:

    1. Parent→child tree walk (standard ppid traversal)
    2. Process group membership (catches Isaac Sim forkserver workers
       that get reparented to PID 1 but keep the executor's PGID)
    """
    children: list[int] = []
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return children

    try:
        root_pgid = os.getpgid(root_pid)
    except (ProcessLookupError, PermissionError):
        root_pgid = None

    parent_map: dict[int, list[int]] = {}
    pgid_matched: set[int] = set()
    my_pid = os.getpid()

    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid or pid == root_pid:
            continue
        try:
            stat = (entry / "stat").read_text()
            parts = stat.split(") ")[1].split()
            ppid = int(parts[1])
            pgrp = int(parts[2])
            parent_map.setdefault(ppid, []).append(pid)
            if root_pgid is not None and pgrp == root_pgid:
                pgid_matched.add(pid)
        except (OSError, IndexError, ValueError):
            continue

    tree_children: set[int] = set()
    queue = [root_pid]
    while queue:
        p = queue.pop(0)
        for child in parent_map.get(p, []):
            tree_children.add(child)
            queue.append(child)

    children = list(tree_children | pgid_matched)
    if pgid_matched - tree_children:
        logger.debug(
            "Found %d orphaned processes by PGID %s (reparented to init)",
            len(pgid_matched - tree_children),
            root_pgid,
        )
    return children


def force_kill_pids(pids: list[int]) -> None:
    """SIGKILL a list of PIDs, ignoring already-dead processes."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def safe_killpg(pid: int, *, expected_cmdline: str | None = None) -> bool:
    """Verify PID/PGID ownership, then SIGTERM the process group.

    Shared guard used by all kill paths to prevent collateral damage
    from recycled PIDs (issue #380).  The sequence is:

    1. Verify ``/proc/{pid}/cmdline`` matches *expected_cmdline*.
    2. Verify PGID == PID (expected for ``start_new_session=True``).
    3a. If PGID verified: ``os.killpg(pgid, SIGTERM)``, wait up to 5 s,
        then escalate to ``SIGKILL``.
    3b. If PGID mismatch: ``os.kill(pid, SIGTERM)`` only (no escalation).

    Returns True if a signal was sent, False if verification failed
    or the signal could not be delivered.
    """
    if not verify_pid_ownership(pid, expected_cmdline=expected_cmdline):
        logger.warning(
            "PID %d failed ownership check (expected %r), skipping kill",
            pid,
            expected_cmdline,
        )
        return False

    pgid = verify_pgid_ownership(pid)
    if pgid is None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already dead — objective achieved
        except OSError as exc:
            logger.error("Failed to kill PID %d: %s — process may still be running", pid, exc)
            return False
        return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True  # already dead
    except OSError as exc:
        logger.error(
            "killpg(%d, SIGTERM) failed: %s — process group may still be running",
            pgid,
            exc,
        )
        return False

    for _ in range(50):  # 5 seconds
        if not is_pid_alive(pid):
            return True
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # exited between SIGTERM and SIGKILL
    except OSError as exc:
        logger.error(
            "SIGKILL escalation failed for PGID %d: %s — process group may still be running",
            pgid,
            exc,
        )
        return False
    return True
