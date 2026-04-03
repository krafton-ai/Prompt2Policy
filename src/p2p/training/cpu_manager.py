"""Thread-safe CPU core allocator for parallel experiments.

Manages a pool of logical CPU cores, assigning contiguous blocks to
experiment subprocesses. Uses Linux ``taskset`` for affinity pinning.

Typical usage::

    mgr = get_cpu_manager()
    cores = mgr.allocate("exp_seed_1", num_cores=5)  # [2, 3, 4, 5, 6]
    # ... launch subprocess with taskset -c 2,3,4,5,6 ...
    mgr.release("exp_seed_1")                         # cores returned to pool
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from p2p.contracts import CpuAllocation, ResourceStatus


@dataclass
class CoreAllocation:
    """A block of CPU cores assigned to one run."""

    cores: list[int]
    run_id: str


class CPUManager:
    """Thread-safe CPU core pool manager.

    Parameters
    ----------
    total_cores:
        Total logical cores on the machine.  ``None`` = auto-detect.
    reserved:
        Number of low-numbered cores to reserve for system / API server.
    """

    def __init__(
        self,
        total_cores: int | None = None,
        reserved: int = 2,
    ) -> None:
        self._total = total_cores or os.cpu_count() or 64
        self._reserved = reserved
        # Reserve the *last* N cores (e.g. 62, 63) for system / API server
        self._available: list[int] = list(range(self._total - reserved))
        self._allocations: dict[str, CoreAllocation] = {}
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def allocate(self, run_id: str, num_cores: int) -> list[int] | None:
        """Try to allocate *num_cores* for *run_id*.

        Returns the list of assigned core numbers, or ``None`` if the pool
        does not have enough free cores.
        """
        with self._lock:
            if len(self._available) < num_cores:
                return None
            cores = self._available[:num_cores]
            self._available = self._available[num_cores:]
            alloc = CoreAllocation(cores=cores, run_id=run_id)
            self._allocations[run_id] = alloc
            return cores

    def release(self, run_id: str) -> None:
        """Return cores for *run_id* back to the pool."""
        with self._lock:
            alloc = self._allocations.pop(run_id, None)
            if alloc:
                self._available.extend(alloc.cores)
                self._available.sort()

    def can_fit(self, num_cores: int) -> bool:
        """Check whether *num_cores* can be allocated right now."""
        with self._lock:
            return len(self._available) >= num_cores

    def available_count(self) -> int:
        """Number of currently free cores."""
        with self._lock:
            return len(self._available)

    def allocation_ids(self) -> list[str]:
        """Return a snapshot of all current allocation IDs."""
        with self._lock:
            return list(self._allocations.keys())

    def status(self) -> ResourceStatus:
        """Snapshot of the manager state (for the /resources/status endpoint)."""
        with self._lock:
            return ResourceStatus(
                total_cores=self._total,
                reserved_cores=self._reserved,
                available_cores=len(self._available),
                active_runs=len(self._allocations),
                allocations=[
                    CpuAllocation(run_id=a.run_id, cores=a.cores)
                    for a in self._allocations.values()
                ],
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cpu_manager: CPUManager | None = None
_singleton_lock = threading.Lock()


def get_cpu_manager() -> CPUManager:
    """Return (or create) the global CPUManager singleton."""
    global _cpu_manager
    if _cpu_manager is None:
        with _singleton_lock:
            if _cpu_manager is None:
                _cpu_manager = CPUManager()
    return _cpu_manager
