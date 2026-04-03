"""High-level run scheduler that orchestrates Backend instances.

Currently **dormant** — controllers use manifest + subprocess (``job_scheduler.py``).
This class will be activated in #220 when the Trainer protocol replaces
``parallel_trainer``, allowing ``loop.py`` to submit training runs through
``Scheduler`` instead of managing subprocesses directly.

Design sketch::

    loop.py → Trainer(Scheduler) → Backend.submit() × N → wait → aggregate
      ├── LocalBackend  (taskset pinning, local Popen)
      ├── SSHBackend    (rsync + remote launch)
      └── (future) RayBackend, K8sBackend, SlurmBackend
"""

from __future__ import annotations

import logging
import time

from p2p.scheduler.backend import Backend
from p2p.scheduler.types import TERMINAL_STATES, RunSpec, RunStatus

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2  # seconds


class Scheduler:
    """Orchestrate run submission and lifecycle via a ``Backend``.

    The scheduler is the single coordinator that translates a batch of
    ``RunSpec`` into running processes, polls for completion, and collects
    results.  It does **not** decide *which* node to use — that is the
    caller's responsibility (or a future node-selector layer).

    Args:
        backend: The execution backend to delegate to.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> Backend:
        return self._backend

    def submit(
        self,
        specs: list[RunSpec],
        *,
        allocated_cores: dict[str, list[int]] | None = None,
    ) -> list[RunStatus]:
        """Submit a batch of RunSpecs.

        Args:
            specs: List of RunSpecs to submit.
            allocated_cores: Optional mapping of ``run_id → core list``
                for CPU affinity pinning.

        Returns:
            List of initial RunStatus (one per spec).
        """
        cores_map = allocated_cores or {}
        results: list[RunStatus] = []
        for spec in specs:
            cores = cores_map.get(spec["run_id"])
            status = self._backend.submit(spec, allocated_cores=cores)
            results.append(status)
        return results

    def wait(
        self,
        run_ids: list[str],
        *,
        poll_interval: float = _POLL_INTERVAL,
        timeout: float = 0,
        abort_on_all_failed: bool = True,
    ) -> list[RunStatus]:
        """Poll until all runs reach a terminal state.

        Args:
            run_ids: IDs to wait for.
            poll_interval: Seconds between polls.
            timeout: Max seconds to wait (0 = unlimited).
            abort_on_all_failed: If True, cancel remaining runs when all
                currently-finished runs have failed.

        Returns:
            Final RunStatus for each run_id.
        """
        start = time.monotonic()

        while True:
            statuses = [self._backend.status(rid) for rid in run_ids]
            states = [s["state"] for s in statuses]

            if all(s in TERMINAL_STATES for s in states):
                break

            # Early abort: all finished runs failed → cancel the rest
            if abort_on_all_failed:
                finished = [s for s in states if s in TERMINAL_STATES]
                if finished and all(s == "error" for s in finished):
                    logger.warning("All finished runs failed — aborting remaining")
                    for rid, state in zip(run_ids, states):
                        if state not in TERMINAL_STATES:
                            self._backend.cancel(rid)
                    # Re-poll to get final states
                    statuses = [self._backend.status(rid) for rid in run_ids]
                    break

            if timeout > 0 and (time.monotonic() - start) >= timeout:
                logger.warning("Scheduler.wait timed out after %.0fs", timeout)
                break

            time.sleep(poll_interval)

        return statuses

    def cancel_all(self, run_ids: list[str]) -> None:
        """Cancel all given runs."""
        for rid in run_ids:
            self._backend.cancel(rid)
