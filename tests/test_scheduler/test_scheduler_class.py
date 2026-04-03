"""Tests for the Scheduler class."""

from __future__ import annotations

from p2p.scheduler.scheduler import Scheduler
from p2p.scheduler.types import RunSpec, RunStatus

# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class _FakeBackend:
    """In-memory backend that records calls and returns canned responses."""

    def __init__(self) -> None:
        self.submitted: list[tuple[RunSpec, list[int] | None]] = []
        self.cancelled: list[str] = []
        self._statuses: dict[str, RunStatus] = {}

    def submit(self, spec: RunSpec, *, allocated_cores: list[int] | None = None) -> RunStatus:
        self.submitted.append((spec, allocated_cores))
        status: RunStatus = {"run_id": spec["run_id"], "state": "running"}
        self._statuses[spec["run_id"]] = status
        return status

    def status(self, run_id: str) -> RunStatus:
        return dict(self._statuses.get(run_id, {"run_id": run_id, "state": "error"}))  # type: ignore[return-value]

    def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        if run_id in self._statuses:
            self._statuses[run_id]["state"] = "cancelled"
            return True
        return False

    def sync_results(self, run_id: str) -> bool:
        return run_id in self._statuses

    def complete(self, run_id: str) -> None:
        """Helper: mark a run as completed."""
        if run_id in self._statuses:
            self._statuses[run_id]["state"] = "completed"

    def fail(self, run_id: str) -> None:
        """Helper: mark a run as errored."""
        if run_id in self._statuses:
            self._statuses[run_id]["state"] = "error"


def _spec(run_id: str) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        entry_point="p2p.session.run_session",
        parameters={"session_id": run_id, "prompt": "test"},
        cpu_cores=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchedulerSubmit:
    def test_submit_batch(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)

        specs = [_spec("r1"), _spec("r2"), _spec("r3")]
        results = sched.submit(specs)

        assert len(results) == 3
        assert all(r["state"] == "running" for r in results)
        assert len(fb.submitted) == 3

    def test_submit_with_allocated_cores(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)

        specs = [_spec("r1"), _spec("r2")]
        cores = {"r1": [0, 1], "r2": [2, 3]}
        sched.submit(specs, allocated_cores=cores)

        assert fb.submitted[0][1] == [0, 1]
        assert fb.submitted[1][1] == [2, 3]

    def test_submit_partial_cores(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)

        specs = [_spec("r1"), _spec("r2")]
        cores = {"r1": [0, 1]}  # r2 has no cores
        sched.submit(specs, allocated_cores=cores)

        assert fb.submitted[0][1] == [0, 1]
        assert fb.submitted[1][1] is None

    def test_backend_property(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        assert sched.backend is fb


class TestSchedulerWait:
    def test_wait_all_completed(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        sched.submit([_spec("r1"), _spec("r2")])

        fb.complete("r1")
        fb.complete("r2")

        results = sched.wait(["r1", "r2"])
        assert all(r["state"] == "completed" for r in results)

    def test_wait_timeout(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        sched.submit([_spec("r1")])
        # r1 stays running → wait should time out

        results = sched.wait(["r1"], poll_interval=0.01, timeout=0.05)
        assert results[0]["state"] == "running"

    def test_wait_abort_on_all_failed(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        sched.submit([_spec("r1"), _spec("r2")])

        fb.fail("r1")
        # r2 still running → should be cancelled since all finished = error

        results = sched.wait(["r1", "r2"], poll_interval=0.01)
        assert results[0]["state"] == "error"
        assert results[1]["state"] == "cancelled"
        assert "r2" in fb.cancelled

    def test_wait_no_abort_when_some_succeed(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        sched.submit([_spec("r1"), _spec("r2"), _spec("r3")])

        fb.complete("r1")
        fb.fail("r2")
        fb.complete("r3")

        results = sched.wait(["r1", "r2", "r3"], poll_interval=0.01)
        assert results[0]["state"] == "completed"
        assert results[1]["state"] == "error"
        assert results[2]["state"] == "completed"
        assert fb.cancelled == []


class TestSchedulerCancelAll:
    def test_cancel_all(self) -> None:
        fb = _FakeBackend()
        sched = Scheduler(fb)
        sched.submit([_spec("r1"), _spec("r2")])

        sched.cancel_all(["r1", "r2"])

        assert set(fb.cancelled) == {"r1", "r2"}
        assert fb.status("r1")["state"] == "cancelled"
        assert fb.status("r2")["state"] == "cancelled"
