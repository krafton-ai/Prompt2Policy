"""Manifest I/O for file-based job state.

Each job gets its own directory under ``runs/scheduler/jobs/<job_id>/``
with a ``manifest.json`` file as the single source of truth.

All writes are atomic (tmp + rename) to prevent corruption on crash.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from p2p.scheduler.types import JobManifest
from p2p.settings import RUNS_DIR

logger = logging.getLogger(__name__)

_JOBS_DIR = RUNS_DIR / "scheduler" / "jobs"


def _jobs_dir() -> Path:
    return _JOBS_DIR


def set_jobs_dir(path: Path) -> None:
    """Override the jobs directory (for testing)."""
    global _JOBS_DIR  # noqa: PLW0603
    _JOBS_DIR = path


def _manifest_path(job_id: str) -> Path:
    return _jobs_dir() / job_id / "manifest.json"


def read_job_manifest(job_id: str) -> JobManifest | None:
    """Read manifest.json for *job_id*, or ``None`` if missing/corrupt."""
    path = _manifest_path(job_id)
    try:
        return json.loads(path.read_text())  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_job_manifest(manifest: JobManifest) -> None:
    """Atomically write manifest.json (tmp + rename)."""
    job_id = manifest["job_id"]
    job_dir = _jobs_dir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(path)


def list_job_ids() -> list[str]:
    """Return all job_ids by scanning the jobs directory."""
    jobs_dir = _jobs_dir()
    if not jobs_dir.exists():
        return []
    return sorted(
        d.name for d in jobs_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()
    )
