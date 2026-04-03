"""Entity lifecycle: metadata, soft-delete, restore, trash, hard-delete."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from p2p.scheduler.manifest_io import _jobs_dir
from p2p.session.iteration_record import read_json_safe
from p2p.settings import RUNS_DIR, resolve_session_dir

_RUNNING_STATUSES = {"running", "pending"}


def _is_session_dir(d: Path) -> bool:
    return d.name.startswith("session_") or (d / "loop_history.json").exists()


def _is_benchmark_dir(d: Path) -> bool:
    return (d.name.startswith("benchmark_") or d.name.startswith("bm_")) and (
        d / "benchmark.json"
    ).exists()


def _parse_session_dir_timestamp(name: str) -> datetime | None:
    """Extract creation timestamp embedded in session directory names.

    Supports ``session_YYYYMMDD_HHMMSS_*`` and ``bm_YYYYMMDD_HHMMSS_*``.
    """
    m = re.match(r"(?:session|bm)_(\d{8})_(\d{6})_", name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _get_session_created_at(session_dir: Path) -> str:
    # Prefer the timestamp embedded in the directory name (immutable).
    dt = _parse_session_dir_timestamp(session_dir.name)
    if dt:
        return dt.isoformat()
    # Fall back to directory mtime (may drift when files are added).
    try:
        ts = session_dir.stat().st_mtime
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat()
    except OSError:
        return ""


def _get_entity_status(entity_dir: Path) -> str | None:
    """Read status from status.json."""
    status_data = read_json_safe(entity_dir / "status.json")
    if status_data:
        return status_data.get("status")
    return None


# -- Core metadata helpers ---------------------------------------------------


def read_entity_metadata(entity_dir: Path) -> dict:
    """Read metadata.json from an entity directory."""
    meta_path = entity_dir / "metadata.json"
    return read_json_safe(meta_path) or {}


def inject_metadata(model: dict | object, entity_dir: Path) -> None:
    """Inject metadata fields (alias, starred, tags) into a Pydantic model."""
    meta = read_entity_metadata(entity_dir)
    if isinstance(model, dict):
        model["alias"] = meta.get("alias", "")
        model["starred"] = meta.get("starred", False)
        model["tags"] = meta.get("tags", [])
    else:
        model.alias = meta.get("alias", "")  # type: ignore[attr-defined]
        model.starred = meta.get("starred", False)  # type: ignore[attr-defined]
        model.tags = meta.get("tags", [])  # type: ignore[attr-defined]


def is_entity_deleted(entity_dir: Path) -> bool:
    """Check if an entity has been soft-deleted."""
    meta = read_entity_metadata(entity_dir)
    return bool(meta.get("deleted_at"))


def update_entity_metadata(
    entity_dir: Path,
    alias: str | None = None,
    starred: bool | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Update metadata.json fields (merge, not replace)."""
    meta = read_entity_metadata(entity_dir)
    if alias is not None:
        meta["alias"] = alias
    if starred is not None:
        meta["starred"] = starred
    if tags is not None:
        meta["tags"] = tags
    meta_path = entity_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def soft_delete_entity(entity_dir: Path) -> bool:
    """Soft-delete by setting deleted_at in metadata.json. Returns True if done."""
    meta = read_entity_metadata(entity_dir)
    meta["deleted_at"] = datetime.now(timezone.utc).isoformat()
    meta_path = entity_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def restore_entity(entity_dir: Path) -> bool:
    """Remove deleted_at from metadata.json. Returns True if done."""
    meta = read_entity_metadata(entity_dir)
    meta.pop("deleted_at", None)
    meta_path = entity_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


# -- Session metadata --------------------------------------------------------


def update_session_metadata(
    session_id: str,
    alias: str | None = None,
    starred: bool | None = None,
    tags: list[str] | None = None,
) -> dict:
    entity_dir = resolve_session_dir(session_id)
    if not entity_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    return update_entity_metadata(entity_dir, alias=alias, starred=starred, tags=tags)


def delete_session(session_id: str) -> None:
    entity_dir = resolve_session_dir(session_id)
    if not entity_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    status = _get_entity_status(entity_dir)
    if status in _RUNNING_STATUSES:
        raise ValueError("Cannot delete a running session. Stop it first.")
    soft_delete_entity(entity_dir)


def restore_session(session_id: str) -> None:
    entity_dir = resolve_session_dir(session_id)
    if not entity_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    restore_entity(entity_dir)


# -- Benchmark metadata ------------------------------------------------------


def update_benchmark_metadata(
    benchmark_id: str,
    alias: str | None = None,
    starred: bool | None = None,
    tags: list[str] | None = None,
) -> dict:
    entity_dir = RUNS_DIR / benchmark_id
    if not entity_dir.exists():
        raise FileNotFoundError(f"Benchmark {benchmark_id} not found")
    return update_entity_metadata(entity_dir, alias=alias, starred=starred, tags=tags)


def delete_benchmark(benchmark_id: str) -> None:
    entity_dir = RUNS_DIR / benchmark_id
    if not entity_dir.exists():
        raise FileNotFoundError(f"Benchmark {benchmark_id} not found")
    status = _get_entity_status(entity_dir)
    if status in _RUNNING_STATUSES:
        raise ValueError("Cannot delete a running benchmark. Stop it first.")
    soft_delete_entity(entity_dir)


def restore_benchmark(benchmark_id: str) -> None:
    entity_dir = RUNS_DIR / benchmark_id
    if not entity_dir.exists():
        raise FileNotFoundError(f"Benchmark {benchmark_id} not found")
    restore_entity(entity_dir)


# -- Job metadata ------------------------------------------------------------


_VALID_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _validate_entity_id(entity_id: str) -> None:
    if not _VALID_ID_RE.match(entity_id):
        raise ValueError("Invalid entity id")


def delete_job(job_id: str) -> None:
    _validate_entity_id(job_id)
    job_dir = _jobs_dir() / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job {job_id} not found")
    manifest = read_json_safe(job_dir / "manifest.json") or {}
    if manifest.get("status") in _RUNNING_STATUSES:
        raise ValueError("Cannot delete a running job. Cancel it first.")
    soft_delete_entity(job_dir)


def restore_job(job_id: str) -> None:
    _validate_entity_id(job_id)
    job_dir = _jobs_dir() / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job {job_id} not found")
    restore_entity(job_dir)


def bulk_trash_jobs(job_ids: list[str]) -> tuple[int, list[str]]:
    """Soft-delete multiple jobs. Returns (trashed_count, failed_ids)."""
    trashed = 0
    failed: list[str] = []
    for jid in job_ids:
        try:
            delete_job(jid)
            trashed += 1
        except (FileNotFoundError, ValueError):
            failed.append(jid)
    return trashed, failed


# -- Hard delete & trash -----------------------------------------------------


def _get_job_benchmark_id(job_dir: Path) -> str | None:
    """Return the benchmark_id linked to a job, if any."""
    manifest = read_json_safe(job_dir / "manifest.json") or {}
    return (manifest.get("metadata") or {}).get("benchmark_id")


def hard_delete_entity(entity_id: str) -> None:
    """Permanently delete an entity directory from disk.

    Only allowed for soft-deleted entities (must be in trash first).
    For jobs, the linked benchmark pointer directory is also removed.
    """
    _validate_entity_id(entity_id)
    entity_dir = RUNS_DIR / entity_id
    if not entity_dir.exists():
        # Check jobs directory
        entity_dir = _jobs_dir() / entity_id
    if not entity_dir.exists():
        raise FileNotFoundError(f"Entity {entity_id} not found")
    if not is_entity_deleted(entity_dir):
        raise ValueError("Entity must be in trash before permanent deletion")
    status = _get_entity_status(entity_dir)
    if status in _RUNNING_STATUSES:
        raise ValueError("Cannot delete a running entity")
    # For jobs, also remove the linked benchmark pointer directory
    is_job = entity_dir.parent == _jobs_dir()
    bm_id = _get_job_benchmark_id(entity_dir) if is_job else None
    shutil.rmtree(entity_dir)
    if bm_id:
        bm_dir = RUNS_DIR / bm_id
        if bm_dir.exists():
            bm_status = _get_entity_status(bm_dir)
            if bm_status not in _RUNNING_STATUSES:
                shutil.rmtree(bm_dir)


def hard_delete_all_trash() -> int:
    """Permanently delete all soft-deleted entities. Returns count deleted."""
    items = list_trash()
    deleted = 0
    for item in items:
        try:
            hard_delete_entity(item["entity_id"])
            deleted += 1
        except (FileNotFoundError, ValueError):
            continue
    return deleted


def list_trash() -> list[dict]:
    """List all soft-deleted entities across sessions, benchmarks, and jobs."""
    items: list[dict] = []

    # Scan RUNS_DIR for sessions and benchmarks
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            meta = read_entity_metadata(d)
            if not meta.get("deleted_at"):
                continue

            # Determine entity type
            if _is_session_dir(d):
                entity_type = "session"
                history = read_json_safe(d / "loop_history.json") or {}
                prompt = history.get("prompt", "")
                status_data = read_json_safe(d / "status.json") or {}
                status = status_data.get("status", history.get("status", ""))
                created_at = _get_session_created_at(d)
            elif _is_benchmark_dir(d):
                entity_type = "benchmark"
                manifest = read_json_safe(d / "benchmark.json") or {}
                prompt = ""
                status = manifest.get("status", "")
                created_at = manifest.get("created_at", "")
            else:
                continue

            items.append(
                {
                    "entity_id": d.name,
                    "entity_type": entity_type,
                    "alias": meta.get("alias", ""),
                    "deleted_at": meta.get("deleted_at", ""),
                    "created_at": created_at,
                    "prompt": prompt,
                    "status": status,
                }
            )

    # Scan jobs directory
    jobs_dir = _jobs_dir()
    if jobs_dir.exists():
        for d in jobs_dir.iterdir():
            if not d.is_dir():
                continue
            meta = read_entity_metadata(d)
            if not meta.get("deleted_at"):
                continue
            manifest = read_json_safe(d / "manifest.json") or {}
            if not manifest:
                continue
            items.append(
                {
                    "entity_id": d.name,
                    "entity_type": "job",
                    "alias": meta.get("alias", ""),
                    "deleted_at": meta.get("deleted_at", ""),
                    "created_at": manifest.get("created_at", ""),
                    "prompt": "",
                    "status": manifest.get("status", ""),
                }
            )

    items.sort(key=lambda x: x.get("deleted_at", ""), reverse=True)
    return items
