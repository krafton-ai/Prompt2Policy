"""Node configuration CRUD backed by a JSON file.

Storage path: ``runs/scheduler/nodes.json``
Thread-safe via a module-level lock. In-memory cache avoids repeated disk reads.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from p2p.scheduler.types import NodeConfig
from p2p.settings import RUNS_DIR

logger = logging.getLogger(__name__)

_STORE_DIR = RUNS_DIR / "scheduler"
_STORE_PATH = _STORE_DIR / "nodes.json"
_lock = threading.Lock()
_cache: list[NodeConfig] | None = None


def _localhost_seed() -> NodeConfig:
    import getpass

    return {
        "node_id": "localhost",
        "host": "127.0.0.1",
        "user": getpass.getuser(),
        "port": 22,
        "max_cores": 60,
    }


def _load() -> list[NodeConfig]:
    global _cache  # noqa: PLW0603
    if _cache is not None:
        return _cache
    if not _STORE_PATH.exists():
        seed = _localhost_seed()
        _save([seed])
        _cache = [seed]
        return _cache
    try:
        data = json.loads(_STORE_PATH.read_text())
        nodes = data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt nodes.json, returning empty list")
        _cache = []
        return _cache
    _cache = nodes
    return _cache


def _save(nodes: list[NodeConfig]) -> None:
    global _cache  # noqa: PLW0603
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(nodes, indent=2))
    tmp.rename(_STORE_PATH)
    _cache = nodes


def list_nodes() -> list[NodeConfig]:
    with _lock:
        return list(_load())


def get_node(node_id: str) -> NodeConfig | None:
    with _lock:
        for node in _load():
            if node["node_id"] == node_id:
                return node
    return None


def add_node(config: NodeConfig) -> None:
    with _lock:
        nodes = _load()
        for n in nodes:
            if n["node_id"] == config["node_id"]:
                msg = f"Node '{config['node_id']}' already exists"
                raise ValueError(msg)
        nodes = [*nodes, config]
        _save(nodes)


def update_node(node_id: str, updates: dict) -> NodeConfig:
    with _lock:
        nodes = list(_load())
        for i, n in enumerate(nodes):
            if n["node_id"] == node_id:
                n.update(updates)  # type: ignore[typeddict-item]
                nodes[i] = n
                _save(nodes)
                return n
        msg = f"Node '{node_id}' not found"
        raise KeyError(msg)


def remove_node(node_id: str) -> bool:
    with _lock:
        nodes = _load()
        filtered = [n for n in nodes if n["node_id"] != node_id]
        if len(filtered) == len(nodes):
            return False
        _save(filtered)
        return True


def set_store_path(path: Path) -> None:
    """Override the store path (for testing)."""
    global _STORE_DIR, _STORE_PATH, _cache  # noqa: PLW0603
    _STORE_DIR = path.parent
    _STORE_PATH = path
    _cache = None  # invalidate cache
