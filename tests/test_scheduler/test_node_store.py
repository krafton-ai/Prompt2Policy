"""Tests for node_store CRUD operations."""

import pytest

from p2p.scheduler import node_store
from p2p.scheduler.types import NodeConfig


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path):
    """Redirect node_store to a temp file."""
    store_path = tmp_path / "nodes.json"
    node_store.set_store_path(store_path)
    yield
    # Reset handled by next test's fixture


def _make_node(node_id: str = "test-node") -> NodeConfig:
    return {
        "node_id": node_id,
        "host": "10.0.0.1",
        "user": "user",
        "port": 22,
        "base_dir": "/home/user/p2p",
        "max_cores": 60,
    }


def test_list_seeds_localhost() -> None:
    """First call seeds localhost node when file does not exist."""
    nodes = node_store.list_nodes()
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "localhost"


def test_add_and_list() -> None:
    node_store.add_node(_make_node("n1"))
    node_store.add_node(_make_node("n2"))
    nodes = node_store.list_nodes()
    # localhost seed + n1 + n2
    assert {n["node_id"] for n in nodes} >= {"n1", "n2"}


def test_add_duplicate_raises() -> None:
    node_store.add_node(_make_node("n1"))
    with pytest.raises(ValueError, match="already exists"):
        node_store.add_node(_make_node("n1"))


def test_get_node() -> None:
    node_store.add_node(_make_node("n1"))
    node = node_store.get_node("n1")
    assert node is not None
    assert node["host"] == "10.0.0.1"


def test_get_node_not_found() -> None:
    assert node_store.get_node("nonexistent") is None


def test_update_node() -> None:
    node_store.add_node(_make_node("n1"))
    updated = node_store.update_node("n1", {"host": "10.0.0.2", "max_cores": 32})
    assert updated["host"] == "10.0.0.2"
    assert updated["max_cores"] == 32
    # Verify persisted
    fetched = node_store.get_node("n1")
    assert fetched is not None
    assert fetched["host"] == "10.0.0.2"


def test_update_node_not_found_raises() -> None:
    with pytest.raises(KeyError, match="not found"):
        node_store.update_node("nonexistent", {"host": "x"})


def test_remove_node() -> None:
    node_store.add_node(_make_node("n1"))
    assert node_store.remove_node("n1") is True
    # Only localhost seed remains
    remaining = node_store.list_nodes()
    assert all(n["node_id"] != "n1" for n in remaining)


def test_remove_node_not_found() -> None:
    assert node_store.remove_node("nonexistent") is False
