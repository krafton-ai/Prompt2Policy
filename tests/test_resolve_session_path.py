"""Tests for resolve_session_subpath / resolve_session_dir."""

from __future__ import annotations

import pytest

from p2p.settings import RUNS_DIR, resolve_session_dir, resolve_session_subpath


class TestResolveSessionSubpath:
    """resolve_session_subpath returns relative path strings."""

    def test_regular_session_unchanged(self) -> None:
        assert resolve_session_subpath("session_abc123") == "session_abc123"

    def test_benchmark_case_nested(self) -> None:
        assert resolve_session_subpath("bm_abc12345_case0") == "bm_abc12345/case0"
        assert resolve_session_subpath("bm_abc12345_case12") == "bm_abc12345/case12"

    def test_benchmark_case_nested_timestamped(self) -> None:
        assert (
            resolve_session_subpath("bm_00010101_000000_abc12345_case0")
            == "bm_00010101_000000_abc12345/case0"
        )
        assert (
            resolve_session_subpath("bm_00010101_000000_abc12345_case12")
            == "bm_00010101_000000_abc12345/case12"
        )

    def test_benchmark_root_unchanged(self) -> None:
        # bm_abc12345 (no _caseN suffix) should pass through unchanged
        assert resolve_session_subpath("bm_abc12345") == "bm_abc12345"

    def test_path_traversal_dotdot_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            resolve_session_subpath("../etc/passwd")

    def test_path_traversal_leading_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            resolve_session_subpath("/tmp/evil")

    def test_path_traversal_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            resolve_session_subpath("foo/bar")

    def test_path_traversal_backslash_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            resolve_session_subpath("foo\\bar")

    def test_path_traversal_embedded_dotdot_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            resolve_session_subpath("session_abc/../../../etc/passwd")


class TestResolveSessionDir:
    """resolve_session_dir returns RUNS_DIR / subpath."""

    def test_regular_session(self) -> None:
        assert resolve_session_dir("session_abc123") == RUNS_DIR / "session_abc123"

    def test_benchmark_case_nested(self) -> None:
        assert resolve_session_dir("bm_abc12345_case0") == RUNS_DIR / "bm_abc12345" / "case0"

    def test_benchmark_case_nested_timestamped(self) -> None:
        assert (
            resolve_session_dir("bm_00010101_000000_abc12345_case0")
            == RUNS_DIR / "bm_00010101_000000_abc12345" / "case0"
        )
