"""Tests for p2p.utils.utils."""

from pathlib import Path

import pytest

from p2p.utils.utils import extract_code_block, read_log_tail


class TestExtractCodeBlock:
    def test_fenced_python_block(self):
        text = "```python\ndef foo():\n    pass\n```"
        assert extract_code_block(text, "def foo") == "def foo():\n    pass"

    def test_fenced_no_lang(self):
        text = "```\ndef bar():\n    return 0\n```"
        assert extract_code_block(text, "def bar") == "def bar():\n    return 0"

    def test_raw_fallback(self):
        text = "def foo():\n    return 1"
        assert extract_code_block(text, "def foo") == "def foo():\n    return 1"

    def test_no_code_raises(self):
        with pytest.raises(ValueError, match="No valid code found"):
            extract_code_block("just random text", "def foo")

    def test_multiple_fn_names(self):
        text = "def _make_reward():\n    pass"
        result = extract_code_block(text, ["def reward_fn", "_make_reward"])
        assert "_make_reward" in result

    def test_multiple_fn_names_none_found(self):
        with pytest.raises(ValueError):
            extract_code_block("nothing here", ["def reward_fn", "_make_reward"])

    def test_fenced_block_preferred_over_raw(self):
        text = (
            "Some preamble\n```python\ndef foo():\n    return 42\n```\n"
            "Some trailing text with def foo"
        )
        assert extract_code_block(text, "def foo") == "def foo():\n    return 42"


class TestReadLogTail:
    def test_returns_last_n_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("\n".join(f"line {i}" for i in range(30)))

        assert read_log_tail(log, n=5) == "\n".join(f"line {i}" for i in range(25, 30))

    def test_default_20_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("\n".join(f"line {i}" for i in range(50)))

        result = read_log_tail(log)
        assert result.count("\n") == 19  # 20 lines = 19 newlines

    def test_fewer_lines_than_n(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("only\ntwo")

        assert read_log_tail(log, n=20) == "only\ntwo"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_log_tail(tmp_path / "nonexistent.log") == ""

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("")

        assert read_log_tail(log) == ""
