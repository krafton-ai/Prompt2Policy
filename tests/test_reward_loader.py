"""Tests for pure functions in reward_loader module."""

from __future__ import annotations

from pathlib import Path

from p2p.training.reward_loader import (
    _fix_escapes_in_string_token,
    _sanitize_escape_sequences,
    _strip_numpy_imports,
    load_from_code,
    load_from_file,
)
from p2p.training.reward_loader import (
    _parse_docstring as parse_docstring,
)

# ---------------------------------------------------------------------------
# _strip_numpy_imports
# ---------------------------------------------------------------------------


def test_strip_numpy_imports_removes_import_numpy():
    code = "import numpy\nx = 1"
    assert _strip_numpy_imports(code) == "\nx = 1"


def test_strip_numpy_imports_removes_import_numpy_as_np():
    code = "import numpy as np\nx = np.zeros(3)"
    assert _strip_numpy_imports(code) == "\nx = np.zeros(3)"


def test_strip_numpy_imports_preserves_from_numpy():
    code = "from numpy import array\nx = array([1])"
    assert _strip_numpy_imports(code) == code


def test_strip_numpy_imports_handles_indented_import():
    code = "def f():\n    import numpy as np\n    return np.zeros(1)"
    result = _strip_numpy_imports(code)
    assert "import numpy" not in result
    assert "return np.zeros(1)" in result


def test_strip_numpy_imports_no_match():
    code = "import math\nx = math.pi"
    assert _strip_numpy_imports(code) == code


# ---------------------------------------------------------------------------
# _fix_escapes_in_string_token
# ---------------------------------------------------------------------------


def test_fix_escapes_in_string_token_valid_escapes_unchanged():
    token = r"'hello\nworld'"
    assert _fix_escapes_in_string_token(token) == token


def test_fix_escapes_in_string_token_invalid_escape_doubled():
    # \p is not a valid escape
    token = "'hello\\pworld'"
    result = _fix_escapes_in_string_token(token)
    assert result == "'hello\\\\pworld'"


def test_fix_escapes_in_string_token_triple_quoted():
    token = '"""line\\pone"""'
    result = _fix_escapes_in_string_token(token)
    assert result == '"""line\\\\pone"""'


def test_fix_escapes_in_string_token_raw_string_unchanged():
    # Raw strings are skipped upstream in _sanitize_escape_sequences.
    # 'r' is not in "bBuUfF", so prefix_end=0 and rest[0]='r' is not a quote.
    token = "r'hello\\pworld'"
    assert _fix_escapes_in_string_token(token) == token


def test_fix_escapes_in_string_token_f_string_prefix():
    token = "f'val\\p'"
    result = _fix_escapes_in_string_token(token)
    assert result == "f'val\\\\p'"


# ---------------------------------------------------------------------------
# _sanitize_escape_sequences
# ---------------------------------------------------------------------------


def test_sanitize_escape_sequences_no_strings():
    code = "x = 1 + 2"
    assert _sanitize_escape_sequences(code) == code


def test_sanitize_escape_sequences_valid_escapes_unchanged():
    code = 'x = "hello\\nworld"'
    assert _sanitize_escape_sequences(code) == code


def test_sanitize_escape_sequences_fixes_invalid_escape():
    # \p is not a valid Python escape
    code = 'x = "hello\\pworld"'
    result = _sanitize_escape_sequences(code)
    assert "\\\\p" in result


def test_sanitize_escape_sequences_raw_string_unchanged():
    code = "x = r'hello\\pworld'"
    assert _sanitize_escape_sequences(code) == code


def test_sanitize_escape_sequences_malformed_code_returns_original():
    code = 'x = "unterminated'
    result = _sanitize_escape_sequences(code)
    assert result == code


def test_sanitize_escape_sequences_latex_in_docstring():
    r"""LaTeX escapes like \cdot, \sum, \omega in docstrings (#200)."""
    code = (
        "def reward_fn(obs, action, next_obs, info):\n"
        '    """LaTeX: r = \\sum w_i \\cdot r_i + \\omega \\min(x, 1)"""\n'
        '    return 1.0, {"a": 1.0}\n'
    )
    result = _sanitize_escape_sequences(code)
    assert "\\\\c" in result  # \cdot → \\cdot
    assert "\\\\s" in result  # \sum → \\sum
    assert "\\\\o" in result  # \omega → \\omega
    assert "\\\\m" in result  # \min → \\min


# ---------------------------------------------------------------------------
# load_from_code / load_from_file with LaTeX escapes (#200)
# ---------------------------------------------------------------------------

_LATEX_DOCSTRING_CODE = (
    "def reward_fn(obs, action, next_obs, info):\n"
    '    """Reward for running.\n'
    "    LaTeX: r = \\sum w_i \\cdot r_i + \\omega \\min(x, 1)\n"
    "    Terms:\n"
    "      speed: forward velocity reward\n"
    '    """\n'
    '    return 1.0, {"speed": 1.0}\n'
)


def test_load_from_code_latex_escapes_no_error():
    r"""load_from_code must not raise on LaTeX escapes like \cdot (#200)."""
    fn = load_from_code(_LATEX_DOCSTRING_CODE)
    r, terms = fn(None, None, None, {})
    assert r == 1.0
    assert terms == {"speed": 1.0}


def test_load_from_file_latex_escapes_no_error(tmp_path: Path):
    r"""load_from_file must not raise on LaTeX escapes like \cdot (#200)."""
    reward_file = tmp_path / "reward_fn.py"
    reward_file.write_text(_LATEX_DOCSTRING_CODE)
    fn = load_from_file(reward_file)
    r, terms = fn(None, None, None, {})
    assert r == 1.0
    assert terms == {"speed": 1.0}


# ---------------------------------------------------------------------------
# parse_docstring
# ---------------------------------------------------------------------------


def _make_fn_with_doc(doc: str):
    """Create a dummy function with a given docstring."""

    def fn(obs, action, next_obs, info):
        pass

    fn.__doc__ = doc
    return fn


def test_parse_docstring_extracts_latex():
    fn = _make_fn_with_doc("Reward function.\n\nLaTeX: r = v_x - 0.1 \\|a\\|^2")
    latex, terms = parse_docstring(fn)
    assert latex == "r = v_x - 0.1 \\|a\\|^2"
    assert terms == []


def test_parse_docstring_extracts_terms():
    doc = """\
Reward function.

Terms:
    forward: forward velocity reward
    ctrl: control cost penalty
"""
    fn = _make_fn_with_doc(doc)
    latex, terms = parse_docstring(fn)
    assert len(terms) == 2
    assert terms[0]["name"] == "forward"
    assert terms[0]["description"] == "forward velocity reward"
    assert terms[1]["name"] == "ctrl"
    assert terms[1]["description"] == "control cost penalty"


def test_parse_docstring_extracts_both():
    doc = """\
Reward function.

LaTeX: r = v_x

Terms:
    speed: x velocity
"""
    fn = _make_fn_with_doc(doc)
    latex, terms = parse_docstring(fn)
    assert latex == "r = v_x"
    assert len(terms) == 1
    assert terms[0]["name"] == "speed"
    assert terms[0]["description"] == "x velocity"


def test_parse_docstring_prefers_source_over_runtime_doc():
    fn = _make_fn_with_doc("Runtime doc.\n\nLaTeX: r_runtime")
    source = '"""Source doc.\n\nLaTeX: r_source\n"""'
    latex, _ = parse_docstring(fn, source=source)
    assert latex == "r_source"


def test_parse_docstring_empty():
    fn = _make_fn_with_doc("")
    latex, terms = parse_docstring(fn)
    assert latex == ""
    assert terms == []


def test_parse_docstring_no_docstring():
    fn = _make_fn_with_doc(None)  # type: ignore[arg-type]
    latex, terms = parse_docstring(fn)
    assert latex == ""
    assert terms == []


def test_parse_docstring_terms_with_equation_lines():
    doc = """\
Reward.

Terms:
    forward: velocity reward
        r_forward = v_x
"""
    fn = _make_fn_with_doc(doc)
    _, terms = parse_docstring(fn)
    assert len(terms) == 1
    assert terms[0]["name"] == "forward"
    assert terms[0].get("latex") == "r_forward = v_x"
