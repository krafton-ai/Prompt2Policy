"""Load reward functions from .py files or code strings, with legacy wrapping."""

from __future__ import annotations

import inspect
import logging
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from p2p.contracts import RewardTerm
from p2p.training.reward_function import RewardFunction

logger = logging.getLogger(__name__)


class LegacyRewardWrapper(RewardFunction):
    """Wrap a plain ``reward_fn(obs, action, next_obs, info)`` function.

    Always calls with 4 args — MuJoCo state is passed via ``info["mj_data"]``
    and ``info["mj_model"]`` when side_info is enabled.
    """

    def __init__(self, fn: Callable, source: str | None = None, engine: str = "mujoco") -> None:
        self._fn = fn
        self._source = source or ""
        self._engine = engine
        self._latex, self._structured_terms = _parse_docstring(fn, self._source)

    def reset(self) -> None:
        """Re-create the closure from source to get a clean state dict.

        LLM-generated reward functions use ``_make_reward()`` closures with
        internal ``state`` dicts.  Without this, stateful variables (milestone
        flags, cumulative counters) leak across episodes because the closure
        is never re-initialised.  ``CustomRewardWrapper.reset()`` calls this
        at every episode boundary.
        """
        if not self._source:
            return
        try:
            fresh = _reload_fn_from_source(self._source, engine=self._engine)
            if fresh is not None:
                self._fn = fresh
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to re-create reward closure on reset; keeping old state",
                exc_info=True,
            )

    def compute(self, obs, action, next_obs, info):
        return self._fn(obs, action, next_obs, info)

    def __call__(self, *args):
        """Forward to the underlying function with 4 args."""
        return self._fn(*args[:4])

    @property
    def latex(self) -> str:
        return self._latex

    @property
    def terms(self) -> dict[str, str]:
        """Backward-compatible dict view of structured terms."""
        return {t["name"]: t.get("description", "") for t in self._structured_terms}

    @property
    def structured_terms(self) -> list[RewardTerm]:
        """Structured term list for RewardSpec."""
        return list(self._structured_terms)

    @property
    def description(self) -> str:
        return inspect.getdoc(self._fn) or ""


def load_from_file(path: Path, engine: str = "mujoco") -> RewardFunction:
    """Load a RewardFunction from a ``.py`` file.

    The file may define either:
    - A ``RewardFunction`` subclass (first one found is instantiated), or
    - A plain ``reward_fn`` function (auto-wrapped in ``LegacyRewardWrapper``).

    Delegates to :func:`load_from_code` so that LLM-generated code with
    invalid escape sequences (e.g. LaTeX ``\\\\cdot`` in docstrings) is
    automatically sanitized before compilation.
    """
    source = path.read_text()
    return load_from_code(source, engine=engine)


def load_from_code(code: str, engine: str = "mujoco") -> RewardFunction:
    """Load a RewardFunction from a code string (legacy ``reward_fn`` style).

    This replaces the previous ``reward_author.load_reward_fn`` for cases where
    code is generated dynamically (e.g. by LLM).
    """
    original_code = _strip_numpy_imports(code)
    sanitized_code = _sanitize_escape_sequences(original_code)
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    from p2p.training.simulator import get_simulator

    get_simulator(engine).inject_reward_namespace(namespace)
    # Dynamic execution of LLM-generated reward code in sandboxed namespace
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=SyntaxWarning)
        compiled = compile(sanitized_code, "<reward_fn>", "exec")  # noqa: S102
    _exec_compiled(compiled, namespace)

    # Check for class-based reward first
    for obj in namespace.values():
        if isinstance(obj, type) and issubclass(obj, RewardFunction) and obj is not RewardFunction:
            return obj()

    # Legacy function
    fn = namespace.get("reward_fn")
    if fn is None or not callable(fn):
        msg = "Code does not define a RewardFunction subclass or reward_fn function"
        raise ValueError(msg)
    # Pass original (pre-sanitized) source so _parse_docstring extracts
    # LaTeX with single backslashes (e.g. \cdot, not \\cdot).
    return LegacyRewardWrapper(fn, source=original_code, engine=engine)


def _reload_fn_from_source(source: str, engine: str = "mujoco") -> Callable | None:
    """Re-exec reward source to get a fresh closure.  Lightweight version of
    ``load_from_code`` that suppresses SyntaxWarnings (the code already
    compiled successfully once — docstring escape sequences are harmless).
    """
    code = _sanitize_escape_sequences(source)
    code = _strip_numpy_imports(code)
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    from p2p.training.simulator import get_simulator

    get_simulator(engine).inject_reward_namespace(namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        compiled = compile(code, "<reward_fn>", "exec")  # noqa: S102
    _exec_compiled(compiled, namespace)
    fn = namespace.get("reward_fn")
    return fn if callable(fn) else None


_NUMPY_IMPORT_RE = re.compile(
    r"^\s*import\s+numpy(?:\s+as\s+\w+)?\s*$",
    re.MULTILINE,
)


def _strip_numpy_imports(code: str) -> str:
    """Remove ``import numpy`` lines — np is pre-injected into the exec namespace.

    LLMs sometimes place ``import numpy as np`` inside functions, which
    creates a local binding that shadows the outer-scope ``np`` for the
    entire function (Python determines scope at parse time). This causes
    ``UnboundLocalError`` if ``np`` is used before the import statement.

    Only strips ``import numpy`` / ``import numpy as X`` lines.
    Does NOT strip ``from numpy import ...`` — those bind bare names
    (e.g. ``array``, ``zeros``) which are not in the pre-injected namespace.
    """
    return _NUMPY_IMPORT_RE.sub("", code)


# Valid Python escape sequences (single char after backslash)
_VALID_ESCAPES = frozenset("\\'\"\nabfnrtv01234567xuUN")


def _sanitize_escape_sequences(code: str) -> str:
    r"""Fix invalid escape sequences in string literals of LLM-generated code.

    Scans string literals and replaces ``\X`` (where X is not a valid escape
    character) with ``\\X``.  This prevents SyntaxWarning on Python 3.12+.
    """
    import io
    import tokenize

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        logger.debug("Tokenize failed during escape sanitization, skipping")
        return code  # let compile() handle malformed code

    # Each replacement: (start_pos, end_pos, old_text, new_text)
    # where start_pos/end_pos are (line, col) tuples from the tokenizer.
    replacements: list[tuple[tuple[int, int], tuple[int, int], str, str]] = []
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        s = tok.string
        # Skip raw strings — they don't process escapes
        prefix_end = 0
        while prefix_end < len(s) and s[prefix_end] in "bBuUfF":
            prefix_end += 1
        if prefix_end < len(s) and s[prefix_end] in "rR":
            continue
        fixed = _fix_escapes_in_string_token(s)
        if fixed != s:
            replacements.append((tok.start, tok.end, s, fixed))

    if not replacements:
        return code

    # Build a line-offset table so we can convert (line, col) → absolute offset.
    lines = code.splitlines(keepends=True)
    line_offsets = [0]
    for ln in lines:
        line_offsets.append(line_offsets[-1] + len(ln))

    # Apply replacements in reverse order (by start position) to keep offsets stable.
    result = code
    for tok_start, tok_end, old, new in reversed(replacements):
        start_off = line_offsets[tok_start[0] - 1] + tok_start[1]
        end_off = line_offsets[tok_end[0] - 1] + tok_end[1]
        result = result[:start_off] + new + result[end_off:]

    return result


def _fix_escapes_in_string_token(token: str) -> str:
    r"""Fix invalid escape sequences within a single string token."""
    prefix_end = 0
    while prefix_end < len(token) and token[prefix_end] in "bBuUfF":
        prefix_end += 1
    prefix = token[:prefix_end]
    rest = token[prefix_end:]

    if rest.startswith('"""') or rest.startswith("'''"):
        quote = rest[:3]
    elif rest.startswith('"') or rest.startswith("'"):
        quote = rest[0]
    else:
        return token

    inner = rest[len(quote) : -len(quote)]

    fixed_parts: list[str] = []
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            next_char = inner[i + 1]
            if next_char not in _VALID_ESCAPES:
                fixed_parts.append("\\\\")
                i += 1
                continue
        fixed_parts.append(inner[i])
        i += 1

    fixed_inner = "".join(fixed_parts)
    if fixed_inner == inner:
        return token
    return prefix + quote + fixed_inner + quote


def _exec_compiled(compiled, namespace):  # noqa: S102
    """Execute pre-compiled code in the given namespace."""
    exec(compiled, namespace)  # noqa: S102


def _split_equation_description(text: str) -> tuple[str, str]:
    """Split ``r_{x} = ... (description)`` into ``(latex, description)``.

    Uses balanced parenthesis counting from the right so that LaTeX like
    ``\\max(0, z)`` is not broken.  The trailing ``(...)`` is only treated as a
    description if its content is plain text (no ``\\``, ``{``, ``}``, ``=``,
    ``_``).
    """
    text = text.strip()
    if not text.endswith(")"):
        return text, ""
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                candidate_desc = text[i + 1 : -1].strip()
                candidate_latex = text[:i].strip()
                # Plain-text description should not contain math symbols
                if not any(c in candidate_desc for c in "\\{}=_"):
                    return candidate_latex, candidate_desc
                # It's part of the equation — no separate description
                return text, ""
    return text, ""


def _parse_docstring(fn: Callable, source: str = "") -> tuple[str, list[RewardTerm]]:
    """Extract LaTeX and structured Terms from docstring.

    Returns ``(overall_latex, list_of_reward_terms)`` where each term is a
    :class:`RewardTerm` dict with ``name``, ``description``, and optionally
    ``latex`` (extracted from the equation line after the description).

    Prefers raw source text to avoid escape-sequence issues
    (e.g. ``\\t`` in ``\\tau`` interpreted as tab).
    """
    doc = ""
    if source:
        # Find the docstring containing "LaTeX:" — closures may have
        # multiple docstrings (helper functions before reward_fn).
        for pattern in (r'"""(.*?)"""', r"'''(.*?)'''"):
            candidates = re.findall(pattern, source, re.DOTALL)
            for candidate in candidates:
                if "LaTeX:" in candidate:
                    doc = candidate.strip()
                    break
            if doc:
                break
        # Fall back to first docstring if none contains "LaTeX:"
        if not doc:
            doc_match = re.search(r'"""(.*?)"""', source, re.DOTALL)
            if not doc_match:
                doc_match = re.search(r"'''(.*?)'''", source, re.DOTALL)
            doc = doc_match.group(1).strip() if doc_match else ""
    if not doc:
        doc = inspect.getdoc(fn) or ""

    # When extracted from raw source, Python string escapes like \\cdot
    # need unescaping to get proper LaTeX: \cdot
    from_source = bool(source and doc)

    def _unescape(s: str) -> str:
        return s.replace("\\\\", "\\") if from_source else s

    latex = ""
    latex_match = re.search(r"LaTeX:\s*(.+)", doc)
    if latex_match:
        latex = _unescape(latex_match.group(1).strip())

    terms: list[RewardTerm] = []
    in_terms = False
    current_term: RewardTerm | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("terms:"):
            in_terms = True
            continue
        is_equation = stripped.startswith(("r_", "r{", "r\\", "r ="))
        if in_terms and ":" in stripped and not is_equation:
            key, desc = stripped.split(":", 1)
            desc_stripped = desc.strip()
            # If description starts with an equation (e.g. "r_{height} = ..."),
            # extract it as the term's LaTeX and use any trailing text as description.
            if re.match(r"r[_{\\]", desc_stripped) or desc_stripped.startswith("r ="):
                eq_latex, eq_desc = _split_equation_description(desc_stripped)
                current_term = {
                    "name": key.strip(),
                    "latex": _unescape(eq_latex),
                    "description": eq_desc,
                }
            else:
                current_term = {"name": key.strip(), "description": desc_stripped}
            terms.append(current_term)
        elif in_terms and current_term is not None and is_equation:
            # Per-term LaTeX equation line
            current_term["latex"] = _unescape(stripped)
        elif in_terms and not stripped:
            in_terms = False
            current_term = None

    return latex, terms
