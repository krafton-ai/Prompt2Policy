"""Tests for RewardFunction ABC and reward_loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from p2p.training.reward_function import RewardFunction
from p2p.training.reward_loader import (
    LegacyRewardWrapper,
    _sanitize_escape_sequences,
    load_from_code,
    load_from_file,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SpeedReward(RewardFunction):
    """Concrete subclass for testing."""

    def compute(self, obs, action, next_obs, info):
        speed = float(info.get("x_velocity", 0.0))
        ctrl = -0.1 * float(np.sum(np.square(action)))
        return speed + ctrl, {"speed": speed, "ctrl": ctrl}

    @property
    def latex(self) -> str:
        return r"r = v_x - 0.1 \|a\|^2"

    @property
    def terms(self) -> dict[str, str]:
        return {"speed": "forward velocity", "ctrl": "control penalty"}


_LEGACY_CODE = textwrap.dedent("""\
    import numpy as np

    def reward_fn(obs, action, next_obs, info):
        \"\"\"
        LaTeX: r = v_x
        Terms:
            speed: forward velocity
        \"\"\"
        v = float(info.get("x_velocity", 0.0))
        return v, {"speed": v}
""")

_CLASS_CODE = textwrap.dedent("""\
    import numpy as np
    from p2p.training.reward_function import RewardFunction

    class MyReward(RewardFunction):
        def compute(self, obs, action, next_obs, info):
            v = float(info.get("x_velocity", 0.0))
            return v, {"speed": v}

        @property
        def latex(self):
            return "r = v_x"

        @property
        def terms(self):
            return {"speed": "forward velocity"}
""")


def _dummy_step():
    """Return (obs, action, next_obs, info) for testing."""
    obs = np.zeros(17)
    action = np.ones(6)
    next_obs = np.zeros(17)
    info = {"x_velocity": 3.5, "x_position": 1.0}
    return obs, action, next_obs, info


# ---------------------------------------------------------------------------
# RewardFunction ABC
# ---------------------------------------------------------------------------


class TestRewardFunctionABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            RewardFunction()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        reward = SpeedReward()
        obs, action, next_obs, info = _dummy_step()
        total, terms = reward(obs, action, next_obs, info)
        assert total == pytest.approx(3.5 + (-0.1 * 6.0))
        assert "speed" in terms
        assert "ctrl" in terms

    def test_call_delegates_to_compute(self):
        reward = SpeedReward()
        obs, action, next_obs, info = _dummy_step()
        assert reward(obs, action, next_obs, info) == reward.compute(obs, action, next_obs, info)

    def test_latex_and_terms_properties(self):
        reward = SpeedReward()
        assert "v_x" in reward.latex
        assert "speed" in reward.terms

    def test_description_defaults_to_docstring(self):
        reward = SpeedReward()
        assert reward.description == "Concrete subclass for testing."

    def test_description_is_not_latex(self):
        reward = SpeedReward()
        assert reward.description != reward.latex


# ---------------------------------------------------------------------------
# LegacyRewardWrapper
# ---------------------------------------------------------------------------


class TestLegacyWrapper:
    def test_wraps_plain_function(self):
        def reward_fn(obs, action, next_obs, info):
            """
            LaTeX: r = v
            Terms:
                speed: velocity
            """
            return float(info["x_velocity"]), {"speed": float(info["x_velocity"])}

        wrapper = LegacyRewardWrapper(reward_fn)
        obs, action, next_obs, info = _dummy_step()
        total, terms = wrapper(obs, action, next_obs, info)
        assert total == pytest.approx(3.5)
        assert wrapper.latex == "r = v"
        assert "speed" in wrapper.terms

    def test_description_from_wrapped_function(self):
        def reward_fn(obs, action, next_obs, info):
            """Reward speed."""
            return float(info["x_velocity"]), {"speed": float(info["x_velocity"])}

        wrapper = LegacyRewardWrapper(reward_fn)
        assert wrapper.description == "Reward speed."
        assert wrapper.description != wrapper.latex

    def test_source_preserves_latex_escapes(self):
        source = textwrap.dedent("""\
            def reward_fn(obs, action, next_obs, info):
                \"\"\"
                LaTeX: r = \\tau + v
                Terms:
                    tau: torque
                \"\"\"
                return 0.0, {"tau": 0.0}
        """)

        def reward_fn(obs, action, next_obs, info):
            return 0.0, {"tau": 0.0}

        wrapper = LegacyRewardWrapper(reward_fn, source=source)
        assert "\\tau" in wrapper.latex


# ---------------------------------------------------------------------------
# load_from_code
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _sanitize_escape_sequences (#53)
# ---------------------------------------------------------------------------


class TestSanitizeEscapeSequences:
    def test_fixes_invalid_escape_in_string(self):
        code = 'x = "\\|hello"\n'
        result = _sanitize_escape_sequences(code)
        assert "\\\\|" in result

    def test_preserves_valid_escapes(self):
        code = 'x = "\\n\\t\\\\"\n'
        result = _sanitize_escape_sequences(code)
        assert result == code

    def test_preserves_raw_strings(self):
        code = 'x = r"\\|hello"\n'
        result = _sanitize_escape_sequences(code)
        assert result == code

    def test_handles_multiline_strings(self):
        code = '"""\\|test"""\n'
        result = _sanitize_escape_sequences(code)
        assert "\\\\|" in result

    def test_multiline_docstring_spanning_lines(self):
        """Regression: multi-line docstring with invalid escape must not duplicate lines."""
        code = (
            "def reward_fn(obs, action, next_obs, info):\n"
            '    """LaTeX: r = \\omega * v\n'
            "    Terms:\n"
            "        speed: forward velocity\n"
            '    """\n'
            '    return 1.0, {"speed": 1.0}\n'
        )
        result = _sanitize_escape_sequences(code)
        # \omega's \o is invalid → should become \\omega
        assert "\\\\omega" in result
        # Must not duplicate the continuation lines of the docstring
        assert result.count("Terms:") == 1
        # Must still compile
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=SyntaxWarning)
            compile(result, "<test>", "exec")

    def test_handles_malformed_code(self):
        code = 'x = "unterminated\n'
        result = _sanitize_escape_sequences(code)
        assert result == code  # returns unchanged on tokenize error

    def test_full_reward_fn_with_invalid_escape(self):
        code = (
            "def reward_fn(obs, action, next_obs, info):\n"
            '    """LaTeX: r = \\|a\\|^2"""\n'
            '    return 1.0, {"a": 1.0}\n'
        )
        result = _sanitize_escape_sequences(code)
        # Should compile without warnings
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=SyntaxWarning)
            compile(result, "<test>", "exec")


class TestLoadFromCode:
    def test_load_legacy_function(self):
        reward = load_from_code(_LEGACY_CODE)
        assert isinstance(reward, LegacyRewardWrapper)
        obs, action, next_obs, info = _dummy_step()
        total, terms = reward(obs, action, next_obs, info)
        assert total == pytest.approx(3.5)
        assert reward.latex == "r = v_x"

    def test_no_reward_raises(self):
        with pytest.raises(ValueError, match="does not define"):
            load_from_code("x = 1")


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_load_legacy_file(self, tmp_path: Path):
        p = tmp_path / "reward.py"
        p.write_text(_LEGACY_CODE)
        reward = load_from_file(p)
        assert isinstance(reward, LegacyRewardWrapper)
        obs, action, next_obs, info = _dummy_step()
        total, _ = reward(obs, action, next_obs, info)
        assert total == pytest.approx(3.5)

    def test_load_class_file(self, tmp_path: Path):
        p = tmp_path / "reward.py"
        p.write_text(_CLASS_CODE)
        reward = load_from_file(p)
        assert not isinstance(reward, LegacyRewardWrapper)
        assert isinstance(reward, RewardFunction)
        obs, action, next_obs, info = _dummy_step()
        total, terms = reward(obs, action, next_obs, info)
        assert total == pytest.approx(3.5)
        assert reward.latex == "r = v_x"

    def test_invalid_file_raises(self, tmp_path: Path):
        p = tmp_path / "bad.py"
        p.write_text("x = 42")
        with pytest.raises(ValueError, match="does not define"):
            load_from_file(p)
