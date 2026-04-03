"""Tests for p2p.utils.subprocess_utils.python_cmd()."""

from __future__ import annotations

import sys
from unittest.mock import patch

from p2p.utils.subprocess_utils import python_cmd


class TestPythonCmdConda:
    """When a non-base conda environment is active, use sys.executable."""

    def test_conda_non_base_with_conda_prefix_1(self) -> None:
        env = {"CONDA_PREFIX": "/opt/conda/envs/rl", "CONDA_PREFIX_1": "/opt/conda"}
        with patch.dict("os.environ", env, clear=False):
            assert python_cmd() == [sys.executable]

    def test_conda_non_base_with_envs_in_path(self) -> None:
        env = {"CONDA_PREFIX": "/opt/conda/envs/rl", "CONDA_PREFIX_1": ""}
        with patch.dict("os.environ", env, clear=False):
            assert python_cmd() == [sys.executable]

    def test_conda_false_positive_envs_in_project_name(self) -> None:
        """Path like /home/user/my-envs-project should NOT trigger conda mode."""
        env = {"CONDA_PREFIX": "/home/user/my-envs-project", "CONDA_PREFIX_1": ""}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("shutil.which", return_value="/usr/bin/uv"),
        ):
            assert python_cmd() == ["uv", "run", "python"]

    def test_conda_base_env_without_prefix_1(self) -> None:
        """Base conda env (no CONDA_PREFIX_1, no /envs/) should fall through."""
        env = {"CONDA_PREFIX": "/opt/conda", "CONDA_PREFIX_1": ""}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("shutil.which", return_value="/usr/bin/uv"),
        ):
            assert python_cmd() == ["uv", "run", "python"]


class TestPythonCmdUv:
    """When not in conda, prefer uv if available."""

    def test_uv_available(self) -> None:
        env = {"CONDA_PREFIX": "", "CONDA_PREFIX_1": ""}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("shutil.which", return_value="/usr/bin/uv"),
        ):
            assert python_cmd() == ["uv", "run", "python"]

    def test_uv_not_available(self) -> None:
        env = {"CONDA_PREFIX": "", "CONDA_PREFIX_1": ""}
        with patch.dict("os.environ", env, clear=False), patch("shutil.which", return_value=None):
            assert python_cmd() == [sys.executable]


class TestPythonCmdUnbuffered:
    """The unbuffered flag appends -u."""

    def test_unbuffered_with_uv(self) -> None:
        env = {"CONDA_PREFIX": "", "CONDA_PREFIX_1": ""}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("shutil.which", return_value="/usr/bin/uv"),
        ):
            assert python_cmd(unbuffered=True) == ["uv", "run", "python", "-u"]

    def test_unbuffered_with_conda(self) -> None:
        env = {"CONDA_PREFIX": "/opt/conda/envs/rl", "CONDA_PREFIX_1": "/opt/conda"}
        with patch.dict("os.environ", env, clear=False):
            assert python_cmd(unbuffered=True) == [sys.executable, "-u"]

    def test_unbuffered_with_sys_executable(self) -> None:
        env = {"CONDA_PREFIX": "", "CONDA_PREFIX_1": ""}
        with patch.dict("os.environ", env, clear=False), patch("shutil.which", return_value=None):
            assert python_cmd(unbuffered=True) == [sys.executable, "-u"]

    def test_unbuffered_false_by_default(self) -> None:
        env = {"CONDA_PREFIX": "", "CONDA_PREFIX_1": ""}
        with patch.dict("os.environ", env, clear=False), patch("shutil.which", return_value=None):
            result = python_cmd()
            assert "-u" not in result
