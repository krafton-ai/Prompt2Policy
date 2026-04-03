"""Shared subprocess utilities for launching Python child processes."""

from __future__ import annotations

import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)


def python_cmd(*, unbuffered: bool = False) -> list[str]:
    """Return the command prefix to run a Python module.

    When running inside a non-base conda environment, use ``sys.executable``
    directly so that the correct Python and packages are used.  Otherwise,
    prefer ``uv run python`` when ``uv`` is on PATH.

    Parameters
    ----------
    unbuffered:
        If *True*, add ``-u`` to force unbuffered stdout/stderr.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_default = os.environ.get("CONDA_PREFIX_1", "")

    # CONDA_PREFIX_1 is set when a non-base env is active.
    # Check "/envs/" with path separators to avoid false positives like
    # "/home/user/my-envs-project/".
    is_conda = bool(conda_prefix and (conda_default or "/envs/" in conda_prefix))

    if is_conda:
        cmd = [sys.executable]
        logger.debug("Python command: %s (conda=True)", cmd)
    elif shutil.which("uv"):
        cmd = ["uv", "run", "python"]
        logger.debug("Python command: %s (conda=False, uv=True)", cmd)
    else:
        cmd = [sys.executable]
        logger.debug("Python command: %s (conda=False, uv=False)", cmd)

    if unbuffered:
        cmd.append("-u")

    return cmd
