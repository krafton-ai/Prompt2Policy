"""Auto-configure CPU/env resources from calibration data.

Given ``(num_configs, num_seeds)``, find the optimal
``(cores_per_run, num_envs, max_parallel)`` that minimizes estimated
wall-clock time, using benchmarked SPS data from ``calibration.json``.

When no calibration file is found the module falls back to conservative
rule-based defaults (4 cores, 16 envs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2p.contracts import ResourceAllocation

# Default calibration file: <project>/docs/cpu_stress_test/calibration.json
_DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[3] / "docs" / "cpu_stress_test" / "calibration.json"
)

_FALLBACK_CORES = 4
_FALLBACK_ENVS_PER_CORE = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_calibration(path: Path | None = None) -> dict | None:
    """Load calibration.json, returning ``None`` if not found."""
    p = path or _DEFAULT_CALIBRATION
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _best_sps_for_cores(
    profiles: dict,
    cores: int,
    env_id: str | None = None,
) -> float:
    """Peak SPS for *cores*, optionally scoped to one environment.

    When *env_id* is ``None``, returns the **average** peak SPS across all
    calibrated environments.
    """
    cores_key = str(cores)

    if env_id and env_id in profiles:
        env_profile = profiles[env_id].get(cores_key, {})
        return max(env_profile.values()) if env_profile else 0.0

    peaks: list[float] = []
    for env_data in profiles.values():
        core_data = env_data.get(cores_key, {})
        if core_data:
            peaks.append(max(core_data.values()))

    return sum(peaks) / len(peaks) if peaks else 0.0


def _candidate_cores(profiles: dict) -> set[int]:
    """Extract the set of core counts present in the calibration profiles."""
    cores: set[int] = set()
    for env_data in profiles.values():
        for key in env_data:
            cores.add(int(key))
    return cores or {2, 4, 8}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_best_allocation(
    num_configs: int,
    num_seeds: int,
    env_id: str | None = None,
    usable_cores: int | None = None,
    calibration_path: Path | None = None,
) -> ResourceAllocation:
    """Find the optimal resource allocation for the given experiment size.

    Evaluates every calibrated core count and picks the one that minimizes
    ``time_score = num_batches / best_sps`` (proportional to wall-clock time).

    Parameters
    ----------
    num_configs:
        Number of hyperparameter configurations.
    num_seeds:
        Number of random seeds per configuration.
    env_id:
        MuJoCo environment ID.  ``None`` = average across all calibrated envs.
    usable_cores:
        Total usable CPU cores.  ``None`` = read from ``calibration.json``.
    calibration_path:
        Override path to ``calibration.json``.

    Returns
    -------
    ResourceAllocation
        Dict with ``cores_per_run``, ``num_envs``, ``max_parallel``,
        ``total_runs``, ``num_batches``, and ``time_score``.
    """
    total_runs = max(1, num_configs * num_seeds)
    cal = _load_calibration(calibration_path)

    if cal is None:
        return _fallback_allocation(total_runs, usable_cores or 60)

    total_cores = usable_cores or cal.get("usable_cores", 60)
    envs_per_core = cal.get("envs_per_core_rule", _FALLBACK_ENVS_PER_CORE)
    profiles = cal.get("profiles", {})
    candidates = _candidate_cores(profiles)

    best: ResourceAllocation | None = None

    for cores in sorted(candidates):
        max_parallel = total_cores // cores
        if max_parallel < 1:
            continue

        num_batches = math.ceil(total_runs / max_parallel)
        best_sps = _best_sps_for_cores(profiles, cores, env_id)
        if best_sps <= 0:
            continue

        time_score = num_batches / best_sps

        num_envs = cores * envs_per_core
        alloc: ResourceAllocation = {
            "cores_per_run": cores,
            "num_envs": num_envs,
            "max_parallel": max_parallel,
            "total_runs": total_runs,
            "num_batches": num_batches,
            "time_score": time_score,
            "estimated_processes": num_envs * min(max_parallel, total_runs),
            "usable_cores": total_cores,
        }

        if best is None or time_score < best["time_score"]:
            best = alloc

    return best or _fallback_allocation(total_runs, total_cores)


def _fallback_allocation(total_runs: int, total_cores: int) -> ResourceAllocation:
    """Conservative rule-based allocation (no calibration data)."""
    cores = _FALLBACK_CORES
    num_envs = cores * _FALLBACK_ENVS_PER_CORE
    max_parallel = max(1, total_cores // cores)
    effective_parallel = min(max_parallel, total_runs)
    return {
        "cores_per_run": cores,
        "num_envs": num_envs,
        "max_parallel": max_parallel,
        "total_runs": total_runs,
        "num_batches": math.ceil(total_runs / max_parallel),
        "time_score": 0.0,
        "estimated_processes": num_envs * effective_parallel,
        "usable_cores": total_cores,
    }
