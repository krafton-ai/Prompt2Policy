"""Job controllers: manifest-based orchestration via subprocess schedulers.

Each controller converts a high-level job config into RunSpecs, writes a
manifest file, and spawns an independent ``job_scheduler`` subprocess.
The subprocess handles all submit/wait/sync logic — the API server is
stateless with respect to running jobs.

Hierarchy:
- **Run**: 1 env_id + 1 config + 1 seed (smallest unit, = 1 RunSpec)
- **Session**: multi-config × multi-seed. Creates N Runs, all on the same node.
- **Benchmark**: CSV test cases. Each test case is a Session.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from p2p.benchmark.benchmark_helpers import build_default_stages
from p2p.config import LoopConfig
from p2p.scheduler.job_queries import _manifest_to_job
from p2p.scheduler.manifest_io import write_job_manifest
from p2p.scheduler.spec_builder import (
    DEFAULT_CONFIG,
    benchmark_case_to_spec,
    session_to_spec,
)
from p2p.scheduler.types import BackendType, Job, JobManifest, RunRecord, now_iso
from p2p.settings import RUNS_DIR

logger = logging.getLogger(__name__)


def _loop_config_summary(
    loop_config: LoopConfig,
    num_configs: int,
) -> dict[str, Any]:
    """Extract display-friendly config fields for the job manifest."""
    return {
        "cores_per_run": loop_config.cores_per_run,
        "num_envs": loop_config.train.num_envs,
        "total_timesteps": loop_config.train.total_timesteps,
        "max_iterations": loop_config.max_iterations,
        "pass_threshold": loop_config.pass_threshold,
        "num_configs": num_configs,
        "vlm_model": loop_config.vlm_model or None,
        "side_info": loop_config.train.side_info,
        "use_zoo_preset": loop_config.use_zoo_preset,
        "hp_tuning": loop_config.hp_tuning,
        "use_code_judge": loop_config.use_code_judge,
        "device": loop_config.train.device,
        "thinking_effort": loop_config.thinking_effort or None,
        "refined_initial_frame": loop_config.refined_initial_frame,
        "criteria_diagnosis": loop_config.criteria_diagnosis,
        "motion_trail_dual": loop_config.motion_trail_dual,
        "model": loop_config.model or None,
        "review_reward": loop_config.review_reward,
        "review_judge": loop_config.review_judge,
    }


# ---------------------------------------------------------------------------
# Subprocess spawner (shared by all controllers)
# ---------------------------------------------------------------------------


def _spawn_job_scheduler(job_id: str) -> subprocess.Popen:
    """Spawn the job scheduler as an independent subprocess."""
    jobs_dir = RUNS_DIR / "scheduler" / "jobs" / job_id
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_path = jobs_dir / "scheduler.log"
    log_file = open(log_path, "w", buffering=1)  # noqa: SIM115

    cmd = [
        sys.executable,
        "-m",
        "p2p.scheduler.job_scheduler",
        "--job-id",
        job_id,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survive server restarts
    )
    log_file.close()  # subprocess inherited the fd
    logger.info("Spawned job scheduler: %s (pid=%d)", job_id, proc.pid)
    return proc


# ---------------------------------------------------------------------------
# SessionController
# ---------------------------------------------------------------------------


class SessionController:
    """Run a training session (multi-config × multi-seed) via manifest + subprocess.

    Creates a single RunSpec with all configs and seeds embedded in LoopConfig.
    The core loop (``run_loop()``) handles the full matrix internally so that
    multi-config logic (``aggregate_judgments``, ``revise_multi``) operates on
    jointly-informed comparisons.

    After ``run()`` completes, ``scheduler_proc`` holds the Popen handle of
    the spawned job-scheduler subprocess (for process tracking / cleanup).
    """

    scheduler_proc: subprocess.Popen | None = None

    def run(
        self,
        *,
        prompt: str,
        loop_config: LoopConfig,
        backend: BackendType = "local",
        node_id: str | None = None,
        configs: list[dict] | None = None,
        seeds: list[int] | None = None,
        cores_per_run: int = 0,
        max_parallel: int = 0,
        session_id: str | None = None,
        allowed_nodes: list[str] | None = None,
        spawn: bool = True,
    ) -> Job:
        if loop_config is None:
            loop_config = LoopConfig()

        configs = configs or loop_config.configs or [DEFAULT_CONFIG]
        if not seeds:
            seeds = loop_config.seeds or [loop_config.train.seed]
        cores_per_run = cores_per_run or loop_config.cores_per_run
        max_parallel = max_parallel or loop_config.max_parallel

        now = datetime.now().astimezone()
        job_id = f"job_{now:%Y%m%d}_{now:%H%M%S}_{uuid.uuid4().hex[:8]}"
        session_group = f"sg_{uuid.uuid4().hex[:8]}"

        spec = session_to_spec(
            prompt=prompt,
            loop_config=loop_config,
            session_id=session_id,
            configs=configs,
            seeds=seeds,
        )
        if node_id:
            spec.setdefault("tags", {})["node_id"] = node_id

        run_records: list[RunRecord] = [
            RunRecord(
                run_id=spec["run_id"],
                spec=spec,
                state="pending",
                node_id="",
                remote_dir="",
                synced=False,
                session_group=session_group,
            )
        ]

        config: dict[str, Any] = {
            "configs": configs,
            "seeds": seeds,
            "cores_per_run": cores_per_run,
            "max_parallel": max_parallel,
            "env_id": loop_config.train.env_id,
            "allowed_nodes": allowed_nodes,
            **_loop_config_summary(loop_config, len(configs)),
        }

        manifest: JobManifest = {
            "job_id": job_id,
            "job_type": "session",
            "status": "running",
            "created_at": now_iso(),
            "backend": backend,
            "config": config,
            "runs": run_records,
            "metadata": {
                "total_runs": len(run_records),
            },
        }
        write_job_manifest(manifest)
        if spawn:
            self.scheduler_proc = _spawn_job_scheduler(job_id)

        return _manifest_to_job(manifest)


# ---------------------------------------------------------------------------
# BenchmarkController
# ---------------------------------------------------------------------------


class BenchmarkController:
    """Run benchmark test cases via manifest + subprocess.

    Each test case is a session (configs × seeds). All runs for the same
    test case share a ``session_group`` for node affinity.
    """

    def run(
        self,
        *,
        loop_config: LoopConfig,
        backend: BackendType = "ssh",
        test_cases: list[dict] | None = None,
        csv_file: str | None = None,
        mode: str = "flat",
        num_stages: int = 25,
        gate_threshold: float = 0.7,
        start_from_stage: int = 1,
        max_parallel: int = 30,
        configs: list[dict] | None = None,
        seeds: list[int] | None = None,
        stages: list | None = None,
        filter_envs: list[str] | None = None,
        filter_categories: list[str] | None = None,
        filter_difficulties: list[str] | None = None,
        allowed_nodes: list[str] | None = None,
        spawn: bool = True,
    ) -> Job:
        if test_cases is None:
            test_cases = self._load_test_cases(
                csv_file=csv_file,
                filter_envs=filter_envs,
                filter_categories=filter_categories,
                filter_difficulties=filter_difficulties,
            )

        if loop_config is None:
            loop_config = LoopConfig()

        now = datetime.now().astimezone()
        benchmark_id = f"bm_{now:%Y%m%d}_{now:%H%M%S}_{uuid.uuid4().hex[:8]}"
        job_id = f"job_{now:%Y%m%d}_{now:%H%M%S}_{uuid.uuid4().hex[:8]}"

        configs = configs or loop_config.configs or [DEFAULT_CONFIG]
        seeds = seeds or loop_config.seeds or [loop_config.train.seed]

        # Build stage definitions for staged mode
        stage_defs: list[dict] | None = None
        if mode == "staged":
            stage_defs = build_default_stages(
                test_cases,
                num_stages=num_stages,
                gate_threshold=gate_threshold,
                max_parallel=max_parallel,
            )
            # Apply user-provided overrides (thresholds, names, max_parallel)
            if stages:
                overrides = {s.stage: s for s in stages}
                for sd in stage_defs:
                    ov = overrides.get(sd["stage"])
                    if ov:
                        sd["gate_threshold"] = ov.gate_threshold
                        sd["max_parallel"] = ov.max_parallel
                        sd["name"] = ov.name

        # Build RunSpecs: 1 per test case, with all configs × seeds in LoopConfig
        run_records: list[RunRecord] = []
        for i, tc in enumerate(test_cases):
            case_group = f"case_{i}_{uuid.uuid4().hex[:6]}"
            spec = benchmark_case_to_spec(
                benchmark_id=benchmark_id,
                case_index=i,
                env_id=tc["env_id"],
                instruction=tc.get("instruction", ""),
                base_loop_config=loop_config,
                configs=configs,
                seeds=seeds,
            )
            # Tag with stage number if staged mode
            if stage_defs:
                for sd in stage_defs:
                    if i in sd["case_indices"]:
                        spec.setdefault("tags", {})["stage"] = str(sd["stage"])
                        break
            run_records.append(
                RunRecord(
                    run_id=spec["run_id"],
                    spec=spec,
                    state="pending",
                    node_id="",
                    remote_dir="",
                    synced=False,
                    session_group=case_group,
                )
            )

        total_stages = len(stage_defs) if stage_defs else 0

        config_dict: dict[str, Any] = {
            "mode": mode,
            "num_stages": num_stages,
            "gate_threshold": gate_threshold,
            "start_from_stage": start_from_stage,
            "max_parallel": max_parallel,
            "configs": configs,
            "seeds": seeds,
            "allowed_nodes": allowed_nodes,
            **_loop_config_summary(loop_config, len(configs)),
        }

        manifest: JobManifest = {
            "job_id": job_id,
            "job_type": "benchmark",
            "status": "running",
            "created_at": now_iso(),
            "backend": backend,
            "config": config_dict,
            "runs": run_records,
            "metadata": {
                "benchmark_id": benchmark_id,
                "mode": mode,
                "total_cases": len(test_cases),
                "total_runs": len(run_records),
                "total_stages": total_stages,
                "current_stage": 0,
                "gate_threshold": gate_threshold,
                "stages": stage_defs or [],
                "test_cases": [
                    {
                        "index": i,
                        "env_id": tc["env_id"],
                        "instruction": tc.get("instruction", ""),
                        "category": tc.get("category", ""),
                        "difficulty": tc.get("difficulty", ""),
                    }
                    for i, tc in enumerate(test_cases)
                ],
            },
        }
        write_job_manifest(manifest)

        # Write discovery pointer file so list/read endpoints can find this
        bm_dir = RUNS_DIR / benchmark_id
        bm_dir.mkdir(parents=True, exist_ok=True)
        pointer = {
            "type": "pointer",
            "job_id": job_id,
            "benchmark_id": benchmark_id,
            "created_at": manifest["created_at"],
            "status": "running",
        }
        (bm_dir / "benchmark.json").write_text(json.dumps(pointer, indent=2))

        if spawn:
            _spawn_job_scheduler(job_id)

        return _manifest_to_job(manifest)

    def _load_test_cases(
        self,
        *,
        csv_file: str | None = None,
        filter_envs: list[str] | None = None,
        filter_categories: list[str] | None = None,
        filter_difficulties: list[str] | None = None,
    ) -> list[dict]:
        """Load test cases from CSV."""
        filename = csv_file or "test_cases.csv"
        if "/" in filename or "\\" in filename or ".." in filename:
            msg = f"Invalid CSV filename: {filename}"
            raise ValueError(msg)
        csv_path = Path("benchmark") / filename
        if not csv_path.exists():
            msg = f"Benchmark test cases file not found: {csv_path.resolve()}"
            raise FileNotFoundError(msg)

        cases: list[dict] = []
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if filter_envs and row.get("env_id") not in filter_envs:
                    continue
                if filter_categories and row.get("category") not in filter_categories:
                    continue
                if filter_difficulties and row.get("difficulty") not in filter_difficulties:
                    continue

                cases.append(row)
        return cases
