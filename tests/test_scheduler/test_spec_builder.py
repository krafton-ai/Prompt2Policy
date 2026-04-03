"""Tests for spec_builder functions."""

from p2p.config import LoopConfig, TrainConfig
from p2p.scheduler.spec_builder import (
    benchmark_case_to_spec,
    session_to_spec,
)


class TestSessionToSpec:
    def test_basic(self) -> None:
        lc = LoopConfig(train=TrainConfig(seed=42))
        spec = session_to_spec(prompt="test prompt", loop_config=lc)
        assert spec["entry_point"] == "p2p.session.run_session"
        assert spec["parameters"]["prompt"] == "test prompt"
        assert "loop_config" in spec["parameters"]
        assert spec["cpu_cores"] >= 2
        assert spec["tags"]["job_type"] == "session"

    def test_custom_session_id(self) -> None:
        spec = session_to_spec(
            session_id="my_session",
            prompt="test",
            loop_config=LoopConfig(),
        )
        assert spec["run_id"] == "my_session"
        assert spec["parameters"]["session_id"] == "my_session"


class TestBenchmarkCaseToSpec:
    def test_basic(self) -> None:
        lc = LoopConfig(train=TrainConfig())
        spec = benchmark_case_to_spec(
            benchmark_id="bm_001",
            case_index=3,
            env_id="Ant-v5",
            instruction="Walk forward",
            base_loop_config=lc,
        )
        assert spec["entry_point"] == "p2p.session.run_session"
        assert "loop_config" in spec["parameters"]
        assert spec["tags"]["benchmark_id"] == "bm_001"
        assert spec["tags"]["case_index"] == "3"
        assert spec["tags"]["env_id"] == "Ant-v5"

    def test_run_id_format(self) -> None:
        lc = LoopConfig(train=TrainConfig())
        spec = benchmark_case_to_spec(
            benchmark_id="bm_001",
            case_index=0,
            env_id="HalfCheetah-v5",
            instruction="test",
            base_loop_config=lc,
            seeds=[5],
        )
        # Simplified run_id: {benchmark_id}_case{case_index}
        assert spec["run_id"] == "bm_001_case0"

    def test_per_case_env_id_in_config(self) -> None:
        """Each case should get its own env_id in the serialized LoopConfig."""
        import json

        lc = LoopConfig(train=TrainConfig(env_id="HalfCheetah-v5"))
        spec = benchmark_case_to_spec(
            benchmark_id="bm_001",
            case_index=0,
            env_id="Ant-v5",
            instruction="test",
            base_loop_config=lc,
        )
        config_json = json.loads(spec["parameters"]["loop_config"])
        assert config_json["train"]["env_id"] == "Ant-v5"
