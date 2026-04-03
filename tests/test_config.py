from pathlib import Path

from p2p.config import (
    TARGET_EPISODE_DURATION_S,
    LoopConfig,
    TrainConfig,
    loop_config_from_params,
)
from p2p.training.env_spec import max_steps_for_duration
from p2p.training.hp_presets import ZOO_PRESETS, available_envs, get_preset


def test_default_config_derived_fields():
    cfg = TrainConfig()
    assert cfg.batch_size == cfg.num_envs * cfg.num_steps
    assert cfg.minibatch_size == cfg.batch_size // cfg.num_minibatches
    assert cfg.num_iterations == cfg.total_timesteps // cfg.batch_size


def test_json_roundtrip():
    cfg = TrainConfig(seed=42, total_timesteps=50_000, learning_rate=1e-3)
    json_str = cfg.to_json()
    restored = TrainConfig.from_json(json_str)

    assert restored.seed == 42
    assert restored.total_timesteps == 50_000
    assert restored.learning_rate == 1e-3
    # Derived fields recomputed
    assert restored.batch_size == restored.num_envs * restored.num_steps
    assert restored.num_iterations == restored.total_timesteps // restored.batch_size


def test_custom_hyperparams():
    cfg = TrainConfig(
        num_envs=4,
        num_steps=512,
        num_minibatches=8,
        total_timesteps=100_000,
    )
    assert cfg.batch_size == 4 * 512
    assert cfg.minibatch_size == (4 * 512) // 8
    assert cfg.num_iterations == 100_000 // (4 * 512)


# ---------------------------------------------------------------------------
# Zoo HP presets
# ---------------------------------------------------------------------------


def test_zoo_preset_halfcheetah():
    preset = get_preset("HalfCheetah-v5")
    assert preset is not None
    assert preset["learning_rate"] == 2.0633e-05
    assert preset["num_steps"] == 512
    assert preset["gamma"] == 0.98
    assert preset["update_epochs"] == 20


def test_zoo_preset_v5_alias():
    """v5 aliases should reference the same dict as v4."""
    assert get_preset("HalfCheetah-v5") is get_preset("HalfCheetah-v4")
    assert get_preset("Hopper-v5") is get_preset("Hopper-v4")
    assert get_preset("Walker2d-v5") is get_preset("Walker2d-v4")


def test_zoo_preset_unknown_env():
    assert get_preset("NonExistentEnv-v99") is None


def test_available_envs_has_all_mujoco():
    envs = available_envs()
    for name in [
        "HalfCheetah-v4",
        "HalfCheetah-v5",
        "Hopper-v4",
        "Walker2d-v4",
        "Ant-v4",
        "Humanoid-v4",
        "Swimmer-v4",
    ]:
        assert name in envs, f"{name} missing from available_envs()"


def test_from_preset_applies_zoo_hps():
    cfg = TrainConfig.from_preset(env_id="HalfCheetah-v5", total_timesteps=500_000, seed=42)
    # Zoo values applied
    assert cfg.learning_rate == 2.0633e-05
    assert cfg.num_steps == 512
    assert cfg.gamma == 0.98
    # User overrides applied on top
    assert cfg.total_timesteps == 500_000
    assert cfg.seed == 42
    # Derived fields recomputed
    assert cfg.batch_size == cfg.num_envs * cfg.num_steps


def test_from_preset_fallback_for_unknown_env():
    cfg = TrainConfig.from_preset(env_id="UnknownEnv-v99", total_timesteps=100_000)
    # Should use dataclass defaults (no Zoo preset)
    assert cfg.learning_rate == 3e-4  # default
    assert cfg.total_timesteps == 100_000
    assert cfg.env_id == "UnknownEnv-v99"


def test_zoo_presets_all_have_required_keys():
    """Every preset must include core HP keys."""
    required = {"learning_rate", "num_steps", "gamma", "gae_lambda"}
    for env_id, preset in ZOO_PRESETS.items():
        for key in required:
            assert key in preset, f"{env_id} missing key: {key}"


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------


def test_loop_config_json_roundtrip():
    """LoopConfig survives to_json → from_json with all fields intact."""
    train = TrainConfig(total_timesteps=200_000, seed=7, learning_rate=1e-3)
    lc = LoopConfig(
        train=train,
        configs=[{"config_id": "a", "label": "a", "params": {"learning_rate": 0.01}}],
        seeds=[1, 2, 3],
        max_iterations=10,
        pass_threshold=0.9,
        runs_dir=Path("/tmp/test-runs"),
        cores_pool=[0, 1, 2],
        use_zoo_preset=False,
    )
    restored = LoopConfig.from_json(lc.to_json())

    assert restored.train.total_timesteps == 200_000
    assert restored.train.seed == 7
    assert restored.train.learning_rate == 1e-3
    # Derived fields recomputed
    assert restored.train.batch_size == restored.train.num_envs * restored.train.num_steps
    expected_cfg = {"config_id": "a", "label": "a", "params": {"learning_rate": 0.01}}
    assert restored.configs == [expected_cfg]
    assert restored.seeds == [1, 2, 3]
    assert restored.max_iterations == 10
    assert restored.runs_dir == Path("/tmp/test-runs")
    assert restored.cores_pool == [0, 1, 2]
    assert restored.use_zoo_preset is False


def test_loop_config_defaults_roundtrip():
    """Default LoopConfig survives roundtrip (None fields, default Path)."""
    lc = LoopConfig()
    restored = LoopConfig.from_json(lc.to_json())

    assert restored.configs is None
    assert restored.seeds is None
    assert restored.cores_pool is None
    assert restored.runs_dir == Path("runs")
    assert restored.use_zoo_preset is True


# ---------------------------------------------------------------------------
# loop_config_from_params factory (#303)
# ---------------------------------------------------------------------------


def test_factory_defaults():
    """Default construction produces valid LoopConfig."""
    lc = loop_config_from_params()
    assert lc.train.total_timesteps == 1_000_000
    assert lc.train.seed == 1
    assert lc.train.env_id == "HalfCheetah-v5"
    assert lc.max_iterations == 5
    assert lc.hp_tuning is True
    assert lc.use_zoo_preset is True


def test_factory_checkpoint_interval_auto():
    """checkpoint_interval=None auto-computes max(100_000, timesteps // 5)."""
    lc = loop_config_from_params(total_timesteps=1_000_000)
    assert lc.train.checkpoint_interval == 200_000

    lc2 = loop_config_from_params(total_timesteps=100_000)
    assert lc2.train.checkpoint_interval == 100_000  # max(100_000, 20_000)


def test_factory_checkpoint_interval_explicit():
    """Explicit checkpoint_interval is respected."""
    lc = loop_config_from_params(total_timesteps=1_000_000, checkpoint_interval=50_000)
    assert lc.train.checkpoint_interval == 50_000


def test_factory_vlm_model_none_uses_default():
    """vlm_model=None falls back to VLM_MODEL."""
    from p2p.settings import VLM_MODEL

    lc = loop_config_from_params()
    assert lc.vlm_model == VLM_MODEL


def test_factory_vlm_model_explicit():
    """Explicit vlm_model is respected, including empty string."""
    lc = loop_config_from_params(vlm_model="my-model")
    assert lc.vlm_model == "my-model"

    lc2 = loop_config_from_params(vlm_model="")
    assert lc2.vlm_model == ""


def test_factory_model_empty_uses_default():
    """model='' falls back to LLM_MODEL."""
    from p2p.settings import LLM_MODEL

    lc = loop_config_from_params()
    assert lc.model == LLM_MODEL


def test_factory_model_explicit():
    """Explicit model is respected."""
    lc = loop_config_from_params(model="custom-model")
    assert lc.model == "custom-model"


def test_factory_zoo_preset():
    """use_zoo_preset=True triggers TrainConfig.from_preset()."""
    lc = loop_config_from_params(env_id="HalfCheetah-v5", use_zoo_preset=True)
    assert lc.train.learning_rate == 2.0633e-05


def test_factory_no_zoo_preset():
    """use_zoo_preset=False uses direct TrainConfig()."""
    lc = loop_config_from_params(env_id="HalfCheetah-v5", use_zoo_preset=False)
    assert lc.train.learning_rate == 3e-4


def test_factory_max_episode_steps():
    """max_episode_steps is passed through when provided."""
    lc = loop_config_from_params(max_episode_steps=500)
    assert lc.train.max_episode_steps == 500

    # Default for HalfCheetah-v5 (dt=0.05): 5s / 0.05 = 100 steps
    lc2 = loop_config_from_params()
    assert lc2.train.max_episode_steps == int(TARGET_EPISODE_DURATION_S / 0.05)


# ---------------------------------------------------------------------------
# dt-based max_episode_steps (#348)
# ---------------------------------------------------------------------------


def test_max_steps_for_duration_known_env():
    """Known MuJoCo envs return int(duration / dt)."""
    assert max_steps_for_duration("HalfCheetah-v5", 5.0) == 100  # 5/0.05
    assert max_steps_for_duration("Hopper-v5", 5.0) == 625  # 5/0.008
    assert max_steps_for_duration("Humanoid-v5", 5.0) == 333  # 5/0.015


def test_max_steps_for_duration_unknown_env():
    """Unknown envs return None."""
    assert max_steps_for_duration("NonExistent-v99", 5.0) is None


def test_from_preset_dt_based_default():
    """from_preset() computes max_episode_steps from dt for known envs."""
    cfg = TrainConfig.from_preset(env_id="Hopper-v5")
    assert cfg.max_episode_steps == int(TARGET_EPISODE_DURATION_S / 0.008)

    cfg2 = TrainConfig.from_preset(env_id="Humanoid-v5")
    assert cfg2.max_episode_steps == int(TARGET_EPISODE_DURATION_S / 0.015)


def test_from_preset_explicit_override():
    """Explicit max_episode_steps overrides dt-based default."""
    cfg = TrainConfig.from_preset(env_id="HalfCheetah-v5", max_episode_steps=999)
    assert cfg.max_episode_steps == 999


def test_from_preset_unknown_env_keeps_default():
    """Unknown envs (dt=0) fall back to TrainConfig default (300)."""
    cfg = TrainConfig.from_preset(env_id="UnknownEnv-v99")
    assert cfg.max_episode_steps == 300


# ---------------------------------------------------------------------------
# Sqrt-scaling num_minibatches cap
# ---------------------------------------------------------------------------


def test_sqrt_scaling_cap_ant_32_envs():
    """Ant (zoo num_minibatches=16) at num_envs=32 must be capped at 32."""
    cfg = TrainConfig.from_preset(env_id="Ant-v5", num_envs=32)
    assert cfg.num_minibatches <= 32


def test_sqrt_scaling_no_cap_ant_4_envs():
    """Ant at num_envs=4 stays below cap — no clamping needed."""
    cfg = TrainConfig.from_preset(env_id="Ant-v5", num_envs=4)
    assert cfg.num_minibatches <= 32
    # Should still scale up from zoo base (16)
    assert cfg.num_minibatches > 16


def test_sqrt_scaling_base_case_unchanged():
    """num_envs=1 (zoo base) preserves the preset value exactly."""
    cfg = TrainConfig.from_preset(env_id="Ant-v5", num_envs=1)
    preset = get_preset("Ant-v5")
    assert preset is not None
    assert cfg.num_minibatches == preset["num_minibatches"]


def test_factory_extra_kwargs_ignored():
    """Extra kwargs are safely ignored via **_extra."""
    lc = loop_config_from_params(
        prompt="test",
        backend="ssh",
        unknown_key=42,
    )
    assert lc.train.total_timesteps == 1_000_000
