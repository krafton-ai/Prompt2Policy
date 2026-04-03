import gymnasium as gym
import numpy as np

from p2p.training.env import CustomRewardWrapper


def _simple_reward_fn(obs, action, next_obs, info):
    energy = float(np.sum(np.square(action)))
    terms = {"energy_penalty": -0.1 * energy, "alive_bonus": 1.0}
    return sum(terms.values()), terms


def test_custom_reward_wrapper_replaces_reward():
    env = gym.make("HalfCheetah-v5")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    wrapped = CustomRewardWrapper(env, _simple_reward_fn)

    wrapped.reset(seed=0)
    action = wrapped.action_space.sample()
    _obs, reward, _term, _trunc, info = wrapped.step(action)

    # Reward should come from our function, not the environment
    expected_energy = float(np.sum(np.square(action)))
    expected_reward = -0.1 * expected_energy + 1.0
    assert abs(reward - expected_reward) < 1e-5

    wrapped.close()


def test_custom_reward_wrapper_adds_reward_terms_to_info():
    env = gym.make("HalfCheetah-v5")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    wrapped = CustomRewardWrapper(env, _simple_reward_fn)

    wrapped.reset(seed=0)
    action = wrapped.action_space.sample()
    _obs, _reward, _term, _trunc, info = wrapped.step(action)

    assert "reward_terms" in info
    assert "energy_penalty" in info["reward_terms"]
    assert "alive_bonus" in info["reward_terms"]
    assert info["reward_terms"]["alive_bonus"] == 1.0

    wrapped.close()


def test_custom_reward_wrapper_preserves_done_signals():
    env = gym.make("HalfCheetah-v5")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    wrapped = CustomRewardWrapper(env, _simple_reward_fn)

    obs, _ = wrapped.reset(seed=0)
    for _ in range(5):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(reward, float)

    wrapped.close()
