import pytest
import numpy as np
import torch
import os

from agent.rewards import PlatformerRewardCalculator
from agent.env import MockPlatformerEnv
from agent.model import ActorCriticPPO
from env.rewards import RetroRewardEngine
from env.retro_env import HeadlessRetroEnv
from agent.train import StableBaselines3PPOTrainer

def test_platformer_reward_calculator():
    calc = PlatformerRewardCalculator(distance_weight=1.0, score_weight=0.1)
    calc.reset({"x_pos": 0.0, "score": 0, "lives": 3, "health": 16})

    # Forward movement + score
    reward1 = calc.calculate_reward({"x_pos": 10.0, "score": 100, "lives": 3, "health": 16})
    assert reward1 > 0  # 10*1.0 + 100*0.1 - 0.01

    # Life loss penalty
    reward_death = calc.calculate_reward({"x_pos": 10.0, "score": 100, "lives": 2, "health": 16})
    assert reward_death < -90.0

def test_mock_platformer_env_gymnasium_specs():
    env = MockPlatformerEnv(observation_shape=(84, 84, 1))
    obs, info = env.reset()

    assert obs.shape == (84, 84, 1)
    assert info["lives"] == 3

    next_obs, reward, terminated, truncated, next_info = env.step(1)  # Move RIGHT
    assert next_obs.shape == (84, 84, 1)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["x_pos"] == 2.0

def test_pytorch_ppo_actor_critic_shapes():
    model = ActorCriticPPO(input_channels=1, num_actions=6)
    fake_obs = torch.zeros((1, 1, 84, 84), dtype=torch.float32)

    logits, value = model(fake_obs)
    assert logits.shape == (1, 6)
    assert value.shape == (1, 1)

    action, log_prob, val_est = model.get_action(fake_obs)
    assert 0 <= action < 6
    assert isinstance(log_prob, torch.Tensor)

def test_retro_reward_engine():
    engine = RetroRewardEngine()
    engine.reset({"x_pos": 0.0, "hearts": 0, "score": 0, "health": 16, "lives": 3})

    r1 = engine.calculate_reward({"x_pos": 5.0, "hearts": 2, "score": 50, "health": 16, "lives": 3})
    assert r1 > 0.0

    # Damage penalty
    r_dmg = engine.calculate_reward({"x_pos": 5.0, "hearts": 2, "score": 50, "health": 12, "lives": 3})
    assert r_dmg < 0.0

def test_headless_retro_env():
    env = HeadlessRetroEnv(obs_shape=(84, 84, 3))
    obs, info = env.reset()

    assert obs.shape == (84, 84, 3)
    assert info["health"] == 16

    obs2, reward, terminated, truncated, info2 = env.step(4)  # WHIP
    assert obs2.shape == (84, 84, 3)
    assert info2["score"] == 20

def test_stable_baselines3_ppo_trainer(tmp_path):
    ckpt_dir = os.path.join(tmp_path, "ckpts")
    log_dir = os.path.join(tmp_path, "logs")

    trainer = StableBaselines3PPOTrainer(
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir
    )

    trainer.train(total_timesteps=100, checkpoint_freq=50)

    assert os.path.exists(ckpt_dir)
    assert len(os.listdir(ckpt_dir)) > 0
