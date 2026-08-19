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
from audit.audit_logger import AuditLogger

def test_platformer_reward_calculator():
    calc = PlatformerRewardCalculator(distance_weight=1.0, score_weight=0.1)
    calc.reset({"x_pos": 0.0, "score": 0, "lives": 3, "health": 16})

    # Forward movement + score
    reward1 = calc.calculate_reward({"x_pos": 10.0, "score": 100, "lives": 3, "health": 16})
    assert reward1 > 0  # 10*1.0 + 100*0.1 - 0.02

    # Life loss penalty (balanced around -25.0)
    reward_death = calc.calculate_reward({"x_pos": 10.0, "score": 100, "lives": 2, "health": 16})
    assert -30.0 < reward_death < -20.0

def test_mock_platformer_env_gymnasium_specs():
    env = MockPlatformerEnv(frame_shape=(84, 84), num_stack=4)
    obs, info = env.reset()

    # 4-frame stacked observation shape (4, 84, 84)
    assert obs.shape == (4, 84, 84)
    assert info["lives"] == 3
    assert info["max_steps"] == 400

    next_obs, reward, terminated, truncated, next_info = env.step(1)  # Move RIGHT
    assert next_obs.shape == (4, 84, 84)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["x_pos"] == 2.0

def test_dynamic_timeout_and_reward_hacking_detection():
    env = MockPlatformerEnv(frame_shape=(84, 84), num_stack=4, base_max_steps=400)
    obs, info = env.reset()

    # Advance x_pos beyond 100px to trigger dynamic timeout extension (+50 steps)
    for _ in range(55):
        obs, reward, terminated, truncated, info = env.step(6) # RIGHT+JUMP

    assert info["max_x_pos"] >= 100.0
    assert info["max_steps"] == 450  # 400 base + 50 extension

def test_pytorch_ppo_actor_critic_shapes():
    model = ActorCriticPPO(input_channels=4, num_actions=8)
    fake_obs = torch.zeros((1, 4, 84, 84), dtype=torch.float32)

    logits, value = model(fake_obs)
    assert logits.shape == (1, 8)
    assert value.shape == (1, 1)

    action, log_prob, val_est, entropy = model.get_action(fake_obs)
    assert 0 <= action < 8
    assert isinstance(log_prob, torch.Tensor)
    assert isinstance(entropy, torch.Tensor)

def test_retro_reward_engine():
    engine = RetroRewardEngine()
    engine.reset({"x_pos": 0.0, "hearts": 0, "score": 0, "health": 16, "lives": 3})

    r1 = engine.calculate_reward({"x_pos": 5.0, "hearts": 2, "score": 50, "health": 16, "lives": 3})
    assert r1 > 0.0

    # Damage penalty
    r_dmg = engine.calculate_reward({"x_pos": 5.0, "hearts": 2, "score": 50, "health": 12, "lives": 3})
    assert r_dmg < 0.0

def test_headless_retro_env():
    env = HeadlessRetroEnv(frame_shape=(84, 84), num_stack=4)
    obs, info = env.reset()

    assert obs.shape == (4, 84, 84)
    assert info["health"] == 16

    obs2, reward, terminated, truncated, info2 = env.step(5)  # WHIP
    assert obs2.shape == (4, 84, 84)
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

def test_anti_reward_hacking_detection(tmp_path):
    audit_file = str(tmp_path / "test_audit.jsonl")
    logger = AuditLogger(log_filepath=audit_file)

    env = MockPlatformerEnv(frame_shape=(84, 84), num_stack=4)
    obs, info = env.reset()

    # Perform 30 WHIP actions without moving
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(5)  # Action 5 = WHIP

    assert info["reward_hacking_detected"] is True

    # Log to audit logger
    logger.log_event("reward_hacking_detected", {
        "episode": 1,
        "total_reward": 100.0,
        "max_x_pos": info["max_x_pos"],
        "warning": "Agent accumulating rewards without horizontal progression."
    })

    events = logger.read_all_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "reward_hacking_detected"
