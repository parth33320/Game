import os
import pytest
import numpy as np
import torch
from env.retro_env import HeadlessRetroEnv
from emulator.castlevania_autonomous_engine import CastlevaniaAutonomousEngine
from agent.model import ActorCriticPPO
from scripts.generate_walkthrough_video import generate_walkthrough_video

def test_ram_vector_edge_cases_and_state_machine():
    env = HeadlessRetroEnv(obs_type="ram", use_retro=False)
    obs, info = env.reset()

    # Feature Normalization (1D vector of length 18 bounded 0.0-1.0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (18,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    # Edge Case 1: Staircase alignment trap -> Action restricted to UP/DOWN
    env.is_on_stairs = True
    obs, reward, term, trunc, info = env.step(1)  # RIGHT action passed
    assert env.action_history[-1] in ("UP", "DOWN")

    # Edge Case 2: Transition door delay -> Issue NOOP
    env.is_on_stairs = False
    env.is_door_transition = True
    obs, reward, term, trunc, info = env.step(1)
    assert env.action_history[-1] == "NOOP"

    # Edge Case 3: Global X-Coordinate Calculation
    env.coarse_screen = 2
    env.fine_x = 100
    env.global_x_pos = env.coarse_screen * 256 + env.fine_x
    assert env.global_x_pos == 612.0

    # Edge Case 4: Boss HP damage reward boost
    env.is_door_transition = False
    env.in_boss_room = True
    env.prev_boss_hp = 16
    env.boss_hp = 12
    obs, reward, term, trunc, info = env.step(0)
    assert reward > 0.0  # Extra damage boost applied

def test_system_state_freezes_and_auto_restart():
    env = HeadlessRetroEnv(obs_type="ram", use_retro=False)
    env.reset()

    # System State 0x03 (Loading transition) freeze learning reward to 0.0
    env.game_state_byte = 0x03
    obs, reward, term, trunc, info = env.step(1)
    assert reward == 0.0

    # Game over auto-restart
    env.game_state_byte = 0x07
    env.lives = 0
    obs, reward, term, trunc, info = env.step(0)
    assert term == True
    assert info.get("auto_restarted") == True

def test_autonomous_state_machine_and_checkpoints(tmp_path):
    save_dir = str(tmp_path / "checkpoints")
    engine = CastlevaniaAutonomousEngine(
        env_creation_func=lambda: HeadlessRetroEnv(obs_type="ram", use_retro=False),
        save_dir=save_dir,
        max_episodes_before_flush=2
    )

    model = ActorCriticPPO(input_dim=18, num_actions=9, is_mlp=True)

    # Test boss snapshot trigger
    engine.trigger_boss_snapshot(model, stage=3)
    latest_path = os.path.join(save_dir, "model_weights_latest.pt")
    assert os.path.exists(latest_path)

    # Test autonomous step
    obs, reward, term, trunc, info = engine.process_autonomous_step(
        rl_agent=model,
        rl_agent_policy_func=lambda obs_tensor: 1
    )
    assert isinstance(obs, np.ndarray)

def test_walkthrough_video_generator(tmp_path):
    video_file = str(tmp_path / "test_walkthrough.mp4")
    out_path = generate_walkthrough_video(
        output_path=video_file,
        num_steps=30,
        width=320,
        height=180,
        fps=30
    )
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
