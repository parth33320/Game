import os
import pytest
import torch
import numpy as np
from agent.pretrain_imitation import parse_p1_buttons_to_action, extract_tas_dataset, train_imitation_baseline
from agent.model import ActorCriticPPO
from emulator.castlevania_autonomous_engine import CastlevaniaAutonomousEngine
from env.retro_env import HeadlessRetroEnv


def test_parse_p1_buttons_to_action():
    assert parse_p1_buttons_to_action("........") == 0  # NOOP
    assert parse_p1_buttons_to_action("...R....") == 1  # RIGHT
    assert parse_p1_buttons_to_action("..L.....") == 2  # LEFT
    assert parse_p1_buttons_to_action(".D......") == 3  # DOWN
    assert parse_p1_buttons_to_action(".......A") == 4  # JUMP
    assert parse_p1_buttons_to_action("......B.") == 5  # WHIP
    assert parse_p1_buttons_to_action("...R...A") == 6  # RIGHT+JUMP
    assert parse_p1_buttons_to_action("...R..B.") == 7  # RIGHT+WHIP
    assert parse_p1_buttons_to_action("U.......") == 8  # UP


def test_extract_tas_dataset():
    if not os.path.exists("CastlevaniaTAS.bk2"):
        pytest.skip("CastlevaniaTAS.bk2 archive not found.")

    X, y, total_frames = extract_tas_dataset("CastlevaniaTAS.bk2")
    assert total_frames == 40573
    assert X.shape == (40573, 15)
    assert y.shape == (40573,)
    assert torch.is_tensor(X)
    assert torch.is_tensor(y)


def test_train_imitation_baseline(tmp_path):
    if not os.path.exists("CastlevaniaTAS.bk2"):
        pytest.skip("CastlevaniaTAS.bk2 archive not found.")

    ckpt_path = str(tmp_path / "imitation_baseline.pt")
    res = train_imitation_baseline(
        bk2_path="CastlevaniaTAS.bk2",
        output_checkpoint=ckpt_path,
        epochs=1,
        batch_size=512,
        lr=1e-3
    )
    assert os.path.exists(ckpt_path)
    assert os.path.getsize(ckpt_path) > 0

    # Verify model weight loading into ActorCriticPPO
    model = ActorCriticPPO(input_dim=15, num_actions=9, is_mlp=True)
    loaded = model.load_checkpoint_weights(ckpt_path)
    assert loaded == True


def test_autonomous_engine_auto_loads_baseline(tmp_path):
    ckpt_path = str(tmp_path / "imitation_baseline.pt")
    model = ActorCriticPPO(input_dim=15, num_actions=9, is_mlp=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    engine = CastlevaniaAutonomousEngine(
        env_creation_func=lambda: HeadlessRetroEnv(obs_type="ram", use_retro=False),
        save_dir=str(tmp_path),
        baseline_checkpoint=ckpt_path
    )

    test_model = ActorCriticPPO(input_dim=15, num_actions=9, is_mlp=True)
    assert engine.baseline_loaded == False

    engine.process_autonomous_step(
        rl_agent=test_model,
        rl_agent_policy_func=lambda obs_tensor: 1
    )
    assert engine.baseline_loaded == True
