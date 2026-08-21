import os
import sys
import json
import time
import pytest
import shutil
import torch

from audit.audit_logger import AuditLogger
from agent.rewards import PlatformerRewardCalculator
from agent.env import MockPlatformerEnv
from scripts.auto_tuner import (
    load_training_audit,
    load_active_params,
    adjust_hyperparameters,
    save_active_params,
    resolve_best_distance_checkpoint,
    run_auto_tuner
)
from scripts.train_agent import trigger_early_stopping, load_active_params as load_train_active_params
from scripts.watchdog import ProcessWatchdog


def test_platformer_reward_calculator_auto_tuning_params():
    calc = PlatformerRewardCalculator(
        distance_weight=1.0,
        time_penalty=-0.05,
        progress_multiplier=1.5,
        penalize_zero_velocity=True,
        zero_velocity_penalty=-0.05,
        score_weight=0.0
    )
    calc.reset({"x_pos": 0.0, "score": 0, "lives": 3, "health": 16})

    # Standing still with zero velocity penalty
    reward_still = calc.calculate_reward({"x_pos": 0.0, "score": 100, "lives": 3, "health": 16})
    # Expect -0.05 (time penalty) + -0.05 (zero velocity penalty) = -0.10, score ignored
    assert pytest.approx(reward_still, abs=1e-4) == -0.10

    # Progress reward with progress_multiplier=1.5
    reward_move = calc.calculate_reward({"x_pos": 10.0, "score": 100, "lives": 3, "health": 16})
    assert reward_move > 0.0


def test_mock_env_repetitive_action_ratio():
    env = MockPlatformerEnv(frame_shape=(84, 84), num_stack=4)
    obs, info = env.reset()

    # Perform 30 WHIP actions
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(5)  # Action 5 = WHIP

    assert info["repetitive_action_ratio"] == 1.0
    assert info["reward_hacking_detected"] is True


def test_trigger_early_stopping_and_audit(tmp_path, monkeypatch):
    audit_file = str(tmp_path / "test_audit.jsonl")
    audit_json = str(tmp_path / "training_audit.json")
    logger = AuditLogger(log_filepath=audit_file)

    monkeypatch.setattr("scripts.train_agent.TRAINING_AUDIT_FILE", audit_json)

    active_params = {"initial_lr": 0.00015, "ent_coef": 0.05}

    with pytest.raises(SystemExit) as exc_info:
        trigger_early_stopping(
            failure_reason="STAGNATION_PLATEAU",
            episode=305,
            total_reward=150.0,
            max_x_pos=80.0,
            entropy_val=0.04,
            active_params=active_params,
            audit_logger=logger,
            exit_code=42
        )

    assert exc_info.value.code == 42
    assert os.path.exists(audit_json)

    with open(audit_json, "r") as f:
        data = json.load(f)
        assert data["failure_reason"] == "STAGNATION_PLATEAU"
        assert data["episode"] == 305
        assert data["max_x_pos"] == 80.0


def test_auto_tuner_hyperparameter_adaptations():
    # Test STAGNATION_PLATEAU
    base_params = {
        "initial_lr": 0.0002,
        "ent_coef": 0.01,
        "progress_multiplier": 1.0
    }
    stagnation_adapted = adjust_hyperparameters("STAGNATION_PLATEAU", base_params)
    assert stagnation_adapted["ent_coef"] == 0.015
    assert stagnation_adapted["initial_lr"] == 0.00015
    assert stagnation_adapted["progress_multiplier"] == 1.5

    # Test REWARD_HACKING
    reward_hacking_adapted = adjust_hyperparameters("REWARD_HACKING", base_params)
    assert reward_hacking_adapted["time_penalty"] == -0.05
    assert reward_hacking_adapted["penalize_zero_velocity"] is True
    assert reward_hacking_adapted["zero_velocity_penalty"] == -0.05
    assert reward_hacking_adapted["score_weight"] == 0.0

    # Test COLLAPSED_EXPLORATION
    collapsed_adapted = adjust_hyperparameters("COLLAPSED_EXPLORATION", base_params)
    assert collapsed_adapted["reinit_policy_entropy"] is True
    assert collapsed_adapted["ent_coef"] == 0.05
    assert collapsed_adapted["lr_warmup_steps"] == 100


def test_resolve_best_distance_checkpoint(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_target = str(tmp_path / "resume_target.pt")

    # Save 3 checkpoints with different max_x_pos
    ckpt_1 = os.path.join(ckpt_dir, "best_ppo_agent_dist_120.pt")
    ckpt_2 = os.path.join(ckpt_dir, "best_ppo_agent_dist_480.pt")
    ckpt_3 = os.path.join(ckpt_dir, "best_ppo_agent_dist_300.pt")

    torch.save({"episode": 10, "max_x_pos": 120.0}, ckpt_1)
    torch.save({"episode": 25, "max_x_pos": 480.0}, ckpt_2)
    torch.save({"episode": 18, "max_x_pos": 300.0}, ckpt_3)

    selected = resolve_best_distance_checkpoint(checkpoint_dir=ckpt_dir, resume_target=resume_target)
    assert selected == resume_target
    assert os.path.exists(resume_target)

    # Verify resume_target content matches highest max_x_pos (480.0)
    target_data = torch.load(resume_target, map_location="cpu", weights_only=False)
    assert target_data["max_x_pos"] == 480.0


def test_watchdog_closed_loop_orchestration(tmp_path):
    log_file = str(tmp_path / "watchdog_audit.jsonl")
    audit = AuditLogger(log_filepath=log_file)

    # Script exiting with 42
    exit_42_cmd = [sys.executable, "-c", "import sys; sys.exit(42)"]

    watchdog = ProcessWatchdog(
        managed_services={"test_train_agent": exit_42_cmd},
        check_interval=0.05,
        audit_logger=audit,
        max_retrain_attempts=2
    )

    watchdog.start_all()
    time.sleep(0.2)

    # Cycle 1 -> detects 42 -> triggers auto_tuner -> relaunch -> retrain count 1
    status1 = watchdog.check_health_and_recover()
    assert status1["test_train_agent"]["status"] == "retraining_relaunched"
    assert status1["test_train_agent"]["retrains"] == 1

    time.sleep(0.2)

    # Cycle 2 -> detects 42 -> triggers auto_tuner -> relaunch -> retrain count 2
    status2 = watchdog.check_health_and_recover()
    assert status2["test_train_agent"]["status"] == "retraining_relaunched"
    assert status2["test_train_agent"]["retrains"] == 2

    time.sleep(0.2)

    # Cycle 3 -> exceeds cap (2) -> retrain_cap_reached
    status3 = watchdog.check_health_and_recover()
    assert status3["test_train_agent"]["status"] == "retrain_cap_reached"

    watchdog.stop_all()

    events = audit.read_all_events()
    event_types = [e.get("event_type") for e in events]
    assert "watchdog_retrain_adaptation" in event_types
    assert "watchdog_retrain_cap_reached" in event_types
