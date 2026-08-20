import sys
import os
import glob
import time
import json
import numpy as np
import torch
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.env import MockPlatformerEnv
from agent.model import ActorCriticPPO
from audit.audit_logger import AuditLogger

ACTIVE_PARAMS_FILE = "config/active_training_params.json"
RESUME_TARGET_FILE = "checkpoints/resume_target.pt"
TRAINING_AUDIT_FILE = "training_audit.json"


def load_active_params(params_file: str = ACTIVE_PARAMS_FILE) -> dict:
    default_params = {
        "initial_lr": 1.5e-4,
        "ent_coef": 0.05,
        "gamma": 0.99,
        "time_penalty": -0.02,
        "score_weight": 0.01,
        "progress_multiplier": 1.0,
        "penalize_zero_velocity": False,
        "zero_velocity_penalty": -0.05,
        "lr_warmup_steps": 0,
        "reinit_policy_entropy": False
    }
    if os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                saved_params = json.load(f)
                default_params.update(saved_params)
                print(f"Loaded active training parameters from {params_file}: {default_params}")
        except Exception as e:
            print(f"Warning: Failed to load {params_file} ({e}). Using defaults.")
    return default_params


def find_target_checkpoint(checkpoint_dir: str, default_target: str = "checkpoints/ppo_agent_ep5000.pt") -> str:
    if os.path.exists(RESUME_TARGET_FILE):
        return RESUME_TARGET_FILE

    if os.path.exists(default_target):
        return default_target

    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime, reverse=True)
            return ckpts[0]

    return default_target


def trigger_early_stopping(
    failure_reason: str,
    episode: int,
    total_reward: float,
    max_x_pos: float,
    entropy_val: float,
    active_params: dict,
    audit_logger: AuditLogger,
    exit_code: int = 42
):
    print(f"\n=======================================================")
    print(f"EARLY STOPPING TRIGGERED: Failure Reason = '{failure_reason}'")
    print(f"Episode: {episode} | Reward: {total_reward:.2f} | Max X: {max_x_pos:.1f} | Entropy: {entropy_val:.4f}")
    print(f"=======================================================\n")

    audit_payload = {
        "status": "EARLY_STOPPING",
        "failure_reason": failure_reason,
        "episode": episode,
        "total_reward": total_reward,
        "max_x_pos": max_x_pos,
        "entropy": entropy_val,
        "active_params": active_params,
        "timestamp": time.time()
    }

    # Save current run metrics to training_audit.json
    with open(TRAINING_AUDIT_FILE, "w") as f:
        json.dump(audit_payload, f, indent=2)

    # Log to append-only JSONL audit trail
    audit_logger.log_event("early_stopping_triggered", audit_payload)

    sys.exit(exit_code)


def train_ppo_agent(
    total_episodes: int = 10000,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5,
    target_checkpoint: str = "checkpoints/ppo_agent_ep5000.pt",
    params_file: str = ACTIVE_PARAMS_FILE
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    active_params = load_active_params(params_file)

    initial_lr = float(active_params.get("initial_lr", active_params.get("lr", 1.5e-4)))
    ent_coef = float(active_params.get("ent_coef", 0.05))
    gamma = float(active_params.get("gamma", 0.99))

    rc_params = {
        "time_penalty": float(active_params.get("time_penalty", -0.02)),
        "score_weight": float(active_params.get("score_weight", 0.01)),
        "progress_multiplier": float(active_params.get("progress_multiplier", 1.0)),
        "penalize_zero_velocity": bool(active_params.get("penalize_zero_velocity", False)),
        "zero_velocity_penalty": float(active_params.get("zero_velocity_penalty", -0.05))
    }

    env = MockPlatformerEnv(
        frame_shape=(84, 84),
        num_stack=4,
        base_max_steps=400,
        reward_calculator_params=rc_params
    )

    model = ActorCriticPPO(input_channels=4, num_actions=8)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    audit_logger = AuditLogger("training_audit.jsonl")

    # Load target checkpoint if available
    ckpt_to_load = find_target_checkpoint(checkpoint_dir, target_checkpoint)
    start_episode = 1
    best_max_x_pos = 0.0

    if os.path.exists(ckpt_to_load):
        print(f"Loading pre-trained checkpoint from {ckpt_to_load}...")
        loaded = model.load_checkpoint_weights(ckpt_to_load, optimizer=optimizer)
        if loaded and ckpt_to_load.endswith(".pt"):
            try:
                raw_ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
                if isinstance(raw_ckpt, dict):
                    if "episode" in raw_ckpt:
                        start_episode = int(raw_ckpt["episode"]) + 1
                    if "max_x_pos" in raw_ckpt:
                        best_max_x_pos = float(raw_ckpt["max_x_pos"])
            except Exception as e:
                print(f"Warning reading checkpoint metadata: {e}")

    # Re-initialize policy network actor layer AFTER checkpoint loading if requested by auto-tuner
    if active_params.get("reinit_policy_entropy", False):
        print("Auto-tuner signal detected: Re-initializing PPO policy actor layer weights...")
        torch.nn.init.orthogonal_(model.actor.weight, gain=0.01)
        if model.actor.bias is not None:
            torch.nn.init.constant_(model.actor.bias, 0.0)

    print(f"Starting PyTorch PPO Platformer Agent Training from episode {start_episode} for {total_episodes} episodes (Best Max X: {best_max_x_pos:.1f})...")

    # Rolling window histories for Early Stopping detection
    rolling_max_x_history = []
    rolling_reward_history = []

    for episode in range(start_episode, start_episode + total_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_step_entropy = 0.05

        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, value, entropy = model.get_action(obs_tensor, ent_coef=ent_coef)
            last_step_entropy = float(entropy.item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            total_reward += reward
            steps += 1
            obs = next_obs

        # Calculate discounted returns and advantages for rollout optimization
        returns = []
        discounted_sum = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted_sum = 0.0
            discounted_sum = r + (gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.cat(values).squeeze(-1)
        advantages = returns_t - values_t.detach()

        # Advantage Normalization
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = 0.5 * (returns_t - values_t).pow(2).mean()
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        curr_max_x = float(info.get("max_x_pos", info.get("x_pos", 0.0)))
        reward_hacking = bool(info.get("reward_hacking_detected", False))
        repetitive_ratio = float(info.get("repetitive_action_ratio", 0.0))

        rolling_max_x_history.append(curr_max_x)
        rolling_reward_history.append(total_reward)

        log_data = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "final_x_pos": info["x_pos"],
            "max_x_pos": curr_max_x,
            "score": info["score"],
            "reward_hacking_detected": reward_hacking,
            "repetitive_action_ratio": repetitive_ratio,
            "entropy": last_step_entropy
        }
        audit_logger.log_event("ppo_training_episode", log_data)

        # -------------------------------------------------------------
        # EARLY STOPPING CALLBACK & ANOMALY EVALUATOR
        # -------------------------------------------------------------

        # 1. Distance Plateau Detection:
        # Rolling window of max_x_pos over last 300 episodes.
        # If max_x_pos fails to increase by >= 20 pixels across 300 episodes, flag STAGNATION_PLATEAU.
        if len(rolling_max_x_history) >= 300:
            window_300 = rolling_max_x_history[-300:]
            max_x_gain_300 = max(window_300) - window_300[0]
            if max_x_gain_300 < 20.0:
                trigger_early_stopping(
                    failure_reason="STAGNATION_PLATEAU",
                    episode=episode,
                    total_reward=total_reward,
                    max_x_pos=curr_max_x,
                    entropy_val=last_step_entropy,
                    active_params=active_params,
                    audit_logger=audit_logger
                )

        # 2. Reward Hacking Detection:
        # Flag REWARD_HACKING if cumulative reward increases by >25% while max_x_pos remains flat,
        # OR if repetitive action ratios exceed 40% of episode steps while stationary (max_x_pos < 10px).
        if reward_hacking or (repetitive_ratio > 0.40 and curr_max_x < 10.0):
            trigger_early_stopping(
                failure_reason="REWARD_HACKING",
                episode=episode,
                total_reward=total_reward,
                max_x_pos=curr_max_x,
                entropy_val=last_step_entropy,
                active_params=active_params,
                audit_logger=audit_logger
            )

        if len(rolling_reward_history) >= 50:
            past_50_reward = np.mean(rolling_reward_history[-50:-25])
            recent_25_reward = np.mean(rolling_reward_history[-25:])
            past_50_max_x = np.max(rolling_max_x_history[-50:-25])
            recent_25_max_x = np.max(rolling_max_x_history[-25:])

            if past_50_reward > 0 and (recent_25_reward - past_50_reward) / abs(past_50_reward) > 0.25:
                if (recent_25_max_x - past_50_max_x) < 5.0:
                    trigger_early_stopping(
                        failure_reason="REWARD_HACKING",
                        episode=episode,
                        total_reward=total_reward,
                        max_x_pos=curr_max_x,
                        entropy_val=last_step_entropy,
                        active_params=active_params,
                        audit_logger=audit_logger
                    )

        # 3. Entropy Drop Detection:
        # Flag COLLAPSED_EXPLORATION if PPO entropy drops below 0.005 while distance gained is < 250px.
        if last_step_entropy < 0.005 and curr_max_x < 250.0:
            trigger_early_stopping(
                failure_reason="COLLAPSED_EXPLORATION",
                episode=episode,
                total_reward=total_reward,
                max_x_pos=curr_max_x,
                entropy_val=last_step_entropy,
                active_params=active_params,
                audit_logger=audit_logger
            )

        print(f"Episode {episode}/{start_episode + total_episodes - 1} - Reward: {total_reward:.2f} - Steps: {steps} - Max Distance: {curr_max_x:.1f}")

        # Distance-Based Best Model Checkpointing
        if curr_max_x > best_max_x_pos:
            best_max_x_pos = curr_max_x
            best_dist_path = os.path.join(checkpoint_dir, "best_ppo_agent_dist.pt")
            best_dist_numbered = os.path.join(checkpoint_dir, f"best_ppo_agent_dist_{int(curr_max_x)}.pt")
            save_payload = {
                "episode": episode,
                "max_x_pos": curr_max_x,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(save_payload, best_dist_path)
            torch.save(save_payload, best_dist_numbered)
            print(f"New Distance Record! Saved Best Model checkpoint to {best_dist_path} (Max X: {curr_max_x:.1f})")

        # Regular periodic checkpointing
        if episode % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_agent_ep{episode}.pt")
            torch.save({
                "episode": episode,
                "max_x_pos": curr_max_x,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved periodic checkpoint to {ckpt_path}")

    print("PyTorch PPO Platformer Agent Training Completed Successfully!")

if __name__ == "__main__":
    train_ppo_agent(total_episodes=5, checkpoint_interval=5)
