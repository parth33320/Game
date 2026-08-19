import sys
import os
import glob
import time
import numpy as np
import torch
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.env import MockPlatformerEnv
from agent.model import ActorCriticPPO
from audit.audit_logger import AuditLogger

def find_target_checkpoint(checkpoint_dir: str, default_target: str = "checkpoints/ppo_agent_ep5000.pt") -> str:
    if os.path.exists(default_target):
        return default_target

    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime, reverse=True)
            return ckpts[0]

    return default_target

def train_ppo_agent(
    total_episodes: int = 10,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5,
    target_checkpoint: str = "checkpoints/ppo_agent_ep5000.pt",
    initial_lr: float = 1.5e-4,
    ent_coef: float = 0.05,
    gamma: float = 0.99
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = MockPlatformerEnv(frame_shape=(84, 84), num_stack=4, base_max_steps=400)

    model = ActorCriticPPO(input_channels=4, num_actions=8)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    audit_logger = AuditLogger("training_audit.jsonl")

    # Load target checkpoint if available
    ckpt_to_load = find_target_checkpoint(checkpoint_dir, target_checkpoint)
    start_episode = 1
    if os.path.exists(ckpt_to_load):
        print(f"Loading pre-trained checkpoint from {ckpt_to_load}...")
        loaded = model.load_checkpoint_weights(ckpt_to_load, optimizer=optimizer)
        if loaded and ckpt_to_load.endswith(".pt"):
            try:
                raw_ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
                if isinstance(raw_ckpt, dict) and "episode" in raw_ckpt:
                    start_episode = int(raw_ckpt["episode"]) + 1
            except Exception:
                pass

    best_max_x_pos = 0.0
    print(f"Starting PyTorch PPO Platformer Agent Training from episode {start_episode} for {total_episodes} episodes...")

    for episode in range(start_episode, start_episode + total_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        while not done:
            # Observation tensor shape: [1, 4, 84, 84]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, value, entropy = model.get_action(obs_tensor, ent_coef=ent_coef)

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

        # Strict Advantage Normalization
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

        # Audit logging & Anti-Reward Hacking alert
        curr_max_x = info.get("max_x_pos", info.get("x_pos", 0.0))
        reward_hacking = bool(info.get("reward_hacking_detected", False))

        log_data = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "final_x_pos": info["x_pos"],
            "max_x_pos": curr_max_x,
            "score": info["score"],
            "reward_hacking_detected": reward_hacking
        }
        audit_logger.log_event("ppo_training_episode", log_data)

        if reward_hacking:
            audit_logger.log_event("reward_hacking_detected", {
                "episode": episode,
                "total_reward": total_reward,
                "max_x_pos": curr_max_x,
                "warning": "Agent accumulating rewards without horizontal progression or repeating stagnant action loops."
            })
            print(f"WARNING: Episode {episode} flagged for REWARD HACKING! (Reward: {total_reward:.2f}, Max X: {curr_max_x:.1f})")

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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved periodic checkpoint to {ckpt_path}")

    print("PyTorch PPO Platformer Agent Training Completed Successfully!")

if __name__ == "__main__":
    train_ppo_agent(total_episodes=5, checkpoint_interval=5)
