import torch
import torch.optim as optim
import os
import time
from agent.env import MockPlatformerEnv
from agent.model import ActorCriticPPO
from audit.audit_logger import AuditLogger

def train_ppo_agent(
    total_episodes: int = 10,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = MockPlatformerEnv()
    model = ActorCriticPPO(input_channels=1, num_actions=6)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    audit_logger = AuditLogger("training_audit.jsonl")

    print(f"Starting PyTorch PPO Platformer Agent Training for {total_episodes} episodes...")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Format observation tensor [B, C, H, W]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            action, log_prob, value = model.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            obs = next_obs

        # Log episode metrics
        log_data = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "final_x_pos": info["x_pos"],
            "score": info["score"]
        }
        audit_logger.log_event("ppo_training_episode", log_data)
        print(f"Episode {episode}/{total_episodes} - Reward: {total_reward:.2f} - Steps: {steps} - Distance: {info['x_pos']:.1f}")

        # Save Checkpoint
        if episode % checkpoint_interval == 0 or episode == total_episodes:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_agent_ep{episode}.pt")
            torch.save({
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training Completed Successfully!")

if __name__ == "__main__":
    train_ppo_agent(total_episodes=5, checkpoint_interval=5)
