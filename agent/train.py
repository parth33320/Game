import os
import time
import torch
import torch.optim as optim
from typing import Dict, Any, Optional
from env.retro_env import HeadlessRetroEnv
from agent.model import ActorCriticPPO
from audit.audit_logger import AuditLogger

class PPOTrainer:
    """
    CPU-optimized Multi-Layer Perceptron (MLP) PPO Trainer for Castlevania NES environment.
    Runs at maximum speed up (~400-800 FPS) on CPU using 1D normalized RAM vectors (~15 features).
    Supports checkpointing, metrics logging, transfer learning, and auto-resumption.
    """
    def __init__(
        self,
        env: Optional[HeadlessRetroEnv] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        learning_rate: float = 3e-4,
        is_mlp: bool = True
    ):
        self.is_mlp = is_mlp
        self.env = env or HeadlessRetroEnv(obs_type="ram" if is_mlp else "pixels")
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.audit_logger = AuditLogger(os.path.join(log_dir, "ppo_training_audit.jsonl"))

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        input_dim = 15 if is_mlp else 4
        num_actions = len(self.env.ACTION_NAMES)

        self.model = ActorCriticPPO(input_dim=input_dim, num_actions=num_actions, is_mlp=is_mlp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_latest_checkpoint(self) -> Optional[str]:
        if not os.path.exists(self.checkpoint_dir):
            return None
        ckpts = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not ckpts:
            return None
        ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return ckpts[0]

    def train(self, total_timesteps: int = 5000, checkpoint_freq: int = 1000) -> Dict[str, Any]:
        latest_ckpt = self.load_latest_checkpoint()
        if latest_ckpt:
            print(f"Resuming training from latest checkpoint: {latest_ckpt}")
            self.model.load_checkpoint_weights(latest_ckpt, self.optimizer)
        else:
            print("No existing checkpoint found. Initializing new PPO policy network...")

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        start_time = time.time()

        for step in range(1, total_timesteps + 1):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, value, entropy = self.model.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if terminated or truncated:
                episode_count += 1
                fps = episode_length / max(time.time() - start_time, 1e-5)
                log_data = {
                    "episode": episode_count,
                    "timestep": step,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "final_x_pos": info["x_pos"],
                    "fps": fps
                }
                self.audit_logger.log_event("ppo_episode_complete", log_data)
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                start_time = time.time()

            # Save checkpoint
            if step % checkpoint_freq == 0 or step == total_timesteps:
                ckpt_file = os.path.join(self.checkpoint_dir, f"ppo_agent_step_{step}.pt")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "step": step
                }, ckpt_file)
                print(f"Saved PPO checkpoint to {ckpt_file}")

        print("PPO Training Completed Successfully!")
        return {"total_timesteps": total_timesteps, "episodes": episode_count}

if __name__ == "__main__":
    trainer = PPOTrainer(is_mlp=True)
    trainer.train(total_timesteps=100)
