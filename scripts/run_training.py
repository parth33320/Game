import sys
import os
from agent.train import StableBaselines3PPOTrainer

def main():
    print("Kicking off headless Castlevania / NES PPO training pipeline in Codespace environment...")
    trainer = StableBaselines3PPOTrainer(
        checkpoint_dir="checkpoints_sb3",
        log_dir="logs_sb3"
    )
    trainer.train(total_timesteps=2000, checkpoint_freq=1000)

if __name__ == "__main__":
    main()
