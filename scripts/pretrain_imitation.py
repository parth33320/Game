import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.imitation import ACTION_NAMES, parse_walkthroughs
from agent.model import ActorCriticPPO
from env.retro_env import HeadlessRetroEnv


def train_imitation(paths, output_path, epochs, learning_rate):
    actions = parse_walkthroughs(paths)
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True)
    model = ActorCriticPPO(input_dim=15, num_actions=len(ACTION_NAMES), is_mlp=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    observations = []
    labels = []
    obs, _ = env.reset(seed=0)
    for action in actions:
        observations.append(obs.copy())
        labels.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=0)
    env.close()

    inputs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.long)
    model.train()
    for epoch in range(epochs):
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Imitation epoch {epoch + 1}/{epochs}: loss={loss.item():.5f}")

    torch.save({
        "episode": 0,
        "max_x_pos": 0.0,
        "source_walkthroughs": paths,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, output_path)
    print(f"Saved behavioral-cloning checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain the RAM policy from BK2 walkthroughs")
    parser.add_argument("paths", nargs="*", help="BK2 or FM2 walkthrough files")
    parser.add_argument("--glob", dest="pattern", default=None, help="Glob pattern for BK2/FM2 files")
    parser.add_argument("--output", default="checkpoints/imitation_baseline.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    paths = args.paths or (glob.glob(args.pattern) if args.pattern else [])
    if not paths:
        default_bk2 = "CastlevaniaTAS.bk2" if os.path.exists("CastlevaniaTAS.bk2") else "Castlevania TAS.bk2"
        if os.path.exists(default_bk2):
            paths = [default_bk2]
        else:
            parser.error(f"No BK2 paths provided and default file '{default_bk2}' not found")
    print(f"Pre-training behavioral cloning on full walkthrough trajectory from: {paths}")
    train_imitation(paths, args.output, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
