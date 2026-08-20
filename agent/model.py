import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

class ActorCriticPPO(nn.Module):
    """
    PyTorch PPO (Proximal Policy Optimization) Actor-Critic architecture.
    Supports both:
    1. 1D CPU MLP vector input (e.g. ~15 normalized RAM state features) -> 2-layer hidden MLP ([128, 128] units).
    2. 2D Stacked Frame Conv input (4-frame stacked observations: [B, 4, 84, 84]).
    Includes transfer learning weight loading adaptation to seamlessly resume from previous checkpoints.
    """
    def __init__(self, input_dim: int = 15, num_actions: int = 9, is_mlp: bool = True):
        super(ActorCriticPPO, self).__init__()

        self.is_mlp = is_mlp
        self.num_actions = num_actions

        if self.is_mlp:
            # High-speed CPU 2-layer Multi-Layer Perceptron (MLP)
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            feature_dim = 128
        else:
            # Feature extractor network for 2D stacked frame pixels
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU()
            )
            feature_dim = 512

        # Policy head (Actor)
        self.actor = nn.Linear(feature_dim, num_actions)

        # Value head (Critic)
        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, state: torch.Tensor):
        if not self.is_mlp:
            if state.dim() == 3:
                state = state.unsqueeze(0)
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)

        features = self.feature_extractor(state)
        logits = self.actor(features)
        value = self.critic(features)

        return logits, value

    def get_action(self, state: torch.Tensor, ent_coef: float = 0.05):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value, entropy

    def load_checkpoint_weights(self, checkpoint_path: str, optimizer: torch.optim.Optimizer = None) -> bool:
        """
        Loads state dict from checkpoint with automatic shape adaptation for transfer learning.
        Returns True if checkpoint successfully loaded.
        """
        if not os.path.exists(checkpoint_path):
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Warning: Unable to load binary PyTorch checkpoint from {checkpoint_path} ({e}). Initializing clean model weights.")
            return False

        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            return False

        model_dict = self.state_dict()
        adapted_dict = {}

        for k, v in state_dict.items():
            if k in model_dict:
                target_shape = model_dict[k].shape
                source_shape = v.shape

                if target_shape == source_shape:
                    adapted_dict[k] = v
                else:
                    # Transfer learning adaptation
                    if k.endswith("weight") and len(target_shape) == 2 and len(source_shape) == 2:
                        new_weight = torch.zeros(target_shape, dtype=v.dtype)
                        r_copy = min(target_shape[0], source_shape[0])
                        c_copy = min(target_shape[1], source_shape[1])
                        new_weight[:r_copy, :c_copy] = v[:r_copy, :c_copy]
                        adapted_dict[k] = new_weight
                    elif k.endswith("bias") and len(target_shape) == 1 and len(source_shape) == 1:
                        new_bias = torch.zeros(target_shape, dtype=v.dtype)
                        n_copy = min(target_shape[0], source_shape[0])
                        new_bias[:n_copy] = v[:n_copy]
                        adapted_dict[k] = new_bias

        model_dict.update(adapted_dict)
        self.load_state_dict(model_dict)
        print(f"Successfully loaded and adapted checkpoint weights from {checkpoint_path}")

        if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Successfully loaded optimizer state dict.")
            except Exception as opt_e:
                print(f"Warning: Could not load optimizer state dict ({opt_e}). Continuing with fresh optimizer state.")

        return True
