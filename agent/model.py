import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

class ActorCriticPPO(nn.Module):
    """
    PyTorch PPO (Proximal Policy Optimization) Actor-Critic architecture.
    Processes state input tensors (4-frame stacked observations: [B, 4, 84, 84])
    and outputs policy action logits for 8 masked discrete actions & value estimates.
    Includes transfer learning weight loading adaptation to seamlessly resume from previous checkpoints.
    """
    def __init__(self, input_channels: int = 4, num_actions: int = 8):
        super(ActorCriticPPO, self).__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # Feature extractor network
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Linear projection assuming 84x84 input frame
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

        # Policy head (Actor)
        self.actor = nn.Linear(512, num_actions)

        # Value head (Critic)
        self.critic = nn.Linear(512, 1)

    def forward(self, state: torch.Tensor):
        # Expect state shape [B, C, H, W]
        if state.dim() == 3:
            state = state.unsqueeze(0)

        features = self.conv(state)
        hidden = self.fc(features)

        logits = self.actor(hidden)
        value = self.critic(hidden)

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
                    # Transfer learning adaptation for Conv1 (channel mismatch: e.g. 1 -> 4)
                    if k == "conv.0.weight" and target_shape[1] != source_shape[1]:
                        new_weight = torch.zeros(target_shape, dtype=v.dtype)
                        c_copy = min(target_shape[1], source_shape[1])
                        new_weight[:, :c_copy, :, :] = v[:, :c_copy, :, :]
                        # Repeat weights across stacked channels if target > source
                        for c in range(c_copy, target_shape[1]):
                            new_weight[:, c, :, :] = v[:, 0, :, :]
                        adapted_dict[k] = new_weight
                    # Transfer learning adaptation for Actor output logits (action mismatch: e.g. 6 -> 8)
                    elif k == "actor.weight" and target_shape[0] != source_shape[0]:
                        new_weight = torch.zeros(target_shape, dtype=v.dtype)
                        n_copy = min(target_shape[0], source_shape[0])
                        new_weight[:n_copy, :] = v[:n_copy, :]
                        adapted_dict[k] = new_weight
                    elif k == "actor.bias" and target_shape[0] != source_shape[0]:
                        new_bias = torch.zeros(target_shape, dtype=v.dtype)
                        n_copy = min(target_shape[0], source_shape[0])
                        new_bias[:n_copy] = v[:n_copy]
                        adapted_dict[k] = new_bias
                    else:
                        print(f"Skipping key {k} due to unhandled shape difference: {source_shape} vs {target_shape}")

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
