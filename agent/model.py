import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticPPO(nn.Module):
    """
    PyTorch PPO (Proximal Policy Optimization) Actor-Critic architecture.
    Processes state input tensors (vector/pixel) and outputs policy action logits & value estimates.
    """
    def __init__(self, input_channels: int = 1, num_actions: int = 6):
        super(ActorCriticPPO, self).__init__()

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

    def get_action(self, state: torch.Tensor):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
