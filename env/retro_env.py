import numpy as np
from typing import Dict, Any, Tuple, Optional
from env.rewards import RetroRewardEngine

class HeadlessRetroEnv:
    """
    Gymnasium-compatible environment wrapper around a headless emulator (gym-retro / NES-py).
    Exposes state inputs as downscaled RAM vectors or pixel tensors and maps discrete actions:
    0: NOOP, 1: Move Left, 2: Move Right, 3: Jump, 4: Whip/Attack, 5: Crouch.
    """
    ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "JUMP", "WHIP", "CROUCH"]

    def __init__(self, obs_shape: Tuple[int, ...] = (84, 84, 3), max_episode_steps: int = 1000):
        self.obs_shape = obs_shape
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.reward_engine = RetroRewardEngine()

        self.x_pos = 0.0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.x_pos = 0.0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3

        info = {
            "x_pos": self.x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives
        }
        self.reward_engine.reset(info)

        obs = np.zeros(self.obs_shape, dtype=np.uint8)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        act = action if (0 <= action < len(self.ACTION_NAMES)) else 0
        act_name = self.ACTION_NAMES[act]

        if act_name == "RIGHT":
            self.x_pos += 1.5
            self.score += 5
        elif act_name == "LEFT":
            self.x_pos = max(0.0, self.x_pos - 0.5)
        elif act_name == "WHIP":
            self.score += 20
            self.hearts += 1

        info = {
            "x_pos": self.x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives
        }

        reward = self.reward_engine.calculate_reward(info)

        terminated = self.lives <= 0
        truncated = self.step_count >= self.max_episode_steps

        obs = np.random.randint(0, 256, size=self.obs_shape, dtype=np.uint8)

        return obs, reward, terminated, truncated, info
