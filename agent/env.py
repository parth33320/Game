import numpy as np
from typing import Dict, Any, Tuple, Optional
from agent.rewards import PlatformerRewardCalculator

class MockPlatformerEnv:
    """
    Gymnasium-compatible headless platformer training environment.
    Exposes state inputs as RAM feature vectors or downscaled pixel tensors,
    mapping discrete button actions: 0=NOOP, 1=Move Right, 2=Move Left, 3=Jump, 4=Attack/Whip, 5=Crouch.
    """
    ACTION_MAP = {
        0: "NOOP",
        1: "RIGHT",
        2: "LEFT",
        3: "JUMP",
        4: "ATTACK",
        5: "CROUCH"
    }

    def __init__(self, observation_shape: Tuple[int, ...] = (84, 84, 1), max_steps: int = 500):
        self.observation_shape = observation_shape
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_calculator = PlatformerRewardCalculator()

        self.x_pos = 0.0
        self.score = 0
        self.lives = 3
        self.health = 16

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_step = 0
        self.x_pos = 0.0
        self.score = 0
        self.lives = 3
        self.health = 16

        initial_info = {
            "x_pos": self.x_pos,
            "score": self.score,
            "lives": self.lives,
            "health": self.health,
            "fell_in_pit": False
        }
        self.reward_calculator.reset(initial_info)

        obs = np.zeros(self.observation_shape, dtype=np.float32)
        return obs, initial_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_name = self.ACTION_MAP.get(action, "NOOP")

        # Simulate physics & movement
        if action_name == "RIGHT":
            self.x_pos += 2.0
            self.score += 10
        elif action_name == "LEFT":
            self.x_pos = max(0.0, self.x_pos - 1.0)
        elif action_name == "JUMP":
            self.x_pos += 1.0

        # Simulate rare damage or pit fall condition
        fell_in_pit = False
        if self.current_step == 450:
            fell_in_pit = True
            self.lives -= 1

        state_info = {
            "x_pos": self.x_pos,
            "score": self.score,
            "lives": self.lives,
            "health": self.health,
            "fell_in_pit": fell_in_pit
        }

        reward = self.reward_calculator.calculate_reward(state_info)

        terminated = self.lives <= 0 or fell_in_pit
        truncated = self.current_step >= self.max_steps

        # Generated synthetic observation tensor
        obs = np.random.uniform(0.0, 1.0, size=self.observation_shape).astype(np.float32)

        return obs, reward, terminated, truncated, state_info
