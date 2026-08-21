import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
from agent.rewards import PlatformerRewardCalculator

class MockPlatformerEnv:
    """
    Gymnasium-compatible headless platformer training environment.
    Restricts action space strictly to 8 essential Castlevania actions:
    0=NOOP, 1=RIGHT, 2=LEFT, 3=DOWN, 4=JUMP, 5=WHIP, 6=RIGHT+JUMP, 7=RIGHT+WHIP.
    Employs 4-frame stacking producing observations of shape (4, 84, 84) and dynamic timeouts.
    Includes anti-reward hacking detection for static-position reward accumulation.
    """
    ACTION_MAP = {
        0: "NOOP",
        1: "RIGHT",
        2: "LEFT",
        3: "DOWN",
        4: "JUMP",
        5: "WHIP",
        6: "RIGHT+JUMP",
        7: "RIGHT+WHIP"
    }

    def __init__(
        self,
        frame_shape: Tuple[int, int] = (84, 84),
        num_stack: int = 4,
        base_max_steps: int = 400,
        reward_calculator_params: Optional[Dict[str, Any]] = None
    ):
        self.frame_shape = frame_shape
        self.num_stack = num_stack
        self.base_max_steps = base_max_steps
        self.max_steps = base_max_steps
        self.current_step = 0

        rc_params = reward_calculator_params or {}
        self.reward_calculator = PlatformerRewardCalculator(**rc_params)

        self.frame_buffer = deque(maxlen=num_stack)
        self.action_history = deque(maxlen=30)
        self.episode_action_history = []

        self.x_pos = 0.0
        self.max_x_pos = 0.0
        self.last_milestone = 0
        self.score = 0
        self.lives = 3
        self.health = 16
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

    def _get_single_frame(self) -> np.ndarray:
        # Fetch active frame buffer slice from the core
        return self.env.unwrapped.get_screen() if hasattr(self.env.unwrapped, "get_screen") else np.zeros((84,84,3), dtype=np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        return np.stack(list(self.frame_buffer), axis=0)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_step = 0
        self.max_steps = self.base_max_steps
        self.x_pos = 0.0
        self.max_x_pos = 0.0
        self.last_milestone = 0
        self.score = 0
        self.lives = 3
        self.health = 16
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.action_history.clear()
        self.episode_action_history.clear()
        self.frame_buffer.clear()

        # Initialize 4 stacked frames
        initial_frame = np.zeros(self.frame_shape, dtype=np.float32)
        for _ in range(self.num_stack):
            self.frame_buffer.append(initial_frame)

        initial_info = {
            "x_pos": self.x_pos,
            "max_x_pos": self.max_x_pos,
            "score": self.score,
            "lives": self.lives,
            "health": self.health,
            "fell_in_pit": False,
            "reward_hacking_detected": False,
            "max_steps": self.max_steps,
            "repetitive_action_ratio": 0.0
        }
        self.reward_calculator.reset(initial_info)

        return self._get_stacked_obs(), initial_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_name = self.ACTION_MAP.get(action, "NOOP")
        self.action_history.append(action_name)
        self.episode_action_history.append(action_name)

        # Simulate physics & movement
        if action_name == "RIGHT":
            self.x_pos += 2.0
            self.score += 5
        elif action_name == "LEFT":
            self.x_pos = max(0.0, self.x_pos - 1.0)
        elif action_name == "JUMP":
            self.x_pos += 0.5
        elif action_name == "WHIP":
            self.score += 10
        elif action_name == "RIGHT+JUMP":
            self.x_pos += 3.0
            self.score += 5
        elif action_name == "RIGHT+WHIP":
            self.x_pos += 2.5
            self.score += 15

        if self.x_pos > self.max_x_pos:
            self.max_x_pos = self.x_pos

        # Dynamic timeout time extensions (+50 steps every 100 pixels)
        current_milestone = int(self.max_x_pos // 100)
        if current_milestone > self.last_milestone:
            milestone_diff = current_milestone - self.last_milestone
            self.max_steps += milestone_diff * 50
            self.last_milestone = current_milestone

        # Simulate rare damage or pit fall condition
        fell_in_pit = False
        if self.current_step == 450:
            fell_in_pit = True
            self.lives -= 1

        state_info = {
            "x_pos": self.x_pos,
            "max_x_pos": self.max_x_pos,
            "score": self.score,
            "lives": self.lives,
            "health": self.health,
            "fell_in_pit": fell_in_pit,
            "max_steps": self.max_steps
        }

        reward = self.reward_calculator.calculate_reward(state_info)
        self.accumulated_reward += reward

        # Calculate repetitive action ratio (stationary/attack actions e.g. WHIP, NOOP, DOWN without moving)
        non_moving_actions = sum(1 for a in self.episode_action_history if a in ("WHIP", "NOOP", "DOWN"))
        repetitive_ratio = non_moving_actions / len(self.episode_action_history) if self.episode_action_history else 0.0

        # Anti-Reward Hacking & Stagnation Detection
        repetitive_loop = (len(self.action_history) == 30 and
                           all(a == "WHIP" for a in self.action_history) and
                           self.x_pos < 10.0)
        static_reward_hack = (self.accumulated_reward > 50.0 and self.max_x_pos < 10.0)
        high_repetitive_ratio = (len(self.episode_action_history) >= 20 and repetitive_ratio > 0.40 and self.max_x_pos < 10.0)

        if repetitive_loop or static_reward_hack or high_repetitive_ratio:
            self.reward_hacking_detected = True

        state_info["reward_hacking_detected"] = self.reward_hacking_detected
        state_info["repetitive_action_ratio"] = repetitive_ratio

        terminated = self.lives <= 0 or fell_in_pit
        truncated = self.current_step >= self.max_steps

        # Generate synthetic frame and append to frame buffer
        new_frame = np.random.uniform(0.0, 1.0, size=self.frame_shape).astype(np.float32)
        self.frame_buffer.append(new_frame)

        return self._get_stacked_obs(), reward, terminated, truncated, state_info
