import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
from env.rewards import RetroRewardEngine

class HeadlessRetroEnv:
    """
    Gymnasium-compatible environment wrapper around a headless emulator (gym-retro / NES-py).
    Restricts action space strictly to 8 essential Castlevania actions:
    0: NOOP, 1: RIGHT, 2: LEFT, 3: DOWN, 4: JUMP, 5: WHIP, 6: RIGHT+JUMP, 7: RIGHT+WHIP.
    Applies 4-frame stacking giving observation shape (4, 84, 84) and dynamic timeouts (+50 steps per 100px).
    Includes anti-reward hacking detection for static-position reward exploitation.
    """
    ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "DOWN", "JUMP", "WHIP", "RIGHT+JUMP", "RIGHT+WHIP"]

    def __init__(self, frame_shape: Tuple[int, int] = (84, 84), num_stack: int = 4, base_max_steps: int = 400):
        self.frame_shape = frame_shape
        self.num_stack = num_stack
        self.base_max_steps = base_max_steps
        self.max_episode_steps = base_max_steps
        self.step_count = 0
        self.reward_engine = RetroRewardEngine()

        self.frame_buffer = deque(maxlen=num_stack)
        self.action_history = deque(maxlen=30)

        self.x_pos = 0.0
        self.max_x_pos = 0.0
        self.last_milestone = 0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

    def _get_stacked_obs(self) -> np.ndarray:
        return np.stack(list(self.frame_buffer), axis=0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.max_episode_steps = self.base_max_steps
        self.x_pos = 0.0
        self.max_x_pos = 0.0
        self.last_milestone = 0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.action_history.clear()
        self.frame_buffer.clear()

        # Initialize frame buffer with 4 stacked frames
        initial_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        for _ in range(self.num_stack):
            self.frame_buffer.append(initial_frame)

        info = {
            "x_pos": self.x_pos,
            "max_x_pos": self.max_x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "reward_hacking_detected": False,
            "max_steps": self.max_episode_steps
        }
        self.reward_engine.reset(info)

        return self._get_stacked_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        act = action if (0 <= action < len(self.ACTION_NAMES)) else 0
        act_name = self.ACTION_NAMES[act]
        self.action_history.append(act_name)

        if act_name == "RIGHT":
            self.x_pos += 1.5
            self.score += 5
        elif act_name == "LEFT":
            self.x_pos = max(0.0, self.x_pos - 0.5)
        elif act_name == "WHIP":
            self.score += 20
            self.hearts += 1
        elif act_name == "RIGHT+JUMP":
            self.x_pos += 2.5
            self.score += 5
        elif act_name == "RIGHT+WHIP":
            self.x_pos += 2.0
            self.score += 15

        if self.x_pos > self.max_x_pos:
            self.max_x_pos = self.x_pos

        # Dynamic timeout extensions (+50 steps per 100 pixels)
        current_milestone = int(self.max_x_pos // 100)
        if current_milestone > self.last_milestone:
            milestone_diff = current_milestone - self.last_milestone
            self.max_episode_steps += milestone_diff * 50
            self.last_milestone = current_milestone

        info = {
            "x_pos": self.x_pos,
            "max_x_pos": self.max_x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "max_steps": self.max_episode_steps
        }

        reward = self.reward_engine.calculate_reward(info)
        self.accumulated_reward += reward

        # Anti-Reward Hacking & Stagnation Check
        repetitive_loop = (len(self.action_history) == 30 and
                           all(a == "WHIP" for a in self.action_history) and
                           self.x_pos < 10.0)
        static_reward_hack = (self.accumulated_reward > 50.0 and self.max_x_pos < 10.0)

        if repetitive_loop or static_reward_hack:
            self.reward_hacking_detected = True

        info["reward_hacking_detected"] = self.reward_hacking_detected

        terminated = self.lives <= 0
        truncated = self.step_count >= self.max_episode_steps

        # Generate synthetic frame and append to frame buffer
        new_frame = np.random.randint(0, 256, size=self.frame_shape, dtype=np.uint8)
        self.frame_buffer.append(new_frame)

        return self._get_stacked_obs(), reward, terminated, truncated, info
