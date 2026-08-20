import os
import shutil
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
from env.rewards import RetroRewardEngine

try:
    import stable_retro
    HAS_STABLE_RETRO = True
except ImportError:
    HAS_STABLE_RETRO = False


def _ensure_rom_imported():
    if not HAS_STABLE_RETRO:
        return
    rom_source = "roms/Castlevania (USA) (Rev 1).nes"
    if os.path.exists(rom_source):
        try:
            sha_path = stable_retro.data.get_file_path("Castlevania-Nes-v0", "rom.sha")
            target_dir = os.path.dirname(sha_path)
            target_rom = os.path.join(target_dir, "rom.nes")
            if not os.path.exists(target_rom):
                shutil.copy(rom_source, target_rom)
        except Exception:
            pass


_ensure_rom_imported()


def _process_frame(frame: np.ndarray, shape: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """Converts an RGB image frame to grayscale uint8 image of size (84, 84)."""
    if frame is None:
        return np.zeros(shape, dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0].astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)

    try:
        from PIL import Image
        img = Image.fromarray(gray)
        img = img.resize(shape, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except Exception:
        h_indices = np.linspace(0, gray.shape[0] - 1, shape[0]).astype(int)
        w_indices = np.linspace(0, gray.shape[1] - 1, shape[1]).astype(int)
        return gray[np.ix_(h_indices, w_indices)]


class HeadlessRetroEnv:
    """
    Gymnasium-compatible environment wrapper around a headless stable-retro emulator
    using actual Castlevania NES ROM ('Castlevania (USA) (Rev 1).nes').
    Restricts action space strictly to 8 essential Castlevania actions:
    0: NOOP, 1: RIGHT, 2: LEFT, 3: DOWN, 4: JUMP, 5: WHIP, 6: RIGHT+JUMP, 7: RIGHT+WHIP.
    Applies 4-frame stacking giving observation shape (4, 84, 84) and dynamic timeouts (+50 steps per 100px).
    Includes anti-reward hacking detection for static-position reward exploitation and autonomous auto-restart on completion.
    """
    ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "DOWN", "JUMP", "WHIP", "RIGHT+JUMP", "RIGHT+WHIP"]

    def __init__(self, frame_shape: Tuple[int, int] = (84, 84), num_stack: int = 4, base_max_steps: int = 400, use_retro: bool = True):
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
        self.stage = 0
        self.game_completed = False
        self.auto_restarted = False
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.retro_env = None
        self.use_retro = use_retro and HAS_STABLE_RETRO

        if self.use_retro:
            _ensure_rom_imported()
            try:
                self.retro_env = stable_retro.make(game="Castlevania-Nes-v0", render_mode=None)
            except Exception as e:
                print(f"Warning: Could not initialize stable_retro ({e}). Falling back to internal engine.")
                self.retro_env = None

        # Button index map for Castlevania-Nes-v0:
        # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
        self._button_map = {
            "NOOP":       [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "RIGHT":      [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "LEFT":       [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "DOWN":       [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "JUMP":       [0, 0, 0, 0, 0, 0, 0, 0, 1], # 'A' is jump
            "WHIP":       [1, 0, 0, 0, 0, 0, 0, 0, 0], # 'B' is whip
            "RIGHT+JUMP": [0, 0, 0, 0, 0, 0, 0, 1, 1],
            "RIGHT+WHIP": [1, 0, 0, 0, 0, 0, 0, 1, 0]
        }

    def _get_stacked_obs(self) -> np.ndarray:
        return np.stack(list(self.frame_buffer), axis=0)

    def _action_to_buttons(self, act_name: str) -> list:
        return self._button_map.get(act_name, [0] * 9)

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
        self.stage = 0
        self.game_completed = False
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.action_history.clear()
        self.frame_buffer.clear()

        raw_frame = None
        if self.retro_env is not None:
            raw_obs, _ = self.retro_env.reset(seed=seed)
            raw_frame = raw_obs
            self._update_ram_state()
        else:
            raw_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        processed = _process_frame(raw_frame, shape=self.frame_shape)
        for _ in range(self.num_stack):
            self.frame_buffer.append(processed)

        info = {
            "x_pos": self.x_pos,
            "max_x_pos": self.max_x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "stage": self.stage,
            "game_completed": self.game_completed,
            "auto_restarted": self.auto_restarted,
            "reward_hacking_detected": False,
            "max_steps": self.max_episode_steps
        }
        self.reward_engine.reset(info)

        return self._get_stacked_obs(), info

    def _update_ram_state(self):
        """Reads real RAM from the retro emulator instance if available."""
        if self.retro_env is None:
            return
        try:
            ram = self.retro_env.get_ram()
            if ram is not None and len(ram) >= 2048:
                player_x = int(ram[0x0026])
                screen_x = int(ram[0x0028])
                raw_x = float(screen_x * 256 + player_x)
                if raw_x > self.x_pos or self.step_count == 0:
                    self.x_pos = raw_x
                self.lives = int(ram[0x002A])
                self.health = int(ram[0x0044])
                self.hearts = int(ram[0x0040])
                self.stage = int(ram[0x0070])
                if self.stage >= 18 or ram[0x001A] == 1:
                    self.game_completed = True
        except Exception:
            pass

    def auto_restart(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes automated button sequence (START) to auto-restart the game
        upon completing the game or game over, ensuring autonomous non-blocking play.
        """
        self.auto_restarted = True
        if self.retro_env is not None:
            start_btn = [0, 0, 0, 1, 0, 0, 0, 0, 0]
            for _ in range(5):
                self.retro_env.step(start_btn)
            for _ in range(5):
                self.retro_env.step([0] * 9)
        return self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        act = action if (0 <= action < len(self.ACTION_NAMES)) else 0
        act_name = self.ACTION_NAMES[act]
        self.action_history.append(act_name)

        if self.retro_env is not None:
            btn_arr = self._action_to_buttons(act_name)
            raw_obs, retro_reward, retro_term, retro_trunc, retro_info = self.retro_env.step(btn_arr)
            new_frame = _process_frame(raw_obs, shape=self.frame_shape)
            self._update_ram_state()
            if "score" in retro_info:
                self.score = retro_info["score"]
        else:
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

            new_frame = np.random.randint(0, 256, size=self.frame_shape, dtype=np.uint8)

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
            "stage": self.stage,
            "game_completed": self.game_completed,
            "auto_restarted": self.auto_restarted,
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

        terminated = self.lives <= 0 or self.game_completed
        truncated = self.step_count >= self.max_episode_steps

        # Auto restart if game over or game completed
        if terminated and not self.auto_restarted:
            self.auto_restart()
            info["auto_restarted"] = True

        self.frame_buffer.append(new_frame)

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def close(self):
        if self.retro_env is not None:
            self.retro_env.close()
            self.retro_env = None
