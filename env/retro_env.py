import os
import shutil
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional, List
from env.rewards import RetroRewardEngine

try:
    import stable_retro
    rom_source = "roms/Castlevania (USA) (Rev 1).nes"
    if os.path.exists(rom_source):
        try:
            target_dir = os.path.dirname(stable_retro.data.get_file_path("Castlevania-Nes-v0", "rom.sha"))
            target_rom = os.path.join(target_dir, "rom.nes")
            if not os.path.exists(target_rom):
                shutil.copy(rom_source, target_rom)
        except Exception:
            pass
    HAS_STABLE_RETRO = True
except ImportError:
    HAS_STABLE_RETRO = False


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

    Supports both 2D image observations and 1D CPU-normalized RAM-vector observations (~15 features).
    Addresses Castlevania RAM edge cases:
    1. Staircase Alignment Traps: Restricts action space to UP/DOWN when on stairs ($0020 == 0x08 or 0x0A).
    2. Transition Door Delays: Freezes environment timer and executes NOOP when door state ($0018 == 0x08) is active.
    3. Global X-Position: Calculates coarse + fine position (RAM $0041 * 256 + RAM $0040).
    4. Boss Room Soft-Locks: Dynamically shifts progress reward to Boss HP damage when in boss room.
    5. Feature Scaling: Normalizes RAM vector elements between 0.0 and 1.0 for CPU MLP training.
    """
    ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "DOWN", "JUMP", "WHIP", "RIGHT+JUMP", "RIGHT+WHIP", "UP"]

    def __init__(
        self,
        frame_shape: Tuple[int, int] = (84, 84),
        num_stack: int = 4,
        base_max_steps: int = 400,
        use_retro: bool = True,
        obs_type: str = "ram",  # "ram" for 1D CPU MLP vector or "pixels" for 2D stacked frame
        reward_params: Optional[Dict[str, Any]] = None
    ):
        self.frame_shape = frame_shape
        self.num_stack = num_stack
        self.base_max_steps = base_max_steps
        self.max_episode_steps = base_max_steps
        self.step_count = 0
        self.obs_type = obs_type
        reward_params = reward_params or {}
        self.reward_engine = RetroRewardEngine(
            progress_weight=float(reward_params.get("distance_weight", 1.0)),
            score_weight=float(reward_params.get("score_weight", 0.05)),
            time_penalty=float(reward_params.get("time_penalty", -0.02)),
            progress_multiplier=float(reward_params.get("progress_multiplier", 1.0)),
            stage_reward=float(reward_params.get("stage_reward", 100.0)),
            completion_reward=float(reward_params.get("completion_reward", 500.0)),
        )

        self.frame_buffer = deque(maxlen=num_stack)
        self.action_history = deque(maxlen=30)

        # Game State Variables
        self.global_x_pos = 0.0
        self.max_x_pos = 0.0
        self.fine_x = 0
        self.coarse_screen = 0
        self.y_pos = 0
        self.last_milestone = 0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3
        self.stage = 0
        self.boss_hp = 16
        self.prev_boss_hp = 16
        self.in_boss_room = False
        self.is_on_stairs = False
        self.is_door_transition = False
        self.game_state_byte = 0x05
        self.movement_state_byte = 0x00
        self.game_completed = False
        self.auto_restarted = False
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.retro_env = None
        self.use_retro = use_retro and HAS_STABLE_RETRO

        if self.use_retro:
            try:
                render_mode = "rgb_array" if self.obs_type != "ram" else None
                self.retro_env = stable_retro.make(game="Castlevania-Nes-v0", render_mode=render_mode)
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
            "UP":         [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "JUMP":       [0, 0, 0, 0, 0, 0, 0, 0, 1], # 'A' is jump
            "WHIP":       [1, 0, 0, 0, 0, 0, 0, 0, 0], # 'B' is whip
            "RIGHT+JUMP": [0, 0, 0, 0, 0, 0, 0, 1, 1],
            "RIGHT+WHIP": [1, 0, 0, 0, 0, 0, 0, 1, 0]
        }

    def _get_ram_vector(self) -> np.ndarray:
        """
        Returns a 1D normalized float32 RAM observation vector (~15 features bounded between 0.0 and 1.0)
        suitable for high-speed CPU Multi-Layer Perceptron (MLP) training.
        """
        vector = np.array([
            min(self.global_x_pos / 10000.0, 1.0),            # Scaled global X position
            min(self.y_pos / 240.0, 1.0),                     # Scaled Y position
            min(self.health / 16.0, 1.0),                     # Simon Health (0.0 to 1.0)
            min(self.lives / 3.0, 1.0),                       # Lives (0.0 to 1.0)
            min(self.hearts / 99.0, 1.0),                     # Hearts count
            min(self.boss_hp / 16.0, 1.0),                    # Boss HP (0.0 to 1.0)
            min(self.stage / 18.0, 1.0),                      # Stage progression
            1.0 if self.is_on_stairs else 0.0,                # Stairwalking state flag
            1.0 if self.is_door_transition else 0.0,          # Transition door flag
            1.0 if self.in_boss_room else 0.0,                # Boss room flag
            1.0 if self.game_completed else 0.0,              # Game completed flag
            min(self.coarse_screen / 50.0, 1.0),              # Screen section count
            min(self.fine_x / 255.0, 1.0),                    # Fine screen X position
            min(self.game_state_byte / 255.0, 1.0),           # Raw game mode byte
            min(self.movement_state_byte / 255.0, 1.0)        # Raw movement state byte
        ], dtype=np.float32)
        return vector

    def _get_stacked_obs(self) -> np.ndarray:
        if self.obs_type == "ram":
            return self._get_ram_vector()
        return np.stack(list(self.frame_buffer), axis=0)

    def _action_to_buttons(self, act_name: str) -> list:
        return self._button_map.get(act_name, [0] * 9)

    def _read_ram_and_update(self):
        """Reads RAM addresses directly and applies Castlevania domain logic."""
        if self.retro_env is not None:
            try:
                ram = self.retro_env.get_ram()
                if ram is not None and len(ram) >= 2048:
                    self.fine_x = int(ram[0x0040])
                    self.coarse_screen = int(ram[0x0041])
                    self.global_x_pos = float(self.coarse_screen * 256 + self.fine_x)

                    self.y_pos = int(ram[0x0038]) if len(ram) > 0x0038 else int(ram[0x0028])
                    self.lives = int(ram[0x002A])
                    self.health = int(ram[0x0044])
                    self.hearts = int(ram[0x0040]) if len(ram) <= 0x0040 else int(ram[0x0042])
                    self.stage = int(ram[0x0070])

                    # RAM Edge Case 1: Movement State & Stairs ($0020 == 0x08 or 0x0A)
                    self.movement_state_byte = int(ram[0x0020])
                    self.is_on_stairs = self.movement_state_byte in (0x08, 0x0A)

                    # RAM Edge Case 2: Door Transition ($0018 == 0x08)
                    self.game_state_byte = int(ram[0x0018])
                    self.is_door_transition = (self.game_state_byte == 0x08)

                    # RAM Edge Case 4: Boss HP ($01AA) & Boss Room Detection
                    if len(ram) > 0x01AA:
                        self.boss_hp = int(ram[0x01AA])
                        self.in_boss_room = (self.boss_hp > 0 and self.boss_hp <= 16 and self.stage in (3, 6, 9, 12, 15, 18))
                    else:
                        self.boss_hp = 16
                        self.in_boss_room = False

                    # Completion is confirmed by the final stage marker. The old
                    # 0x001A and 0x0A checks also occur during normal gameplay.
                    if self.stage >= 18:
                        self.game_completed = True
            except Exception:
                pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.max_episode_steps = self.base_max_steps
        self.global_x_pos = 0.0
        self.max_x_pos = 0.0
        self.fine_x = 0
        self.coarse_screen = 0
        self.y_pos = 0
        self.last_milestone = 0
        self.hearts = 0
        self.score = 0
        self.health = 16
        self.lives = 3
        self.stage = 0
        self.boss_hp = 16
        self.prev_boss_hp = 16
        self.in_boss_room = False
        self.is_on_stairs = False
        self.is_door_transition = False
        self.game_completed = False
        self.accumulated_reward = 0.0
        self.reward_hacking_detected = False

        self.action_history.clear()
        self.frame_buffer.clear()

        raw_frame = None
        if self.retro_env is not None:
            raw_obs, _ = self.retro_env.reset(seed=seed)
            raw_frame = raw_obs
            self._read_ram_and_update()
        else:
            raw_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.obs_type != "ram":
            processed = _process_frame(raw_frame, shape=self.frame_shape)
            for _ in range(self.num_stack):
                self.frame_buffer.append(processed)

        info = {
            "x_pos": self.global_x_pos,
            "max_x_pos": self.max_x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "stage": self.stage,
            "boss_hp": self.boss_hp,
            "in_boss_room": self.in_boss_room,
            "is_on_stairs": self.is_on_stairs,
            "is_door_transition": self.is_door_transition,
            "game_completed": self.game_completed,
            "auto_restarted": self.auto_restarted,
            "game_state_byte": self.game_state_byte,
            "reward_hacking_detected": False,
            "max_steps": self.max_episode_steps,
            "termination_reason": "running"
        }
        self.reward_engine.reset(info)

        return self._get_stacked_obs(), info

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
        act = action if (0 <= action < len(self.ACTION_NAMES)) else 0
        act_name = self.ACTION_NAMES[act]

        # System State Override 1: Loading transition (0x03) or stage clear scoring screen (0x0C)
        # Freeze learning steps and return 0.0 reward to avoid policy gradient noise
        if self.game_state_byte in (0x03, 0x0C):
            act_name = "NOOP"

        # System State Override 2: Game over (0x07) or title/menu screen -> pulse START to restart
        elif self.game_state_byte == 0x07 or self.lives <= 0:
            act_name = "NOOP"
            # Auto press START logic to move past Title or Game Over Continue screen
            if self.step_count % 10 < 5:
                act_name = "NOOP"

        # RAM Edge Case 1: Staircase alignment trap -> Restrict actions to UP/DOWN on stairs
        elif self.is_on_stairs and act_name not in ("UP", "DOWN"):
            act_name = "UP" if act % 2 == 0 else "DOWN"

        # RAM Edge Case 2: Transition door delay -> Issue NOOP action and pause step count/penalties
        elif self.is_door_transition:
            act_name = "NOOP"
        else:
            self.step_count += 1

        self.action_history.append(act_name)

        if self.retro_env is not None:
            btn_arr = self._action_to_buttons(act_name)
            raw_obs, retro_reward, retro_term, retro_trunc, retro_info = self.retro_env.step(btn_arr)
            if self.obs_type != "ram":
                new_frame = _process_frame(raw_obs, shape=self.frame_shape)
            self._read_ram_and_update()
            if "score" in retro_info:
                self.score = retro_info["score"]
        else:
            # Fallback simulated progression for environments without NES ROM binaries
            if not self.is_door_transition and self.game_state_byte not in (0x03, 0x0C):
                if act_name in ("RIGHT", "RIGHT+JUMP", "RIGHT+WHIP"):
                    self.global_x_pos += 2.0
                    self.score += 5
                elif act_name == "LEFT":
                    self.global_x_pos = max(0.0, self.global_x_pos - 0.5)
                elif act_name == "WHIP":
                    self.score += 20
                    self.hearts += 1

            new_frame = np.random.randint(0, 256, size=self.frame_shape, dtype=np.uint8)

        if self.global_x_pos > self.max_x_pos:
            self.max_x_pos = self.global_x_pos

        # Dynamic timeout extensions (+50 steps per 100 pixels)
        current_milestone = int(self.max_x_pos // 100)
        if current_milestone > self.last_milestone:
            milestone_diff = current_milestone - self.last_milestone
            self.max_episode_steps += milestone_diff * 50
            self.last_milestone = current_milestone

        info = {
            "x_pos": self.global_x_pos,
            "max_x_pos": self.max_x_pos,
            "hearts": self.hearts,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "stage": self.stage,
            "boss_hp": self.boss_hp,
            "in_boss_room": self.in_boss_room,
            "is_on_stairs": self.is_on_stairs,
            "is_door_transition": self.is_door_transition,
            "game_completed": self.game_completed,
            "auto_restarted": self.auto_restarted,
            "game_state_byte": self.game_state_byte,
            "max_steps": self.max_episode_steps
        }

        # System State 0x03 (Transition) or 0x0C (Scoring screen) -> Freeze learning reward to 0.0
        if self.game_state_byte in (0x03, 0x0C):
            reward = 0.0
        else:
            # Calculate reward (with Boss Room damage shift edge case handled in reward engine)
            reward = self.reward_engine.calculate_reward(info)

            # RAM Edge Case 4: Boss damage extra reward boost
            if self.in_boss_room and self.boss_hp < self.prev_boss_hp:
                boss_damage = self.prev_boss_hp - self.boss_hp
                reward += boss_damage * 5.0

        self.prev_boss_hp = self.boss_hp
        self.accumulated_reward += reward

        # Anti-Reward Hacking & Stagnation Check
        repetitive_loop = (len(self.action_history) == 30 and
                           all(a == "WHIP" for a in self.action_history) and
                           self.global_x_pos < 10.0)
        static_reward_hack = (self.accumulated_reward > 50.0 and self.max_x_pos < 10.0)

        if repetitive_loop or static_reward_hack:
            self.reward_hacking_detected = True

        info["reward_hacking_detected"] = self.reward_hacking_detected

        terminated = self.lives <= 0 or self.game_completed or self.game_state_byte == 0x07
        truncated = self.step_count >= self.max_episode_steps
        info["termination_reason"] = (
            "game_completed" if self.game_completed else
            "game_over" if self.game_state_byte == 0x07 else
            "lives_depleted" if self.lives <= 0 else
            "timeout" if truncated else
            "running"
        )

        # Auto restart if game over or game completed
        if terminated and not self.auto_restarted:
            self.auto_restart()
            info["auto_restarted"] = True

        if self.obs_type != "ram":
            self.frame_buffer.append(new_frame)

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def close(self):
        if self.retro_env is not None:
            self.retro_env.close()
            self.retro_env = None
