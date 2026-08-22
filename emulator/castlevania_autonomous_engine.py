import os
import time
import gc
import shutil
import torch
from typing import Callable, Optional, Dict, Any

class CastlevaniaAutonomousEngine:
    """
    Autonomous state machine orchestrator for NES Castlevania CPU RL training and continuous 24/7 streaming.
    - Handles system state freezes (0x03 loading transitions, 0x0C stage clear scoring).
    - Automatically pulses START button for menu registers (Title Screen, Game Over 0x07).
    - Triggers snapshots upon boss defeat and periodically saves model weights every 10 minutes to local/cloud storage.
    - Flushes environment and triggers garbage collection every N episodes to eliminate C++ Libretro memory leaks.
    """
    def __init__(
        self,
        env_creation_func: Callable,
        save_dir: str = "checkpoints",
        stream_key: Optional[str] = None,
        max_episodes_before_flush: int = 50,
        checkpoint_interval_seconds: float = 600.0  # 10 minutes
    ):
        self.create_env = env_creation_func
        self.save_dir = save_dir
        self.stream_key = stream_key
        self.max_episodes_before_flush = max_episodes_before_flush
        self.checkpoint_interval_seconds = checkpoint_interval_seconds

        os.makedirs(self.save_dir, exist_ok=True)

        self.env = self.create_env()
        self.episode_count = 0
        self.state_timer = 0
        self.last_checkpoint_time = time.time()

        self.prev_boss_health = 16
        self.boss_just_defeated = False
        self.current_state = "PLAYING"

    def get_ram_safely(self) -> Optional[Any]:
        try:
            if hasattr(self.env, "retro_env") and self.env.retro_env is not None:
                return self.env.retro_env.get_ram()
            if hasattr(self.env.unwrapped, "get_ram"):
                return self.env.unwrapped.get_ram()
        except Exception:
            pass
        return None

    def trigger_boss_snapshot(self, rl_agent: Any, stage: int):
        timestamp = int(time.time())
        target_path = os.path.join(self.save_dir, f"model_weights_stage_{stage}_{timestamp}.pt")
        latest_path = os.path.join(self.save_dir, f"model_weights_latest.pt")
        print(f"🏆 BOSS DEFEATED on Stage {stage}! Saving automated snapshot to {target_path}...")

        try:
            if hasattr(rl_agent, "model"):
                torch.save(rl_agent.model.state_dict(), target_path)
            elif hasattr(rl_agent, "state_dict"):
                torch.save(rl_agent.state_dict(), target_path)
            elif hasattr(rl_agent, "save_weights"):
                rl_agent.save_weights(target_path)
            else:
                torch.save(rl_agent, target_path)

            shutil.copy(target_path, latest_path)
            print(f"💾 Snapshot synced successfully to {latest_path}")
        except Exception as e:
            print(f"⚠️ Error saving boss snapshot: {e}")

    def periodic_time_checkpoint(self, rl_agent: Any):
        current_time = time.time()
        if current_time - self.last_checkpoint_time >= self.checkpoint_interval_seconds:
            timestamp = int(current_time)
            target_path = os.path.join(self.save_dir, f"model_weights_10min_{timestamp}.pt")
            latest_path = os.path.join(self.save_dir, f"model_weights_latest.pt")
            print(f"⏰ 10-Minute Periodic Checkpoint Triggered! Saving weights to {target_path}...")
            try:
                if hasattr(rl_agent, "model"):
                    torch.save(rl_agent.model.state_dict(), target_path)
                elif hasattr(rl_agent, "state_dict"):
                    torch.save(rl_agent.state_dict(), target_path)
                elif hasattr(rl_agent, "save_weights"):
                    rl_agent.save_weights(target_path)
                else:
                    torch.save(rl_agent, target_path)

                shutil.copy(target_path, latest_path)
                self.last_checkpoint_time = current_time
            except Exception as e:
                print(f"⚠️ Error saving 10-minute periodic checkpoint: {e}")

    def process_autonomous_step(self, rl_agent: Any, rl_agent_policy_func: Callable):
        ram = self.get_ram_safely()

        # Handle periodic 10-min model saving
        self.periodic_time_checkpoint(rl_agent)

        if ram is None:
            # Fallback environment step
            action = 0
            obs, reward, term, trunc, info = self.env.step(action)
            return obs, reward, term, trunc, info

        system_state = ram[0x0018]
        current_stage = ram[0x0028] if len(ram) > 0x0028 else ram[0x002A]
        player_health = ram[0x0045] if len(ram) > 0x0045 else ram[0x0044]
        boss_health = ram[0x01A9] if len(ram) > 0x01A9 else ram[0x01AA]

        # Check boss defeat snapshot trigger
        if boss_health == 0 and self.prev_boss_health > 0 and current_stage in (3, 6, 9, 12, 15, 18):
            self.trigger_boss_snapshot(rl_agent, current_stage)
            self.boss_just_defeated = True

        self.prev_boss_health = boss_health

        if self.boss_just_defeated and system_state == 0x05 and boss_health == 16:
            self.boss_just_defeated = False

        # State machine transition logic
        if system_state == 0x07 or player_health == 0:
            self.current_state = "GAME_OVER"
        elif system_state in (0x03, 0x0C):
            self.current_state = "TRANSITION"
        elif self.boss_just_defeated or (boss_health == 0 and current_stage in (3, 6, 9, 12, 15, 18)):
            self.current_state = "BOSS_CLEAR"
        else:
            self.current_state = "PLAYING"

        if self.current_state == "TRANSITION":
            obs, reward, term, trunc, info = self.env.step(0)  # NOOP
            return obs, 0.0, term, trunc, info

        elif self.current_state == "GAME_OVER":
            self.state_timer += 1
            # Alternate button to pulse START (action 8 = UP/START menu pick)
            act = 8 if (self.state_timer % 10 < 5) else 0
            obs, reward, term, trunc, info = self.env.step(act)
            return obs, 0.0, term, trunc, info

        elif self.current_state == "BOSS_CLEAR":
            # Hold RIGHT (action 1) to walk out of boss room
            obs, reward, term, trunc, info = self.env.step(1)
            return obs, reward, term, trunc, info

        else:
            # Active RL gameplay frame
            action = rl_agent_policy_func(obs_tensor=self.env._get_stacked_obs() if hasattr(self.env, "_get_stacked_obs") else None)
            obs, reward, term, trunc, info = self.env.step(action)

            if term or trunc:
                self.handle_episode_end()
                obs, info = self.env.reset()

            return obs, reward, term, trunc, info

    def handle_episode_end(self):
        self.episode_count += 1
        self.state_timer = 0
        self.prev_boss_health = 16

        if self.episode_count >= self.max_episodes_before_flush:
            print(f"♻️ Self-Healing Buffer Triggered ({self.episode_count} episodes): Reinitializing environment to purge C++ leaks...")
            self.env.close()
            del self.env
            gc.collect()
            time.sleep(1)

            self.env = self.create_env()
            self.episode_count = 0
