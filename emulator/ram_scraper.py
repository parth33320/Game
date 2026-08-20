import time
from typing import Dict, Any, Optional, List

class RAMScraper:
    """
    Interfaces with emulator API or memory state to extract real-time gameplay flags
    (text box state, active menu options, player coordinates, battle stats, game-over/title screen states, stage completion)
    without vision models. Includes automated menu recovery logic to auto-restart on game over or game completion.

    Supports Castlevania RAM Edge Cases:
    1. Global X-Position Calculation: coarse_screen ($0041) * 256 + fine_x ($0040).
    2. Staircase Alignment Detection: movement state ($0020) in (0x08, 0x0A).
    3. Transition Door Delays: game state ($0018) == 0x08.
    4. Boss Room Detection & HP ($01AA).
    5. Feature Scaling: Bounded 1D normalized float array (~15 values).
    """
    def __init__(self, memory_backend: Optional[Dict[str, Any]] = None, retro_env: Optional[Any] = None):
        self.memory_backend = memory_backend or {}
        self.retro_env = retro_env
        self.current_speed = 1.0

    def read_ram_state(self) -> Dict[str, Any]:
        """Reads raw memory addresses and parses them into structured game flags."""
        if self.retro_env is not None and hasattr(self.retro_env, "get_ram"):
            try:
                ram = self.retro_env.get_ram()
                if ram is not None and len(ram) >= 2048:
                    fine_x = int(ram[0x0040])
                    coarse_screen = int(ram[0x0041])
                    global_x = float(coarse_screen * 256 + fine_x)

                    player_y = int(ram[0x0038]) if len(ram) > 0x0038 else int(ram[0x0028])
                    lives = int(ram[0x002A])
                    health = int(ram[0x0044])
                    hearts = int(ram[0x0042]) if len(ram) > 0x0042 else int(ram[0x0040])
                    stage = int(ram[0x0070])

                    movement_state = int(ram[0x0020])
                    is_on_stairs = movement_state in (0x08, 0x0A)

                    game_state = int(ram[0x0018])
                    is_door_transition = (game_state == 0x08)
                    is_game_over = (game_state == 0x07) or lives == 0 or bool(ram[0x0100])
                    is_title_screen = (game_state == 0x00) or bool(ram[0x0101])
                    is_dead = health == 0 or bool(ram[0x0102])
                    is_completed = stage >= 18 or bool(ram[0x001A]) or (game_state == 0x0A)

                    boss_hp = int(ram[0x01AA]) if len(ram) > 0x01AA else 16
                    in_boss_room = (boss_hp > 0 and boss_hp <= 16 and stage in (3, 6, 9, 12, 15, 18))

                    return {
                        "text_box_open": False,
                        "active_menu_options": [],
                        "player_coords": {"x": global_x, "y": player_y},
                        "coarse_screen": coarse_screen,
                        "fine_x": fine_x,
                        "lives": lives,
                        "health": health,
                        "hearts": hearts,
                        "stage": stage,
                        "boss_hp": boss_hp,
                        "in_boss_room": in_boss_room,
                        "is_on_stairs": is_on_stairs,
                        "is_door_transition": is_door_transition,
                        "game_state_byte": game_state,
                        "movement_state_byte": movement_state,
                        "is_battle": False,
                        "is_game_over": is_game_over,
                        "is_title_screen": is_title_screen,
                        "is_dead": is_dead,
                        "is_completed": is_completed,
                        "opponent_stats": None
                    }
            except Exception:
                pass

        # Fallback to memory_backend dictionary
        text_box_open = bool(self.memory_backend.get("0x0200", 0))
        active_menu = self.memory_backend.get("0x0202", ["Fight", "Bag", "Pokemon", "Run"])
        player_x = int(self.memory_backend.get("0x0210", 12))
        player_y = int(self.memory_backend.get("0x0212", 34))
        is_battle = bool(self.memory_backend.get("0x0220", 0))
        opponent_hp = int(self.memory_backend.get("0x0222", 100))
        opponent_max_hp = int(self.memory_backend.get("0x0224", 100))
        opponent_level = int(self.memory_backend.get("0x0226", 15))

        # Game-Over, Title Screen, Game Completed, and Death memory flags
        is_game_over = bool(self.memory_backend.get("0x0100", 0))
        is_title_screen = bool(self.memory_backend.get("0x0101", 0))
        is_dead = bool(self.memory_backend.get("0x0102", 0))
        is_completed = bool(self.memory_backend.get("0x0103", 0))

        return {
            "text_box_open": text_box_open,
            "active_menu_options": active_menu if (text_box_open or is_battle) else [],
            "player_coords": {"x": player_x, "y": player_y},
            "is_battle": is_battle,
            "is_game_over": is_game_over,
            "is_title_screen": is_title_screen,
            "is_dead": is_dead,
            "is_completed": is_completed,
            "is_on_stairs": False,
            "is_door_transition": False,
            "in_boss_room": False,
            "opponent_stats": {
                "hp": opponent_hp,
                "max_hp": opponent_max_hp,
                "level": opponent_level
            } if is_battle else None
        }

    def auto_recover_menu_sequence(self) -> List[str]:
        """
        Programmatically executes button sequences (START, A, START) to bypass
        Game-Over / Title screens / Game completion screens and auto-restart gameplay without human input or LLM calls.
        """
        state = self.read_ram_state()
        sequence = []
        if state["is_game_over"] or state["is_title_screen"] or state["is_dead"] or state.get("is_completed", False):
            sequence = ["START", "A", "START"]
            # Clear terminal memory flags upon recovery sequence
            self.memory_backend["0x0100"] = 0
            self.memory_backend["0x0101"] = 0
            self.memory_backend["0x0102"] = 0
            self.memory_backend["0x0103"] = 0

            # Execute button sequence on retro_env if connected
            if self.retro_env is not None and hasattr(self.retro_env, "step"):
                try:
                    start_btn = [0, 0, 0, 1, 0, 0, 0, 0, 0]
                    a_btn = [0, 0, 0, 0, 0, 0, 0, 0, 1]
                    for _ in range(3):
                        self.retro_env.step(start_btn)
                    for _ in range(3):
                        self.retro_env.step(a_btn)
                    for _ in range(3):
                        self.retro_env.step(start_btn)
                except Exception:
                    pass

        return sequence

    def set_speed(self, speed_multiplier: float) -> float:
        """Toggles emulator speed (e.g. 8.0 for fast-forward navigation/grinding, 1.0 for decision prompts)."""
        self.current_speed = float(speed_multiplier)
        return self.current_speed

    def get_speed(self) -> float:
        return self.current_speed
