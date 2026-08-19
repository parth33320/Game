import time
from typing import Dict, Any, Optional, List

class RAMScraper:
    """
    Interfaces with emulator API or memory state to extract real-time gameplay flags
    (text box state, active menu options, player coordinates, battle stats, game-over/title screen states)
    without vision models. Includes automated menu recovery logic to auto-restart on game over.
    """
    def __init__(self, memory_backend: Optional[Dict[str, Any]] = None):
        self.memory_backend = memory_backend or {}
        self.current_speed = 1.0

    def read_ram_state(self) -> Dict[str, Any]:
        """Reads raw memory addresses and parses them into structured game flags."""
        text_box_open = bool(self.memory_backend.get("0x0200", 0))
        active_menu = self.memory_backend.get("0x0202", ["Fight", "Bag", "Pokemon", "Run"])
        player_x = int(self.memory_backend.get("0x0210", 12))
        player_y = int(self.memory_backend.get("0x0212", 34))
        is_battle = bool(self.memory_backend.get("0x0220", 0))
        opponent_hp = int(self.memory_backend.get("0x0222", 100))
        opponent_max_hp = int(self.memory_backend.get("0x0224", 100))
        opponent_level = int(self.memory_backend.get("0x0226", 15))

        # Game-Over, Title Screen, and Death memory flags
        is_game_over = bool(self.memory_backend.get("0x0100", 0))
        is_title_screen = bool(self.memory_backend.get("0x0101", 0))
        is_dead = bool(self.memory_backend.get("0x0102", 0))

        return {
            "text_box_open": text_box_open,
            "active_menu_options": active_menu if (text_box_open or is_battle) else [],
            "player_coords": {"x": player_x, "y": player_y},
            "is_battle": is_battle,
            "is_game_over": is_game_over,
            "is_title_screen": is_title_screen,
            "is_dead": is_dead,
            "opponent_stats": {
                "hp": opponent_hp,
                "max_hp": opponent_max_hp,
                "level": opponent_level
            } if is_battle else None
        }

    def auto_recover_menu_sequence(self) -> List[str]:
        """
        Programmatically executes button sequences (START, A, START) to bypass
        Game-Over / Title screens and auto-restart gameplay without human input or LLM calls.
        """
        state = self.read_ram_state()
        sequence = []
        if state["is_game_over"] or state["is_title_screen"] or state["is_dead"]:
            sequence = ["START", "A", "START"]
            # Clear terminal memory flags upon recovery sequence
            self.memory_backend["0x0100"] = 0
            self.memory_backend["0x0101"] = 0
            self.memory_backend["0x0102"] = 0
        return sequence

    def set_speed(self, speed_multiplier: float) -> float:
        """Toggles emulator speed (e.g. 8.0 for fast-forward navigation/grinding, 1.0 for decision prompts)."""
        self.current_speed = float(speed_multiplier)
        return self.current_speed

    def get_speed(self) -> float:
        return self.current_speed
