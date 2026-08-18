import time
from typing import Dict, Any, Optional

class RAMScraper:
    """
    Interfaces with emulator API or memory state to extract real-time gameplay flags
    (text box state, active menu options, player coordinates, battle stats) without vision models.
    """
    def __init__(self, memory_backend: Optional[Dict[str, Any]] = None):
        # memory_backend simulates or references emulator RAM pointer / state dictionary
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

        return {
            "text_box_open": text_box_open,
            "active_menu_options": active_menu if (text_box_open or is_battle) else [],
            "player_coords": {"x": player_x, "y": player_y},
            "is_battle": is_battle,
            "opponent_stats": {
                "hp": opponent_hp,
                "max_hp": opponent_max_hp,
                "level": opponent_level
            } if is_battle else None
        }

    def set_speed(self, speed_multiplier: float) -> float:
        """Toggles emulator speed (e.g. 8.0 for fast-forward navigation/grinding, 1.0 for decision prompts)."""
        self.current_speed = float(speed_multiplier)
        return self.current_speed

    def get_speed(self) -> float:
        return self.current_speed
