import sys
import os
import asyncio
import time
import logging
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emulator.ram_scraper import RAMScraper
from ai.llm_player import LLMPlayer
from chat.chat_listener import RestreamChatListener
from audit.audit_logger import AuditLogger
from telemetry.telemetry_overlay import TelemetryPublisher

logger = logging.getLogger(__name__)

class PPOPolicyRunner:
    """
    Autonomous PPO Policy / Script Logic Runner for 24/7 non-blocking gameplay progression.
    Evaluates RAM state or decision prompts and outputs optimal actions autonomously
    without depending on external LLM APIs or chat inputs.
    """
    def __init__(self, ppo_model=None):
        self.ppo_model = ppo_model

    def select_action(self, ram_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Determines the next action using the PPO policy or deterministic script logic.
        Ensures continuous progression when decision boxes (dialogue/battle) are active.
        """
        if ram_state.get("is_battle"):
            return {"action": "A", "dialogue": "PPO Policy: Auto-selecting battle action (A)"}
        elif ram_state.get("text_box_open"):
            return {"action": "A", "dialogue": "PPO Policy: Auto-advancing dialogue prompt (A)"}
        
        return {"action": "RIGHT", "dialogue": "PPO Policy: Overworld navigation (RIGHT)"}

class GameplayAutomationLoop:
    """
    Main asynchronous game automation loop:
    1. Runs game at high speed (8x) during overworld exploration.
    2. Intercepts battle/dialogue states or active decision prompts, dropping speed to 1x.
    3. Enforces Chat Override Priority (if chat commands exist).
    4. Evaluates optional LLM decision engine with full rate-limit resilience & exception safety.
    5. Falls back to autonomous PPO policy / script logic for non-blocking 24/7 progression.
    6. Publishes real-time telemetry state and logs to audit trail.
    """
    def __init__(
        self,
        ram_scraper: Optional[RAMScraper] = None,
        llm_player: Optional[LLMPlayer] = None,
        chat_listener: Optional[RestreamChatListener] = None,
        audit_logger: Optional[AuditLogger] = None,
        telemetry_publisher: Optional[TelemetryPublisher] = None,
        ppo_runner: Optional[PPOPolicyRunner] = None,
        enable_llm_fallback: bool = True
    ):
        self.ram_scraper = ram_scraper or RAMScraper()
        self.llm_player = llm_player or LLMPlayer()
        self.chat_listener = chat_listener or RestreamChatListener()
        self.audit_logger = audit_logger or AuditLogger()
        self.telemetry_publisher = telemetry_publisher
        self.ppo_runner = ppo_runner or PPOPolicyRunner()
        self.enable_llm_fallback = enable_llm_fallback
        
        self.running = False
        self.executed_actions = []

    async def step(self) -> Dict[str, Any]:
        """Performs one iteration of the automation loop."""
        state = self.ram_scraper.read_ram_state()
        requires_decision = bool(state.get("text_box_open", False) or state.get("is_battle", False))

        source = "idle_navigation"
        action = None
        dialogue = ""

        # Safe fetch of exclusive winner override status
        override_info = (
            self.chat_listener.get_exclusive_override_info()
            if hasattr(self.chat_listener, "get_exclusive_override_info")
            else {"active_override": False, "winner": None, "remaining_seconds": 0.0}
        )

        # 1. Top Priority: Chat Override / Exclusive Winner Command
        if self.chat_listener.has_pending_commands():
            chat_cmd = self.chat_listener.get_next_command()
            if chat_cmd:
                action = chat_cmd["command"]
                if override_info["active_override"] and chat_cmd["author"] == override_info["winner"]:
                    source = "exclusive_winner_override"
                    dialogue = f"EXCLUSIVE WINNER CONTROL by {chat_cmd['author']}: {action}"
                else:
                    source = "chat_override"
                    dialogue = f"Chat override by {chat_cmd['author']} via {chat_cmd.get('platform', 'unified_chat')}: {action}"
                self.ram_scraper.set_speed(1.0)

        # 2. Secondary Priority (when decision required and chat idle): Optional LLM Player with non-blocking resilience
        if action is None and requires_decision and self.enable_llm_fallback:
            self.ram_scraper.set_speed(1.0)
            try:
                llm_result = self.llm_player.select_action(state)
                if llm_result and llm_result.get("action"):
                    action = llm_result["action"]
                    dialogue = llm_result.get("dialogue", "")
                    source = "local_llm"
            except Exception as e:
                logger.warning(f"LLM evaluation failed or rate-limited ({e}). Falling back to autonomous PPO policy.")

        # 3. Tertiary Fallback for 24/7 Autonomous Progression: PPO Policy / Script Logic
        if action is None and requires_decision:
            self.ram_scraper.set_speed(1.0)
            ppo_result = self.ppo_runner.select_action(state)
            action = ppo_result["action"]
            dialogue = ppo_result["dialogue"]
            source = "ppo_policy"
        elif action is None:
            # Overworld fast-forward mode (unless exclusive winner override is active)
            speed = 1.0 if override_info["active_override"] else 8.0
            self.ram_scraper.set_speed(speed)
            ppo_result = self.ppo_runner.select_action(state)
            action = ppo_result["action"]
            source = "exclusive_winner_idle" if override_info["active_override"] else "high_speed_exploration"

        # Record action execution
        record = {
            "timestamp": time.time(),
            "source": source,
            "action": action,
            "dialogue": dialogue,
            "speed": self.ram_scraper.get_speed(),
            "ram_state": state
        }

        self.executed_actions.append(record)
        self.audit_logger.log_event("action_executed", record)

        # Update telemetry publisher if present
        if self.telemetry_publisher:
            self.telemetry_publisher.update_telemetry(
                ram_stats={
                    "hp": state.get("opponent_stats", {}).get("hp", 100) if state.get("is_battle") else 100,
                    "max_hp": state.get("opponent_stats", {}).get("max_hp", 100) if state.get("is_battle") else 100,
                    "score": state.get("score", 0),
                    "player_coords": state.get("player_coords", {"x": 0, "y": 0}),
                    "active_threats": [state.get("opponent_stats")] if state.get("is_battle") else []
                },
                ai_status={
                    "last_decision_source": source,
                    "last_action": action,
                    "last_dialogue": dialogue
                },
                recent_log_entry=record,
                override_status={
                    "active_override": override_info["active_override"],
                    "override_type": "exclusive_winner" if override_info["active_override"] else None,
                    "winner": override_info["winner"],
                    "remaining_seconds": override_info["remaining_seconds"]
                },
                speed_mode=self.ram_scraper.get_speed()
            )

        return record

    async def run_loop(self, max_steps: int = 10, interval: float = 0.05):
        self.running = True
        step_count = 0
        while self.running and step_count < max_steps:
            await self.step()
            step_count += 1
            await asyncio.sleep(interval)

if __name__ == "__main__":
    loop = GameplayAutomationLoop()
    asyncio.run(loop.run_loop(max_steps=5))
