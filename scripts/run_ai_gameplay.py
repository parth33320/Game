import asyncio
import time
from typing import Dict, Any, Optional

from emulator.ram_scraper import RAMScraper
from ai.llm_player import LLMPlayer
from chat.chat_listener import RestreamChatListener
from audit.audit_logger import AuditLogger

class GameplayAutomationLoop:
    """
    Main asynchronous game automation loop:
    1. Runs game at high speed (8x) during exploration/walking.
    2. Intercepts battle/dialogue states or active decision prompts, dropping speed to 1x.
    3. Enforces strict Chat Override Priority: viewer chat commands preempt local LLM / AI decisions.
    4. Falls back to LLM player for decision making when chat is idle.
    5. Injects key inputs into emulator and resumes loop safely.
    """
    def __init__(
        self,
        ram_scraper: Optional[RAMScraper] = None,
        llm_player: Optional[LLMPlayer] = None,
        chat_listener: Optional[RestreamChatListener] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.ram_scraper = ram_scraper or RAMScraper()
        self.llm_player = llm_player or LLMPlayer()
        self.chat_listener = chat_listener or RestreamChatListener()
        self.audit_logger = audit_logger or AuditLogger()
        self.running = False
        self.executed_actions = []

    async def step(self) -> Dict[str, Any]:
        """Performs one iteration of the automation loop."""
        state = self.ram_scraper.read_ram_state()
        requires_decision = state["text_box_open"] or state["is_battle"]

        source = "idle_navigation"
        action = None
        dialogue = ""

        # Check Chat Override Priority
        if self.chat_listener.has_pending_commands():
            chat_cmd = self.chat_listener.get_next_command()
            if chat_cmd:
                action = chat_cmd["command"]
                source = "chat_override"
                dialogue = f"Chat override by {chat_cmd['author']} via {chat_cmd['platform']}: {action}"
                self.ram_scraper.set_speed(1.0)

        # Fallback to LLM Decision Engine if decision required and no chat input
        if action is None and requires_decision:
            self.ram_scraper.set_speed(1.0)
            llm_result = self.llm_player.select_action(state)
            action = llm_result["action"]
            dialogue = llm_result["dialogue"]
            source = "local_llm"
        elif action is None:
            # Overworld fast-forward mode
            self.ram_scraper.set_speed(8.0)
            action = "RIGHT"  # Default overworld exploration step
            source = "high_speed_exploration"

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
