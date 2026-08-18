import threading
import queue
import time
import json
import re
from typing import Dict, Any, List, Optional

VALID_COMMANDS = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}

class RestreamChatListener:
    """
    Client connector for Twitch, YouTube Live, and Restream Chat API / Webhook services.
    Ingests unified chat messages across all connected channels into a thread-safe command queue.
    Supports democracy/anarchy voting modes and instant execution.
    """
    def __init__(self, mode: str = "anarchy", voting_window_seconds: float = 2.0):
        self.mode = mode.lower()  # "anarchy" (instant) or "democracy" (windowed vote)
        self.voting_window_seconds = voting_window_seconds
        self.command_queue: queue.Queue = queue.Queue()
        self.raw_vote_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._running = False

    def parse_chat_message(self, raw_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalizes multi-platform chat payloads (Twitch, YouTube Live, Restream API / Webhooks)
        into a standardized event object.
        """
        text = str(raw_payload.get("text") or raw_payload.get("message") or "").strip().upper()
        author = str(raw_payload.get("author") or raw_payload.get("username") or raw_payload.get("user") or "anonymous")
        platform = str(raw_payload.get("platform") or raw_payload.get("service") or "restream").lower()
        timestamp = float(raw_payload.get("timestamp") or time.time())

        # Extract command token if present
        matched_cmd = None
        for cmd in sorted(list(VALID_COMMANDS), key=len, reverse=True):
            if re.search(rf"\b{cmd}\b", text):
                matched_cmd = cmd
                break

        if not matched_cmd:
            return None

        return {
            "command": matched_cmd,
            "author": author,
            "platform": platform,
            "timestamp": timestamp,
            "raw_text": text
        }

    def ingest_payload(self, raw_payload: Dict[str, Any]) -> bool:
        """Processes and adds a raw chat payload to queue or vote buffer."""
        parsed = self.parse_chat_message(raw_payload)
        if not parsed:
            return False

        with self._lock:
            if self.mode == "anarchy":
                self.command_queue.put(parsed)
            else:
                self.raw_vote_buffer.append(parsed)
        return True

    def get_next_command(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next command.
        In anarchy mode, pops the immediate command from queue.
        In democracy mode, tallies votes over the window and returns the winning command.
        """
        with self._lock:
            if self.mode == "anarchy":
                if not self.command_queue.empty():
                    return self.command_queue.get_nowait()
                return None
            else:
                # Democracy voting tally
                if not self.raw_vote_buffer:
                    return None

                tally: Dict[str, int] = {}
                for vote in self.raw_vote_buffer:
                    cmd = vote["command"]
                    tally[cmd] = tally.get(cmd, 0) + 1

                winning_cmd = max(tally.items(), key=lambda x: x[1])[0]
                winning_event = {
                    "command": winning_cmd,
                    "author": "democracy_vote",
                    "platform": "unified_chat",
                    "timestamp": time.time(),
                    "votes": tally[winning_cmd],
                    "total_votes": len(self.raw_vote_buffer)
                }
                self.raw_vote_buffer.clear()
                return winning_event

    def has_pending_commands(self) -> bool:
        with self._lock:
            if self.mode == "anarchy":
                return not self.command_queue.empty()
            return len(self.raw_vote_buffer) > 0

    def clear(self):
        with self._lock:
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            self.raw_vote_buffer.clear()
