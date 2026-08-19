import threading
import queue
import time
import json
import re
from typing import Dict, Any, List, Optional
from chat.challenge_evaluator import ChallengeEvaluator

VALID_COMMANDS = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}

class RestreamChatListener:
    """
    Client connector for Twitch, YouTube Live, and Restream Chat API / Webhook services.
    Ingests unified chat messages across all connected channels into a thread-safe command queue.
    Integrates ChallengeEvaluator for automated quiz & coding challenges in chat.
    Supports anarchy/democracy voting modes and exclusive challenge winner control.
    """
    def __init__(
        self,
        mode: str = "anarchy",
        voting_window_seconds: float = 2.0,
        challenge_evaluator: Optional[ChallengeEvaluator] = None
    ):
        self.mode = mode.lower()
        self.voting_window_seconds = voting_window_seconds
        self.challenge_evaluator = challenge_evaluator or ChallengeEvaluator()
        
        self.command_queue: queue.Queue = queue.Queue()
        self.raw_vote_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._running = False

        self.exclusive_winner: Optional[str] = None
        self.exclusive_expires_at: float = 0.0

    def parse_chat_message(self, raw_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = str(raw_payload.get("text") or raw_payload.get("message") or "").strip()
        author = str(raw_payload.get("author") or raw_payload.get("username") or raw_payload.get("user") or "anonymous")
        platform = str(raw_payload.get("platform") or raw_payload.get("service") or "restream").lower()
        timestamp = float(raw_payload.get("timestamp") or time.time())

        if text.startswith("!answer ") or text.startswith("!solve "):
            submission = text.split(" ", 1)[1]
            eval_result = self.challenge_evaluator.evaluate_submission(author, submission)
            
            if eval_result["correct"]:
                with self._lock:
                    self.exclusive_winner = author
                    self.exclusive_expires_at = time.time() + eval_result["reward_duration"]
            
            return {
                "type": "challenge_submission",
                "author": author,
                "platform": platform,
                "timestamp": timestamp,
                "eval_result": eval_result,
                "raw_text": text
            }

        matched_cmd = None
        text_upper = text.upper()
        for cmd in sorted(list(VALID_COMMANDS), key=len, reverse=True):
            if re.search(rf"\b{cmd}\b", text_upper):
                matched_cmd = cmd
                break

        if not matched_cmd:
            return None

        return {
            "type": "command",
            "command": matched_cmd,
            "author": author,
            "platform": platform,
            "timestamp": timestamp,
            "raw_text": text
        }

    def ingest_payload(self, raw_payload: Dict[str, Any]) -> bool:
        parsed = self.parse_chat_message(raw_payload)
        if not parsed:
            return False

        if parsed.get("type") == "challenge_submission":
            return True

        with self._lock:
            now = time.time()
            if self.exclusive_winner and now < self.exclusive_expires_at:
                if parsed["author"] == self.exclusive_winner:
                    self.command_queue.put(parsed)
                return True

            if self.mode == "anarchy":
                self.command_queue.put(parsed)
            else:
                self.raw_vote_buffer.append(parsed)
        return True

    def get_exclusive_override_info(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            active = bool(self.exclusive_winner and now < self.exclusive_expires_at)
            remaining = max(0.0, round(self.exclusive_expires_at - now, 1)) if active else 0.0
            return {
                "active_override": active,
                "winner": self.exclusive_winner if active else None,
                "remaining_seconds": remaining
            }

    def get_next_command(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            if self.exclusive_winner and now < self.exclusive_expires_at:
                if not self.command_queue.empty():
                    return self.command_queue.get_nowait()
                return None

            if self.mode == "anarchy":
                if not self.command_queue.empty():
                    return self.command_queue.get_nowait()
                return None
            else:
                if not self.raw_vote_buffer:
                    return None

                tally: Dict[str, int] = {}
                for vote in self.raw_vote_buffer:
                    cmd = vote["command"]
                    tally[cmd] = tally.get(cmd, 0) + 1

                winning_cmd = max(tally.items(), key=lambda x: x[1])[0]
                winning_event = {
                    "type": "command",
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
            if self.mode == "anarchy" or (self.exclusive_winner and time.time() < self.exclusive_expires_at):
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
            self.exclusive_winner = None
            self.exclusive_expires_at = 0.0
