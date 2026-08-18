import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

VALID_ACTIONS = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}

class LLMPlayer:
    """
    Ollama-powered client connecting to a local text model (Llama 3 / Qwen 2.5).
    Structures prompts with extracted RAM state and comedic persona, parsing responses into valid button inputs.
    """
    def __init__(
        self,
        model_name: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        persona: str = "sarcastic retro gamer streamer"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.persona = persona

    def build_prompt(self, ram_state: Dict[str, Any]) -> str:
        prompt = (
            f"You are playing a turn-based retro RPG on stream as a {self.persona}.\n"
            f"CURRENT RAM STATE:\n"
            f"- Text Box Open: {ram_state.get('text_box_open')}\n"
            f"- Is Battle: {ram_state.get('is_battle')}\n"
            f"- Active Menu Options: {ram_state.get('active_menu_options')}\n"
            f"- Player Position: {ram_state.get('player_coords')}\n"
            f"- Opponent Stats: {ram_state.get('opponent_stats')}\n\n"
            f"Choose ONE valid button input from: {sorted(list(VALID_ACTIONS))}.\n"
            f"Format your response strictly as JSON with keys 'action' and 'dialogue':\n"
            f'{{"action": "A", "dialogue": "Take that, pixel monster!"}}'
        )
        return prompt

    def query_ollama(self, prompt: str) -> str:
        """Sends prompt to local Ollama API endpoint."""
        url = f"{self.ollama_url}/api/generate"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "")
        except Exception:
            # Fallback mock response if Ollama server is offline or unreachable in tests
            return '{"action": "A", "dialogue": "Ollama offline, pressing A!"}'

    def parse_response(self, raw_text: str) -> Dict[str, str]:
        """Parses LLM output text into sanitized button action and comedic dialogue string."""
        action = "A"
        dialogue = "Let's keep moving!"

        # Try JSON extraction
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                act = str(parsed.get("action", "")).upper().strip()
                if act in VALID_ACTIONS:
                    action = act
                dialogue = str(parsed.get("dialogue", dialogue))
                return {"action": action, "dialogue": dialogue}
            except Exception:
                pass

        # First try finding uppercase actions in text order
        words = re.findall(r"\b[A-Za-z]+\b", raw_text)
        for w in words:
            if w in VALID_ACTIONS:
                return {"action": w, "dialogue": raw_text}

        # Fallback case-insensitive match in word order
        for w in words:
            if w.upper() in VALID_ACTIONS:
                return {"action": w.upper(), "dialogue": raw_text}

        return {"action": action, "dialogue": dialogue}

    def select_action(self, ram_state: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(ram_state)
        response_text = self.query_ollama(prompt)
        return self.parse_response(response_text)
