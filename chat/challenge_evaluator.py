import re
import ast
import random
import time
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_CHALLENGE_BANK = [
    {
        "id": "leetcode_01",
        "category": "LeetCode",
        "prompt": "LeetCode #1: What is the time complexity of searching an element in a balanced Binary Search Tree? (Answer in Big-O notation, e.g., O(log n))",
        "expected": r"^O\(\s*LOG\s*N\s*\)$",
        "type": "regex"
    },
    {
        "id": "python_01",
        "category": "Python",
        "prompt": "Python Challenge: Evaluate list expression `[1, 9]`",
        "expected_value": [1, 9],
        "type": "ast_eval"
    },
    {
        "id": "excel_01",
        "category": "Excel Macro",
        "prompt": "Excel Challenge: What Excel formula sums cells A1 through A10?",
        "expected": r"^=SUM\(\s*A1\s*:\s*A10\s*\)$",
        "type": "regex"
    },
    {
        "id": "stem_01",
        "category": "STEM Trivia",
        "prompt": "STEM Trivia: What is the atomic number of Carbon?",
        "expected": r"^6$",
        "type": "regex"
    },
    {
        "id": "python_02",
        "category": "Python",
        "prompt": "Python Challenge: What keyword is used to define an anonymous/inline function in Python?",
        "expected": r"^LAMBDA$",
        "type": "regex"
    }
]

class SafeASTEvaluator:
    """Safe AST-based expression evaluator for user coding challenge submissions."""
    @staticmethod
    def safe_eval(expr: str) -> Tuple[bool, Any]:
        try:
            tree = ast.parse(expr, mode='eval')
            allowed_nodes = (
                ast.Expression, ast.List, ast.Tuple, ast.Set, ast.Dict,
                ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.Load, ast.Store
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    return False, "Disallowed expression syntax"
            result = eval(compile(tree, filename='', mode='eval'), {"__builtins__": {}})
            return True, result
        except Exception as e:
            return False, str(e)

class ChallengeEvaluator:
    """
    Automated technical challenge & quiz module for Restream chat integration.
    Periodically triggers technical prompts (LeetCode, Python, Excel, STEM trivia)
    and evaluates viewer submissions securely.
    """
    def __init__(
        self,
        challenge_bank: Optional[List[Dict[str, Any]]] = None,
        cooldown_seconds: float = 10.0,
        reward_duration_seconds: float = 30.0
    ):
        self.challenge_bank = challenge_bank or DEFAULT_CHALLENGE_BANK
        self.cooldown_seconds = cooldown_seconds
        self.reward_duration_seconds = reward_duration_seconds
        self.active_challenge: Optional[Dict[str, Any]] = None
        self.user_cooldowns: Dict[str, float] = {}

    def trigger_new_challenge(self) -> Dict[str, Any]:
        """Picks and triggers a random challenge prompt from the question bank."""
        challenge = random.choice(self.challenge_bank)
        self.active_challenge = {
            "details": challenge,
            "start_time": time.time(),
            "solved": False,
            "winner": None
        }
        return challenge

    def get_active_challenge_prompt(self) -> Optional[str]:
        if self.active_challenge and not self.active_challenge["solved"]:
            return self.active_challenge["details"]["prompt"]
        return None

    def evaluate_submission(self, username: str, submission_text: str) -> Dict[str, Any]:
        """
        Evaluates a viewer submission against the current active challenge.
        Checks user cooldowns, verifies answer via regex or safe AST evaluation,
        and manages challenge resolution state.
        """
        now = time.time()
        
        if username in self.user_cooldowns:
            remaining = self.user_cooldowns[username] - now
            if remaining > 0:
                return {
                    "status": "cooldown",
                    "correct": False,
                    "message": f"User {username} is on cooldown for {round(remaining, 1)}s.",
                    "reward_duration": 0.0
                }

        if not self.active_challenge or self.active_challenge["solved"]:
            return {
                "status": "no_active_challenge",
                "correct": False,
                "message": "No active technical challenge right now.",
                "reward_duration": 0.0
            }

        details = self.active_challenge["details"]
        cleaned_input = submission_text.strip()
        is_correct = False

        if details["type"] == "regex":
            pattern = details["expected"]
            if re.search(pattern, cleaned_input, re.IGNORECASE):
                is_correct = True
        elif details["type"] == "ast_eval":
            success, val = SafeASTEvaluator.safe_eval(cleaned_input)
            if success and val == details["expected_value"]:
                is_correct = True

        if is_correct:
            self.active_challenge["solved"] = True
            self.active_challenge["winner"] = username
            return {
                "status": "success",
                "correct": True,
                "winner": username,
                "challenge_id": details["id"],
                "message": f"CONGRATULATIONS {username}! Correct answer! You get {self.reward_duration_seconds}s exclusive character control!",
                "reward_duration": self.reward_duration_seconds
            }
        else:
            self.user_cooldowns[username] = now + self.cooldown_seconds
            return {
                "status": "incorrect",
                "correct": False,
                "username": username,
                "message": f"Incorrect answer by {username}. Penalty: {self.cooldown_seconds}s cooldown.",
                "reward_duration": 0.0
            }
