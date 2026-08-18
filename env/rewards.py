from typing import Dict, Any

class RetroRewardEngine:
    """
    Reward shaping engine for NES/Castlevania Gym environment:
    - Positive rewards for horizontal progression, collecting hearts/score, and surviving.
    - Heavy penalties for taking damage, losing lives, or getting stuck.
    """
    def __init__(
        self,
        progress_weight: float = 1.0,
        heart_weight: float = 0.5,
        score_weight: float = 0.05,
        damage_penalty: float = -15.0,
        death_penalty: float = -200.0,
        stuck_penalty: float = -2.0
    ):
        self.progress_weight = progress_weight
        self.heart_weight = heart_weight
        self.score_weight = score_weight
        self.damage_penalty = damage_penalty
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty

        self.last_x = 0.0
        self.last_hearts = 0
        self.last_score = 0
        self.last_health = 16
        self.last_lives = 3
        self.stuck_counter = 0

    def reset(self, info: Dict[str, Any]):
        self.last_x = float(info.get("x_pos", 0.0))
        self.last_hearts = int(info.get("hearts", 0))
        self.last_score = int(info.get("score", 0))
        self.last_health = int(info.get("health", 16))
        self.last_lives = int(info.get("lives", 3))
        self.stuck_counter = 0

    def calculate_reward(self, info: Dict[str, Any]) -> float:
        curr_x = float(info.get("x_pos", 0.0))
        curr_hearts = int(info.get("hearts", 0))
        curr_score = int(info.get("score", 0))
        curr_health = int(info.get("health", 16))
        curr_lives = int(info.get("lives", 3))

        reward = 0.0

        # Horizontal progress
        x_diff = curr_x - self.last_x
        if x_diff > 0:
            reward += x_diff * self.progress_weight
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1
            if self.stuck_counter > 50:
                reward += self.stuck_penalty

        # Item collections (hearts/score)
        if curr_hearts > self.last_hearts:
            reward += (curr_hearts - self.last_hearts) * self.heart_weight
        if curr_score > self.last_score:
            reward += (curr_score - self.last_score) * self.score_weight

        # Damage / Death penalties
        if curr_health < self.last_health and curr_lives == self.last_lives:
            reward += (self.last_health - curr_health) * self.damage_penalty
        if curr_lives < self.last_lives:
            reward += self.death_penalty

        self.last_x = curr_x
        self.last_hearts = curr_hearts
        self.last_score = curr_score
        self.last_health = curr_health
        self.last_lives = curr_lives

        return reward
