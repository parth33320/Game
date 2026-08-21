from typing import Dict, Any

class RetroRewardEngine:
    """
    Reward shaping engine for NES/Castlevania Gym environment:
    - Strict per-step time penalty (-0.02) to force progression.
    - Exponential progression reward when exceeding maximum horizontal position.
    - Reduced death penalty to promote exploration over passive waiting.
    - Secondary rewards for collecting items/score.
    """
    def __init__(
        self,
        progress_weight: float = 1.0,
        heart_weight: float = 0.5,
        score_weight: float = 0.05,
        time_penalty: float = -0.02,
        damage_penalty: float = -5.0,
        death_penalty: float = -30.0,
        stuck_penalty: float = -0.5,
        progress_multiplier: float = 1.0,
        stage_reward: float = 100.0,
        completion_reward: float = 500.0
    ):
        self.progress_weight = progress_weight
        self.heart_weight = heart_weight
        self.score_weight = score_weight
        self.time_penalty = time_penalty
        self.damage_penalty = damage_penalty
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty
        self.progress_multiplier = progress_multiplier
        self.stage_reward = stage_reward
        self.completion_reward = completion_reward

        self.max_x = 0.0
        self.last_x = 0.0
        self.last_hearts = 0
        self.last_score = 0
        self.last_health = 16
        self.last_lives = 3
        self.last_stage = 0
        self.last_completed = False
        self.stuck_counter = 0

    def reset(self, info: Dict[str, Any]):
        self.max_x = float(info.get("x_pos", 0.0))
        self.last_x = float(info.get("x_pos", 0.0))
        self.last_hearts = int(info.get("hearts", 0))
        self.last_score = int(info.get("score", 0))
        self.last_health = int(info.get("health", 16))
        self.last_lives = int(info.get("lives", 3))
        self.last_stage = int(info.get("stage", 0))
        self.last_completed = bool(info.get("game_completed", False))
        self.stuck_counter = 0

    def calculate_reward(self, info: Dict[str, Any]) -> float:
        curr_x = float(info.get("x_pos", 0.0))
        curr_hearts = int(info.get("hearts", 0))
        curr_score = int(info.get("score", 0))
        curr_health = int(info.get("health", 16))
        curr_lives = int(info.get("lives", 3))
        curr_stage = int(info.get("stage", 0))
        curr_completed = bool(info.get("game_completed", False))

        reward = 0.0

        # Step time penalty to enforce urgency
        reward += self.time_penalty

        # Progression reward strictly when exceeding max_x with milestone multipliers
        if curr_x > self.max_x:
            x_diff = curr_x - self.max_x
            progress_multiplier = 1.0 + (self.max_x / 100.0)
            reward += x_diff * self.progress_weight * progress_multiplier * self.progress_multiplier
            self.max_x = curr_x
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1
            if self.stuck_counter > 30:
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

        if curr_stage > self.last_stage:
            reward += (curr_stage - self.last_stage) * self.stage_reward
        if curr_completed and not self.last_completed:
            reward += self.completion_reward

        self.last_x = curr_x
        self.last_hearts = curr_hearts
        self.last_score = curr_score
        self.last_health = curr_health
        self.last_lives = curr_lives
        self.last_stage = curr_stage
        self.last_completed = curr_completed

        return reward
