from typing import Dict, Any

class PlatformerRewardCalculator:
    """
    Reward engineering logic for platformer RL agent:
    - Strict per-step time penalty (-0.02) to penalize standing still.
    - Exponential progression reward when exceeding maximum horizontal position (max_x_pos).
    - Reduced death penalty to encourage exploration over safe timing out.
    - Score and item collection secondary rewards.
    """
    def __init__(
        self,
        distance_weight: float = 1.0,
        score_weight: float = 0.01,
        time_penalty: float = -0.02,
        life_loss_penalty: float = -25.0,
        damage_penalty: float = -5.0,
        pit_fall_penalty: float = -35.0
    ):
        self.distance_weight = distance_weight
        self.score_weight = score_weight
        self.time_penalty = time_penalty
        self.life_loss_penalty = life_loss_penalty
        self.damage_penalty = damage_penalty
        self.pit_fall_penalty = pit_fall_penalty

        self.max_x_pos = 0.0
        self.last_x_pos = 0.0
        self.last_score = 0
        self.last_lives = 3
        self.last_health = 16

    def reset(self, initial_state: Dict[str, Any]):
        self.max_x_pos = float(initial_state.get("x_pos", 0.0))
        self.last_x_pos = float(initial_state.get("x_pos", 0.0))
        self.last_score = int(initial_state.get("score", 0))
        self.last_lives = int(initial_state.get("lives", 3))
        self.last_health = int(initial_state.get("health", 16))

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        curr_x = float(current_state.get("x_pos", 0.0))
        curr_score = int(current_state.get("score", 0))
        curr_lives = int(current_state.get("lives", 3))
        curr_health = int(current_state.get("health", 16))
        fell_in_pit = bool(current_state.get("fell_in_pit", False))

        reward = 0.0

        # Step time penalty to enforce urgency
        reward += self.time_penalty

        # Exponential progress reward strictly when exceeding max_x_pos
        if curr_x > self.max_x_pos:
            x_diff = curr_x - self.max_x_pos
            progress_multiplier = 1.0 + (self.max_x_pos / 100.0)
            reward += x_diff * self.distance_weight * progress_multiplier
            self.max_x_pos = curr_x

        # Score gain
        score_diff = curr_score - self.last_score
        if score_diff > 0:
            reward += score_diff * self.score_weight

        # Life loss / Death penalty
        if curr_lives < self.last_lives or fell_in_pit:
            reward += self.pit_fall_penalty if fell_in_pit else self.life_loss_penalty

        # Health / Damage penalty
        if curr_health < self.last_health and curr_lives == self.last_lives:
            reward += (self.last_health - curr_health) * self.damage_penalty

        # Update state trackers
        self.last_x_pos = curr_x
        self.last_score = curr_score
        self.last_lives = curr_lives
        self.last_health = curr_health

        return reward
