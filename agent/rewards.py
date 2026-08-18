from typing import Dict, Any

class PlatformerRewardCalculator:
    """
    Reward engineering logic for platformer RL agent:
    - Positive rewards for horizontal distance traveled and score accumulation.
    - Heavy penalties for life loss, damage taken, or falling into pits.
    """
    def __init__(
        self,
        distance_weight: float = 1.0,
        score_weight: float = 0.01,
        time_penalty: float = -0.01,
        life_loss_penalty: float = -100.0,
        damage_penalty: float = -10.0,
        pit_fall_penalty: float = -150.0
    ):
        self.distance_weight = distance_weight
        self.score_weight = score_weight
        self.time_penalty = time_penalty
        self.life_loss_penalty = life_loss_penalty
        self.damage_penalty = damage_penalty
        self.pit_fall_penalty = pit_fall_penalty

        self.last_x_pos = 0.0
        self.last_score = 0
        self.last_lives = 3
        self.last_health = 16

    def reset(self, initial_state: Dict[str, Any]):
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

        # Horizontal progression (reward positive x movement)
        x_diff = curr_x - self.last_x_pos
        reward += x_diff * self.distance_weight

        # Score gain
        score_diff = curr_score - self.last_score
        if score_diff > 0:
            reward += score_diff * self.score_weight

        # Step time penalty to encourage speed
        reward += self.time_penalty

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
