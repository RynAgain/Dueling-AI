# ---------------------------------------------------------------------------
# ai/expert_agent.py -- Rule-based heuristic agent (no learning required)
# ---------------------------------------------------------------------------
"""A hand-coded expert agent that plays reasonably well.

Uses simple rules:
  1. Rotate turret toward enemy
  2. Shoot when turret is aligned with enemy
  3. Move toward enemy (approach from an angle)
  4. Dodge when bullet is incoming
"""
from __future__ import annotations
import math
import random
import config as cfg


class ExpertAgent:
    """Deterministic rule-based agent. No learning, no epsilon."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.num_actions = cfg.NUM_ACTIONS
        self.epsilon = 0.0
        self.epsilon_decay = 1.0  # no decay

    def choose_action(self, state: list[float]) -> int:
        """Pick action based on heuristic rules.

        State vector (16 floats per frame, 3 frames stacked = 48):
        We only use the first frame (most recent, indices 0-15).
        """
        turret_sin = state[2]
        turret_cos = state[3]
        dist = state[4]
        wall_front = state[7]
        can_shoot = state[10]
        bullet_threat = state[13]
        incoming_sin = state[14]
        incoming_cos = state[15]

        # --- Turret control: aim at enemy ---
        turret_deg = math.degrees(math.atan2(turret_sin, turret_cos))
        if abs(turret_deg) < 8:
            turret_action = cfg.TURRET_NOOP
        elif turret_deg > 0:
            turret_action = cfg.TURRET_RIGHT
        else:
            turret_action = cfg.TURRET_LEFT

        # --- Fire control ---
        fire_action = cfg.FIRE_NONE
        if can_shoot > 0.5 and abs(turret_deg) < 15:
            fire_action = cfg.FIRE_SHOOT

        # --- Movement control ---
        if bullet_threat > 0.7:
            incoming_deg = math.degrees(math.atan2(incoming_sin, incoming_cos))
            if abs(incoming_deg) < 90:
                if incoming_deg > 0:
                    move_action = cfg.MOVE_ROTATE_LEFT
                else:
                    move_action = cfg.MOVE_ROTATE_RIGHT
            else:
                move_action = cfg.MOVE_FORWARD
        elif wall_front > 0.7:
            move_action = cfg.MOVE_ROTATE_LEFT if random.random() < 0.5 else cfg.MOVE_ROTATE_RIGHT
        elif dist > 0.3:
            hull_sin = state[0]
            hull_cos = state[1]
            hull_angle = math.degrees(math.atan2(hull_sin, hull_cos))
            if abs(hull_angle) < 30:
                move_action = cfg.MOVE_FORWARD
            elif hull_angle > 0:
                move_action = cfg.MOVE_ROTATE_RIGHT
            else:
                move_action = cfg.MOVE_ROTATE_LEFT
        else:
            move_action = cfg.MOVE_FORWARD if random.random() < 0.6 else cfg.MOVE_ROTATE_LEFT

        action = cfg.encode_action(move_action, turret_action, fire_action)
        return min(action, self.num_actions - 1)

    def learn(self, state, action: int, reward: float,
              next_state, done: bool = False) -> None:
        pass

    def decay_epsilon(self) -> None:
        pass

    def episode_end_replay(self) -> None:
        pass

    def save(self, path: str, episode: int = 0) -> None:
        pass

    def load(self, path: str) -> dict:
        return {"episode": 0, "epsilon": 0.0}

    def buffer_size(self) -> int:
        return 0

    def network_param_count(self) -> int:
        return 0
