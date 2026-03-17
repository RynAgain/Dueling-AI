# ---------------------------------------------------------------------------
# ai/expert_agent.py -- Rule-based heuristic agent (no learning required)
# ---------------------------------------------------------------------------
"""A hand-coded expert agent that plays reasonably well.

Uses simple rules:
  1. Rotate turret toward enemy
  2. Shoot when turret is aligned with enemy
  3. Move toward enemy (approach from an angle)
  4. Dodge when bullet is incoming
  5. Pick up nearby power-ups
  6. Lay mines when enemy is close behind

Use with --expert flag or as a training opponent.
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
        self.epsilon = 0.0  # no exploration needed

    def choose_action(self, state: list[float]) -> int:
        """Pick action based on heuristic rules.

        State vector (21 floats) from ContinuousStateEncoder:
          0: hull_rel_sin    1: hull_rel_cos
          2: turret_rel_sin  3: turret_rel_cos
          4: dist_norm
          5: enemy_facing_sin  6: enemy_facing_cos
          7: wall_front  8: wall_left  9: wall_right
          10: can_shoot
          11: own_hp  12: enemy_hp
          13: pu_speed  14: pu_rapid  15: pu_shield
          16: nearby_pu
          17: mine_threat  18: bullet_threat
          19: incoming_sin  20: incoming_cos
        """
        turret_sin = state[2]
        turret_cos = state[3]
        dist = state[4]
        wall_front = state[7]
        can_shoot = state[10]
        bullet_threat = state[18]
        incoming_sin = state[19]
        incoming_cos = state[20]
        nearby_pu = state[16]
        mine_threat = state[17]

        # --- Turret control: aim at enemy ---
        turret_angle = math.atan2(turret_sin, turret_cos)  # radians, + = enemy right
        turret_deg = math.degrees(turret_angle)

        if abs(turret_deg) < 8:
            turret_action = cfg.TURRET_NOOP  # aligned enough
        elif turret_deg > 0:
            turret_action = cfg.TURRET_RIGHT
        else:
            turret_action = cfg.TURRET_LEFT

        # --- Fire control ---
        fire_action = cfg.FIRE_NONE
        if can_shoot > 0.5 and abs(turret_deg) < 15:
            fire_action = cfg.FIRE_SHOOT
        elif dist < 0.15 and random.random() < 0.1:
            # Occasionally lay mine when very close
            fire_action = cfg.FIRE_MINE

        # --- Movement control ---
        # Priority: dodge > avoid mine > pick up power-up > approach enemy

        if bullet_threat > 0.7:
            # Imminent bullet -- dodge perpendicular
            incoming_angle = math.atan2(incoming_sin, incoming_cos)
            incoming_deg = math.degrees(incoming_angle)
            # Move perpendicular: if bullet from front-right, move left
            if abs(incoming_deg) < 90:
                # Bullet from front -- move backward or sideways
                if incoming_deg > 0:
                    move_action = cfg.MOVE_ROTATE_LEFT
                else:
                    move_action = cfg.MOVE_ROTATE_RIGHT
            else:
                # Bullet from behind -- move forward
                move_action = cfg.MOVE_FORWARD
        elif mine_threat > 0.7:
            # Close mine -- back away
            move_action = cfg.MOVE_BACKWARD
        elif wall_front > 0.7:
            # Wall ahead -- turn
            move_action = cfg.MOVE_ROTATE_LEFT if random.random() < 0.5 else cfg.MOVE_ROTATE_RIGHT
        elif dist > 0.3:
            # Far from enemy -- approach
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
            # Close to enemy -- circle strafe
            move_action = cfg.MOVE_FORWARD if random.random() < 0.6 else cfg.MOVE_ROTATE_LEFT

        # Encode composite action: move * 9 + turret * 3 + fire
        action = move_action * (cfg.NUM_TURRET_OPTIONS * cfg.NUM_FIRE_OPTIONS) + \
                 turret_action * cfg.NUM_FIRE_OPTIONS + fire_action
        return min(action, self.num_actions - 1)

    def learn(self, *args, **kwargs) -> None:
        """No-op -- expert doesn't learn."""
        pass

    def decay_epsilon(self) -> None:
        """No-op."""
        pass

    def episode_end_replay(self) -> None:
        """No-op."""
        pass

    def save(self, path: str, **kwargs) -> None:
        """No-op -- nothing to save."""
        pass

    def load(self, path: str) -> dict:
        """No-op."""
        return {"episode": 0, "epsilon": 0.0}

    def buffer_size(self) -> int:
        return 0

    @property
    def q_table_a(self) -> dict:
        return {}

    @property
    def q_table_b(self) -> dict:
        return {}

    def network_param_count(self) -> int:
        return 0
