# ---------------------------------------------------------------------------
# ai/state_encoder_continuous.py -- Continuous state vector for DQN
# ---------------------------------------------------------------------------
"""Encodes the game state as a fixed-length float vector (no discretization).

Frame stacking: the encoder stores the last N frames per tank and returns
them concatenated, so the DQN can perceive velocity and predict trajectories.
"""
from __future__ import annotations
import math
from collections import deque
import config as cfg

_ARENA_DIAG = math.sqrt(cfg.ARENA_WIDTH ** 2 + cfg.ARENA_HEIGHT ** 2)

FRAME_STACK_SIZE = 3
BASE_STATE_DIM = 16  # single-frame vector size
STACKED_STATE_DIM = BASE_STATE_DIM * FRAME_STACK_SIZE  # 48


class ContinuousStateEncoder:
    """Encodes game state into a flat float list for DQN input.

    Single-frame vector (16 floats):
      0.  rel_angle_hull_sin
      1.  rel_angle_hull_cos
      2.  rel_angle_turret_sin
      3.  rel_angle_turret_cos
      4.  dist_to_enemy       [0, 1]
      5.  enemy_facing_sin
      6.  enemy_facing_cos
      7.  wall_front           [0, 1]
      8.  wall_left            [0, 1]
      9.  wall_right           [0, 1]
     10.  can_shoot            0 or 1
     11.  own_hp               [0, 1]
     12.  enemy_hp             [0, 1]
     13.  bullet_threat        [0, 1]
     14.  bullet_incoming_sin
     15.  bullet_incoming_cos

    Stacked output (48 floats): [frame_t, frame_t-1, frame_t-2]
    """

    STATE_DIM = STACKED_STATE_DIM  # 48

    def __init__(self):
        self._history: dict[int, deque[list[float]]] = {}

    def _get_history(self, tank_id: int) -> deque[list[float]]:
        if tank_id not in self._history:
            zero_frame = [0.0] * BASE_STATE_DIM
            self._history[tank_id] = deque(
                [zero_frame[:] for _ in range(FRAME_STACK_SIZE)],
                maxlen=FRAME_STACK_SIZE)
        return self._history[tank_id]

    def reset(self) -> None:
        """Clear frame history (call at round/episode start)."""
        self._history.clear()

    def encode(self, tanks, bullets, walls, arena, tank_id: int,
               **kwargs) -> list[float]:
        """Encode current state and return stacked frame vector."""
        frame = self._encode_single_frame(tanks, bullets, walls, arena, tank_id)
        history = self._get_history(tank_id)
        history.append(frame)

        stacked: list[float] = []
        for f in reversed(history):
            stacked.extend(f)
        return stacked

    def _encode_single_frame(self, tanks, bullets, walls, arena,
                              tank_id: int) -> list[float]:
        me = tanks[tank_id]
        enemy = tanks[1 - tank_id]

        # Angles to enemy
        angle_to_enemy = math.atan2(enemy.y - me.y, enemy.x - me.x)
        hull_rel = angle_to_enemy - math.radians(me.angle)
        turret_rel = angle_to_enemy - math.radians(me.turret_angle)

        # Distance
        dist = math.dist((me.x, me.y), (enemy.x, enemy.y))
        dist_norm = min(dist / _ARENA_DIAG, 1.0)

        # Enemy facing
        enemy_angle_to_me = math.atan2(me.y - enemy.y, me.x - enemy.x)
        enemy_rel = enemy_angle_to_me - math.radians(enemy.angle)

        # Wall raycasts
        wall_front = self._cast_wall_ray(me, me.angle, walls)
        wall_left = self._cast_wall_ray(me, me.angle - 90, walls)
        wall_right = self._cast_wall_ray(me, me.angle + 90, walls)

        # Can shoot
        can_shoot = 1.0 if me.can_shoot() else 0.0

        # HP
        max_hp = cfg.TANK_HP
        own_hp = me.hp / max_hp
        enemy_hp = enemy.hp / max_hp

        # Bullet threat + incoming direction
        bullet_t = 0.0
        incoming_sin = 0.0
        incoming_cos = 0.0
        best_dist = float("inf")
        enemy_id = enemy.id
        for b in bullets:
            if b.owner_id != enemy_id or not b.alive:
                continue
            to_x = me.x - b.x
            to_y = me.y - b.y
            dot = to_x * b.dx + to_y * b.dy
            if dot <= 0:
                continue
            d = math.sqrt(to_x * to_x + to_y * to_y)
            if d < best_dist and d < 300:
                best_dist = d
                bullet_t = 1.0 if d < 120 else 0.5
                angle_from = math.atan2(b.y - me.y, b.x - me.x)
                rel = angle_from - math.radians(me.angle)
                incoming_sin = math.sin(rel)
                incoming_cos = math.cos(rel)

        return [
            math.sin(hull_rel),       # 0
            math.cos(hull_rel),       # 1
            math.sin(turret_rel),     # 2
            math.cos(turret_rel),     # 3
            dist_norm,                # 4
            math.sin(enemy_rel),      # 5
            math.cos(enemy_rel),      # 6
            wall_front,               # 7
            wall_left,                # 8
            wall_right,               # 9
            can_shoot,                # 10
            own_hp,                   # 11
            enemy_hp,                 # 12
            bullet_t,                 # 13
            incoming_sin,             # 14
            incoming_cos,             # 15
        ]

    @staticmethod
    def _cast_wall_ray(tank, angle_deg: float, walls) -> float:
        """Raycast returning continuous value: 0=clear, 0.5=nearby, 1=blocked."""
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)
        for d in range(10, 201, 10):
            px = tank.x + dx * d
            py = tank.y + dy * d
            for w in walls:
                if w.rect.collidepoint(px, py):
                    if d <= 50:
                        return 1.0
                    return 0.5
        return 0.0
