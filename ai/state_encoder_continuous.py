# ---------------------------------------------------------------------------
# ai/state_encoder_continuous.py -- Continuous state vector for DQN
# ---------------------------------------------------------------------------
"""Encodes the game state as a fixed-length float vector (no discretization).

Each component is normalized to roughly [0, 1] or [-1, 1] range for
neural network input stability.  The DQN sees continuous values
instead of bins, so it can learn fine-grained distinctions.

Frame stacking: the encoder stores the last N frames per tank and returns
them concatenated, so the DQN can perceive velocity, acceleration,
and opponent behavior patterns.
"""
from __future__ import annotations
import math
from collections import deque
import config as cfg

# Arena diagonal for distance normalization
_ARENA_DIAG = math.sqrt(cfg.ARENA_WIDTH ** 2 + cfg.ARENA_HEIGHT ** 2)

# Number of frames to stack (1 = no stacking, 3 = current + 2 history)
FRAME_STACK_SIZE = 3
BASE_STATE_DIM = 21  # single-frame vector size
STACKED_STATE_DIM = BASE_STATE_DIM * FRAME_STACK_SIZE  # 63


class ContinuousStateEncoder:
    """Encodes game state into a flat float list for DQN input.

    With frame stacking enabled (FRAME_STACK_SIZE > 1), the output is
    the concatenation of the last N frame vectors, giving the DQN
    temporal context to perceive motion and predict trajectories.

    Single-frame vector (21 floats):
      0.  rel_angle_hull     -- hull-relative angle to enemy, sin
      1.  rel_angle_hull_cos -- ... cos
      2.  rel_angle_turret   -- turret-relative angle to enemy, sin
      3.  rel_angle_turret_cos -- ... cos
      4.  dist_to_enemy      -- distance, normalized [0, 1]
      5.  enemy_facing_sin   -- enemy's facing toward us, sin
      6.  enemy_facing_cos   -- ... cos
      7.  wall_front         -- wall raycast front [0, 1]
      8.  wall_left          -- wall raycast left
      9.  wall_right         -- wall raycast right
     10.  can_shoot          -- 0 or 1
     11.  own_hp             -- normalized [0, 1]
     12.  enemy_hp           -- normalized [0, 1]
     13.  active_powerup_speed  -- 0 or 1
     14.  active_powerup_rapid  -- 0 or 1
     15.  active_powerup_shield -- 0 or 1
     16.  nearby_powerup     -- 0 or 1
     17.  mine_threat         -- [0, 1]
     18.  bullet_threat       -- [0, 1]
     19.  bullet_incoming_sin -- sin of incoming bullet angle
     20.  bullet_incoming_cos -- cos of incoming bullet angle

    Stacked output (63 floats): [frame_t, frame_t-1, frame_t-2]
    """

    STATE_DIM = STACKED_STATE_DIM  # 63 (used by DQN to size its input layer)

    def __init__(self):
        # Per-tank frame history: deque of recent frames
        self._history: dict[int, deque[list[float]]] = {}

    def _get_history(self, tank_id: int) -> deque[list[float]]:
        if tank_id not in self._history:
            # Initialize with zeros (agent has no memory before game starts)
            zero_frame = [0.0] * BASE_STATE_DIM
            self._history[tank_id] = deque(
                [zero_frame[:] for _ in range(FRAME_STACK_SIZE)],
                maxlen=FRAME_STACK_SIZE)
        return self._history[tank_id]

    def reset(self) -> None:
        """Clear frame history (call at round/episode start)."""
        self._history.clear()

    def encode(self, tanks, bullets, walls, arena, tank_id: int,
               powerups=None, mines=None) -> list[float]:
        """Encode current state and return stacked frame vector."""
        frame = self._encode_single_frame(tanks, bullets, walls, arena,
                                           tank_id, powerups, mines)
        history = self._get_history(tank_id)
        history.append(frame)

        # Concatenate: most recent first, then older frames
        stacked: list[float] = []
        # history is [oldest, ..., newest], we want [newest, ..., oldest]
        for f in reversed(history):
            stacked.extend(f)
        return stacked

    def _encode_single_frame(self, tanks, bullets, walls, arena,
                              tank_id: int, powerups=None,
                              mines=None) -> list[float]:
        """Encode a single frame (21 floats)."""
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

        # Wall raycasts (0=clear, 0.5=nearby, 1=blocked)
        wall_front = self._cast_wall_ray(me, me.angle, walls)
        wall_left = self._cast_wall_ray(me, me.angle - 90, walls)
        wall_right = self._cast_wall_ray(me, me.angle + 90, walls)

        # Can shoot
        can_shoot = 1.0 if me.can_shoot() else 0.0

        # HP
        max_hp = cfg.TANK_HP
        own_hp = me.hp / max_hp
        enemy_hp = enemy.hp / max_hp

        # Power-ups (one-hot)
        pu = me.active_powerup
        pu_speed = 1.0 if pu == "speed" else 0.0
        pu_rapid = 1.0 if pu == "rapid" else 0.0
        pu_shield = 1.0 if pu == "shield" else 0.0

        # Nearby power-up
        nearby_pu = 0.0
        if powerups:
            for p in powerups:
                if p.alive and math.dist((me.x, me.y), (p.x, p.y)) < 200:
                    nearby_pu = 1.0
                    break

        # Mine threat
        mine_t = 0.0
        if mines:
            for m in mines:
                if m.alive and m.armed:
                    d = math.dist((me.x, me.y), (m.x, m.y))
                    if d < 80:
                        mine_t = 1.0
                        break
                    elif d < 200:
                        mine_t = max(mine_t, 0.5)

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
                # Threat level
                if d < 120:
                    bullet_t = 1.0
                else:
                    bullet_t = max(bullet_t, 0.5)
                # Direction bullet comes from, relative to hull
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
            pu_speed,                 # 13
            pu_rapid,                 # 14
            pu_shield,                # 15
            nearby_pu,                # 16
            mine_t,                   # 17
            bullet_t,                 # 18
            incoming_sin,             # 19
            incoming_cos,             # 20
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
