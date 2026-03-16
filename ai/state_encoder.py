# ---------------------------------------------------------------------------
# ai/state_encoder.py -- Discretize continuous game state into Q-table key
# ---------------------------------------------------------------------------
from __future__ import annotations
import math
import config as cfg


class StateEncoder:
    """Converts raw game state into a compact hashable tuple for Q-learning.

    State components (15 total):
      0. bin_angle        -- hull-relative angle to enemy (16 bins)
      1. bin_dist         -- distance to enemy (4 bins)
      2. bin_enemy_facing -- enemy facing direction relative to us (4 bins)
      3. wall_front       -- wall raycast front (3 levels)
      4. wall_left        -- wall raycast left  (3 levels)
      5. wall_right       -- wall raycast right (3 levels)
      6. can_shoot        -- cooldown ready (0/1)
      7. threat           -- nearest bullet threat (3 levels)
      8. own_hp           -- own HP bucket (0=low, 1=mid, 2=full)
      9. enemy_hp         -- enemy HP bucket (0=low, 1=mid, 2=full)
     10. turret_angle_bin -- turret-relative angle to enemy (16 bins)
     11. active_powerup   -- active power-up type (0=none, 1=speed, 2=rapid, 3=shield)
     12. nearby_powerup   -- power-up nearby? (0=no, 1=yes)
     13. mine_threat      -- nearest mine threat (0=none, 1=far, 2=close)
     14. bullet_incoming_dir -- direction the nearest threatening bullet comes FROM
                               relative to tank's hull (0=none, 1-8 = 8 compass dirs)
    """

    def __init__(self):
        self.angle_bins: int = cfg.ANGLE_BINS        # 16
        self.dist_bins: int = cfg.DISTANCE_BINS       # 4
        self.dist_step: float = cfg.DISTANCE_STEP     # 200 px
        self.enemy_bins: int = cfg.ENEMY_FACING_BINS  # 4
        self.wall_ray_bins: int = cfg.WALL_RAY_BINS   # 3
        self.threat_bins: int = cfg.THREAT_BINS        # 3
        # Pre-compute bin width for angle discretisation
        self._angle_bin_width: float = 360.0 / self.angle_bins  # 22.5

    # ==================================================================
    def encode(self, tanks, bullets, walls, arena, tank_id: int,
               powerups=None, mines=None) -> tuple:
        """Return a discretized state tuple for the given tank.

        Parameters
        ----------
        tanks : list[Tank]
        bullets : list[Bullet]
        walls : list[Wall]
        arena : Arena
        tank_id : int  (0 or 1)
        powerups : list[PowerUp] | None
        mines : list[Mine] | None
        """
        me = tanks[tank_id]
        enemy = tanks[1 - tank_id]

        # 1. Relative hull angle to enemy (16 bins of 22.5 deg) ------------
        angle_to_enemy = math.degrees(
            math.atan2(enemy.y - me.y, enemy.x - me.x))
        relative_angle = self._normalize_angle(angle_to_enemy - me.angle)
        bin_angle = int(relative_angle / self._angle_bin_width) % self.angle_bins

        # 2. Distance to enemy (4 bins) ---------------------------------
        dist = math.dist((me.x, me.y), (enemy.x, enemy.y))
        bin_dist = min(int(dist / self.dist_step), self.dist_bins - 1)

        # 3. Enemy facing direction relative to us (4 quadrants) --------
        enemy_angle_to_me = math.degrees(
            math.atan2(me.y - enemy.y, me.x - enemy.x))
        enemy_relative = self._normalize_angle(enemy_angle_to_me - enemy.angle)
        bin_enemy_facing = int(enemy_relative / 90.0) % self.enemy_bins

        # 4-6. Wall raycasts (front, left, right) -----------------------
        wall_front = self._cast_wall_ray(me, me.angle, walls)
        wall_left = self._cast_wall_ray(me, me.angle - 90, walls)
        wall_right = self._cast_wall_ray(me, me.angle + 90, walls)

        # 7. Cooldown ready (boolean) -----------------------------------
        can_shoot = 1 if me.can_shoot() else 0

        # 8. Nearest bullet threat (3 levels) ---------------------------
        threat = self._assess_bullet_threat(me, bullets, enemy.id)

        # 9. Own HP bucket (0=low(1), 1=mid(2), 2=full(3)) -------------
        own_hp = self._hp_bucket(me.hp)

        # 10. Enemy HP bucket -------------------------------------------
        enemy_hp = self._hp_bucket(enemy.hp)

        # 11. Turret-relative angle to enemy (16 bins) ------------------
        turret_relative = self._normalize_angle(angle_to_enemy - me.turret_angle)
        turret_angle_bin = int(turret_relative / self._angle_bin_width) % self.angle_bins

        # 12. Active power-up type (0=none, 1=speed, 2=rapid, 3=shield)
        active_powerup = self._encode_powerup(me.active_powerup)

        # 13. Nearby power-up presence (0=no, 1=yes) -------------------
        nearby_powerup = 0
        if powerups:
            for pu in powerups:
                if pu.alive:
                    d = math.dist((me.x, me.y), (pu.x, pu.y))
                    if d < 200:
                        nearby_powerup = 1
                        break

        # 14. Nearest mine threat (0=none, 1=far, 2=close) -------------
        mine_threat = self._assess_mine_threat(me, mines)

        # 15. Incoming bullet direction (0=none, 1-8 compass dirs) -----
        bullet_incoming_dir = self._incoming_bullet_direction(me, bullets, enemy.id)

        return (bin_angle, bin_dist, bin_enemy_facing,
                wall_front, wall_left, wall_right,
                can_shoot, threat,
                own_hp, enemy_hp, turret_angle_bin,
                active_powerup, nearby_powerup, mine_threat,
                bullet_incoming_dir)

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _normalize_angle(deg: float) -> float:
        """Map angle into [0, 360)."""
        return deg % 360.0

    @staticmethod
    def _hp_bucket(hp: int) -> int:
        """Encode HP as bucket: 0=low(1), 1=mid(2), 2=full(3+)."""
        if hp <= 1:
            return 0
        if hp == 2:
            return 1
        return 2

    @staticmethod
    def _encode_powerup(kind: str | None) -> int:
        """Encode active power-up: 0=none, 1=speed, 2=rapid, 3=shield."""
        if kind is None:
            return 0
        if kind == "speed":
            return 1
        if kind == "rapid":
            return 2
        if kind == "shield":
            return 3
        return 0

    def _cast_wall_ray(self, tank, angle_deg: float, walls) -> int:
        """Simple raycast from tank centre along *angle_deg*.

        Returns 0=clear, 1=nearby wall, 2=blocked (wall very close).
        """
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)
        step = 10
        max_dist = 200
        for d in range(step, max_dist + 1, step):
            px = tank.x + dx * d
            py = tank.y + dy * d
            for w in walls:
                if w.rect.collidepoint(px, py):
                    if d <= 50:
                        return 2  # blocked
                    return 1      # nearby
        return 0  # clear

    @staticmethod
    def _assess_bullet_threat(tank, bullets, enemy_id: int) -> int:
        """How threatening is the nearest enemy bullet?

        Returns 0=none, 1=distant, 2=imminent.
        """
        min_dist = float("inf")
        for b in bullets:
            if b.owner_id != enemy_id or not b.alive:
                continue
            # only care if bullet is heading roughly toward us
            to_tank_x = tank.x - b.x
            to_tank_y = tank.y - b.y
            dot = to_tank_x * b.dx + to_tank_y * b.dy
            if dot <= 0:
                continue  # bullet moving away
            d = math.dist((b.x, b.y), (tank.x, tank.y))
            if d < min_dist:
                min_dist = d
        if min_dist > 300:
            return 0  # none
        if min_dist > 120:
            return 1  # distant
        return 2      # imminent

    @staticmethod
    def _incoming_bullet_direction(tank, bullets, enemy_id: int) -> int:
        """Direction the nearest threatening enemy bullet is coming FROM,
        relative to the tank's hull heading.

        Returns 0 if no threatening bullet, or 1-8 for 8 compass directions
        (1=front, 2=front-right, 3=right, 4=back-right, 5=back, 6=back-left,
         7=left, 8=front-left).  This tells the agent WHICH WAY to move to dodge.
        """
        best_dist = float("inf")
        best_angle = None
        for b in bullets:
            if b.owner_id != enemy_id or not b.alive:
                continue
            to_tank_x = tank.x - b.x
            to_tank_y = tank.y - b.y
            dot = to_tank_x * b.dx + to_tank_y * b.dy
            if dot <= 0:
                continue  # moving away
            d = math.sqrt(to_tank_x * to_tank_x + to_tank_y * to_tank_y)
            if d > 250:
                continue  # too far to matter
            if d < best_dist:
                best_dist = d
                # Angle FROM bullet TO tank, relative to tank's hull heading
                angle_from = math.degrees(math.atan2(b.y - tank.y, b.x - tank.x))
                best_angle = (angle_from - tank.angle) % 360.0
        if best_angle is None:
            return 0  # no threat
        # Quantize into 8 compass bins (each 45 degrees), 1-indexed
        bin_idx = int((best_angle + 22.5) / 45.0) % 8
        return bin_idx + 1  # 1-8

    @staticmethod
    def _assess_mine_threat(tank, mines) -> int:
        """Nearest armed mine threat level.

        Returns 0=none, 1=far, 2=close.
        """
        if not mines:
            return 0
        min_dist = float("inf")
        for m in mines:
            if not m.alive or not m.armed:
                continue
            d = math.dist((tank.x, tank.y), (m.x, m.y))
            if d < min_dist:
                min_dist = d
        if min_dist > 200:
            return 0  # none
        if min_dist > 80:
            return 1  # far
        return 2      # close
