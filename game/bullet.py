# ---------------------------------------------------------------------------
# game/bullet.py -- Projectile fired by a tank (with ricochet support)
# ---------------------------------------------------------------------------
from __future__ import annotations
import math
import pygame
import config as cfg


class Bullet:
    """Straight-line projectile with ricochet off arena walls."""

    def __init__(self, x: float, y: float, angle_deg: float, owner_id: int):
        self.x: float = x
        self.y: float = y
        rad = math.radians(angle_deg)
        self.dx: float = math.cos(rad)
        self.dy: float = math.sin(rad)
        self.speed: float = cfg.BULLET_SPEED
        self.owner_id: int = owner_id
        self.alive: bool = True
        self.lifetime: int = cfg.BULLET_LIFETIME
        self.bounces_remaining: int = cfg.BULLET_MAX_BOUNCES
        self.has_bounced: bool = False  # True once bullet has ricocheted
        # Outcome-based dodge tracking: set of tank ids this bullet was
        # projected to hit at some point during its flight.
        self.threatened_tanks: set[int] = set()
        # Set of tank ids this bullet actually hit (damage or kill).
        self.hit_tanks: set[int] = set()

    # ------------------------------------------------------------------
    def try_bounce(self, arena_w: int, arena_h: int) -> bool:
        """Check arena boundary collision and bounce or destroy.

        Returns True if the bullet is still alive after the check.
        Corner hits (crossing both X and Y boundaries simultaneously)
        are treated as a single bounce event.
        """
        hit_x = self.x < 0 or self.x > arena_w
        hit_y = self.y < 0 or self.y > arena_h

        if not hit_x and not hit_y:
            return True  # no boundary contact

        if self.bounces_remaining <= 0:
            self.alive = False
            return False

        # Reflect whichever axes were crossed (costs only 1 bounce)
        if hit_x:
            self.dx = -self.dx
            if self.x < 0:
                self.x = -self.x
            else:
                self.x = 2 * arena_w - self.x

        if hit_y:
            self.dy = -self.dy
            if self.y < 0:
                self.y = -self.y
            else:
                self.y = 2 * arena_h - self.y

        self.bounces_remaining -= 1
        self.has_bounced = True
        return True

    def get_rect(self) -> pygame.Rect:
        r = cfg.BULLET_RADIUS
        return pygame.Rect(self.x - r, self.y - r, r * 2, r * 2)
