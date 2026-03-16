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
        # Outcome-based dodge tracking: set of tank ids this bullet was
        # projected to hit at some point during its flight.
        self.threatened_tanks: set[int] = set()
        # Set of tank ids this bullet actually hit (damage or kill).
        self.hit_tanks: set[int] = set()

    # ------------------------------------------------------------------
    def update(self) -> None:
        """Advance one tick."""
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

    def try_bounce(self, arena_w: int, arena_h: int) -> bool:
        """Check arena boundary collision and bounce or destroy.

        Returns True if the bullet is still alive after the check.
        """
        bounced = False
        # Left/right walls
        if self.x < 0 or self.x > arena_w:
            if self.bounces_remaining > 0:
                self.dx = -self.dx
                # Clamp position back inside
                if self.x < 0:
                    self.x = -self.x
                else:
                    self.x = 2 * arena_w - self.x
                self.bounces_remaining -= 1
                bounced = True
            else:
                self.alive = False
                return False
        # Top/bottom walls
        if self.y < 0 or self.y > arena_h:
            if self.bounces_remaining > 0:
                self.dy = -self.dy
                if self.y < 0:
                    self.y = -self.y
                else:
                    self.y = 2 * arena_h - self.y
                self.bounces_remaining -= 1
                bounced = True
            else:
                self.alive = False
                return False
        return True

    def get_rect(self) -> pygame.Rect:
        r = cfg.BULLET_RADIUS
        return pygame.Rect(self.x - r, self.y - r, r * 2, r * 2)

    def is_out_of_bounds(self) -> bool:
        return (
            self.x < 0
            or self.x > cfg.ARENA_WIDTH
            or self.y < 0
            or self.y > cfg.ARENA_HEIGHT
        )
