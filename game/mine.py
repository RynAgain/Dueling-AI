# ---------------------------------------------------------------------------
# game/mine.py -- Deployable mine
# ---------------------------------------------------------------------------
from __future__ import annotations
import pygame
import config as cfg


class Mine:
    """A mine placed by a tank that detonates on enemy contact after arming.

    Mines never hurt their owner -- only the enemy tank triggers them.
    """

    def __init__(self, x: float, y: float, owner_id: int):
        self.x: float = x
        self.y: float = y
        self.owner_id: int = owner_id
        self.rect: pygame.Rect = pygame.Rect(x - 8, y - 8, 16, 16)
        self.alive: bool = True
        self.arm_delay: int = cfg.MINE_ARM_DELAY  # ticks before mine becomes active
        self.armed: bool = False
        self.lifetime: int = cfg.MINE_LIFETIME  # despawn after this many ticks
        self.age: int = 0  # ticks since placed

    def update(self) -> None:
        """Advance one tick."""
        self.age += 1
        if not self.armed and self.age >= self.arm_delay:
            self.armed = True
        if self.age >= self.lifetime:
            self.alive = False
