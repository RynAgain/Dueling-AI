# ---------------------------------------------------------------------------
# game/powerup.py -- Collectible power-up items
# ---------------------------------------------------------------------------
from __future__ import annotations
import pygame
import config as cfg


class PowerUp:
    """A collectible power-up that spawns on the field."""

    SPEED_BOOST = "speed"     # 1.5x movement speed for duration
    RAPID_FIRE = "rapid"      # halve cooldown for duration
    SHIELD = "shield"         # absorb 1 hit without losing HP

    KINDS = [SPEED_BOOST, RAPID_FIRE, SHIELD]

    def __init__(self, x: float, y: float, kind: str):
        self.x: float = x
        self.y: float = y
        self.kind: str = kind
        self.rect: pygame.Rect = pygame.Rect(x - 12, y - 12, 24, 24)
        self.alive: bool = True
        self.spawn_tick: int = 0
        self.lifetime: int = cfg.POWERUP_LIFETIME  # despawn after this many ticks
