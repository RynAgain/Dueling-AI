# ---------------------------------------------------------------------------
# game/wall.py -- Breakable wall block
# ---------------------------------------------------------------------------
import pygame
import config as cfg


class Wall:
    """Axis-aligned destructible wall block."""

    def __init__(self, x: int, y: int):
        self.rect = pygame.Rect(x, y, cfg.WALL_SIZE, cfg.WALL_SIZE)
        self.max_hp: int = cfg.WALL_MAX_HP
        self.hp: int = self.max_hp

    # ------------------------------------------------------------------
    def take_damage(self, amount: int = 1) -> None:
        self.hp = max(0, self.hp - amount)

    def is_destroyed(self) -> bool:
        return self.hp <= 0

    def damage_ratio(self) -> float:
        """1.0 = full health, 0.0 = destroyed."""
        return self.hp / self.max_hp
