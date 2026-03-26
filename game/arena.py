# ---------------------------------------------------------------------------
# game/arena.py -- Arena boundaries, wall generation
# ---------------------------------------------------------------------------
from __future__ import annotations
import random
import pygame
import config as cfg
from game.wall import Wall


class Arena:
    """Manages the play-field boundaries and breakable wall layout."""

    def __init__(self):
        self.width: int = cfg.ARENA_WIDTH
        self.height: int = cfg.ARENA_HEIGHT
        self.walls: list[Wall] = []

    def generate_walls(self, count: int | None = None) -> None:
        """Create a semi-symmetric wall layout for a new round."""
        self.walls.clear()
        if count is not None and count <= 0:
            return

        if count is None:
            total = random.randint(cfg.WALL_COUNT_MIN, cfg.WALL_COUNT_MAX)
        else:
            total = count

        half = total // 2
        placed: list[pygame.Rect] = []
        margin = 80
        cx = self.width // 2
        cy = self.height // 2

        for _ in range(half):
            for _attempt in range(40):
                gx = random.randint(margin, cx - cfg.WALL_SIZE)
                gy = random.randint(margin, self.height - margin - cfg.WALL_SIZE)
                gx = (gx // cfg.WALL_SIZE) * cfg.WALL_SIZE
                gy = (gy // cfg.WALL_SIZE) * cfg.WALL_SIZE
                rect = pygame.Rect(gx, gy, cfg.WALL_SIZE, cfg.WALL_SIZE)
                if not any(rect.colliderect(p) for p in placed):
                    self.walls.append(Wall(gx, gy))
                    placed.append(rect)
                    mx = self.width - gx - cfg.WALL_SIZE
                    mirror_rect = pygame.Rect(mx, gy, cfg.WALL_SIZE, cfg.WALL_SIZE)
                    if not any(mirror_rect.colliderect(p) for p in placed):
                        self.walls.append(Wall(mx, gy))
                        placed.append(mirror_rect)
                    break

        if total >= 6:
            centre_budget = max(1, total - len(self.walls))
            centre_count = min(random.randint(1, 3), centre_budget)
            for _ in range(centre_count):
                for _attempt in range(30):
                    gx = random.randint(cx - 80, cx + 40)
                    gy = random.randint(cy - 80, cy + 40)
                    gx = (gx // cfg.WALL_SIZE) * cfg.WALL_SIZE
                    gy = (gy // cfg.WALL_SIZE) * cfg.WALL_SIZE
                    rect = pygame.Rect(gx, gy, cfg.WALL_SIZE, cfg.WALL_SIZE)
                    if not any(rect.colliderect(p) for p in placed):
                        self.walls.append(Wall(gx, gy))
                        placed.append(rect)
                        break

    def spawn_positions(self) -> tuple[tuple[float, float, float],
                                        tuple[float, float, float]]:
        """Return (x, y, angle) for each tank (opposite corners, small jitter)."""
        margin = 60
        jitter = 20
        x1 = margin + random.randint(-jitter, jitter)
        y1 = margin + random.randint(-jitter, jitter)
        x2 = self.width - margin + random.randint(-jitter, jitter)
        y2 = self.height - margin + random.randint(-jitter, jitter)
        return (float(x1), float(y1), 45.0), (float(x2), float(y2), 225.0)

    def remove_destroyed_walls(self) -> list[Wall]:
        destroyed = [w for w in self.walls if w.is_destroyed()]
        self.walls = [w for w in self.walls if not w.is_destroyed()]
        return destroyed

    def clamp_position(self, x: float, y: float,
                       half_w: float, half_h: float) -> tuple[float, float]:
        x = max(half_w, min(self.width - half_w, x))
        y = max(half_h, min(self.height - half_h, y))
        return x, y
