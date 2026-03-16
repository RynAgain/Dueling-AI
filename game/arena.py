# ---------------------------------------------------------------------------
# game/arena.py -- Arena boundaries, wall generation, power-up spawn spots
# ---------------------------------------------------------------------------
from __future__ import annotations
import random
import pygame
import config as cfg
from game.wall import Wall
from game.tank import Tank


class Arena:
    """Manages the play-field boundaries and breakable wall layout."""

    def __init__(self):
        self.width: int = cfg.ARENA_WIDTH
        self.height: int = cfg.ARENA_HEIGHT
        self.walls: list[Wall] = []

    # ------------------------------------------------------------------
    def generate_walls(self, count: int | None = None) -> None:
        """Create a semi-symmetric wall layout for a new round.

        Parameters
        ----------
        count : int | None
            Target number of walls to place.
            - 0       -> no walls at all
            - positive int -> try to place that many walls
            - None    -> default random range (WALL_COUNT_MIN .. WALL_COUNT_MAX)
        """
        self.walls.clear()

        if count is not None and count <= 0:
            return  # no walls requested

        if count is None:
            total = random.randint(cfg.WALL_COUNT_MIN, cfg.WALL_COUNT_MAX)
        else:
            total = count

        half = total // 2
        placed: list[pygame.Rect] = []
        margin = 80  # keep walls away from spawn corners
        cx = self.width // 2
        cy = self.height // 2

        for _ in range(half):
            for _attempt in range(40):
                # pick a spot in the left half
                gx = random.randint(margin, cx - cfg.WALL_SIZE)
                gy = random.randint(margin, self.height - margin - cfg.WALL_SIZE)
                # snap to grid for tidiness
                gx = (gx // cfg.WALL_SIZE) * cfg.WALL_SIZE
                gy = (gy // cfg.WALL_SIZE) * cfg.WALL_SIZE
                rect = pygame.Rect(gx, gy, cfg.WALL_SIZE, cfg.WALL_SIZE)
                if not any(rect.colliderect(p) for p in placed):
                    self.walls.append(Wall(gx, gy))
                    placed.append(rect)
                    # mirror on the right side
                    mx = self.width - gx - cfg.WALL_SIZE
                    mirror_rect = pygame.Rect(mx, gy, cfg.WALL_SIZE, cfg.WALL_SIZE)
                    if not any(mirror_rect.colliderect(p) for p in placed):
                        self.walls.append(Wall(mx, gy))
                        placed.append(mirror_rect)
                    break

        # add a few centre walls for cover (only when we have enough budget)
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def get_powerup_spawn_position(self, tanks, walls, powerups, mines) -> tuple[float, float] | None:
        """Find a random open position for a power-up.

        Returns (x, y) or None if no valid spot found after attempts.
        """
        margin = 60
        for _attempt in range(30):
            px = random.randint(margin, self.width - margin)
            py = random.randint(margin, self.height - margin)
            test_rect = pygame.Rect(px - 12, py - 12, 24, 24)
            # Check walls
            if any(test_rect.colliderect(w.rect) for w in walls):
                continue
            # Check tanks
            if any(test_rect.colliderect(t.get_rect()) for t in tanks):
                continue
            # Check existing powerups
            if any(test_rect.colliderect(p.rect) for p in powerups if p.alive):
                continue
            # Check mines
            if any(test_rect.colliderect(m.rect) for m in mines if m.alive):
                continue
            return (float(px), float(py))
        return None

    # ------------------------------------------------------------------
    def remove_destroyed_walls(self) -> list[Wall]:
        """Remove walls with 0 HP and return the list of destroyed ones."""
        destroyed = [w for w in self.walls if w.is_destroyed()]
        self.walls = [w for w in self.walls if not w.is_destroyed()]
        return destroyed

    def clamp_position(self, x: float, y: float,
                       half_w: float, half_h: float) -> tuple[float, float]:
        """Clamp a position so the bounding box stays inside the arena."""
        x = max(half_w, min(self.width - half_w, x))
        y = max(half_h, min(self.height - half_h, y))
        return x, y
