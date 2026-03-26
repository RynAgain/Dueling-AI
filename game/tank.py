# ---------------------------------------------------------------------------
# game/tank.py -- Player tank with HP and independent turret
# ---------------------------------------------------------------------------
from __future__ import annotations
import math
import pygame
import config as cfg
from game.bullet import Bullet


class Tank:
    """Top-down tank controlled by an AI agent."""

    def __init__(self, tank_id: int, x: float, y: float, angle: float, color: tuple):
        self.id: int = tank_id
        self.x: float = x
        self.y: float = y
        self.angle: float = angle  # degrees, 0 = right (hull angle)
        self.turret_angle: float = angle  # independent turret angle
        self.color: tuple = color
        self.speed: float = cfg.TANK_SPEED
        self.rotation_speed: float = cfg.TANK_ROTATION_SPEED
        self.turret_rotation_speed: float = cfg.TURRET_ROTATION_SPEED
        self.cooldown_timer: int = 0
        self.alive: bool = True
        self.bullet_count: int = 0

        # HP system
        self.max_hp: int = cfg.TANK_HP
        self.hp: int = self.max_hp
        self.flash_timer: int = 0

    # -- HP ---------------------------------------------------------------
    def take_damage(self, amount: int = 1) -> bool:
        """Reduce HP. Returns True if the tank is dead (HP <= 0)."""
        self.hp = max(0, self.hp - amount)
        if self.hp > 0:
            self.flash_timer = 5
        return self.hp <= 0

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def reset_hp(self) -> None:
        self.hp = self.max_hp
        self.flash_timer = 0
        self.alive = True

    # -- movement --------------------------------------------------------
    def move_forward(self) -> tuple[float, float]:
        rad = math.radians(self.angle)
        return (self.x + math.cos(rad) * self.speed,
                self.y + math.sin(rad) * self.speed)

    def move_backward(self) -> tuple[float, float]:
        rad = math.radians(self.angle)
        bspeed = self.speed * cfg.TANK_REVERSE_FACTOR
        return (self.x - math.cos(rad) * bspeed,
                self.y - math.sin(rad) * bspeed)

    def rotate_left(self) -> None:
        self.angle = (self.angle - self.rotation_speed) % 360

    def rotate_right(self) -> None:
        self.angle = (self.angle + self.rotation_speed) % 360

    def set_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    # -- turret ----------------------------------------------------------
    def rotate_turret_left(self) -> None:
        self.turret_angle = (self.turret_angle - self.turret_rotation_speed) % 360

    def rotate_turret_right(self) -> None:
        self.turret_angle = (self.turret_angle + self.turret_rotation_speed) % 360

    # -- shooting --------------------------------------------------------
    def can_shoot(self) -> bool:
        return self.cooldown_timer <= 0 and self.bullet_count < cfg.MAX_BULLETS_PER_TANK

    def shoot(self) -> Bullet | None:
        if not self.can_shoot():
            return None
        tip_x, tip_y = self.get_turret_tip()
        bullet = Bullet(tip_x, tip_y, self.turret_angle, self.id)
        self.cooldown_timer = cfg.SHOOT_COOLDOWN
        self.bullet_count += 1
        return bullet

    def update_cooldown(self) -> None:
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -- geometry --------------------------------------------------------
    def get_turret_tip(self) -> tuple[float, float]:
        rad = math.radians(self.turret_angle)
        tip_dist = cfg.TANK_WIDTH // 2 + 4
        return (self.x + math.cos(rad) * tip_dist,
                self.y + math.sin(rad) * tip_dist)

    def get_rect(self) -> pygame.Rect:
        hw = cfg.TANK_WIDTH / 2
        hh = cfg.TANK_HEIGHT / 2
        return pygame.Rect(self.x - hw, self.y - hh, cfg.TANK_WIDTH, cfg.TANK_HEIGHT)
