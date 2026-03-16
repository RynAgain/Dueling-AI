# ---------------------------------------------------------------------------
# game/tank.py -- Player tank with HP, independent turret, and mine support
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
        self.bullet_count: int = 0  # live bullets owned by this tank

        # HP system
        self.max_hp: int = cfg.TANK_HP
        self.hp: int = self.max_hp
        self.flash_timer: int = 0  # frames to flash white when hit

        # Power-up state
        self.active_powerup: str | None = None  # "speed", "rapid", "shield"
        self.powerup_timer: int = 0
        self.has_shield: bool = False  # shield absorbs one hit

        # Mine tracking
        self.mine_count: int = 0  # live mines owned by this tank

    # -- HP ---------------------------------------------------------------
    def take_damage(self, amount: int = 1) -> bool:
        """Reduce HP. Returns True if the tank is dead (HP <= 0)."""
        # Shield absorbs hit
        if self.has_shield:
            self.has_shield = False
            if self.active_powerup == "shield":
                self.active_powerup = None
                self.powerup_timer = 0
            return False
        self.hp = max(0, self.hp - amount)
        if self.hp > 0:
            self.flash_timer = 5  # flash for 5 frames
        return self.hp <= 0

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def reset_hp(self) -> None:
        """Reset HP to full for a new round."""
        self.hp = self.max_hp
        self.flash_timer = 0
        self.alive = True
        self.active_powerup = None
        self.powerup_timer = 0
        self.has_shield = False
        self.mine_count = 0

    # -- power-up application --------------------------------------------
    def apply_powerup(self, kind: str) -> None:
        """Apply a power-up effect to this tank."""
        # Clear previous power-up
        self._clear_powerup_effects()
        self.active_powerup = kind
        self.powerup_timer = cfg.POWERUP_DURATION
        if kind == "speed":
            pass  # speed multiplier applied in movement methods
        elif kind == "rapid":
            pass  # cooldown halving applied in can_shoot / shoot
        elif kind == "shield":
            self.has_shield = True

    def update_powerup(self) -> None:
        """Tick down power-up timer."""
        if self.powerup_timer > 0:
            self.powerup_timer -= 1
            if self.powerup_timer <= 0:
                self._clear_powerup_effects()
                self.active_powerup = None

    def _clear_powerup_effects(self) -> None:
        """Remove active power-up effects."""
        if self.active_powerup == "shield":
            self.has_shield = False

    # -- movement --------------------------------------------------------
    def _effective_speed(self) -> float:
        """Return current speed accounting for power-ups."""
        s = self.speed
        if self.active_powerup == "speed":
            s *= cfg.SPEED_BOOST_MULT
        return s

    def move_forward(self) -> tuple[float, float]:
        """Return the proposed new (x, y) after moving forward."""
        rad = math.radians(self.angle)
        spd = self._effective_speed()
        nx = self.x + math.cos(rad) * spd
        ny = self.y + math.sin(rad) * spd
        return nx, ny

    def move_backward(self) -> tuple[float, float]:
        """Return the proposed new (x, y) after moving backward (slower)."""
        rad = math.radians(self.angle)
        bspeed = self._effective_speed() * cfg.TANK_REVERSE_FACTOR
        nx = self.x - math.cos(rad) * bspeed
        ny = self.y - math.sin(rad) * bspeed
        return nx, ny

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
        cd = self.cooldown_timer
        if self.active_powerup == "rapid":
            # With rapid fire, effective cooldown check is halved
            # But we still use the same timer -- just it was set to half
            pass
        return cd <= 0 and self.bullet_count < cfg.MAX_BULLETS_PER_TANK

    def shoot(self) -> Bullet | None:
        if not self.can_shoot():
            return None
        tip_x, tip_y = self.get_turret_tip()
        bullet = Bullet(tip_x, tip_y, self.turret_angle, self.id)
        # Rapid fire halves cooldown
        cd = cfg.SHOOT_COOLDOWN
        if self.active_powerup == "rapid":
            cd = cd // 2
        self.cooldown_timer = cd
        self.bullet_count += 1
        return bullet

    def update_cooldown(self) -> None:
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -- geometry --------------------------------------------------------
    def get_turret_tip(self) -> tuple[float, float]:
        """Point at the front of the turret, based on turret_angle."""
        rad = math.radians(self.turret_angle)
        tip_dist = cfg.TANK_WIDTH // 2 + 4
        return (self.x + math.cos(rad) * tip_dist,
                self.y + math.sin(rad) * tip_dist)

    def get_rect(self) -> pygame.Rect:
        """AABB around the tank centre (non-rotated bounding box)."""
        hw = cfg.TANK_WIDTH / 2
        hh = cfg.TANK_HEIGHT / 2
        return pygame.Rect(self.x - hw, self.y - hh, cfg.TANK_WIDTH, cfg.TANK_HEIGHT)
