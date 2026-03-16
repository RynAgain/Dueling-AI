# ---------------------------------------------------------------------------
# rendering/renderer.py -- All Pygame draw calls (zero game logic)
# ---------------------------------------------------------------------------
from __future__ import annotations
import math
import pygame
import config as cfg


# Colour palette
COL_BG = (40, 44, 40)
COL_HUD_BG = (20, 22, 20)
COL_TEXT = (220, 220, 220)
COL_SPEED = (120, 255, 120)
COL_PHASE = (255, 200, 80)
COL_WALL_FULL = (139, 90, 43)
COL_WALL_MED = (170, 130, 80)
COL_WALL_LOW = (200, 170, 130)
COL_WALL_BORDER = (80, 50, 20)
COL_TANK_1 = (60, 120, 220)
COL_TANK_2 = (220, 60, 60)
COL_TURRET_1 = (140, 190, 255)
COL_TURRET_2 = (255, 180, 80)
COL_BULLET_1 = (0, 240, 255)
COL_BULLET_2 = (255, 240, 0)
COL_HP_GREEN = (60, 200, 60)
COL_HP_YELLOW = (220, 200, 40)
COL_HP_RED = (220, 50, 50)
COL_HP_BG = (60, 60, 60)
COL_SHIELD = (0, 220, 220)
COL_MINE_ARMED = (80, 80, 80)
COL_MINE_UNARMED = (160, 160, 160)
COL_POWERUP_SPEED = (240, 240, 40)
COL_POWERUP_RAPID = (255, 160, 40)
COL_POWERUP_SHIELD = (40, 220, 220)
COL_FLASH_WHITE = (255, 255, 255)


class Renderer:
    """Handles all visual output via Pygame."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT))
        pygame.display.set_caption("Tank Duel AI")
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_tiny = pygame.font.SysFont("consolas", 10)
        self.clock = pygame.time.Clock()

    # ==================================================================
    def draw_frame(self, game_manager, agents, fast: bool = False,
                   stats: dict | None = None) -> None:
        """Render one complete frame."""
        if fast:
            return
        self.screen.fill(COL_BG)

        # offset everything by HUD_HEIGHT so arena starts below the bar
        offset_y = cfg.HUD_HEIGHT

        # walls
        for wall in game_manager.arena.walls:
            self._draw_wall(wall, offset_y)

        # power-ups
        for pu in game_manager.powerups:
            if pu.alive:
                self._draw_powerup(pu, offset_y)

        # mines
        for mine in game_manager.mines:
            if mine.alive:
                self._draw_mine(mine, offset_y)

        # tanks
        for tank in game_manager.tanks:
            self._draw_tank(tank, offset_y)

        # bullets
        for bullet in game_manager.bullets:
            self._draw_bullet(bullet, offset_y)

        # HUD
        self._draw_hud(game_manager, agents, stats)

        pygame.display.flip()

    # ==================================================================
    # Individual draw helpers
    # ==================================================================
    def _draw_wall(self, wall, oy: int) -> None:
        ratio = wall.damage_ratio()
        if ratio >= 1.0:
            col = COL_WALL_FULL
        elif ratio >= 0.66:
            col = COL_WALL_MED
        else:
            col = COL_WALL_LOW

        r = wall.rect.move(0, oy)
        pygame.draw.rect(self.screen, col, r)
        pygame.draw.rect(self.screen, COL_WALL_BORDER, r, 1)

        # draw crack lines for damaged walls
        if wall.hp == 2:
            cx, cy = r.center
            pygame.draw.line(self.screen, COL_WALL_BORDER,
                             (r.left + 4, cy - 4), (r.right - 4, cy + 4), 1)
        elif wall.hp == 1:
            cx, cy = r.center
            pygame.draw.line(self.screen, COL_WALL_BORDER,
                             (r.left + 3, r.top + 3), (r.right - 3, r.bottom - 3), 1)
            pygame.draw.line(self.screen, COL_WALL_BORDER,
                             (r.right - 3, r.top + 3), (r.left + 3, r.bottom - 3), 1)

    # ------------------------------------------------------------------
    def _draw_tank(self, tank, oy: int) -> None:
        tx = int(tank.x)
        ty = int(tank.y) + oy

        # Flash white when damaged
        if tank.flash_timer > 0:
            body_col = COL_FLASH_WHITE
        else:
            body_col = COL_TANK_1 if tank.id == 0 else COL_TANK_2

        # create a small surface for the tank hull body
        body = pygame.Surface((cfg.TANK_WIDTH, cfg.TANK_HEIGHT), pygame.SRCALPHA)
        body.fill(body_col)

        # rotate hull around centre
        rotated = pygame.transform.rotate(body, -tank.angle)
        rect = rotated.get_rect(center=(tx, ty))
        self.screen.blit(rotated, rect)

        # turret line (independent angle, slightly thicker, different shade)
        turret_col = COL_TURRET_1 if tank.id == 0 else COL_TURRET_2
        tip_x, tip_y = tank.get_turret_tip()
        pygame.draw.line(self.screen, turret_col,
                         (tx, ty),
                         (int(tip_x), int(tip_y) + oy), 4)

        # HP bar above tank
        self._draw_hp_bar(tank, tx, ty)

        # Shield indicator (cyan ring around tank)
        if tank.has_shield:
            pygame.draw.circle(self.screen, COL_SHIELD, (tx, ty),
                               int(cfg.TANK_WIDTH * 0.7), 2)

        # Active power-up indicator (small colored dot next to tank)
        if tank.active_powerup is not None:
            pu_col = self._powerup_indicator_color(tank.active_powerup)
            pygame.draw.circle(self.screen, pu_col,
                               (tx + cfg.TANK_WIDTH // 2 + 6, ty - cfg.TANK_HEIGHT // 2 - 2),
                               4)

    # ------------------------------------------------------------------
    def _draw_hp_bar(self, tank, tx: int, ty: int) -> None:
        """Draw a small HP bar above the tank."""
        bar_w = 28
        bar_h = 4
        bar_x = tx - bar_w // 2
        bar_y = ty - cfg.TANK_HEIGHT // 2 - 8

        # Background
        pygame.draw.rect(self.screen, COL_HP_BG,
                         (bar_x, bar_y, bar_w, bar_h))
        # HP fill
        hp_ratio = tank.hp / tank.max_hp
        fill_w = int(bar_w * hp_ratio)
        if hp_ratio > 0.66:
            col = COL_HP_GREEN
        elif hp_ratio > 0.33:
            col = COL_HP_YELLOW
        else:
            col = COL_HP_RED
        if fill_w > 0:
            pygame.draw.rect(self.screen, col,
                             (bar_x, bar_y, fill_w, bar_h))

    # ------------------------------------------------------------------
    def _draw_bullet(self, bullet, oy: int) -> None:
        col = COL_BULLET_1 if bullet.owner_id == 0 else COL_BULLET_2
        pygame.draw.circle(self.screen, col,
                           (int(bullet.x), int(bullet.y) + oy),
                           cfg.BULLET_RADIUS)

    # ------------------------------------------------------------------
    def _draw_powerup(self, pu, oy: int) -> None:
        """Draw power-up as a colored diamond."""
        cx = int(pu.x)
        cy = int(pu.y) + oy
        size = 10
        if pu.kind == "speed":
            col = COL_POWERUP_SPEED
        elif pu.kind == "rapid":
            col = COL_POWERUP_RAPID
        elif pu.kind == "shield":
            col = COL_POWERUP_SHIELD
        else:
            col = (200, 200, 200)
        # Diamond shape
        points = [(cx, cy - size), (cx + size, cy),
                  (cx, cy + size), (cx - size, cy)]
        pygame.draw.polygon(self.screen, col, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 1)

    # ------------------------------------------------------------------
    def _draw_mine(self, mine, oy: int) -> None:
        """Draw mine as a small circle with X when armed."""
        cx = int(mine.x)
        cy = int(mine.y) + oy
        radius = 7
        col = COL_MINE_ARMED if mine.armed else COL_MINE_UNARMED
        pygame.draw.circle(self.screen, col, (cx, cy), radius)
        if mine.armed:
            # X pattern
            off = 4
            pygame.draw.line(self.screen, (200, 40, 40),
                             (cx - off, cy - off), (cx + off, cy + off), 1)
            pygame.draw.line(self.screen, (200, 40, 40),
                             (cx + off, cy - off), (cx - off, cy + off), 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _powerup_indicator_color(kind: str) -> tuple:
        if kind == "speed":
            return COL_POWERUP_SPEED
        if kind == "rapid":
            return COL_POWERUP_RAPID
        if kind == "shield":
            return COL_POWERUP_SHIELD
        return (200, 200, 200)

    # ------------------------------------------------------------------
    def _draw_hud(self, gm, agents, stats: dict | None = None) -> None:
        hud_rect = pygame.Rect(0, 0, cfg.SCREEN_WIDTH, cfg.HUD_HEIGHT)
        pygame.draw.rect(self.screen, COL_HUD_BG, hud_rect)

        eps_str = f"{agents[0].epsilon:.3f}" if agents else "---"

        # Determine curriculum phase label
        phase_str = ""
        if stats:
            ep = stats.get("ep", 0)
            if ep <= cfg.CURRICULUM_PHASE_1_END:
                phase_str = "P1:NoWall"
            elif ep <= cfg.CURRICULUM_PHASE_2_END:
                phase_str = "P2:FewWall"
            else:
                phase_str = "P3:Full"

        # Tank HP info for HUD
        t0_hp = gm.tanks[0].hp if gm.tanks else 0
        t1_hp = gm.tanks[1].hp if len(gm.tanks) > 1 else 0
        t0_pu = gm.tanks[0].active_powerup or "" if gm.tanks else ""
        t1_pu = gm.tanks[1].active_powerup or "" if len(gm.tanks) > 1 else ""
        t0_mines = gm.tanks[0].mine_count if gm.tanks else 0
        t1_mines = gm.tanks[1].mine_count if len(gm.tanks) > 1 else 0

        parts = [
            f"B:{gm.scores[0]} HP:{t0_hp} M:{t0_mines} {t0_pu[:3]}",
            f"Ep:{gm.current_episode} Rnd:{gm.current_round}",
            f"Eps:{eps_str}",
            f"R:{gm.scores[1]} HP:{t1_hp} M:{t1_mines} {t1_pu[:3]}",
        ]
        total_w = cfg.SCREEN_WIDTH
        segment = total_w // len(parts)
        for i, text in enumerate(parts):
            surf = self.font.render(text, True, COL_TEXT)
            x = segment * i + (segment - surf.get_width()) // 2
            y = (cfg.HUD_HEIGHT - surf.get_height()) // 2
            self.screen.blit(surf, (x, y))

        # Curriculum phase indicator (top-left corner, below score line)
        if phase_str:
            phase_surf = self.font_small.render(phase_str, True, COL_PHASE)
            self.screen.blit(phase_surf, (4, cfg.HUD_HEIGHT - phase_surf.get_height() - 2))

        # Speed indicator (bottom-right of HUD bar)
        if stats:
            ep = stats.get("ep", 0)
            total = stats.get("total", 0)
            eps_sec = stats.get("eps_sec", 0)
            speed_text = f"{ep}/{total} | {eps_sec:.1f} ep/s"
            surf = self.font_small.render(speed_text, True, COL_SPEED)
            x = cfg.SCREEN_WIDTH - surf.get_width() - 6
            y = cfg.HUD_HEIGHT - surf.get_height() - 2
            self.screen.blit(surf, (x, y))

    # ------------------------------------------------------------------
    def tick(self, fps: int = cfg.FPS) -> None:
        self.clock.tick(fps)

    @staticmethod
    def quit() -> None:
        pygame.quit()
