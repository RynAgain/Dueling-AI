# ---------------------------------------------------------------------------
# game/game_manager.py -- Core game loop, scoring, episode lifecycle
# ---------------------------------------------------------------------------
from __future__ import annotations
import math
import random
import pygame
import config as cfg
from game.arena import Arena
from game.tank import Tank
from game.bullet import Bullet


# Event keys used in the events dict returned by step()
EVT_HIT = "hit"              # list of (shooter_id, victim_id)  -- kills only
EVT_DAMAGE = "damage"        # list of (shooter_id, victim_id, remaining_hp, bounced)
EVT_WALL_DESTROY = "wall_destroy"  # list of destroyer_id
EVT_BULLET_EXPIRE = "bullet_expire"  # list of owner_id (missed shots)
EVT_ROUND_OVER = "round_over"  # bool
EVT_ROUND_WINNER = "round_winner"  # int | None
EVT_DODGE = "dodge"          # list of tank_id that dodged a threatening bullet

# Colour palette for the two tanks
TANK_COLORS = [(60, 120, 220), (220, 60, 60)]  # blue, red

# Module-level caches for hot-path config values
_decode_action = cfg.decode_action
_TURRET_LEFT = cfg.TURRET_LEFT
_TURRET_RIGHT = cfg.TURRET_RIGHT
_FIRE_NONE = cfg.FIRE_NONE
_FIRE_SHOOT = cfg.FIRE_SHOOT
_MOVE_FWD = cfg.MOVE_FORWARD
_MOVE_BWD = cfg.MOVE_BACKWARD
_MOVE_ROTL = cfg.MOVE_ROTATE_LEFT
_MOVE_ROTR = cfg.MOVE_ROTATE_RIGHT
_TANK_HW = cfg.TANK_WIDTH / 2
_TANK_HH = cfg.TANK_HEIGHT / 2
_TANK_W = cfg.TANK_WIDTH
_TANK_H = cfg.TANK_HEIGHT
_ROUND_TIMEOUT = cfg.ROUND_TIMEOUT
_SHRINK_START = int(cfg.ROUND_TIMEOUT * cfg.SHRINK_START_RATIO)
_SHRINK_FORCE = cfg.SHRINK_FORCE
_POINTS_TO_WIN = cfg.POINTS_TO_WIN
_HALF_TO_WIN = cfg.POINTS_TO_WIN // 2
_math_dist = math.dist


class GameManager:
    """Orchestrates rounds and episodes, advances physics, detects collisions."""

    def __init__(self):
        self.arena = Arena()
        self.tanks: list[Tank] = []
        self.bullets: list[Bullet] = []
        self.scores: dict[int, int] = {0: 0, 1: 0}
        self.current_episode: int = 0
        self.current_round: int = 0
        self.tick_count: int = 0
        self._prev_positions: dict[int, tuple[float, float]] = {}
        self._prev_dist: float = 0.0
        self._wall_count: int | None = None

        # Spawn positions for movement reward
        self.spawn_positions_xy: dict[int, tuple[float, float]] = {}

        self._reset_tanks_and_arena()

    # ==================================================================
    # Setup helpers
    # ==================================================================
    def set_wall_count(self, count: int | None) -> None:
        self._wall_count = count

    def _reset_tanks_and_arena(self) -> None:
        self.arena.generate_walls(count=self._wall_count)
        self.bullets.clear()
        pos1, pos2 = self.arena.spawn_positions()
        self.tanks = [
            Tank(0, pos1[0], pos1[1], pos1[2], TANK_COLORS[0]),
            Tank(1, pos2[0], pos2[1], pos2[2], TANK_COLORS[1]),
        ]
        self.tick_count = 0
        t0, t1 = self.tanks[0], self.tanks[1]
        self._prev_dist = _math_dist((t0.x, t0.y), (t1.x, t1.y))
        self._prev_positions = {0: (t0.x, t0.y), 1: (t1.x, t1.y)}
        self.spawn_positions_xy = {0: (t0.x, t0.y), 1: (t1.x, t1.y)}

    def new_episode(self) -> None:
        self.current_episode += 1
        self.scores = {0: 0, 1: 0}
        self.current_round = 0
        self._reset_tanks_and_arena()

    def new_round(self) -> None:
        self.current_round += 1
        self._reset_tanks_and_arena()

    # ==================================================================
    # Episode / round queries
    # ==================================================================
    def episode_over(self) -> bool:
        s0 = self.scores[0]
        s1 = self.scores[1]
        if s0 + s1 >= _POINTS_TO_WIN:
            return True
        if s0 > _HALF_TO_WIN or s1 > _HALF_TO_WIN:
            return True
        return False

    def get_episode_winner(self) -> int | None:
        s0 = self.scores[0]
        s1 = self.scores[1]
        if s0 > s1:
            return 0
        if s1 > s0:
            return 1
        return None

    # ==================================================================
    # Core tick  (HOT PATH)
    # ==================================================================
    def step(self, action0: int, action1: int) -> dict:
        """Advance one tick.  Returns an events dict."""
        evt_hit: list = []
        evt_damage: list = []
        evt_wall_destroy: list = []
        evt_bullet_expire: list = []
        evt_dodge: list = []

        move0, turret0, fire0 = _decode_action(action0)
        move1, turret1, fire1 = _decode_action(action1)

        tanks = self.tanks
        t0 = tanks[0]
        t1 = tanks[1]
        arena = self.arena
        walls = arena.walls

        # 1. Process hull movement / rotation
        if move0 == _MOVE_FWD:
            nx, ny = t0.move_forward()
            if self._valid_pos(t0, nx, ny, t1, walls, arena):
                t0.x, t0.y = nx, ny
        elif move0 == _MOVE_BWD:
            nx, ny = t0.move_backward()
            if self._valid_pos(t0, nx, ny, t1, walls, arena):
                t0.x, t0.y = nx, ny
        elif move0 == _MOVE_ROTL:
            t0.rotate_left()
        elif move0 == _MOVE_ROTR:
            t0.rotate_right()

        if move1 == _MOVE_FWD:
            nx, ny = t1.move_forward()
            if self._valid_pos(t1, nx, ny, t0, walls, arena):
                t1.x, t1.y = nx, ny
        elif move1 == _MOVE_BWD:
            nx, ny = t1.move_backward()
            if self._valid_pos(t1, nx, ny, t0, walls, arena):
                t1.x, t1.y = nx, ny
        elif move1 == _MOVE_ROTL:
            t1.rotate_left()
        elif move1 == _MOVE_ROTR:
            t1.rotate_right()

        # 2. Process turret rotation
        if turret0 == _TURRET_LEFT:
            t0.rotate_turret_left()
        elif turret0 == _TURRET_RIGHT:
            t0.rotate_turret_right()
        if turret1 == _TURRET_LEFT:
            t1.rotate_turret_left()
        elif turret1 == _TURRET_RIGHT:
            t1.rotate_turret_right()

        # 3. Clamp tanks inside arena
        t0.x, t0.y = arena.clamp_position(t0.x, t0.y, _TANK_HW, _TANK_HH)
        t1.x, t1.y = arena.clamp_position(t1.x, t1.y, _TANK_HW, _TANK_HH)

        # 4. Update cooldowns
        t0.update_cooldown()
        t1.update_cooldown()

        # 5. Spawn bullets
        bullets = self.bullets
        if fire0 == _FIRE_SHOOT:
            b = t0.shoot()
            if b is not None:
                bullets.append(b)
        if fire1 == _FIRE_SHOOT:
            b = t1.shoot()
            if b is not None:
                bullets.append(b)

        # 6. Move bullets
        aw = arena.width
        ah = arena.height
        for b in bullets:
            b.x += b.dx * b.speed
            b.y += b.dy * b.speed
            b.lifetime -= 1
            if b.lifetime <= 0:
                b.alive = False
            elif not b.try_bounce(aw, ah):
                pass

        # 7. Bullet collisions
        self._check_bullet_collisions(bullets, walls, tanks,
                                       evt_hit, evt_damage, evt_wall_destroy)

        # 8. Remove destroyed walls
        arena.remove_destroyed_walls()

        # 9. Threat-tag alive bullets
        under_threat: set[int] = set()
        _THREAT_LOOKAHEAD = cfg.THREAT_LOOKAHEAD_TICKS
        for b in bullets:
            if not b.alive:
                continue
            enemy_tank = t1 if b.owner_id == 0 else t0
            target_id = enemy_tank.id
            bx, by = b.x, b.y
            bdx, bdy = b.dx * b.speed, b.dy * b.speed
            trect = pygame.Rect(enemy_tank.x - _TANK_HW, enemy_tank.y - _TANK_HH,
                                _TANK_W, _TANK_H)
            br = cfg.BULLET_RADIUS
            for step in range(1, _THREAT_LOOKAHEAD + 1):
                px = bx + bdx * step
                py = by + bdy * step
                brect = pygame.Rect(px - br, py - br, br * 2, br * 2)
                if brect.colliderect(trect):
                    b.threatened_tanks.add(target_id)
                    under_threat.add(target_id)
                    break

        # 10. Collect expired + prune dead bullets + dodge check
        bc0 = 0
        bc1 = 0
        alive_bullets: list[Bullet] = []
        for b in bullets:
            if b.alive:
                alive_bullets.append(b)
                if b.owner_id == 0:
                    bc0 += 1
                else:
                    bc1 += 1
            else:
                if b.lifetime <= 0:
                    evt_bullet_expire.append(b.owner_id)
                for threatened_tid in b.threatened_tanks:
                    if threatened_tid not in b.hit_tanks:
                        evt_dodge.append(threatened_tid)
        self.bullets = alive_bullets
        t0.bullet_count = bc0
        t1.bullet_count = bc1

        # 11. Per-tank closer shaping
        prev0 = self._prev_positions.get(0, (t0.x, t0.y))
        prev1 = self._prev_positions.get(1, (t1.x, t1.y))
        prev_d0 = _math_dist(prev0, (t1.x, t1.y))
        prev_d1 = _math_dist(prev1, (t0.x, t0.y))
        curr_d = _math_dist((t0.x, t0.y), (t1.x, t1.y))
        closer0 = curr_d < prev_d0
        closer1 = curr_d < prev_d1
        self._prev_dist = curr_d
        self._prev_positions = {0: (t0.x, t0.y), 1: (t1.x, t1.y)}

        # 12. Shrink phase
        self.tick_count += 1
        if self.tick_count >= _SHRINK_START:
            cx = arena.width / 2
            cy = arena.height / 2
            progress = (self.tick_count - _SHRINK_START) / max(_ROUND_TIMEOUT - _SHRINK_START, 1)
            force = _SHRINK_FORCE * (1.0 + progress * 2.0)
            for tank in tanks:
                dx = cx - tank.x
                dy = cy - tank.y
                dist_to_center = math.sqrt(dx * dx + dy * dy)
                if dist_to_center > 10:
                    tank.x += (dx / dist_to_center) * force
                    tank.y += (dy / dist_to_center) * force

        # 13. Scoring / round-end
        round_over = False
        round_winner = None
        for tank in tanks:
            if not tank.is_alive:
                round_over = True
                round_winner = 1 - tank.id
                self.scores[round_winner] = self.scores.get(round_winner, 0) + 1
                break

        if not round_over and self.tick_count >= _ROUND_TIMEOUT:
            round_over = True
            if t0.hp != t1.hp:
                round_winner = 0 if t0.hp > t1.hp else 1
                self.scores[round_winner] = self.scores.get(round_winner, 0) + 1

        return {
            EVT_HIT: evt_hit,
            EVT_DAMAGE: evt_damage,
            EVT_WALL_DESTROY: evt_wall_destroy,
            EVT_BULLET_EXPIRE: evt_bullet_expire,
            EVT_ROUND_OVER: round_over,
            EVT_ROUND_WINNER: round_winner,
            EVT_DODGE: evt_dodge,
            "under_threat": under_threat,
            "closer": {0: closer0, 1: closer1},
        }

    # ==================================================================
    # Collision helpers
    # ==================================================================
    @staticmethod
    def _valid_pos(tank: Tank, nx: float, ny: float,
                   other: Tank, walls, arena) -> bool:
        hw = _TANK_HW
        hh = _TANK_HH
        if nx - hw < 0 or nx + hw > arena.width or ny - hh < 0 or ny + hh > arena.height:
            return False
        proposed = pygame.Rect(nx - hw, ny - hh, _TANK_W, _TANK_H)
        for w in walls:
            if proposed.colliderect(w.rect):
                return False
        other_rect = pygame.Rect(other.x - hw, other.y - hh, _TANK_W, _TANK_H)
        if proposed.colliderect(other_rect):
            return False
        return True

    @staticmethod
    def _check_bullet_collisions(bullets: list[Bullet], walls,
                                  tanks: list[Tank],
                                  evt_hit: list, evt_damage: list,
                                  evt_wall_destroy: list) -> None:
        br = cfg.BULLET_RADIUS
        br2 = br * 2
        for b in bullets:
            if not b.alive:
                continue
            brect = pygame.Rect(b.x - br, b.y - br, br2, br2)
            hit_wall = False
            for w in walls:
                if brect.colliderect(w.rect):
                    w.take_damage(1)
                    b.alive = False
                    hit_wall = True
                    if w.hp <= 0:
                        evt_wall_destroy.append(b.owner_id)
                    break
            if hit_wall:
                continue
            owner = b.owner_id
            for tank in tanks:
                if tank.id == owner or not tank.is_alive:
                    continue
                trect = pygame.Rect(tank.x - _TANK_HW, tank.y - _TANK_HH,
                                    _TANK_W, _TANK_H)
                if brect.colliderect(trect):
                    dead = tank.take_damage(1)
                    b.hit_tanks.add(tank.id)
                    evt_damage.append((owner, tank.id, tank.hp, b.has_bounced))
                    if dead:
                        evt_hit.append((owner, tank.id))
                    b.alive = False
                    break
