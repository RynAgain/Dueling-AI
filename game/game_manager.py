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
from game.powerup import PowerUp
from game.mine import Mine


# Event keys used in the events dict returned by step()
EVT_HIT = "hit"              # list of (shooter_id, victim_id)  -- kills only
EVT_DAMAGE = "damage"        # list of (shooter_id, victim_id, remaining_hp)
EVT_WALL_DESTROY = "wall_destroy"  # list of destroyer_id
EVT_BULLET_EXPIRE = "bullet_expire"  # list of owner_id (missed shots)
EVT_ROUND_OVER = "round_over"  # bool
EVT_ROUND_WINNER = "round_winner"  # int | None
EVT_MINE_HIT = "mine_hit"    # list of (mine_owner_id, victim_id)
EVT_DODGE = "dodge"          # list of tank_id that dodged a threatening bullet

# Colour palette for the two tanks
TANK_COLORS = [(60, 120, 220), (220, 60, 60)]  # blue, red

# Module-level caches for hot-path config values
_NUM_FIRE = cfg.NUM_FIRE_OPTIONS
_NUM_TURRET = cfg.NUM_TURRET_OPTIONS
_MOVE_FWD = cfg.MOVE_FORWARD
_MOVE_BWD = cfg.MOVE_BACKWARD
_MOVE_ROTL = cfg.MOVE_ROTATE_LEFT
_MOVE_ROTR = cfg.MOVE_ROTATE_RIGHT
_TURRET_LEFT = cfg.TURRET_LEFT
_TURRET_RIGHT = cfg.TURRET_RIGHT
_FIRE_NONE = cfg.FIRE_NONE
_FIRE_SHOOT = cfg.FIRE_SHOOT
_FIRE_MINE = cfg.FIRE_MINE
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


# Use shared decode from config
_decode_action = cfg.decode_action


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
        self._prev_positions: dict[int, tuple[float, float]] = {}  # per-tank previous pos
        self._prev_dist: float = 0.0  # inter-tank distance (for reference)
        self._wall_count: int | None = None  # curriculum wall count

        # Curriculum feature toggles
        self.powerups_enabled: bool = True
        self.mines_enabled: bool = True

        # Power-ups
        self.powerups: list[PowerUp] = []
        self._powerup_spawn_timer: int = 0
        self._next_powerup_interval: int = random.randint(300, 500)

        # Mines
        self.mines: list[Mine] = []

        # Spawn positions for movement reward
        self.spawn_positions_xy: dict[int, tuple[float, float]] = {}

        self._reset_tanks_and_arena()

    # ==================================================================
    # Setup helpers
    # ==================================================================
    def set_wall_count(self, count: int | None) -> None:
        """Set the wall count for subsequent rounds (curriculum learning)."""
        self._wall_count = count

    def _reset_tanks_and_arena(self) -> None:
        self.arena.generate_walls(count=self._wall_count)
        self.bullets.clear()
        self.powerups.clear()
        self.mines.clear()
        self._powerup_spawn_timer = 0
        self._next_powerup_interval = random.randint(300, 500)
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
        # Pre-allocate events
        evt_hit: list = []
        evt_damage: list = []
        evt_wall_destroy: list = []
        evt_bullet_expire: list = []
        evt_mine_hit: list = []
        evt_dodge: list = []

        # Decode composite actions
        move0, turret0, fire0 = _decode_action(action0)
        move1, turret1, fire1 = _decode_action(action1)

        tanks = self.tanks
        t0 = tanks[0]
        t1 = tanks[1]
        arena = self.arena
        walls = arena.walls

        # 1. Process hull movement / rotation --------------------------------
        if move0 == _MOVE_FWD:
            nx, ny = t0.move_forward()
            if self._valid_pos(t0, nx, ny, t1, walls, arena):
                t0.x = nx
                t0.y = ny
        elif move0 == _MOVE_BWD:
            nx, ny = t0.move_backward()
            if self._valid_pos(t0, nx, ny, t1, walls, arena):
                t0.x = nx
                t0.y = ny
        elif move0 == _MOVE_ROTL:
            t0.rotate_left()
        elif move0 == _MOVE_ROTR:
            t0.rotate_right()

        if move1 == _MOVE_FWD:
            nx, ny = t1.move_forward()
            if self._valid_pos(t1, nx, ny, t0, walls, arena):
                t1.x = nx
                t1.y = ny
        elif move1 == _MOVE_BWD:
            nx, ny = t1.move_backward()
            if self._valid_pos(t1, nx, ny, t0, walls, arena):
                t1.x = nx
                t1.y = ny
        elif move1 == _MOVE_ROTL:
            t1.rotate_left()
        elif move1 == _MOVE_ROTR:
            t1.rotate_right()

        # 2. Process turret rotation -----------------------------------------
        if turret0 == _TURRET_LEFT:
            t0.rotate_turret_left()
        elif turret0 == _TURRET_RIGHT:
            t0.rotate_turret_right()

        if turret1 == _TURRET_LEFT:
            t1.rotate_turret_left()
        elif turret1 == _TURRET_RIGHT:
            t1.rotate_turret_right()

        # 3. Clamp tanks inside arena bounds --------------------------------
        cx0, cy0 = arena.clamp_position(t0.x, t0.y, _TANK_HW, _TANK_HH)
        t0.x = cx0
        t0.y = cy0
        cx1, cy1 = arena.clamp_position(t1.x, t1.y, _TANK_HW, _TANK_HH)
        t1.x = cx1
        t1.y = cy1

        # 4. Update cooldowns and power-ups ----------------------------------
        t0.update_cooldown()
        t1.update_cooldown()
        t0.update_powerup()
        t1.update_powerup()

        # 5. Spawn bullets or mines (fire action) ----------------------------
        bullets = self.bullets
        mines_ok = self.mines_enabled
        if fire0 == _FIRE_SHOOT:
            b = t0.shoot()
            if b is not None:
                bullets.append(b)
        elif fire0 == _FIRE_MINE and mines_ok:
            self._try_lay_mine(t0)

        if fire1 == _FIRE_SHOOT:
            b = t1.shoot()
            if b is not None:
                bullets.append(b)
        elif fire1 == _FIRE_MINE and mines_ok:
            self._try_lay_mine(t1)

        # 6. Move bullets ----------------------------------------------------
        aw = arena.width
        ah = arena.height
        for b in bullets:
            b.x += b.dx * b.speed
            b.y += b.dy * b.speed
            b.lifetime -= 1
            if b.lifetime <= 0:
                b.alive = False
            elif not b.try_bounce(aw, ah):
                pass  # bullet destroyed by boundary (bounces exhausted)

        # 7. Bullet collisions (single pass) ---------------------------------
        self._check_bullet_collisions_fast(bullets, walls, tanks,
                                           evt_hit, evt_damage,
                                           evt_wall_destroy)

        # 8. Remove destroyed walls ------------------------------------------
        arena.remove_destroyed_walls()

        # 9. Update mines ----------------------------------------------------
        self._update_mines(tanks, evt_mine_hit, evt_damage, evt_hit)

        # 10. Power-up spawning (gated by curriculum) ------------------------
        if self.powerups_enabled:
            self._update_powerup_spawning(tanks)

        # 11. Power-up pickup ------------------------------------------------
        if self.powerups_enabled:
            self._check_powerup_pickup(tanks)

        # 12. Despawn old power-ups ------------------------------------------
        for pu in self.powerups:
            if pu.alive:
                pu.spawn_tick += 1
                if pu.spawn_tick >= pu.lifetime:
                    pu.alive = False
        self.powerups = [p for p in self.powerups if p.alive]

        # 13. Threat-tag alive bullets (for outcome-based dodge reward) ------
        #     Project each bullet forward a few ticks; if trajectory would
        #     intersect a target tank's hitbox, mark it as "threatening".
        #     Also track per-tick "under threat" for each tank.
        under_threat: set[int] = set()  # tank ids currently under threat
        _THREAT_LOOKAHEAD = cfg.THREAT_LOOKAHEAD_TICKS
        for b in bullets:
            if not b.alive:
                continue
            enemy_tank = t1 if b.owner_id == 0 else t0
            target_id = enemy_tank.id
            # Project bullet position over next N ticks
            bx, by = b.x, b.y
            bdx, bdy = b.dx * b.speed, b.dy * b.speed
            trect = pygame.Rect(enemy_tank.x - _TANK_HW, enemy_tank.y - _TANK_HH,
                                _TANK_W, _TANK_H)
            br = cfg.BULLET_RADIUS
            would_hit = False
            for step in range(1, _THREAT_LOOKAHEAD + 1):
                px = bx + bdx * step
                py = by + bdy * step
                brect = pygame.Rect(px - br, py - br, br * 2, br * 2)
                if brect.colliderect(trect):
                    would_hit = True
                    break
            if would_hit:
                b.threatened_tanks.add(target_id)
                under_threat.add(target_id)

        # 14. Collect expired + prune dead bullets + outcome dodge check -----
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
                # Bullet just died -- check for successful dodges
                if b.lifetime <= 0:
                    evt_bullet_expire.append(b.owner_id)
                # For each tank that was threatened but NOT hit: that's a dodge
                for threatened_tid in b.threatened_tanks:
                    if threatened_tid not in b.hit_tanks:
                        evt_dodge.append(threatened_tid)
        self.bullets = alive_bullets
        t0.bullet_count = bc0
        t1.bullet_count = bc1

        # 15. Per-tank closer shaping ----------------------------------------
        #     Each tank gets credit only if *it* moved closer to the opponent.
        d = _math_dist((t0.x, t0.y), (t1.x, t1.y))
        prev0 = self._prev_positions.get(0, (t0.x, t0.y))
        prev1 = self._prev_positions.get(1, (t1.x, t1.y))
        prev_d0 = _math_dist(prev0, (t1.x, t1.y))  # old t0 pos -> current t1
        prev_d1 = _math_dist(prev1, (t0.x, t0.y))  # old t1 pos -> current t0
        curr_d0 = _math_dist((t0.x, t0.y), (t1.x, t1.y))
        curr_d1 = curr_d0  # same inter-tank distance
        closer0 = curr_d0 < prev_d0  # t0 moved closer to t1
        closer1 = curr_d1 < prev_d1  # t1 moved closer to t0
        self._prev_dist = d
        self._prev_positions = {0: (t0.x, t0.y), 1: (t1.x, t1.y)}

        # 16. Aim quality (angular difference to enemy using turret angle) ----
        aim_quality = {}
        for tid, me, enemy in [(0, t0, t1), (1, t1, t0)]:
            angle_to_enemy = math.degrees(
                math.atan2(enemy.y - me.y, enemy.x - me.x))
            diff = abs(((angle_to_enemy - me.turret_angle) + 180) % 360 - 180)
            aim_quality[tid] = diff

        # 17. Shrink phase -- push tanks toward center after 60% of timeout --
        self.tick_count += 1
        if self.tick_count >= _SHRINK_START:
            cx = arena.width / 2
            cy = arena.height / 2
            # Push strength increases linearly as timeout approaches
            progress = (self.tick_count - _SHRINK_START) / max(_ROUND_TIMEOUT - _SHRINK_START, 1)
            force = _SHRINK_FORCE * (1.0 + progress * 2.0)  # ramps up from 0.3 to 0.9
            for tank in tanks:
                dx = cx - tank.x
                dy = cy - tank.y
                dist_to_center = math.sqrt(dx * dx + dy * dy)
                if dist_to_center > 10:
                    tank.x += (dx / dist_to_center) * force
                    tank.y += (dy / dist_to_center) * force

        # 18. Scoring / round-end check --------------------------------------
        round_over = False
        round_winner = None
        # Check if any tank is dead (HP <= 0)
        for tank in tanks:
            if not tank.is_alive:
                round_over = True
                round_winner = 1 - tank.id
                self.scores[round_winner] = self.scores.get(round_winner, 0) + 1
                break

        if not round_over and self.tick_count >= _ROUND_TIMEOUT:
            round_over = True
            # Timeout: award point to tank with more HP remaining
            if t0.hp != t1.hp:
                round_winner = 0 if t0.hp > t1.hp else 1
                self.scores[round_winner] = self.scores.get(round_winner, 0) + 1
            # If HP is tied, no point awarded (true draw for this round)

        # 18. Spawn distance (how far each tank is from its spawn point) -----
        spawn_dist = {}
        for tid, tank in [(0, t0), (1, t1)]:
            sx, sy = self.spawn_positions_xy.get(tid, (tank.x, tank.y))
            spawn_dist[tid] = _math_dist((tank.x, tank.y), (sx, sy))

        return {
            EVT_HIT: evt_hit,
            EVT_DAMAGE: evt_damage,
            EVT_WALL_DESTROY: evt_wall_destroy,
            EVT_BULLET_EXPIRE: evt_bullet_expire,
            EVT_ROUND_OVER: round_over,
            EVT_ROUND_WINNER: round_winner,
            EVT_MINE_HIT: evt_mine_hit,
            EVT_DODGE: evt_dodge,
            "under_threat": under_threat,
            "closer": {0: closer0, 1: closer1},
            "spawn_dist": spawn_dist,
            "aim_quality": aim_quality,
        }

    # ==================================================================
    # Mine helpers
    # ==================================================================
    def _try_lay_mine(self, tank: Tank) -> None:
        """Attempt to lay a mine at the tank's position."""
        if tank.mine_count >= cfg.MAX_MINES_PER_TANK:
            return
        mine = Mine(tank.x, tank.y, tank.id)
        self.mines.append(mine)
        tank.mine_count += 1

    def _update_mines(self, tanks: list[Tank],
                      evt_mine_hit: list, evt_damage: list, evt_hit: list) -> None:
        """Update mines: tick timers, check collisions.

        Mines never hurt their owner -- only the enemy tank can trigger them.
        """
        alive_mines: list[Mine] = []
        # Recount mines per tank
        mine_counts = {0: 0, 1: 0}
        for mine in self.mines:
            mine.update()
            if not mine.alive:
                continue
            triggered = False
            if mine.armed:
                for tank in tanks:
                    if not tank.is_alive:
                        continue
                    # Mines never hurt their owner
                    if tank.id == mine.owner_id:
                        continue
                    if tank.get_rect().colliderect(mine.rect):
                        # Mine triggers on enemy!
                        dead = tank.take_damage(cfg.MINE_DAMAGE)
                        evt_mine_hit.append((mine.owner_id, tank.id))
                        evt_damage.append((mine.owner_id, tank.id, tank.hp, False))
                        if dead:
                            evt_hit.append((mine.owner_id, tank.id))
                        triggered = True
                        break
            if not triggered:
                alive_mines.append(mine)
                mine_counts[mine.owner_id] = mine_counts.get(mine.owner_id, 0) + 1
        self.mines = alive_mines
        for tank in tanks:
            tank.mine_count = mine_counts.get(tank.id, 0)

    # ==================================================================
    # Power-up helpers
    # ==================================================================
    def _update_powerup_spawning(self, tanks: list[Tank]) -> None:
        """Periodically spawn power-ups."""
        self._powerup_spawn_timer += 1
        if self._powerup_spawn_timer >= self._next_powerup_interval:
            self._powerup_spawn_timer = 0
            self._next_powerup_interval = random.randint(300, 500)
            if len([p for p in self.powerups if p.alive]) < cfg.POWERUP_MAX_ON_FIELD:
                pos = self.arena.get_powerup_spawn_position(
                    tanks, self.arena.walls, self.powerups, self.mines)
                if pos is not None:
                    kind = random.choice(PowerUp.KINDS)
                    pu = PowerUp(pos[0], pos[1], kind)
                    self.powerups.append(pu)

    def _check_powerup_pickup(self, tanks: list[Tank]) -> None:
        """Check if any tank drives over a power-up."""
        for pu in self.powerups:
            if not pu.alive:
                continue
            for tank in tanks:
                if not tank.is_alive:
                    continue
                if tank.get_rect().colliderect(pu.rect):
                    tank.apply_powerup(pu.kind)
                    pu.alive = False
                    break

    # ==================================================================
    # Collision helpers  (optimised)
    # ==================================================================
    @staticmethod
    def _valid_pos(tank: Tank, nx: float, ny: float,
                   other: Tank, walls, arena) -> bool:
        """Check if proposed position collides with walls or the other tank."""
        hw = _TANK_HW
        hh = _TANK_HH
        # Arena bounds first (cheapest check)
        if nx - hw < 0 or nx + hw > arena.width or ny - hh < 0 or ny + hh > arena.height:
            return False
        # Build proposed rect once
        pleft = nx - hw
        ptop = ny - hh
        proposed = pygame.Rect(pleft, ptop, _TANK_W, _TANK_H)
        # Walls
        for w in walls:
            if proposed.colliderect(w.rect):
                return False
        # Other tank (inline rect construction)
        ohw = _TANK_HW
        ohh = _TANK_HH
        other_rect = pygame.Rect(other.x - ohw, other.y - ohh, _TANK_W, _TANK_H)
        if proposed.colliderect(other_rect):
            return False
        return True

    @staticmethod
    def _check_bullet_collisions_fast(bullets: list[Bullet], walls,
                                       tanks: list[Tank],
                                       evt_hit: list,
                                       evt_damage: list,
                                       evt_wall_destroy: list) -> None:
        """Collision detection for all live bullets -- single pass.

        Now supports HP: bullets damage tanks, kills only when HP reaches 0.
        Bullets still destroyed on breakable wall hits (no ricochet off walls).
        """
        br = cfg.BULLET_RADIUS
        br2 = br * 2

        for b in bullets:
            if not b.alive:
                continue
            bx = b.x
            by = b.y
            # Build rect once
            brect = pygame.Rect(bx - br, by - br, br2, br2)
            # Breakable walls -- bullets destroyed on hit (no bounce)
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
            # Tanks (cannot hit own tank)
            owner = b.owner_id
            for tank in tanks:
                if tank.id == owner:
                    continue
                if not tank.is_alive:
                    continue
                # Inline tank rect
                trect = pygame.Rect(tank.x - _TANK_HW, tank.y - _TANK_HH,
                                    _TANK_W, _TANK_H)
                if brect.colliderect(trect):
                    dead = tank.take_damage(1)
                    b.hit_tanks.add(tank.id)  # record hit for dodge tracking
                    evt_damage.append((owner, tank.id, tank.hp, b.has_bounced))
                    if dead:
                        evt_hit.append((owner, tank.id))
                    b.alive = False
                    break
