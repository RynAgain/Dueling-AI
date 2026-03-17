# Tank Duel AI -- System Audit Report

**Date**: 2026-03-17  
**Scope**: Full codebase review -- architecture, bugs, quality, performance, correctness  
**Files reviewed**: 18 source files, ~2,700 LOC

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Assessment](#2-architecture-assessment)
3. [Bugs and Defects](#3-bugs-and-defects)
4. [Design Issues](#4-design-issues)
5. [Performance Concerns](#5-performance-concerns)
6. [Code Quality](#6-code-quality)
7. [Configuration and Consistency](#7-configuration-and-consistency)
8. [Documentation Drift](#8-documentation-drift)
9. [Risk Assessment](#9-risk-assessment)
10. [Recommendations Summary](#10-recommendations-summary)

---

## 1. Executive Summary

The codebase is a well-structured reinforcement learning game with clear separation between game logic, AI, and rendering. It has evolved significantly beyond its original architecture document, adding DQN, power-ups, mines, HP systems, ricochets, and curriculum learning. The implementation is functional and reasonably clean. However, the audit identified **5 bugs** (2 high severity), **8 design issues**, and **6 performance concerns** that should be addressed.

**Overall health: 7/10** -- Solid foundation with targeted issues to fix.

---

## 2. Architecture Assessment

### 2.1 Strengths

| Area | Assessment |
|------|-----------|
| **Module separation** | Clean 3-package split (`game/`, `ai/`, `rendering/`). Renderer has zero game logic. |
| **Agent abstraction** | Three agent types (`QLearningAgent`, `DQNAgent`, `ExpertAgent`) share an implicit interface. `main.py` works with any of them. |
| **Configuration** | All constants centralized in `config.py`. No magic numbers scattered in logic files. |
| **Curriculum learning** | Elegant 5-phase system that gradually enables game features during training. |
| **Parallel training** | Multiple `GameManager` instances feed one shared DQN -- good data throughput design. |
| **Save/resume** | Graceful Ctrl+C handling, auto-resume from checkpoints, epsilon/episode state preserved. |
| **State encoding** | Two encoders for two agent types: discretized for Q-table, continuous for DQN. Both well-normalized. |

### 2.2 Weaknesses

| Area | Assessment |
|------|-----------|
| **No formal agent interface** | The three agent types duck-type the same API but share no base class or protocol. Adding a new agent type requires reverse-engineering the implicit contract. |
| **Reward logic in main.py** | `compute_reward()` is ~60 lines of game-domain logic sitting in the entry point. The architecture doc specified a `RewardCalculator` class, which was never implemented. |
| **Missing modules from architecture** | `game/collision.py`, `ai/reward.py`, `rendering/colors.py`, `training/trainer.py`, `training/stats.py` were all planned but never created. Their logic was inlined into `game_manager.py`, `main.py`, and `renderer.py`. |
| **God method: `GameManager.step()`** | At ~250 lines with 18 numbered phases, this method does too much. It handles movement, collision, mines, power-ups, threat projection, shrink phase, scoring, aim quality, spawn distance, and dodge tracking all in one call. |

---

## 3. Bugs and Defects

### BUG-1 [HIGH] -- Double bullet movement per tick

In [`game_manager.py`](game/game_manager.py:247), bullets are moved manually with inlined physics:

```python
b.x += b.dx * b.speed
b.y += b.dy * b.speed
b.lifetime -= 1
```

The `Bullet.update()` method in [`bullet.py`](game/bullet.py:32) does the exact same thing. If `update()` were ever called (it is not currently), bullets would move twice per tick. However, the real bug is that `Bullet.update()` is dead code -- it exists but is never used, while the `GameManager` reimplements its logic inline. This creates a maintenance trap: anyone modifying `Bullet.update()` would expect it to affect behavior, but it would not.

**Fix**: Use `b.update()` in the game manager loop instead of inlining, or delete the dead method.

---

### BUG-2 [HIGH] -- Ricochet can consume two bounces in one tick

In [`bullet.py`](game/bullet.py:40), `try_bounce()` checks X boundaries and Y boundaries sequentially. If a bullet is in a corner and crosses both boundaries in one tick, it will decrement `bounces_remaining` twice (once for X, once for Y). With `BULLET_MAX_BOUNCES = 1`, this means a corner ricochet uses the one bounce on the X-wall, then the Y-wall check finds `bounces_remaining = 0` and kills the bullet.

**Fix**: Treat corner hits as a single bounce event, or check `bounces_remaining` before the Y-wall branch:

```python
if self.y < 0 or self.y > arena_h:
    if self.bounces_remaining > 0:  # may already be 0 from X bounce above
```

The current code already has this check, but the issue is that a legitimate corner bounce should only cost 1 bounce, not 2. The fix is to `return True` after the X-bounce block before checking Y, or combine both axes into one bounce.

---

### BUG-3 [MEDIUM] -- `_save_model` introspects `save()` signature incorrectly

In [`main.py`](main.py:731):

```python
if hasattr(agent, 'save') and 'episode' in agent.save.__code__.co_varnames:
```

This uses `co_varnames` to check if `save()` accepts an `episode` parameter. `co_varnames` includes **all** local variables, not just parameters. If any future `save()` method happens to use a local variable named `episode`, this check would incorrectly pass. More critically, this technique fails with decorated functions, lambdas, or C-extension methods.

**Fix**: Use `inspect.signature()` or just always pass `episode` as a keyword argument (since all three agent save methods already accept `**kwargs`):

```python
agent.save(path, episode=episode)
```

This already works for `DQNAgent.save()` and `ExpertAgent.save()`. For `QLearningAgent.save()`, the `episode` parameter is not accepted -- this is itself a secondary bug, since training state (episode count, epsilon) is not persisted for the Q-learning agent.

---

### BUG-4 [MEDIUM] -- Q-learning agent `save()` does not persist episode/epsilon

[`agent.py`](ai/agent.py:153) `save()` stores Q-tables and replay buffer but **not** the current epsilon or episode count. When training resumes:

- `_load_model()` in `main.py` reads `meta.get("episode", 0)` which will always be 0
- `agent.epsilon` is never restored from the saved file

Contrast with `DQNAgent.save()` at [`dqn_agent.py`](ai/dqn_agent.py:191) which correctly saves both.

**Fix**: Add `"epsilon": self.epsilon` and `"episode": episode` to the Q-learning agent's save payload, and accept `episode` as a parameter.

---

### BUG-5 [LOW] -- `closer` shaping reward is symmetric but shouldn't be

In [`game_manager.py`](game/game_manager.py:339):

```python
closer0 = d < prev
closer1 = closer0  # symmetric
```

Both tanks always get the same `closer` flag. If the distance shrank, both are told "you moved closer," even if only one moved and the other stood still (or moved farther but the net distance still decreased). This dilutes the reward signal -- the stationary tank gets rewarded for the other tank's approach.

**Fix**: Track each tank's individual contribution to distance change. Store previous per-tank positions and compute whether each tank moved toward the opponent.

---

## 4. Design Issues

### DESIGN-1 -- No agent protocol/base class

The three agent types (`QLearningAgent`, `DQNAgent`, `ExpertAgent`) share an implicit duck-typed interface:

- `choose_action(state) -> int`
- `learn(state, action, reward, next_state, ...)`
- `decay_epsilon()`
- `episode_end_replay()`
- `save(path, ...)` / `load(path) -> dict`
- `epsilon: float`
- `q_table_a`, `q_table_b` (compatibility properties)
- `buffer_size() -> int`

But there is no shared base class, no Protocol, and no ABC. The DQN and Expert agents add dummy `q_table_a`/`q_table_b` properties just for compatibility with `main.py`'s reporting code.

**Recommendation**: Define a `BaseAgent` ABC or `typing.Protocol` to formalize the contract.

---

### DESIGN-2 -- `learn()` signature differs between agents

- `QLearningAgent.learn(state, action, reward, next_state)` -- no `done` parameter
- `DQNAgent.learn(state, action, reward, next_state, done=False)` -- has `done`
- `ExpertAgent.learn(*args, **kwargs)` -- swallows everything

This forces `main.py` to branch on `use_dqn` when calling learn:

```python
if use_dqn:
    _learn0(s0, a0, r0, sn0, done=round_done)
else:
    _learn0(s0, a0, r0, sn0)
```

**Recommendation**: Give all agents the same `learn(state, action, reward, next_state, done=False)` signature. The Q-learning agent can simply ignore `done` if not needed, or use it to zero out the next-state value on terminal transitions (which would be more correct anyway).

---

### DESIGN-3 -- Expert agent encodes actions differently

In [`expert_agent.py`](ai/expert_agent.py:115):

```python
action = move_action * (cfg.NUM_TURRET_OPTIONS * cfg.NUM_FIRE_OPTIONS) + \
         turret_action * cfg.NUM_FIRE_OPTIONS + fire_action
```

But in [`game_manager.py`](game/game_manager.py:54), `_decode_action()` does:

```python
fire_action = action % _NUM_FIRE
remaining = action // _NUM_FIRE
turret_action = remaining % _NUM_TURRET
move_action = remaining // _NUM_TURRET
```

These are consistent: `action = move * (turret * fire) + turret * fire + fire` matches the decoding order `fire = action % fire_count; turret = (action // fire_count) % turret_count; move = action // (fire_count * turret_count)`. So the encoding is correct. However, the encoding logic should be extracted to a shared utility since it must stay in sync across files.

---

### DESIGN-4 -- Collision detection is O(bullets * walls)

`_check_bullet_collisions_fast()` iterates every bullet against every wall. With up to 6 bullets and 14 walls, this is 84 collision checks per tick at worst -- acceptable for current scale, but the method name contains "fast" which is misleading. If wall count or bullet count ever increase, a spatial partitioning approach (grid cells) would be needed.

---

### DESIGN-5 -- Reward function has too many shaping terms

The reward function in [`main.py`](main.py:44) sums 12+ different reward components per tick. Dense reward shaping can help early learning but can also cause agents to get stuck in local optima (e.g., optimizing for aim bonus instead of actually shooting). The `REWARD_MISSED_SHOT = -0.5` is particularly aggressive -- it's half the magnitude of dealing damage, which could discourage shooting entirely.

---

### DESIGN-6 -- Dead property in QLearningAgent

In [`agent.py`](ai/agent.py:47):

```python
@property
def q_table(self_inner):
    return self_inner.q_table_a
self.q_table = self.q_table_a  # default reference for migration/save
```

The `@property` decorator is applied to a nested function that is never used as a property (it's defined inside `__init__`, not on the class). The `self.q_table` instance attribute immediately shadows it. The decorated function is dead code.

---

### DESIGN-7 -- `is_out_of_bounds()` on Bullet is dead code

[`bullet.py`](game/bullet.py:81) defines `is_out_of_bounds()` that checks against arena boundaries. This method is never called anywhere -- `try_bounce()` handles boundary checking instead.

---

### DESIGN-8 -- Mine owner tracking on mine despawn

When a mine despawns due to lifetime expiry in [`game_manager.py`](game/game_manager.py:428), `mine.alive` is set to `False` by `mine.update()`. The mine count recount handles this. However, if a mine triggers and then `continue` skips adding it to `alive_mines`, that mine's `alive` flag is never set to False. This is benign because the mine is already removed from the list, but leaves an inconsistent object state if references to it exist elsewhere.

---

## 5. Performance Concerns

### PERF-1 -- Wall raycast is O(walls * steps) per ray, 3 rays per tank per tick

Each [`_cast_wall_ray()`](ai/state_encoder.py:152) casts 20 steps against all walls, so 3 rays * 20 steps * 14 walls = **840 `collidepoint()` calls** per tank per tick during state encoding. With 2 tanks, that's 1,680 calls. In parallel training with 2+ games, this doubles again.

**Recommendation**: Pre-build a simple grid lookup for wall occupancy (40px cells), reducing raycast to O(steps) per ray.

---

### PERF-2 -- Threat projection is O(bullets * lookahead * rect_construction)

In [`game_manager.py`](game/game_manager.py:289), for each alive bullet, a `pygame.Rect` is constructed and tested at each of 20 lookahead steps. With 6 bullets * 20 steps = 120 Rect constructions per tick. Not critical but avoidable with vector math (distance-to-line calculation).

---

### PERF-3 -- `pygame.event.get()` called in inner loop

In [`main.py`](main.py:139), `pygame.event.get()` is called on **every tick** of every round, even in `--fast` mode. The Pygame event pump is relatively expensive and only needed for rendering or to prevent the OS from declaring the window unresponsive. In fast mode, it could be called every N ticks instead.

---

### PERF-4 -- PER sampling uses Efraimidis-Spirakis on every sample call

[`replay_buffer.py`](ai/replay_buffer.py:96) computes `u ** (1/p)` for every buffer entry on every sample call. With 50,000 entries and sampling every 8 steps, this is millions of power operations per episode. A sum-tree data structure would reduce this from O(n) to O(log n).

---

### PERF-5 -- FPS set to 600 in config but labelled as 60

[`config.py`](config.py:133) has `FPS = 600`, but the README and architecture doc reference 60 FPS. The renderer's `tick()` method uses this value: `self.clock.tick(fps)`. At 600 FPS, the visual demo runs 10x faster than intended, consuming more CPU for no perceptual benefit (monitors max at 60-144Hz).

**Fix**: Set `FPS = 60` in config, or use separate `RENDER_FPS = 60` and `TRAINING_FPS = 600`.

---

### PERF-6 -- `defaultdict` lambda captures `self` in closure

In [`agent.py`](ai/agent.py:38):

```python
self.q_table_a: dict = defaultdict(lambda: [0.0] * self.num_actions)
```

The lambda captures `self`, which prevents garbage collection of the agent if only the Q-table reference is held. This is minor but worth noting for long training runs.

---

## 6. Code Quality

### Positive

- **Type hints** used consistently (`list[Bullet]`, `dict[int, int]`, `tuple[float, float]`, union types via `|`)
- **Docstrings** on all public classes and most public methods
- **Comment density** is high and helpful, especially the numbered phases in `step()`
- **No circular imports** -- dependency graph is acyclic
- **Config caching** -- hot-path config values are cached as module-level variables in `game_manager.py` and `main.py`
- **Error handling** -- graceful fallback from DQN to Q-table when PyTorch is missing

### Negative

- **No tests** -- zero test files exist. For a project with complex physics and reward logic, this is the highest-risk gap.
- **No `requirements.txt`** or `pyproject.toml` -- dependencies are only documented in README prose.
- **No type checking** setup (`mypy.ini`, `py.typed`, etc.)
- **No linting** configuration (`.flake8`, `ruff.toml`, etc.)
- **Mixed string conventions** -- some files use `'single'`, others use `"double"` quotes inconsistently.
- **Dead code** -- `Bullet.update()`, `Bullet.is_out_of_bounds()`, the nested `@property` in `QLearningAgent.__init__()`.

---

## 7. Configuration and Consistency

### 7.1 Config values that don't match documentation

| Setting | Config value | README/Architecture value | File |
|---------|-------------|--------------------------|------|
| `FPS` | 600 | 60 | [`config.py:133`](config.py:133) |
| `POINTS_TO_WIN` | 3 | 10 (architecture) | [`config.py:47`](config.py:47) |
| `ROUND_TIMEOUT` | 1000 | 1800 (architecture) | [`config.py:48`](config.py:48) |
| `EPSILON_DECAY` | 0.998 | 0.9995 (architecture) | [`config.py:57`](config.py:57) |
| `EPSILON_MIN` | 0.02 | 0.05 (architecture) | [`config.py:58`](config.py:58) |
| Action space | 45 actions | 6 actions (architecture) | [`config.py:64`](config.py:64) |
| State dim | 21 (DQN) / 15 (Q-table) | 8 (architecture) | Multiple |

These are all intentional evolution since the architecture doc was written, but the architecture doc is now significantly out of date.

### 7.2 Unused config values

- `REWARD_MINE_SELF = -5.0` at [`config.py:77`](config.py:77) -- labelled "(legacy -- mines no longer hurt owner)" but still defined. Referenced in `main.py` as `_R_MINE_SELF` but **never used** in `compute_reward()`.
- `NUM_MOVE_ACTIONS = 5` at [`config.py:143`](config.py:143) -- labelled "kept for legacy reference", duplicates `NUM_MOVE_OPTIONS`.

### 7.3 Missing dependency specification

No `requirements.txt` exists. The project needs:

```
pygame>=2.0
torch>=2.0  # optional, for DQN mode
matplotlib  # optional, for training plots
```

---

## 8. Documentation Drift

The [`ARCHITECTURE.md`](plans/ARCHITECTURE.md) is the original design document and is now substantially outdated:

| Architecture Doc Says | Reality |
|----------------------|---------|
| Tabular Q-learning only | DQN is the default, Q-learning is fallback |
| 6 discrete actions | 45 composite actions (5 move x 3 turret x 3 fire) |
| 8-component state tuple | 15-component (Q-table) or 21-float vector (DQN) |
| `collision.py` module | Collisions are inline in `game_manager.py` |
| `reward.py` / `RewardCalculator` | Reward computation is in `main.py` |
| `training/trainer.py`, `training/stats.py` | Training loop is in `main.py` |
| `rendering/colors.py` | Colors are inline in `renderer.py` |
| 1 HP, instant kill | 3 HP with damage tracking |
| No power-ups, no mines | Both implemented with 3 power-up types |
| No ricochet | Bullets ricochet once off arena walls |
| Best-of-10 scoring | Best-of-3 scoring |
| Single game | Parallel training with N game instances |
| No independent turret | Independent turret rotation |
| No curriculum learning | 5-phase curriculum |
| No expert agent | Rule-based expert opponent for mixed training |

The [`README.md`](README.md) is accurate and up-to-date with the current implementation.

**Recommendation**: Either update `ARCHITECTURE.md` to reflect reality, or rename it to `ARCHITECTURE_v1.md` and create a new version.

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Impact |
|------|----------|-----------|--------|
| No tests -- regressions go undetected | HIGH | HIGH | Breaking changes to reward/collision logic silently degrade training |
| BUG-1 dead `Bullet.update()` -- maintenance trap | HIGH | MEDIUM | Future developer modifies wrong code path |
| BUG-4 Q-table agent can't resume training | MEDIUM | HIGH | Wasted training time on Q-table mode |
| BUG-2 corner ricochet double-bounce | MEDIUM | LOW | Occasional unfair bullet death |
| BUG-5 symmetric closer reward | MEDIUM | HIGH | Diluted learning signal every tick |
| PERF-4 O(n) PER sampling | MEDIUM | HIGH | Training slowdown scales with buffer size |
| PERF-5 FPS=600 vs intended 60 | LOW | HIGH | Demo plays too fast |
| Architecture doc is misleading | LOW | MEDIUM | Onboarding confusion |

---

## 10. Recommendations Summary

### Priority 1 -- Fix Bugs

1. **BUG-1**: Delete `Bullet.update()` and `Bullet.is_out_of_bounds()` (dead code), or refactor `GameManager.step()` to use them.
2. **BUG-2**: Handle corner ricochets as single-bounce events in `Bullet.try_bounce()`.
3. **BUG-4**: Add `epsilon` and `episode` persistence to `QLearningAgent.save()`/`load()`.
4. **BUG-5**: Compute per-tank closer signals instead of symmetric.
5. **BUG-3**: Simplify `_save_model()` to always pass `episode=` keyword argument.

### Priority 2 -- Reduce Risk

6. Add `requirements.txt` with pinned dependencies.
7. Add at least unit tests for: `_decode_action()`, `compute_reward()`, `Bullet.try_bounce()`, state encoder output shapes.
8. Clean up dead code: unused `_R_MINE_SELF`, `NUM_MOVE_ACTIONS`, nested `@property` in `QLearningAgent`.

### Priority 3 -- Improve Design

9. Define a `BaseAgent` protocol or ABC.
10. Extract `compute_reward()` from `main.py` into `ai/reward.py`.
11. Extract action encoding/decoding into a shared `config.py` helper.
12. Split `GameManager.step()` into smaller methods (movement, collision, scoring phases).
13. Fix `FPS = 600` to `FPS = 60` (or add separate render/training FPS).

### Priority 4 -- Performance

14. Replace PER linear sampling with a sum-tree for O(log n) sampling.
15. Pre-build wall grid for O(1) raycast lookups.
16. Throttle `pygame.event.get()` in fast mode.

### Priority 5 -- Documentation

17. Update or archive `ARCHITECTURE.md` to match current implementation.
18. Add inline module-level docstrings to `__init__.py` files.

---

*End of audit.*
