# Tank Duel AI

A top-down tank dueling game where two AI-controlled tanks learn to fight each other through reinforcement learning. Built with Pygame and PyTorch.

---

## Quick Start

```bash
# Install dependencies
pip install pygame matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Train (DQN, 2 parallel games, auto-resumes if interrupted)
python main.py --fast

# Watch trained agents fight
python main.py --demo
```

---

## How It Works

Two tanks spawn on opposite sides of an arena filled with breakable walls. They learn to move, aim, shoot, and dodge -- all through self-play reinforcement learning. Both tanks share the same neural network (DQN), so every tick produces training data from both perspectives simultaneously.

### Game Mechanics

| Feature | Details |
|---------|---------|
| **Tank HP** | 3 hit points per tank; point scored when enemy reaches 0 |
| **Scoring** | Best of 3 points = 1 episode |
| **Movement** | Hull: forward/backward + rotation (5 deg/tick) |
| **Turret** | Independent rotation (8 deg/tick), aims separately from hull |
| **Bullets** | Straight-line travel, destroyed on arena/wall contact |
| **Breakable Walls** | 3 HP each, degrade visually, block movement and bullets |

### AI System: Deep Q-Network (DQN)

- 3-layer MLP (48 -> 256 -> 256 -> 128 -> 30 actions), ~120K parameters
- Double DQN with target network for stable training
- **Frame stacking**: 3 frames of 16 floats = 48-dim state vector (perceives motion)
- **Prioritized Experience Replay** (sum-tree, O(log n) sampling)
- **Reward normalization** (running mean/std, Welford's algorithm)
- Self-play: one shared network controls both tanks
- Parallel training: multiple games feed the same network

### Action Space (30 actions)

5 movement x 3 turret x 2 fire:

- **Movement**: forward, backward, rotate hull left, rotate hull right, no movement
- **Turret**: rotate turret left, rotate turret right, no rotation
- **Fire**: shoot, don't shoot

### Curriculum Learning (2 phases)

| Phase | Episodes | Features |
|-------|----------|----------|
| P1: Combat | 1-300 | No walls -- learn basic aiming and shooting |
| P2: Walls | 301+ | Full walls -- learn navigation and cover |

### Reward Structure (4 core signals)

| Event | Reward |
|-------|--------|
| Kill enemy (HP -> 0) | +10.0 |
| Get killed | -10.0 |
| Deal 1 HP damage | +5.0 |
| Take 1 HP damage | -5.0 |
| Missed shot (bullet expired) | -0.5 |
| Per-tick penalty | -0.01 |

---

## Command Reference

### Training

```bash
# Default: DQN with 2 parallel games, renders at 60 FPS
python main.py

# Fast training (skip rendering)
python main.py --fast

# More episodes
python main.py --episodes 5000 --fast

# More parallel games (4 games = 8 agents/tick)
python main.py --parallel 4 --fast
```

### Population-Based Training (PBT)

```bash
# 6 agents compete in round-robin, losers replaced by mutated winners
python main.py --population --fast

# Larger population, more generations
python main.py --population --fast --pop-size 8 --generations 60
```

PBT creates a population of N agents that play round-robin tournaments each generation. Bottom ~67% are replaced by mutated clones of top ~33%. Produces stronger strategies than single self-play.

### Demo / Watching

```bash
# Watch trained DQN agents
python main.py --demo

# Watch more episodes
python main.py --demo --episodes 20
```

### Auto-Resume

Training automatically resumes from the last saved checkpoint:

```bash
python main.py --episodes 500 --fast   # trains 0-500, saves
python main.py --episodes 500 --fast   # loads ep=500, trains 500-1000
```

Epsilon, network weights, and curriculum phase all carry over between runs.

---

## Project Structure

```
main.py                          Entry point, training loop
config.py                        All tunable constants and hyperparameters
requirements.txt                 Python dependencies

ai/
  base_agent.py                  Agent protocol (interface contract)
  dqn_agent.py                   Double DQN + PER + reward normalization
  population.py                  Population-Based Training (generational selection)
  reward.py                      Reward computation (events -> scalar)
  expert_agent.py                Rule-based heuristic opponent
  state_encoder_continuous.py    16-float state vector with 3-frame stacking

game/
  game_manager.py                Core tick loop, collision detection, scoring
  tank.py                        Tank: HP, turret, movement
  bullet.py                      Bullet: straight-line travel
  wall.py                        Breakable wall: 3 HP, visual degradation
  arena.py                       Wall generation, spawn positions

rendering/
  renderer.py                    Pygame drawing: tanks, bullets, walls, HUD

saved_models/
  dqn_shared.pt                  DQN model checkpoint (auto-created)

plans/
  ARCHITECTURE.md                Original design document
  AUDIT.md                       System audit report
```

---

## Configuration

All tunable values are in [`config.py`](config.py). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `ARENA_WIDTH/HEIGHT` | 800x600 | Arena dimensions |
| `TANK_HP` | 3 | Hit points per tank |
| `TANK_SPEED` | 2.0 | Pixels per tick |
| `BULLET_SPEED` | 6.0 | Pixels per tick |
| `POINTS_TO_WIN` | 3 | Best-of-3 per episode |
| `ROUND_TIMEOUT` | 1000 | Ticks per round (~17 seconds) |
| `NUM_ACTIONS` | 30 | 5 move x 3 turret x 2 fire |
| `DQN_STATE_DIM` | 48 | 16 floats x 3 frames stacked |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `DQN_EPSILON_DECAY` | 0.997 | Per-episode decay |
| `EPSILON_MIN` | 0.02 | Final exploration rate |
| `DQN_LR` | 0.001 | Adam learning rate |
| `REPLAY_CAPACITY` | 50000 | Replay buffer size |

---

## Dependencies

- **Python 3.10+**
- **pygame** -- game engine and rendering
- **torch** (CPU) -- DQN neural network
- **matplotlib** (optional) -- training graphs

```bash
pip install pygame matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
