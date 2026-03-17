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

Two tanks spawn on opposite sides of an arena filled with breakable walls. They learn to move, aim, shoot, dodge, use power-ups, and lay mines -- all through self-play reinforcement learning. Both tanks share the same neural network (DQN), so every tick produces training data from both perspectives simultaneously.

### Game Mechanics

| Feature | Details |
|---------|---------|
| **Tank HP** | 3 hit points per tank; point scored when enemy reaches 0 |
| **Scoring** | Best of 3 points = 1 episode |
| **Movement** | Hull: forward/backward + rotation (5 deg/tick) |
| **Turret** | Independent rotation (8 deg/tick), aims separately from hull |
| **Bullets** | Straight-line travel, ricochet once off arena walls, destroy breakable walls |
| **Breakable Walls** | 3 HP each, degrade visually, block movement and bullets |
| **Power-ups** | Speed boost (1.5x), rapid fire (half cooldown), shield (absorb 1 hit) |
| **Mines** | Drop at current position, arm after 60 ticks, only hurt the enemy |

### AI System

**Default: Deep Q-Network (DQN)**
- 3-layer MLP (21 -> 128 -> 128 -> 64 -> 45 actions), ~30K parameters
- Double DQN with target network for stable training
- Continuous state vector (21 floats) -- no discretization loss
- Experience replay buffer (50K transitions)
- Self-play: one shared network controls both tanks
- Parallel training: multiple games feed the same network

**Fallback: Tabular Q-learning** (use `--qtable`)
- Double Q-learning with two Q-tables
- Prioritized Experience Replay
- 15-component discretized state (bins for angles, distance, etc.)

### Action Space (45 actions)

5 movement options x 3 turret options x 3 fire options:

- **Movement**: forward, backward, rotate hull left, rotate hull right, no movement
- **Turret**: rotate turret left, rotate turret right, no rotation
- **Fire**: don't shoot, shoot, lay mine

### Curriculum Learning (5 phases)

Training gradually introduces game complexity:

| Phase | Episodes | Features |
|-------|----------|----------|
| P1: Shoot | 1-200 | No walls, no power-ups, no mines -- learn basic combat |
| P2: Walls | 201-500 | Few walls (4) -- learn navigation |
| P3: Power-ups | 501-800 | Full walls + power-ups -- learn strategy |
| P4: Mines | 801-1200 | Everything enabled -- learn mine tactics |
| P5: Full | 1201+ | Full game, epsilon near minimum |

### Reward Structure

| Event | Reward |
|-------|--------|
| Kill enemy (HP -> 0) | +10.0 |
| Get killed | -10.0 |
| Deal 1 HP damage | +5.0 |
| Take 1 HP damage | -5.0 |
| Successful dodge (bullet missed) | +1.5 |
| Per-tick under bullet threat | -0.1 |
| Mine hits enemy | +5.0 |
| Good aim (turret within 15 deg) | +0.3 |
| Okay aim (within 30 deg) | +0.1 |
| Moving closer to enemy | +0.05 |
| Away from spawn | +0.02 (capped) |
| Destroy wall | +0.5 |
| Missed shot (bullet expired) | -0.2 |
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

# Single game (no parallel)
python main.py --parallel 1

# Use tabular Q-learning instead of DQN
python main.py --qtable --fast
```

### Demo / Watching

```bash
# Watch trained DQN agents
python main.py --demo

# Watch more episodes
python main.py --demo --episodes 20

# Watch Q-learning agents
python main.py --qtable --demo
```

### Auto-Resume

Training automatically resumes from the last saved checkpoint:

```bash
python main.py --episodes 500 --fast   # trains 0-500, saves
python main.py --episodes 500 --fast   # loads ep=500, trains 500-1000
python main.py --episodes 500 --fast   # loads ep=1000, trains 1000-1500
```

Epsilon, network weights, and curriculum phase all carry over between runs.

---

## Project Structure

```
main.py                          Entry point, training loop, reward computation
config.py                        All tunable constants and hyperparameters

ai/
  dqn_agent.py                   Double DQN agent (PyTorch, CPU)
  state_encoder_continuous.py    21-float continuous state vector for DQN
  agent.py                       Tabular Double Q-learning agent (fallback)
  state_encoder.py               15-component discretized state for Q-table
  replay_buffer.py               Prioritized Experience Replay buffer

game/
  game_manager.py                Core tick loop, collision detection, scoring
  tank.py                        Tank: HP, turret, movement, power-up effects
  bullet.py                      Bullet: ricochet, threat tracking
  wall.py                        Breakable wall: 3 HP, visual degradation
  arena.py                       Wall generation, spawn positions
  mine.py                        Deployable mine: arm delay, enemy-only trigger
  powerup.py                     Speed boost, rapid fire, shield

rendering/
  renderer.py                    Pygame drawing: tanks, bullets, walls, HUD

saved_models/
  dqn_shared.pt                  DQN model checkpoint (auto-created)
  agent_shared.pkl               Q-learning model checkpoint (auto-created)

plans/
  ARCHITECTURE.md                Original design document
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
| `BULLET_MAX_BOUNCES` | 1 | Ricochets off arena walls |
| `POINTS_TO_WIN` | 3 | Best-of-3 per episode |
| `ROUND_TIMEOUT` | 1000 | Ticks per round (~17 seconds) |
| `NUM_ACTIONS` | 45 | 5 move x 3 turret x 3 fire |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_DECAY` | 0.998 | Per-episode decay |
| `EPSILON_MIN` | 0.02 | Final exploration rate |
| `DQN_LR` | 0.001 | Adam learning rate |
| `REPLAY_CAPACITY` | 50000 | Replay buffer size |

---

## Dependencies

- **Python 3.10+**
- **pygame** -- game engine and rendering
- **torch** (CPU) -- DQN neural network (optional: falls back to Q-table)
- **matplotlib** (optional) -- training graphs

```bash
pip install pygame matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Training Output

During training, progress is reported every 50 episodes:

```
[*] training (fast) [DQN x2 parallel]: 2000 episodes
  Ep   50/2000  |  Wins B:26 R:24 D:0  |  eps=0.9048  |  buf=50000  |  P1:shoot  |  1.2 ep/s
  Ep  100/2000  |  Wins B:51 R:49 D:0  |  eps=0.8187  |  buf=50000  |  P1:shoot  |  1.1 ep/s
  ...
```

At the end of training, a 4-panel plot is saved to `training_results.png`:
1. Win rate (50-episode rolling window)
2. Total reward per episode
3. Epsilon decay curve
4. Replay buffer / Q-table size
