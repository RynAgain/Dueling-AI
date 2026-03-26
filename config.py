# ---------------------------------------------------------------------------
# config.py -- All tunable constants for the Tank Duel AI game
# ---------------------------------------------------------------------------

# Arena
ARENA_WIDTH = 800
ARENA_HEIGHT = 600
HUD_HEIGHT = 40  # extra strip at top for scores / info

# Tank
TANK_WIDTH = 32
TANK_HEIGHT = 24
TANK_SPEED = 2.0
TANK_REVERSE_FACTOR = 0.6
TANK_ROTATION_SPEED = 5.0  # degrees per tick (hull)
TURRET_ROTATION_SPEED = 8  # degrees per tick (turret -- faster than hull)
SHOOT_COOLDOWN = 30  # ticks
MAX_BULLETS_PER_TANK = 3
TANK_HP = 3  # hit points per tank

# Bullet
BULLET_RADIUS = 4
BULLET_SPEED = 6.0
BULLET_LIFETIME = 180  # ticks
BULLET_MAX_BOUNCES = 0  # 0 = no ricochet (direct-line only, fully learnable)

# Wall
WALL_SIZE = 40
WALL_MAX_HP = 3
WALL_COUNT_MIN = 8
WALL_COUNT_MAX = 14

# Scoring
POINTS_TO_WIN = 3
ROUND_TIMEOUT = 1000  # ticks (~17s at 60 FPS)
SHRINK_START_RATIO = 0.6
SHRINK_FORCE = 0.3
MAX_ROUNDS_PER_EPISODE = 10

# RL hyperparameters
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.02

# Action space: 5 movement x 3 turret x 2 fire = 30
NUM_MOVE_OPTIONS = 5
NUM_TURRET_OPTIONS = 3
NUM_FIRE_OPTIONS = 2   # shoot / don't shoot (no mines)
NUM_ACTIONS = NUM_MOVE_OPTIONS * NUM_TURRET_OPTIONS * NUM_FIRE_OPTIONS  # 30

# Rewards (simplified: 4 core signals)
REWARD_HIT_ENEMY = 10.0    # kill (HP reaches 0)
REWARD_GOT_HIT = -10.0     # got killed
REWARD_DAMAGE = 5.0         # deal 1 HP damage
REWARD_TOOK_DAMAGE = -5.0   # take 1 HP damage
REWARD_MISSED_SHOT = -0.5   # bullet expired unused
REWARD_TIMESTEP = -0.01     # per-tick urgency

# Threat projection
THREAT_LOOKAHEAD_TICKS = 20

# Experience replay
REPLAY_CAPACITY = 50000
REPLAY_EPISODE_BURST = 128

# DQN hyperparameters
DQN_STATE_DIM = 48           # stacked state: 16 floats x 3 frames
DQN_HIDDEN_1 = 256
DQN_HIDDEN_2 = 256
DQN_HIDDEN_3 = 128
DQN_LR = 1e-3
DQN_BATCH_SIZE = 256
DQN_TARGET_UPDATE = 200
DQN_MIN_REPLAY = 256
DQN_EPSILON_DECAY = 0.997
DQN_TRAIN_FREQ = 2
DQN_GRAD_STEPS = 3

# Expert opponent mixing
EXPERT_MIX_RATIO = 0.25

# Curriculum learning (2 phases -- simple)
CURRICULUM_PHASE_1_END = 300   # episodes 1-300: no walls (pure combat)
# episodes 301+: full walls

# Rendering
FPS = 60
TRAINING_FPS = 600
SCREEN_WIDTH = ARENA_WIDTH
SCREEN_HEIGHT = ARENA_HEIGHT + HUD_HEIGHT

# Movement sub-actions (0-4)
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
MOVE_ROTATE_LEFT = 2
MOVE_ROTATE_RIGHT = 3
MOVE_NOOP = 4

# Turret sub-actions (0-2)
TURRET_LEFT = 0
TURRET_RIGHT = 1
TURRET_NOOP = 2

# Fire sub-actions (0-1)
FIRE_NONE = 0
FIRE_SHOOT = 1

# Population-Based Training (PBT)
PBT_POPULATION_SIZE = 6
PBT_GENERATION_EPISODES = 50
PBT_MATCHES_PER_PAIR = 3
PBT_ELITE_RATIO = 0.33
PBT_MUTATE_LR_RANGE = (0.5, 2.0)
PBT_MUTATE_EPSILON_RANGE = (0.8, 1.0)
PBT_WEIGHT_NOISE_STD = 0.01


# ---------------------------------------------------------------------------
# Action encoding / decoding helpers
# ---------------------------------------------------------------------------

def encode_action(move: int, turret: int, fire: int) -> int:
    """Encode sub-actions into a single composite action index."""
    return move * (NUM_TURRET_OPTIONS * NUM_FIRE_OPTIONS) + \
           turret * NUM_FIRE_OPTIONS + fire


def decode_action(action: int) -> tuple[int, int, int]:
    """Decode composite action into (move, turret, fire) sub-actions."""
    fire = action % NUM_FIRE_OPTIONS
    remaining = action // NUM_FIRE_OPTIONS
    turret = remaining % NUM_TURRET_OPTIONS
    move = remaining // NUM_TURRET_OPTIONS
    return move, turret, fire
