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
BULLET_MAX_BOUNCES = 1  # ricochet off arena walls

# Wall
WALL_SIZE = 40
WALL_MAX_HP = 3
WALL_COUNT_MIN = 8
WALL_COUNT_MAX = 14

# Power-ups
POWERUP_SPAWN_INTERVAL = 400   # ticks between spawn attempts (300-500 random)
POWERUP_MAX_ON_FIELD = 2
POWERUP_DURATION = 300          # ticks (5 seconds at 60fps)
POWERUP_LIFETIME = 600          # despawn after 10 seconds
SPEED_BOOST_MULT = 1.5

# Mines
MAX_MINES_PER_TANK = 2
MINE_ARM_DELAY = 60   # ticks before mine becomes active
MINE_LIFETIME = 900   # despawn after 15 seconds
MINE_DAMAGE = 1

# Scoring
POINTS_TO_WIN = 10
ROUND_TIMEOUT = 1000  # ticks (~17s at 60 FPS -- faster resolution)
MAX_ROUNDS_PER_EPISODE = 20  # safety cap to prevent infinite episodes

# Q-Learning hyperparameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.998
EPSILON_MIN = 0.02

# Action space: 5 movement x 3 turret x 3 fire = 45
NUM_MOVE_OPTIONS = 5
NUM_TURRET_OPTIONS = 3
NUM_FIRE_OPTIONS = 3
NUM_ACTIONS = NUM_MOVE_OPTIONS * NUM_TURRET_OPTIONS * NUM_FIRE_OPTIONS  # 45

# Rewards
REWARD_HIT_ENEMY = 10.0    # only on kill (HP reaches 0)
REWARD_GOT_HIT = -10.0     # only on kill
REWARD_DAMAGE = 5.0         # dealing 1 HP damage (not kill)
REWARD_TOOK_DAMAGE = -5.0   # taking 1 HP damage (not kill)
REWARD_WALL_DESTROY = 0.5
REWARD_MISSED_SHOT = -0.2
REWARD_TIMESTEP = -0.01
REWARD_CLOSER = 0.05
REWARD_MINE_HIT = 5.0       # mine you placed hit the enemy
REWARD_MINE_SELF = -5.0     # you stepped on a mine

# Aim reward shaping
REWARD_AIM_GOOD = 0.3
REWARD_AIM_OKAY = 0.1
AIM_GOOD_THRESHOLD = 15.0   # degrees
AIM_OKAY_THRESHOLD = 30.0   # degrees

# Dodge reward shaping
REWARD_DODGE = 1.5           # successful dodge (bullet was on course but missed)
REWARD_UNDER_THREAT = -0.1   # per-tick penalty while a bullet is on collision course

# Threat projection
THREAT_LOOKAHEAD_TICKS = 20  # how many ticks ahead to project bullet trajectory

# State encoding bins
ANGLE_BINS = 16              # 22.5 degree resolution
DISTANCE_BINS = 4
DISTANCE_STEP = 200          # pixels per distance bin
ENEMY_FACING_BINS = 4
WALL_RAY_BINS = 3
THREAT_BINS = 3
BULLET_DIR_BINS = 9          # 8 directional + 0=none (direction bullet comes from)

# Experience replay
REPLAY_CAPACITY = 50000
REPLAY_BATCH_SIZE = 64       # increased from 32 for better learning
REPLAY_INTERVAL = 8          # replay every N agent steps (was 10)
REPLAY_MIN_SIZE = 500        # don't replay until buffer has this many
REPLAY_EPISODE_BURST = 128   # large replay burst at end of each episode
PER_ALPHA = 0.6              # prioritisation exponent (0=uniform, 1=full priority)

# Curriculum learning
CURRICULUM_PHASE_1_END = 300  # episodes 0-300: no walls
CURRICULUM_PHASE_2_END = 800  # episodes 301-800: few walls (4)
# episodes 801+: full walls (8-12)

# Rendering
FPS = 60
SCREEN_WIDTH = ARENA_WIDTH
SCREEN_HEIGHT = ARENA_HEIGHT + HUD_HEIGHT

# Movement sub-actions (0-4)
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
MOVE_ROTATE_LEFT = 2
MOVE_ROTATE_RIGHT = 3
MOVE_NOOP = 4
NUM_MOVE_ACTIONS = 5  # kept for legacy reference

# Turret sub-actions (0-2)
TURRET_LEFT = 0
TURRET_RIGHT = 1
TURRET_NOOP = 2

# Fire sub-actions (0-2)
FIRE_NONE = 0
FIRE_SHOOT = 1
FIRE_MINE = 2
