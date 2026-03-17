# ---------------------------------------------------------------------------
# ai/reward.py -- Reward computation (translates frame events to scalar)
# ---------------------------------------------------------------------------
"""Simplified reward function: 4 core signals only.

Kill, damage, got-hit, and timestep penalty. All shaping rewards
(aim, dodge, closer, movement, wall destroy) are removed to give
the DQN a clean gradient that points directly at winning.

The reward normalizer in DQNAgent handles scale balancing.
"""
from __future__ import annotations
import config as cfg
from game.game_manager import (EVT_HIT, EVT_DAMAGE, EVT_MINE_HIT,
                                EVT_BULLET_EXPIRE)


# Core rewards only
_R_TIMESTEP = cfg.REWARD_TIMESTEP        # -0.01  (small urgency pressure)
_R_HIT_ENEMY = cfg.REWARD_HIT_ENEMY      # +10.0  (kill)
_R_GOT_HIT = cfg.REWARD_GOT_HIT          # -10.0  (got killed)
_R_DAMAGE = cfg.REWARD_DAMAGE            # +5.0   (deal 1 HP damage)
_R_TOOK_DAMAGE = cfg.REWARD_TOOK_DAMAGE  # -5.0   (take 1 HP damage)
_R_MINE_HIT = cfg.REWARD_MINE_HIT        # +5.0   (mine hit enemy)
_R_MISSED_SHOT = cfg.REWARD_MISSED_SHOT  # -0.5   (bullet expired unused)


def compute_reward(events: dict, tank_id: int) -> float:
    """Translate frame events into a scalar reward for *tank_id*.

    Only 4 categories of reward:
      1. Kill / got killed          (+10 / -10)
      2. Damage dealt / taken       (+5 / -5)
      3. Mine hit on enemy          (+5)
      4. Missed shot penalty        (-0.5)
      + timestep cost               (-0.01)
    """
    r = _R_TIMESTEP

    # Kill events (HP reached 0)
    for shooter, victim in events[EVT_HIT]:
        if shooter == tank_id:
            r += _R_HIT_ENEMY
        if victim == tank_id:
            r += _R_GOT_HIT

    # Damage events (HP reduced but not dead yet)
    for shooter, victim, remaining_hp, bounced in events[EVT_DAMAGE]:
        if remaining_hp > 0:
            if shooter == tank_id:
                r += _R_DAMAGE
            if victim == tank_id:
                r += _R_TOOK_DAMAGE

    # Mine hit events
    for mine_owner, victim in events[EVT_MINE_HIT]:
        if mine_owner == tank_id:
            r += _R_MINE_HIT

    # Missed shots (bullet expired without hitting anything)
    for owner_id in events[EVT_BULLET_EXPIRE]:
        if owner_id == tank_id:
            r += _R_MISSED_SHOT

    return r
