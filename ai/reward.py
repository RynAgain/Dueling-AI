# ---------------------------------------------------------------------------
# ai/reward.py -- Reward computation (translates frame events to scalar)
# ---------------------------------------------------------------------------
"""Simplified reward function: 4 core signals only."""
from __future__ import annotations
from game.game_manager import EVT_HIT, EVT_DAMAGE, EVT_BULLET_EXPIRE
import config as cfg

_R_TIMESTEP = cfg.REWARD_TIMESTEP
_R_HIT_ENEMY = cfg.REWARD_HIT_ENEMY
_R_GOT_HIT = cfg.REWARD_GOT_HIT
_R_DAMAGE = cfg.REWARD_DAMAGE
_R_TOOK_DAMAGE = cfg.REWARD_TOOK_DAMAGE
_R_MISSED_SHOT = cfg.REWARD_MISSED_SHOT


def compute_reward(events: dict, tank_id: int) -> float:
    """Translate frame events into a scalar reward for *tank_id*."""
    r = _R_TIMESTEP

    for shooter, victim in events[EVT_HIT]:
        if shooter == tank_id:
            r += _R_HIT_ENEMY
        if victim == tank_id:
            r += _R_GOT_HIT

    for shooter, victim, remaining_hp, bounced in events[EVT_DAMAGE]:
        if remaining_hp > 0:
            if shooter == tank_id:
                r += _R_DAMAGE
            if victim == tank_id:
                r += _R_TOOK_DAMAGE

    for owner_id in events[EVT_BULLET_EXPIRE]:
        if owner_id == tank_id:
            r += _R_MISSED_SHOT

    return r
