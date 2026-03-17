# ---------------------------------------------------------------------------
# ai/population.py -- Population-Based Training (PBT) for DQN agents
# ---------------------------------------------------------------------------
"""Manages a population of DQN agents that compete in round-robin tournaments.

Each generation:
  1. All agents play round-robin matches against each other.
  2. Agents are ranked by composite fitness (Elo + win rate).
  3. Top performers (elite) survive unchanged.
  4. Bottom performers are replaced by clones of top performers,
     with small weight noise and hyperparameter mutations.

This forces agents to develop robust strategies that beat *diverse*
opponents rather than co-adapting with a single self-play partner.
"""
from __future__ import annotations
import math
import os
import random
import time
from dataclasses import dataclass, field
from itertools import combinations

import pygame
import config as cfg
from ai.dqn_agent import DQNAgent
from ai.reward import compute_reward
from game.game_manager import (GameManager, EVT_ROUND_OVER, EVT_HIT,
                                EVT_DAMAGE, EVT_MINE_HIT, EVT_DODGE)


# ===========================================================================
# Fitness tracking
# ===========================================================================

@dataclass
class AgentFitness:
    """Tracks all performance metrics for a single agent across a generation."""

    agent_idx: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    damage_dealt: int = 0      # total HP damage inflicted
    damage_taken: int = 0      # total HP damage received
    kills: int = 0             # times reduced enemy to 0 HP
    deaths: int = 0            # times own HP reached 0
    rounds_won: int = 0        # individual rounds won (not just episodes)
    rounds_lost: int = 0
    dodges: int = 0            # successful bullet dodges
    total_reward: float = 0.0  # cumulative reward across all matches
    elo: float = 1000.0        # Elo rating (persistent across generations)

    @property
    def matches(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.matches, 1)

    @property
    def kd_ratio(self) -> float:
        return self.kills / max(self.deaths, 1)

    @property
    def damage_ratio(self) -> float:
        return self.damage_dealt / max(self.damage_taken, 1)

    @property
    def composite_score(self) -> float:
        """Weighted composite used for ranking.

        Elo dominates, but damage ratio and win rate break ties.
        """
        return self.elo * 0.6 + self.win_rate * 200 + self.damage_ratio * 50

    def reset_generation(self) -> None:
        """Reset per-generation stats, keeping Elo."""
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.kills = 0
        self.deaths = 0
        self.rounds_won = 0
        self.rounds_lost = 0
        self.dodges = 0
        self.total_reward = 0.0


def _update_elo(winner_elo: float, loser_elo: float,
                k: float = 32.0) -> tuple[float, float]:
    """Standard Elo update. Returns (new_winner_elo, new_loser_elo)."""
    expected_w = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
    expected_l = 1.0 - expected_w
    new_w = winner_elo + k * (1.0 - expected_w)
    new_l = loser_elo + k * (0.0 - expected_l)
    return new_w, new_l


def _update_elo_draw(elo_a: float, elo_b: float,
                     k: float = 16.0) -> tuple[float, float]:
    """Elo update for a draw (each player scores 0.5)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = elo_a + k * (0.5 - expected_a)
    new_b = elo_b + k * (0.5 - expected_b)
    return new_a, new_b


# ===========================================================================
# Population
# ===========================================================================

class Population:
    """A population of DQN agents trained via generational selection."""

    def __init__(self, size: int | None = None, encoder=None):
        self.size = size or cfg.PBT_POPULATION_SIZE
        self.encoder = encoder
        self.agents: list[DQNAgent] = []
        self.fitness: list[AgentFitness] = []
        self.generation: int = 0
        self.cumulative_episode: int = 0

        # Create population
        for i in range(self.size):
            agent = DQNAgent(i)
            self.agents.append(agent)
            self.fitness.append(AgentFitness(agent_idx=i))

        # Shared game manager for matches
        self._gm = GameManager()

    # ==================================================================
    # Round-robin tournament
    # ==================================================================
    def run_tournament(self, matches_per_pair: int | None = None,
                       fast: bool = True,
                       renderer=None) -> list[AgentFitness]:
        """Play every agent pair against each other.

        Returns list of AgentFitness sorted by composite score descending.
        """
        n = self.size
        mpp = matches_per_pair or cfg.PBT_MATCHES_PER_PAIR

        # Reset per-generation stats (keep Elo)
        for f in self.fitness:
            f.reset_generation()

        pairs = list(combinations(range(n), 2))
        total_matches = len(pairs) * mpp

        print(f"    [tournament] {n} agents, {len(pairs)} pairs "
              f"x {mpp} matches = {total_matches} games")

        for i, j in pairs:
            for _ in range(mpp):
                self._play_match(i, j, fast=fast, renderer=renderer)
                self.cumulative_episode += 1

        # Sort by composite score (descending)
        ranked = sorted(self.fitness, key=lambda f: -f.composite_score)
        return ranked

    def _play_match(self, idx_a: int, idx_b: int,
                    fast: bool = True,
                    renderer=None) -> int | None:
        """Play one full episode between agent[idx_a] and agent[idx_b].

        Both agents learn from the match.
        Returns the index of the winning agent, or None for draw.
        """
        gm = self._gm
        agent_a = self.agents[idx_a]
        agent_b = self.agents[idx_b]
        fit_a = self.fitness[idx_a]
        fit_b = self.fitness[idx_b]
        encoder = self.encoder

        gm.new_episode()
        max_rounds = cfg.MAX_ROUNDS_PER_EPISODE
        rounds_played = 0

        # Apply curriculum based on cumulative episode count
        _apply_curriculum_to_gm(gm, self.cumulative_episode)

        match_reward_a = 0.0
        match_reward_b = 0.0

        while not gm.episode_over() and rounds_played < max_rounds:
            gm.new_round()
            if hasattr(encoder, 'reset'):
                encoder.reset()  # clear frame stacking history
            rounds_played += 1

            tanks = gm.tanks
            bullets = gm.bullets
            arena = gm.arena
            walls = arena.walls

            while True:
                # Process events to keep OS happy
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise SystemExit("Window closed")

                s0 = encoder.encode(tanks, bullets, walls, arena, 0,
                                    powerups=gm.powerups, mines=gm.mines)
                s1 = encoder.encode(tanks, bullets, walls, arena, 1,
                                    powerups=gm.powerups, mines=gm.mines)
                a0 = agent_a.choose_action(s0)
                a1 = agent_b.choose_action(s1)
                events = gm.step(a0, a1)

                # Learn from both perspectives
                sn0 = encoder.encode(tanks, bullets, walls, arena, 0,
                                     powerups=gm.powerups, mines=gm.mines)
                sn1 = encoder.encode(tanks, bullets, walls, arena, 1,
                                     powerups=gm.powerups, mines=gm.mines)
                r0 = compute_reward(events, 0)
                r1 = compute_reward(events, 1)
                rd = events[EVT_ROUND_OVER]
                agent_a.learn(s0, a0, r0, sn0, done=rd)
                agent_b.learn(s1, a1, r1, sn1, done=rd)

                match_reward_a += r0
                match_reward_b += r1

                # Track damage events
                # Tank 0 = agent_a, Tank 1 = agent_b
                for shooter, victim, remaining_hp, bounced in events[EVT_DAMAGE]:
                    if shooter == 0:  # agent_a dealt damage
                        fit_a.damage_dealt += 1
                        fit_b.damage_taken += 1
                    elif shooter == 1:  # agent_b dealt damage
                        fit_b.damage_dealt += 1
                        fit_a.damage_taken += 1

                # Track kills
                for shooter, victim in events[EVT_HIT]:
                    if shooter == 0:
                        fit_a.kills += 1
                        fit_b.deaths += 1
                    elif shooter == 1:
                        fit_b.kills += 1
                        fit_a.deaths += 1

                # Track mine hits as damage
                for mine_owner, victim in events[EVT_MINE_HIT]:
                    if mine_owner == 0:
                        fit_a.damage_dealt += 1
                        fit_b.damage_taken += 1
                    elif mine_owner == 1:
                        fit_b.damage_dealt += 1
                        fit_a.damage_taken += 1

                # Track dodges
                for dodger_id in events.get(EVT_DODGE, []):
                    if dodger_id == 0:
                        fit_a.dodges += 1
                    elif dodger_id == 1:
                        fit_b.dodges += 1

                if not fast and renderer is not None:
                    renderer.draw_frame(gm, [agent_a, agent_b])
                    renderer.tick()

                if rd:
                    # Track round winner
                    rw = events.get("round_winner", None)
                    if rw == 0:
                        fit_a.rounds_won += 1
                        fit_b.rounds_lost += 1
                    elif rw == 1:
                        fit_b.rounds_won += 1
                        fit_a.rounds_lost += 1
                    break

        # Episode-end replay burst for both agents
        agent_a.episode_end_replay()
        agent_b.episode_end_replay()

        # Record match result
        fit_a.total_reward += match_reward_a
        fit_b.total_reward += match_reward_b

        winner_tank = gm.get_episode_winner()
        if winner_tank == 0:
            fit_a.wins += 1
            fit_b.losses += 1
            fit_a.elo, fit_b.elo = _update_elo(fit_a.elo, fit_b.elo)
            return idx_a
        elif winner_tank == 1:
            fit_b.wins += 1
            fit_a.losses += 1
            fit_b.elo, fit_a.elo = _update_elo(fit_b.elo, fit_a.elo)
            return idx_b
        else:
            fit_a.draws += 1
            fit_b.draws += 1
            fit_a.elo, fit_b.elo = _update_elo_draw(fit_a.elo, fit_b.elo)
            return None

    # ==================================================================
    # Selection + mutation
    # ==================================================================
    def evolve(self, rankings: list[AgentFitness]) -> None:
        """Replace bottom performers with mutated clones of top performers.

        rankings: list of AgentFitness sorted by composite score descending.
        """
        n = self.size
        elite_count = max(1, int(n * cfg.PBT_ELITE_RATIO))
        elite_indices = [f.agent_idx for f in rankings[:elite_count]]
        bottom_indices = [f.agent_idx for f in rankings[elite_count:]]

        noise_std = cfg.PBT_WEIGHT_NOISE_STD
        lr_range = cfg.PBT_MUTATE_LR_RANGE
        eps_range = cfg.PBT_MUTATE_EPSILON_RANGE

        for bottom_idx in bottom_indices:
            # Pick a random elite to clone from
            parent_idx = random.choice(elite_indices)
            parent = self.agents[parent_idx]
            child = self.agents[bottom_idx]

            child.clone_from(parent, noise_std=noise_std)
            child.mutate_hyperparameters(lr_range=lr_range, eps_range=eps_range)

            # Inherit parent's Elo (slightly regressed toward mean)
            parent_elo = self.fitness[parent_idx].elo
            self.fitness[bottom_idx].elo = 1000 + (parent_elo - 1000) * 0.8

        # Decay epsilon for all agents
        for agent in self.agents:
            agent.decay_epsilon()

        self.generation += 1

    # ==================================================================
    # Fitness display
    # ==================================================================
    def print_fitness(self, rankings: list[AgentFitness]) -> None:
        """Print a formatted fitness table for the current generation."""
        print()
        print(f"    {'Rk':>2}  {'Agent':>5}  {'W-L-D':>7}  {'WR%':>5}  "
              f"{'Elo':>6}  {'K/D':>5}  {'Dmg':>5}  {'Dodg':>4}  "
              f"{'Reward':>8}  {'Score':>7}")
        print(f"    {'--':>2}  {'-----':>5}  {'-------':>7}  {'-----':>5}  "
              f"{'------':>6}  {'-----':>5}  {'-----':>5}  {'----':>4}  "
              f"{'--------':>8}  {'-------':>7}")
        for rank, f in enumerate(rankings):
            marker = " <--best" if rank == 0 else ""
            elite = " [E]" if rank < max(1, int(self.size * cfg.PBT_ELITE_RATIO)) else ""
            wld = f"{f.wins}-{f.losses}-{f.draws}"
            print(f"    {rank+1:>2}  A{f.agent_idx:>3}  {wld:>7}  "
                  f"{f.win_rate*100:>4.0f}%  {f.elo:>6.0f}  "
                  f"{f.kd_ratio:>5.1f}  "
                  f"{f.damage_dealt:>3}/{f.damage_taken:<3}  "
                  f"{f.dodges:>4}  "
                  f"{f.total_reward:>+8.1f}  "
                  f"{f.composite_score:>7.0f}{elite}{marker}")
        print()

    # ==================================================================
    # Get best agent
    # ==================================================================
    def best_agent(self, rankings: list[AgentFitness]) -> DQNAgent:
        """Return the top-ranked agent."""
        return self.agents[rankings[0].agent_idx]

    # ==================================================================
    # Save / load
    # ==================================================================
    def save_best(self, path: str, rankings: list[AgentFitness],
                  episode: int = 0) -> None:
        """Save the best agent's model."""
        best = self.best_agent(rankings)
        best.save(path, episode=episode)

    def save_all(self, model_dir: str, episode: int = 0) -> None:
        """Save all agents in the population."""
        os.makedirs(model_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            path = os.path.join(model_dir, f"pbt_agent_{i}.pt")
            agent.save(path, episode=episode)

    def load_best_into_all(self, path: str) -> dict:
        """Load a single model and distribute it to all agents.

        Used to seed the population from a pre-trained model.
        """
        meta = {"episode": 0, "epsilon": cfg.EPSILON_START}
        if os.path.exists(path):
            result = self.agents[0].load(path)
            if isinstance(result, dict):
                meta.update(result)
            # Copy to all other agents with small noise for diversity
            for i in range(1, self.size):
                self.agents[i].clone_from(
                    self.agents[0],
                    noise_std=cfg.PBT_WEIGHT_NOISE_STD * 2)
                self.agents[i].mutate_hyperparameters()
            print(f"    [population] Seeded {self.size} agents from {path}")
        return meta


def _apply_curriculum_to_gm(gm: GameManager, cumulative_ep: int) -> None:
    """Set game features based on curriculum phase."""
    if cumulative_ep <= cfg.CURRICULUM_PHASE_1_END:
        gm.set_wall_count(0)
        gm.powerups_enabled = False
        gm.mines_enabled = False
    elif cumulative_ep <= cfg.CURRICULUM_PHASE_2_END:
        gm.set_wall_count(4)
        gm.powerups_enabled = False
        gm.mines_enabled = False
    elif cumulative_ep <= cfg.CURRICULUM_PHASE_3_END:
        gm.set_wall_count(None)
        gm.powerups_enabled = True
        gm.mines_enabled = False
    elif cumulative_ep <= cfg.CURRICULUM_PHASE_4_END:
        gm.set_wall_count(None)
        gm.powerups_enabled = True
        gm.mines_enabled = True
    else:
        gm.set_wall_count(None)
        gm.powerups_enabled = True
        gm.mines_enabled = True
