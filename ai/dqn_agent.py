# ---------------------------------------------------------------------------
# ai/dqn_agent.py -- Deep Q-Network agent (PyTorch, CPU-friendly)
# ---------------------------------------------------------------------------
from __future__ import annotations
import os
import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config as cfg


# ===========================================================================
# Neural network
# ===========================================================================

class QNetwork(nn.Module):
    """Small MLP: state_dim -> 128 -> 128 -> 64 -> num_actions."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# ===========================================================================
# Replay buffer (simple, numpy-free for portability)
# ===========================================================================

class DQNReplayBuffer:
    """Simple fixed-size replay buffer storing float-vector transitions."""

    def __init__(self, capacity: int = 50000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: list[float], action: int,
             reward: float, next_state: list[float], done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ===========================================================================
# DQN Agent
# ===========================================================================

class DQNAgent:
    """Double DQN agent with target network and experience replay.

    Runs entirely on CPU -- fast enough for this game (~10K params).
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.num_actions = cfg.NUM_ACTIONS
        self.state_dim = cfg.DQN_STATE_DIM

        # Hyperparameters (DQN-specific overrides)
        self.gamma = cfg.GAMMA
        self.epsilon = cfg.EPSILON_START
        self.epsilon_decay = getattr(cfg, 'DQN_EPSILON_DECAY', cfg.EPSILON_DECAY)
        self.epsilon_min = cfg.EPSILON_MIN
        self.lr = getattr(cfg, 'DQN_LR', 3e-3)
        self.batch_size = getattr(cfg, 'DQN_BATCH_SIZE', 128)
        self.target_update_freq = getattr(cfg, 'DQN_TARGET_UPDATE', 300)
        self.min_replay = getattr(cfg, 'DQN_MIN_REPLAY', 100)
        self._train_freq = getattr(cfg, 'DQN_TRAIN_FREQ', 4)
        self._grad_steps = getattr(cfg, 'DQN_GRAD_STEPS', 2)

        # Networks
        self.device = torch.device("cpu")
        self.policy_net = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = DQNReplayBuffer(
            capacity=getattr(cfg, 'REPLAY_CAPACITY', 50000))

        self._step_count = 0

    # ------------------------------------------------------------------
    def choose_action(self, state: list[float]) -> int:
        """Epsilon-greedy using the policy network."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    def learn(self, state: list[float], action: int,
              reward: float, next_state: list[float],
              done: bool = False) -> None:
        """Store transition and perform training step(s)."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._step_count += 1

        if len(self.replay_buffer) < self.min_replay:
            return

        # Train every N steps (skip some for speed)
        if self._step_count % self._train_freq != 0:
            return

        # Multiple gradient steps per train call
        for _ in range(self._grad_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            self._train_batch(batch)

        # Update target network periodically
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_batch(self, batch: list) -> None:
        """One gradient step on a batch of transitions."""
        if not batch:
            return

        states = torch.tensor([t[0] for t in batch],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in batch],
                               dtype=torch.long, device=self.device)
        rewards = torch.tensor([t[2] for t in batch],
                               dtype=torch.float32, device=self.device)
        next_states = torch.tensor([t[3] for t in batch],
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor([t[4] for t in batch],
                             dtype=torch.float32, device=self.device)

        # Current Q-values for chosen actions
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            q_next = self.target_net(next_states).gather(
                1, best_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Huber loss (smooth L1) -- less sensitive to outliers than MSE
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    def episode_end_replay(self) -> None:
        """Extra training at episode end (burst of gradient steps)."""
        if len(self.replay_buffer) < self.min_replay:
            return
        burst = getattr(cfg, 'REPLAY_EPISODE_BURST', 128)
        steps = max(burst // self.batch_size, 1)
        for _ in range(steps):
            batch = self.replay_buffer.sample(self.batch_size)
            self._train_batch(batch)
        # Sync target network after burst
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str, episode: int = 0) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "epsilon": self.epsilon,
            "episode": episode,
        }, path)

    def load(self, path: str) -> dict:
        """Load model. Returns metadata dict with 'epsilon' and 'episode'."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_state_dim = checkpoint.get("state_dim", self.state_dim)
        saved_num_actions = checkpoint.get("num_actions", self.num_actions)
        if saved_state_dim != self.state_dim or saved_num_actions != self.num_actions:
            print(f"    [migrate] DQN dimensions changed "
                  f"(state {saved_state_dim}->{self.state_dim}, "
                  f"actions {saved_num_actions}->{self.num_actions}), "
                  f"starting fresh network")
            return {"epsilon": self.epsilon, "episode": 0}
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint.get("step_count", 0)
        saved_eps = checkpoint.get("epsilon", self.epsilon)
        self.epsilon = saved_eps
        return {"epsilon": saved_eps, "episode": checkpoint.get("episode", 0)}

    # ------------------------------------------------------------------
    # Compatibility properties (so main.py can work with both agent types)
    @property
    def q_table_a(self) -> dict:
        """Dummy for compatibility with Q-table size reporting."""
        return {}

    @property
    def q_table_b(self) -> dict:
        """Dummy for compatibility."""
        return {}

    def network_param_count(self) -> int:
        """Total trainable parameters in the policy network."""
        return sum(p.numel() for p in self.policy_net.parameters())

    def buffer_size(self) -> int:
        """Current replay buffer size."""
        return len(self.replay_buffer)
