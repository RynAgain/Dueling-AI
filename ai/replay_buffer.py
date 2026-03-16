# ---------------------------------------------------------------------------
# ai/replay_buffer.py -- Prioritized Experience Replay buffer for Q-learning
# ---------------------------------------------------------------------------
from __future__ import annotations
import random
import math


class ReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.

    Stores (state, action, reward, next_state) transitions weighted by
    priority (absolute TD-error + small epsilon).  Higher-priority
    transitions are sampled more frequently, so the agent re-learns
    from surprising / high-reward experiences much more often.

    Falls back to uniform sampling if priorities are not set.
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        """
        Parameters
        ----------
        capacity : max number of transitions to store (FIFO eviction).
        alpha    : prioritisation exponent (0 = uniform, 1 = full priority).
        """
        self.capacity: int = capacity
        self.alpha: float = alpha  # how much prioritisation to use
        self._buffer: list = []
        self._priorities: list[float] = []
        self._pos: int = 0          # circular write position
        self._max_priority: float = 1.0  # new transitions get max priority

    # ------------------------------------------------------------------
    def push(self, state: tuple, action: int, reward: float,
             next_state: tuple) -> None:
        """Add a transition with max priority (will be replayed soon)."""
        entry = (state, action, reward, next_state)
        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
            self._priorities.append(self._max_priority)
        else:
            self._buffer[self._pos] = entry
            self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    # ------------------------------------------------------------------
    def sample(self, batch_size: int = 32) -> list[tuple[int, tuple]]:
        """Sample a batch weighted by priority.

        Returns list of (index, (s, a, r, s')) tuples.
        The index is needed to update priorities after learning.
        """
        n = len(self._buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        # Compute sampling probabilities from priorities
        priorities = self._priorities[:n]
        # Raise to alpha power
        probs = [p ** self.alpha for p in priorities]
        total = sum(probs)
        if total <= 0:
            # Fallback to uniform
            indices = random.sample(range(n), batch_size)
        else:
            # Weighted sampling without replacement
            probs_norm = [p / total for p in probs]
            indices = self._weighted_sample(probs_norm, batch_size)

        return [(idx, self._buffer[idx]) for idx in indices]

    # ------------------------------------------------------------------
    def update_priorities(self, indices: list[int],
                          td_errors: list[float]) -> None:
        """Update priorities for sampled transitions based on TD-error."""
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + 1e-5  # small epsilon to avoid zero
            self._priorities[idx] = priority
            if priority > self._max_priority:
                self._max_priority = priority

    # ------------------------------------------------------------------
    def sample_uniform(self, batch_size: int = 32) -> list[tuple]:
        """Simple uniform sample (legacy compatibility)."""
        n = len(self._buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)
        indices = random.sample(range(n), batch_size)
        return [self._buffer[idx] for idx in indices]

    # ------------------------------------------------------------------
    @staticmethod
    def _weighted_sample(probs: list[float], k: int) -> list[int]:
        """Weighted sampling without replacement using reservoir method."""
        n = len(probs)
        if k >= n:
            return list(range(n))
        # Use the Efraimidis-Spirakis algorithm for speed
        keys = []
        for i, p in enumerate(probs):
            if p <= 0:
                continue
            u = random.random()
            key = u ** (1.0 / p) if p > 0 else 0
            keys.append((key, i))
        keys.sort(reverse=True)
        return [idx for _, idx in keys[:k]]

    # ------------------------------------------------------------------
    def get_data_for_save(self) -> dict:
        """Return serializable snapshot of the buffer."""
        return {
            "buffer": list(self._buffer),
            "priorities": list(self._priorities[:len(self._buffer)]),
            "max_priority": self._max_priority,
        }

    def load_data(self, data: dict) -> None:
        """Restore buffer from saved snapshot."""
        self._buffer = data.get("buffer", [])
        self._priorities = data.get("priorities", [1.0] * len(self._buffer))
        self._max_priority = data.get("max_priority", 1.0)
        self._pos = len(self._buffer) % self.capacity
        # Ensure priorities list matches buffer length
        while len(self._priorities) < len(self._buffer):
            self._priorities.append(self._max_priority)

    def __len__(self) -> int:
        return len(self._buffer)
