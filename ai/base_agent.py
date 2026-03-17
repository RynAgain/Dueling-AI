# ---------------------------------------------------------------------------
# ai/base_agent.py -- Agent protocol (interface contract for all agent types)
# ---------------------------------------------------------------------------
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol defining the interface all agents must implement.

    Used by main.py, run_episode(), and run_parallel_episodes().
    """

    agent_id: int
    num_actions: int
    epsilon: float

    def choose_action(self, state) -> int:
        """Select an action given a state (tuple for Q-table, list for DQN)."""
        ...

    def learn(self, state, action: int, reward: float,
              next_state, done: bool = False) -> None:
        """Update the agent from a single transition."""
        ...

    def decay_epsilon(self) -> None:
        """Decay exploration rate (called once per episode)."""
        ...

    def episode_end_replay(self) -> None:
        """Optional end-of-episode replay burst."""
        ...

    def save(self, path: str, episode: int = 0) -> None:
        """Persist model to disk."""
        ...

    def load(self, path: str) -> dict:
        """Load model from disk.  Returns metadata dict."""
        ...

    def buffer_size(self) -> int:
        """Current replay buffer size (0 if not applicable)."""
        ...
