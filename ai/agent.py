# ---------------------------------------------------------------------------
# ai/agent.py -- Double Q-learning agent with Prioritized Experience Replay
# ---------------------------------------------------------------------------
from __future__ import annotations
import os
import pickle
import random
from collections import defaultdict

import config as cfg
from ai.replay_buffer import ReplayBuffer


class QLearningAgent:
    """Double Q-learning agent with Prioritized Experience Replay.

    Improvements over basic Q-learning:
    - **Double Q-learning**: Two Q-tables (A and B) reduce value overestimation.
      Action selected by one table, evaluated by the other.
    - **Prioritized Experience Replay**: High TD-error transitions are replayed
      more often, focusing learning on surprising experiences.
    - **Episode-end replay burst**: Large batch of replays after each episode
      to consolidate that episode's lessons.
    - **Persistent replay buffer**: Saved/loaded alongside Q-tables so
      training can resume without losing experience.
    """

    def __init__(self, agent_id: int):
        self.agent_id: int = agent_id
        self.alpha: float = cfg.ALPHA
        self.gamma: float = cfg.GAMMA
        self.epsilon: float = cfg.EPSILON_START
        self.epsilon_decay: float = cfg.EPSILON_DECAY
        self.epsilon_min: float = cfg.EPSILON_MIN
        self.num_actions: int = cfg.NUM_ACTIONS

        # Double Q-learning: two Q-tables
        self.q_table_a: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0] * self.num_actions
        )
        self.q_table_b: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0] * self.num_actions
        )

        # Legacy alias -- combined view for action selection
        # (sum of both tables for better estimates)
        @property
        def q_table(self_inner):
            return self_inner.q_table_a
        self.q_table = self.q_table_a  # default reference for migration/save

        # Experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=cfg.REPLAY_CAPACITY,
            alpha=getattr(cfg, 'PER_ALPHA', 0.6)
        )
        self._replay_batch_size: int = cfg.REPLAY_BATCH_SIZE
        self._replay_interval: int = cfg.REPLAY_INTERVAL
        self._replay_min_size: int = cfg.REPLAY_MIN_SIZE
        self._episode_burst_size: int = getattr(cfg, 'REPLAY_EPISODE_BURST', 128)
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    def choose_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection using combined Q-values."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # Use sum of both Q-tables for action selection (better estimate)
        qa = self.q_table_a[state]
        qb = self.q_table_b[state]
        combined = [a + b for a, b in zip(qa, qb)]
        max_q = max(combined)
        best = [a for a, q in enumerate(combined) if q == max_q]
        return random.choice(best)

    # ------------------------------------------------------------------
    def learn(self, state: tuple, action: int,
              reward: float, next_state: tuple) -> None:
        """Double Q-learning update + prioritized replay."""
        # Online double-Q update
        td_error = self._double_q_update(state, action, reward, next_state)

        # Store transition with initial priority based on TD-error
        self.replay_buffer.push(state, action, reward, next_state)

        # Periodic prioritized replay
        self._step_counter += 1
        if (self._step_counter % self._replay_interval == 0
                and len(self.replay_buffer) >= self._replay_min_size):
            self._prioritized_replay()

    # ------------------------------------------------------------------
    def episode_end_replay(self) -> None:
        """Large replay burst at the end of an episode to consolidate learning."""
        if len(self.replay_buffer) >= self._replay_min_size:
            self._prioritized_replay(batch_size=self._episode_burst_size)

    # ------------------------------------------------------------------
    def _double_q_update(self, state: tuple, action: int,
                         reward: float, next_state: tuple) -> float:
        """Double Q-learning: randomly update table A or B.

        If updating A: use A to select best next action, B to evaluate it.
        If updating B: use B to select best next action, A to evaluate it.
        Returns the TD-error for priority computation.
        """
        if random.random() < 0.5:
            # Update table A
            qa = self.q_table_a[state][action]
            # Select best action using A
            best_a = max(range(self.num_actions),
                         key=lambda a: self.q_table_a[next_state][a])
            # Evaluate using B
            td_target = reward + self.gamma * self.q_table_b[next_state][best_a]
            td_error = td_target - qa
            self.q_table_a[state][action] += self.alpha * td_error
        else:
            # Update table B
            qb = self.q_table_b[state][action]
            # Select best action using B
            best_b = max(range(self.num_actions),
                         key=lambda a: self.q_table_b[next_state][a])
            # Evaluate using A
            td_target = reward + self.gamma * self.q_table_a[next_state][best_b]
            td_error = td_target - qb
            self.q_table_b[state][action] += self.alpha * td_error

        return td_error

    def _prioritized_replay(self, batch_size: int | None = None) -> None:
        """Sample a prioritized batch and perform double-Q updates."""
        bs = batch_size or self._replay_batch_size
        samples = self.replay_buffer.sample(bs)
        if not samples:
            return

        indices = []
        td_errors = []
        for idx, (s, a, r, ns) in samples:
            td = self._double_q_update(s, a, r, ns)
            indices.append(idx)
            td_errors.append(td)

        # Update priorities so high-error transitions get replayed more
        self.replay_buffer.update_priorities(indices, td_errors)

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        payload = {
            "q_table_a": dict(self.q_table_a),
            "q_table_b": dict(self.q_table_b),
            "num_actions": self.num_actions,
            "state_size": self._detect_state_size(),
            "replay_buffer": self.replay_buffer.get_data_for_save(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _detect_state_size(self) -> int:
        for key in self.q_table_a:
            return len(key)
        for key in self.q_table_b:
            return len(key)
        return 0

    def load(self, path: str) -> None:
        """Load Q-tables with automatic migration for state/action changes.

        Handles:
        - Old single-table format (plain dict or {q_table: ...})
        - New double-table format ({q_table_a: ..., q_table_b: ...})
        - State tuple padding/truncation
        - Action array resizing
        - Replay buffer restoration
        """
        with open(path, "rb") as f:
            raw = pickle.load(f)

        # Detect format and extract tables
        if isinstance(raw, dict) and "q_table_a" in raw:
            # New double-table format
            saved_a = raw["q_table_a"]
            saved_b = raw["q_table_b"]
            saved_num_actions = raw.get("num_actions", 0)
            saved_state_size = raw.get("state_size", 0)
            replay_data = raw.get("replay_buffer", None)
        elif isinstance(raw, dict) and "q_table" in raw:
            # Old single-table format with metadata
            saved_a = raw["q_table"]
            saved_b = {}  # B starts fresh
            saved_num_actions = raw.get("num_actions", 0)
            saved_state_size = raw.get("state_size", 0)
            replay_data = raw.get("replay_buffer", None)
        else:
            # Legacy: plain dict
            saved_a = raw
            saved_b = {}
            saved_num_actions = 0
            saved_state_size = 0
            replay_data = None
            for k, v in saved_a.items():
                saved_state_size = len(k)
                saved_num_actions = len(v)
                break

        current_num_actions = self.num_actions
        current_state_size = 15  # must match state_encoder output

        # Migrate both tables
        migrated_a, count_a = self._migrate_table(
            saved_a, saved_state_size, current_state_size,
            saved_num_actions, current_num_actions)
        migrated_b, count_b = self._migrate_table(
            saved_b, saved_state_size, current_state_size,
            saved_num_actions, current_num_actions)

        self.q_table_a = defaultdict(
            lambda: [0.0] * self.num_actions, migrated_a)
        self.q_table_b = defaultdict(
            lambda: [0.0] * self.num_actions, migrated_b)
        self.q_table = self.q_table_a  # alias

        # Restore replay buffer
        if replay_data:
            self.replay_buffer.load_data(replay_data)
            print(f"    [replay] Restored {len(self.replay_buffer)} transitions")

        needs_state = (saved_state_size != 0 and
                       saved_state_size != current_state_size)
        needs_action = (saved_num_actions != 0 and
                        saved_num_actions != current_num_actions)
        if needs_state or needs_action:
            parts = []
            if needs_state:
                parts.append(f"state {saved_state_size}->{current_state_size}")
            if needs_action:
                parts.append(f"actions {saved_num_actions}->{current_num_actions}")
            print(f"    [migrate] Q-tables adapted: {', '.join(parts)} "
                  f"(A:{len(migrated_a)}, B:{len(migrated_b)} entries)")

    @staticmethod
    def _migrate_table(data: dict,
                       old_state_size: int, new_state_size: int,
                       old_num_actions: int, new_num_actions: int
                       ) -> tuple[dict, int]:
        """Migrate a single Q-table dict. Returns (migrated_dict, num_migrated)."""
        needs_state = (old_state_size != 0 and old_state_size != new_state_size)
        needs_action = (old_num_actions != 0 and old_num_actions != new_num_actions)

        if not needs_state and not needs_action:
            return data, 0

        migrated: dict[tuple, list[float]] = {}
        count = 0
        for state_key, action_values in data.items():
            new_key = state_key
            if needs_state:
                key_list = list(state_key)
                if len(key_list) < new_state_size:
                    key_list.extend([0] * (new_state_size - len(key_list)))
                elif len(key_list) > new_state_size:
                    key_list = key_list[:new_state_size]
                new_key = tuple(key_list)
                count += 1
            if needs_action:
                vals = list(action_values)
                if len(vals) < new_num_actions:
                    vals.extend([0.0] * (new_num_actions - len(vals)))
                elif len(vals) > new_num_actions:
                    vals = vals[:new_num_actions]
                action_values = vals
            migrated[new_key] = action_values
        return migrated, count
