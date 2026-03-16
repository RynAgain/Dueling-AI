# ---------------------------------------------------------------------------
# main.py -- Entry point for the Tank Duel AI game
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import sys
import os
import time
import pygame
import config as cfg
from game.game_manager import GameManager, EVT_HIT, EVT_DAMAGE, EVT_WALL_DESTROY, \
    EVT_BULLET_EXPIRE, EVT_ROUND_OVER, EVT_ROUND_WINNER, EVT_MINE_HIT, EVT_DODGE
from ai.agent import QLearningAgent
from ai.state_encoder import StateEncoder
from rendering.renderer import Renderer


# ===========================================================================
# Reward computation (inline -- keeps things self-contained)
# ===========================================================================

# Cache config values as module-level locals for speed in the hot path
_R_TIMESTEP = cfg.REWARD_TIMESTEP
_R_HIT_ENEMY = cfg.REWARD_HIT_ENEMY
_R_GOT_HIT = cfg.REWARD_GOT_HIT
_R_DAMAGE = cfg.REWARD_DAMAGE
_R_TOOK_DAMAGE = cfg.REWARD_TOOK_DAMAGE
_R_WALL_DESTROY = cfg.REWARD_WALL_DESTROY
_R_MISSED_SHOT = cfg.REWARD_MISSED_SHOT
_R_CLOSER = cfg.REWARD_CLOSER
_R_AIM_GOOD = cfg.REWARD_AIM_GOOD
_R_AIM_OKAY = cfg.REWARD_AIM_OKAY
_AIM_GOOD_THRESH = cfg.AIM_GOOD_THRESHOLD
_AIM_OKAY_THRESH = cfg.AIM_OKAY_THRESHOLD
_R_DODGE = cfg.REWARD_DODGE
_R_UNDER_THREAT = cfg.REWARD_UNDER_THREAT
_R_MINE_HIT = cfg.REWARD_MINE_HIT
_R_MINE_SELF = cfg.REWARD_MINE_SELF


def compute_reward(events: dict, tank_id: int) -> float:
    """Translate frame events into a scalar reward for *tank_id*."""
    r = _R_TIMESTEP

    # Kill events (HP reached 0)
    for shooter, victim in events[EVT_HIT]:
        if shooter == tank_id:
            r += _R_HIT_ENEMY
        if victim == tank_id:
            r += _R_GOT_HIT

    # Damage events (HP reduced but not dead yet)
    for shooter, victim, remaining_hp in events[EVT_DAMAGE]:
        # Only reward damage that didn't result in a kill (kill already rewarded above)
        if remaining_hp > 0:
            if shooter == tank_id:
                r += _R_DAMAGE
            if victim == tank_id:
                r += _R_TOOK_DAMAGE

    # Mine hit events (mines only hurt the enemy, never the owner)
    for mine_owner, victim in events[EVT_MINE_HIT]:
        if mine_owner == tank_id:
            r += _R_MINE_HIT

    for destroyer_id in events[EVT_WALL_DESTROY]:
        if destroyer_id == tank_id:
            r += _R_WALL_DESTROY

    for owner_id in events[EVT_BULLET_EXPIRE]:
        if owner_id == tank_id:
            r += _R_MISSED_SHOT

    if events["closer"].get(tank_id, False):
        r += _R_CLOSER

    # Aim reward shaping (using turret angle now)
    aim_diff = events.get("aim_quality", {}).get(tank_id, 999.0)
    if aim_diff < _AIM_GOOD_THRESH:
        r += _R_AIM_GOOD
    elif aim_diff < _AIM_OKAY_THRESH:
        r += _R_AIM_OKAY

    # Dodge reward: outcome-based -- awarded when a bullet that was projected
    # to hit this tank ultimately missed (tank moved out of the way).
    for dodger_id in events.get(EVT_DODGE, []):
        if dodger_id == tank_id:
            r += _R_DODGE

    # Per-tick penalty while under threat (bullet on collision course).
    # Creates urgency to move out of the way immediately.
    if tank_id in events.get("under_threat", set()):
        r += _R_UNDER_THREAT

    return r


# ===========================================================================
# Run one full episode
# ===========================================================================

def run_episode(gm: GameManager, agents: list[QLearningAgent],
                encoder: StateEncoder, renderer: Renderer,
                learn: bool = True, fast: bool = False,
                stats: dict | None = None,
                ep_rewards: list[float] | None = None) -> int | None:
    """Play one episode.  Returns winner id (0 or 1) or None for draw."""
    gm.new_episode()
    rounds_played = 0
    max_rounds = cfg.MAX_ROUNDS_PER_EPISODE

    # Track total reward for this episode
    total_reward = 0.0

    # Local refs for the hot loop
    _encode = encoder.encode
    _step = gm.step
    _choose0 = agents[0].choose_action
    _choose1 = agents[1].choose_action
    _learn0 = agents[0].learn if learn else None
    _learn1 = agents[1].learn if learn else None
    _compute_reward = compute_reward

    while not gm.episode_over() and rounds_played < max_rounds:
        gm.new_round()
        rounds_played += 1

        tanks = gm.tanks
        bullets = gm.bullets
        arena = gm.arena
        walls = arena.walls

        while True:
            # -- handle Pygame events so window stays responsive ----------
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    renderer.quit()
                    sys.exit()

            # 1. Encode states (pass powerups and mines for new features)
            s0 = _encode(tanks, bullets, walls, arena, 0,
                         powerups=gm.powerups, mines=gm.mines)
            s1 = _encode(tanks, bullets, walls, arena, 1,
                         powerups=gm.powerups, mines=gm.mines)

            # 2. Choose actions
            a0 = _choose0(s0)
            a1 = _choose1(s1)

            # 3. Step the game
            events = _step(a0, a1)

            # 4. Encode next states + learn
            if learn:
                sn0 = _encode(tanks, bullets, walls, arena, 0,
                              powerups=gm.powerups, mines=gm.mines)
                sn1 = _encode(tanks, bullets, walls, arena, 1,
                              powerups=gm.powerups, mines=gm.mines)
                r0 = _compute_reward(events, 0)
                r1 = _compute_reward(events, 1)
                _learn0(s0, a0, r0, sn0)
                _learn1(s1, a1, r1, sn1)
                total_reward += r0 + r1

            # 5. Render
            if not fast:
                renderer.draw_frame(gm, agents, stats=stats)
                renderer.tick()

            # 6. Check round end
            if events[EVT_ROUND_OVER]:
                break

    # Decay exploration after each episode (only once -- shared agent)
    if learn:
        agents[0].decay_epsilon()
        agents[0].episode_end_replay()  # large replay burst to consolidate

    # Record total episode reward for graphing
    if ep_rewards is not None:
        ep_rewards.append(total_reward)

    return gm.get_episode_winner()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Tank Duel AI")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes (default 2000)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip per-frame rendering for faster training "
                             "(window stays open, draws between episodes)")
    parser.add_argument("--demo", action="store_true",
                        help="Load saved models and play visual demo only")
    parser.add_argument("--demo-episodes", type=int, default=5,
                        help="Episodes for post-training demo (default 5)")
    parser.add_argument("--model-dir", type=str, default="saved_models",
                        help="Directory for saved Q-tables")
    args = parser.parse_args()

    model_dir = args.model_dir

    # -- Create shared agent + encoder + renderer --------------------------
    # Self-play: both slots share the SAME QLearningAgent object.
    # The StateEncoder already encodes state relative to "me" vs "enemy",
    # so the Q-table is symmetric and can be shared.
    shared_agent = QLearningAgent(0)
    agents = [shared_agent, shared_agent]  # same object for both slots
    encoder = StateEncoder()
    renderer = Renderer()

    # ======================================================================
    # MODE: --demo  (load models, visual playback, no learning)
    # ======================================================================
    if args.demo:
        _load_models(agents, model_dir)
        num = args.episodes if args.episodes != 2000 else 5
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=False, fast=False, model_dir=model_dir)
        _print_summary(wins, num)
        renderer.quit()
        return

    # ======================================================================
    # DEFAULT MODE: training with rendering
    # ======================================================================
    num = args.episodes
    wins = _run_loop(agents, encoder, renderer, num,
                     learn=True, fast=args.fast, model_dir=model_dir)
    _save_models(agents, model_dir)
    _print_summary(wins, num)

    # Auto-demo with trained agents
    demo_eps = args.demo_episodes
    print(f"\n[*] Launching visual demo ({demo_eps} episodes) ...")
    for agent in agents:
        agent.epsilon = cfg.EPSILON_MIN  # exploit learned policy
    _run_loop(agents, encoder, renderer, demo_eps,
              learn=False, fast=False, model_dir=model_dir)

    renderer.quit()


# ===========================================================================
# Core training / demo loop
# ===========================================================================

def _run_loop(agents: list[QLearningAgent], encoder: StateEncoder,
              renderer: Renderer, num_episodes: int,
              learn: bool, fast: bool, model_dir: str) -> dict:
    """Run episodes with the renderer.  Returns win tally."""
    gm = GameManager()
    wins = {0: 0, 1: 0, None: 0}
    t_start = time.perf_counter()
    last_report = t_start

    # Tracking data for training graphs
    ep_rewards: list[float] = []
    epsilon_history: list[float] = []
    qtable_size_history: list[int] = []
    win_history: list[int | None] = []  # per-episode winner

    mode = "training" + (" (fast)" if fast else "") if learn else "demo"
    print(f"[*] {mode}: {num_episodes} episodes  "
          f"(eps_decay={cfg.EPSILON_DECAY}, timeout={cfg.ROUND_TIMEOUT})")

    for ep in range(1, num_episodes + 1):
        now = time.perf_counter()
        elapsed = now - t_start
        eps_sec = ep / max(elapsed, 0.001)
        stats = {"ep": ep, "total": num_episodes, "eps_sec": eps_sec}

        # Curriculum learning: set wall count based on episode
        if learn:
            if ep <= cfg.CURRICULUM_PHASE_1_END:
                wall_count = 0
            elif ep <= cfg.CURRICULUM_PHASE_2_END:
                wall_count = 4
            else:
                wall_count = None  # full random
            gm.set_wall_count(wall_count)

        winner = run_episode(gm, agents, encoder, renderer,
                             learn=learn, fast=fast, stats=stats,
                             ep_rewards=ep_rewards if learn else None)
        wins[winner] = wins.get(winner, 0) + 1

        # Record metrics for graphs
        if learn:
            epsilon_history.append(agents[0].epsilon)
            qtable_size_history.append(
                len(agents[0].q_table_a) + len(agents[0].q_table_b))
            win_history.append(winner)

        # Progress every 50 episodes
        if ep % 50 == 0 or ep == num_episodes:
            now2 = time.perf_counter()
            elapsed2 = now2 - t_start
            avg_rate = ep / max(elapsed2, 0.001)
            recent = now2 - last_report
            recent_rate = 50 / max(recent, 0.001)
            last_report = now2
            qtable_sz = len(agents[0].q_table_a) + len(agents[0].q_table_b)
            phase = ("P1" if ep <= cfg.CURRICULUM_PHASE_1_END
                     else "P2" if ep <= cfg.CURRICULUM_PHASE_2_END
                     else "P3")
            print(f"  Ep {ep:>5}/{num_episodes}  |  "
                  f"Wins B:{wins[0]} R:{wins[1]} D:{wins.get(None,0)}  |  "
                  f"eps={agents[0].epsilon:.4f}  |  "
                  f"Q={qtable_sz}  |  {phase}  |  "
                  f"{avg_rate:.1f} ep/s (recent {recent_rate:.1f})")

        # Periodic save every 200 episodes
        if learn and ep % 200 == 0:
            _save_models(agents, model_dir)

    # Generate training plots at end of training
    if learn and num_episodes > 0:
        _generate_training_plots(win_history, ep_rewards,
                                 epsilon_history, qtable_size_history)

    return wins


# ===========================================================================
# Training graphs
# ===========================================================================

def _generate_training_plots(win_history: list,
                             ep_rewards: list[float],
                             epsilon_history: list[float],
                             qtable_size_history: list[int]) -> None:
    """Save training progress plots to training_results.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed -- skipping training graphs. "
              "Install with: pip install matplotlib")
        return

    num_eps = len(win_history)
    if num_eps < 2:
        print("[!] Too few episodes for meaningful graphs, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tank Duel AI -- Training Results", fontsize=14, fontweight="bold")
    episodes = list(range(1, num_eps + 1))

    # 1. Win rate (50-episode rolling window) --------------------------------
    ax = axes[0][0]
    window = 50
    # Compute rolling win rate for agent 0 (Blue)
    win_rate_blue = []
    win_rate_red = []
    for i in range(num_eps):
        start = max(0, i - window + 1)
        chunk = win_history[start:i + 1]
        n = len(chunk)
        blue_wins = sum(1 for w in chunk if w == 0)
        red_wins = sum(1 for w in chunk if w == 1)
        win_rate_blue.append(blue_wins / n * 100)
        win_rate_red.append(red_wins / n * 100)
    ax.plot(episodes, win_rate_blue, label="Blue win%", color="#3C78DC", linewidth=1)
    ax.plot(episodes, win_rate_red, label="Red win%", color="#DC3C3C", linewidth=1)
    ax.set_title("Win Rate (50-ep rolling window)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average reward per episode ------------------------------------------
    ax = axes[0][1]
    ax.plot(episodes, ep_rewards, color="#2ca02c", linewidth=0.5, alpha=0.4)
    # Smoothed line
    if num_eps >= window:
        smoothed = []
        for i in range(num_eps):
            start = max(0, i - window + 1)
            smoothed.append(sum(ep_rewards[start:i + 1]) / (i - start + 1))
        ax.plot(episodes, smoothed, color="#2ca02c", linewidth=1.5,
                label=f"Smoothed ({window}-ep)")
        ax.legend()
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # 3. Epsilon decay curve -------------------------------------------------
    ax = axes[1][0]
    ax.plot(episodes, epsilon_history, color="#ff7f0e", linewidth=1.5)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    # 4. Q-table size growth -------------------------------------------------
    ax = axes[1][1]
    ax.plot(episodes, qtable_size_history, color="#9467bd", linewidth=1.5)
    ax.set_title("Q-table Size (unique states)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("States")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "training_results.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] Training graphs saved to {out_path}")


# ===========================================================================
# Helpers
# ===========================================================================

def _load_models(agents: list[QLearningAgent], model_dir: str) -> None:
    # Try shared model first, then fall back to legacy agent_0 / agent_1
    shared_path = os.path.join(model_dir, "agent_shared.pkl")
    if os.path.exists(shared_path):
        agents[0].load(shared_path)
        agents[0].epsilon = cfg.EPSILON_MIN
        print(f"[*] Loaded shared agent from {shared_path}")
        return
    # Legacy: try loading agent_0.pkl
    legacy_path = os.path.join(model_dir, "agent_0.pkl")
    if os.path.exists(legacy_path):
        agents[0].load(legacy_path)
        agents[0].epsilon = cfg.EPSILON_MIN
        print(f"[*] Loaded shared agent from legacy {legacy_path}")
        return
    print(f"[!] No saved model found in {model_dir}/, using untrained agent")


def _save_models(agents: list[QLearningAgent], model_dir: str) -> None:
    path = os.path.join(model_dir, "agent_shared.pkl")
    agents[0].save(path)
    print(f"[*] Shared Q-table saved to {path}")


def _print_summary(wins: dict, num_episodes: int) -> None:
    print(f"\n{'='*50}")
    print(f"  RESULTS -- {num_episodes} episodes")
    print(f"{'='*50}")
    print(f"  Blue wins : {wins.get(0, 0)}")
    print(f"  Red  wins : {wins.get(1, 0)}")
    print(f"  Draws     : {wins.get(None, 0)}")
    pct0 = wins.get(0, 0) / max(num_episodes, 1) * 100
    pct1 = wins.get(1, 0) / max(num_episodes, 1) * 100
    print(f"  Blue win% : {pct0:.1f}%")
    print(f"  Red  win% : {pct1:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
