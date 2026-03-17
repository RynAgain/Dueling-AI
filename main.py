# ---------------------------------------------------------------------------
# main.py -- Entry point for the Tank Duel AI game
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import sys
import os
import time
import random
import pygame
import config as cfg
from game.game_manager import GameManager, EVT_HIT, EVT_DAMAGE, EVT_WALL_DESTROY, \
    EVT_BULLET_EXPIRE, EVT_ROUND_OVER, EVT_ROUND_WINNER, EVT_MINE_HIT, EVT_DODGE
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
_R_MOVEMENT = cfg.REWARD_MOVEMENT
_MOVEMENT_DIST_MIN = cfg.MOVEMENT_REWARD_DIST
_MOVEMENT_DIST_CAP = cfg.MOVEMENT_REWARD_CAP


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

    # Dodge reward: outcome-based
    for dodger_id in events.get(EVT_DODGE, []):
        if dodger_id == tank_id:
            r += _R_DODGE

    # Per-tick penalty while under threat
    if tank_id in events.get("under_threat", set()):
        r += _R_UNDER_THREAT

    # Movement reward: encourage tanks to leave spawn and move around
    sd = events.get("spawn_dist", {}).get(tank_id, 0.0)
    if sd >= _MOVEMENT_DIST_MIN:
        ratio = min((sd - _MOVEMENT_DIST_MIN) / max(_MOVEMENT_DIST_CAP - _MOVEMENT_DIST_MIN, 1), 1.0)
        r += _R_MOVEMENT * ratio

    return r


# ===========================================================================
# Run one full episode (works with both Q-learning and DQN agents)
# ===========================================================================

def run_episode(gm: GameManager, agents, encoder, renderer: Renderer,
                learn: bool = True, fast: bool = False,
                stats: dict | None = None,
                ep_rewards: list[float] | None = None,
                use_dqn: bool = False) -> int | None:
    """Play one episode (single game).  Returns winner id or None."""
    gm.new_episode()
    rounds_played = 0
    max_rounds = cfg.MAX_ROUNDS_PER_EPISODE
    total_reward = 0.0

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
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise SystemExit("Window closed")

            s0 = _encode(tanks, bullets, walls, arena, 0,
                         powerups=gm.powerups, mines=gm.mines)
            s1 = _encode(tanks, bullets, walls, arena, 1,
                         powerups=gm.powerups, mines=gm.mines)
            a0 = _choose0(s0)
            a1 = _choose1(s1)
            events = _step(a0, a1)

            if learn:
                sn0 = _encode(tanks, bullets, walls, arena, 0,
                              powerups=gm.powerups, mines=gm.mines)
                sn1 = _encode(tanks, bullets, walls, arena, 1,
                              powerups=gm.powerups, mines=gm.mines)
                r0 = _compute_reward(events, 0)
                r1 = _compute_reward(events, 1)
                round_done = events[EVT_ROUND_OVER]
                if use_dqn:
                    _learn0(s0, a0, r0, sn0, done=round_done)
                    _learn1(s1, a1, r1, sn1, done=round_done)
                else:
                    _learn0(s0, a0, r0, sn0)
                    _learn1(s1, a1, r1, sn1)
                total_reward += r0 + r1

            if not fast:
                renderer.draw_frame(gm, agents, stats=stats)
                renderer.tick()

            if events[EVT_ROUND_OVER]:
                break

    if learn:
        agents[0].decay_epsilon()
        agents[0].episode_end_replay()

    if ep_rewards is not None:
        ep_rewards.append(total_reward)

    return gm.get_episode_winner()


# ===========================================================================
# Parallel episode runner (N games, one shared agent)
# ===========================================================================

def run_parallel_episodes(game_managers: list[GameManager], agent,
                          encoder, renderer: Renderer,
                          fast: bool = True,
                          stats: dict | None = None,
                          use_dqn: bool = True) -> list[tuple[int | None, float]]:
    """Run all N games until every game finishes its current episode.

    Returns list of (winner, total_reward) for each game.
    Only the first game is rendered (if not fast).
    """
    n = len(game_managers)
    max_rounds = cfg.MAX_ROUNDS_PER_EPISODE
    _encode = encoder.encode
    _choose = agent.choose_action
    _learn = agent.learn
    _cr = compute_reward

    # Per-game state
    rounds = [0] * n
    rewards = [0.0] * n
    finished = [False] * n
    winners: list[int | None] = [None] * n

    # Start episodes
    for gm in game_managers:
        gm.new_episode()

    # Start first round for each
    for i, gm in enumerate(game_managers):
        gm.new_round()
        rounds[i] = 1

    while not all(finished):
        # Handle Pygame events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit("Window closed")

        # Step each active game
        for i, gm in enumerate(game_managers):
            if finished[i]:
                continue

            tanks = gm.tanks
            walls = gm.arena.walls

            # Encode, act, step
            s0 = _encode(tanks, gm.bullets, walls, gm.arena, 0,
                         powerups=gm.powerups, mines=gm.mines)
            s1 = _encode(tanks, gm.bullets, walls, gm.arena, 1,
                         powerups=gm.powerups, mines=gm.mines)
            a0 = _choose(s0)
            a1 = _choose(s1)
            events = gm.step(a0, a1)

            # Learn from both tanks
            sn0 = _encode(tanks, gm.bullets, walls, gm.arena, 0,
                          powerups=gm.powerups, mines=gm.mines)
            sn1 = _encode(tanks, gm.bullets, walls, gm.arena, 1,
                          powerups=gm.powerups, mines=gm.mines)
            r0 = _cr(events, 0)
            r1 = _cr(events, 1)
            rd = events[EVT_ROUND_OVER]
            if use_dqn:
                _learn(s0, a0, r0, sn0, done=rd)
                _learn(s1, a1, r1, sn1, done=rd)
            else:
                _learn(s0, a0, r0, sn0)
                _learn(s1, a1, r1, sn1)
            rewards[i] += r0 + r1

            # Round over?
            if events[EVT_ROUND_OVER]:
                if gm.episode_over() or rounds[i] >= max_rounds:
                    finished[i] = True
                    winners[i] = gm.get_episode_winner()
                else:
                    gm.new_round()
                    rounds[i] += 1

        # Render first game only
        if not fast and not finished[0]:
            renderer.draw_frame(game_managers[0], [agent, agent], stats=stats)
            renderer.tick()

    # Episode-end processing (once per episode, shared agent)
    agent.decay_epsilon()
    agent.episode_end_replay()

    return list(zip(winners, rewards))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Tank Duel AI")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes (default 2000)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip per-frame rendering for faster training")
    parser.add_argument("--demo", action="store_true",
                        help="Load saved models and play visual demo only")
    parser.add_argument("--demo-episodes", type=int, default=5,
                        help="Episodes for post-training demo (default 5)")
    parser.add_argument("--model-dir", type=str, default="saved_models",
                        help="Directory for saved models")
    parser.add_argument("--qtable", action="store_true",
                        help="Use tabular Q-learning instead of DQN "
                             "(DQN is default, requires PyTorch)")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Number of parallel game instances for training "
                             "(default 2, each has 2 tanks = 4 agents)")
    parser.add_argument("--expert", action="store_true",
                        help="Run rule-based expert agent (no learning, "
                             "immediate smart play for testing)")
    args = parser.parse_args()

    model_dir = args.model_dir

    # -- Expert mode (rule-based, no learning) --------------------------------
    if args.expert:
        from ai.expert_agent import ExpertAgent
        from ai.state_encoder_continuous import ContinuousStateEncoder

        shared_agent = ExpertAgent(0)
        agents = [shared_agent, shared_agent]
        encoder = ContinuousStateEncoder()
        renderer = Renderer()
        print("[*] Using rule-based expert agent (no learning)")

        num = args.episodes if args.episodes != 2000 else 10
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=False, fast=False, model_dir=model_dir,
                         model_name="expert", model_ext=".none",
                         use_dqn=False)
        _print_summary(wins, num)
        renderer.quit()
        return

    use_dqn = not args.qtable  # DQN is default

    # -- Create agent + encoder based on mode --------------------------------
    if use_dqn:
        try:
            from ai.dqn_agent import DQNAgent
            from ai.state_encoder_continuous import ContinuousStateEncoder
        except ImportError as e:
            print(f"[!] DQN requires PyTorch: {e}")
            print("    Install with: pip install torch --index-url "
                   "https://download.pytorch.org/whl/cpu")
            print("    Falling back to tabular Q-learning...")
            use_dqn = False

    if use_dqn:
        from ai.dqn_agent import DQNAgent
        from ai.state_encoder_continuous import ContinuousStateEncoder

        shared_agent = DQNAgent(0)
        agents = [shared_agent, shared_agent]
        encoder = ContinuousStateEncoder()
        model_ext = ".pt"
        model_name = "dqn_shared"
        print(f"[*] Using DQN agent ({shared_agent.network_param_count()} params, CPU)")
    else:
        from ai.agent import QLearningAgent
        from ai.state_encoder import StateEncoder

        shared_agent = QLearningAgent(0)
        agents = [shared_agent, shared_agent]
        encoder = StateEncoder()
        model_ext = ".pkl"
        model_name = "agent_shared"
        print("[*] Using tabular Q-learning agent")

    renderer = Renderer()

    # ======================================================================
    # MODE: --demo
    # ======================================================================
    if args.demo:
        _load_model(agents[0], model_dir, model_name, model_ext, use_dqn,
                    force_epsilon=cfg.EPSILON_MIN)
        num = args.episodes if args.episodes != 2000 else 5
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=False, fast=False, model_dir=model_dir,
                         model_name=model_name, model_ext=model_ext,
                         use_dqn=use_dqn)
        _print_summary(wins, num)
        renderer.quit()
        return

    # ======================================================================
    # DEFAULT MODE: training (auto-resume from existing model)
    # ======================================================================
    start_ep = 0
    model_path = os.path.join(model_dir, model_name + model_ext)
    if os.path.exists(model_path):
        meta = _load_model(agents[0], model_dir, model_name, model_ext, use_dqn)
        start_ep = meta.get("episode", 0)
        if start_ep > 0:
            print(f"[*] Resuming from episode {start_ep}, "
                  f"epsilon={agents[0].epsilon:.4f}")

    num = args.episodes
    n_parallel = args.parallel if use_dqn else 1  # parallel only for DQN

    # Graceful shutdown: save model on Ctrl+C or window close
    _shutdown_save = {
        "agent": agents[0], "model_dir": model_dir,
        "model_name": model_name, "model_ext": model_ext,
        "use_dqn": use_dqn, "start_ep": start_ep, "current_ep": [0],
    }

    try:
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=True, fast=args.fast, model_dir=model_dir,
                         model_name=model_name, model_ext=model_ext,
                         use_dqn=use_dqn, start_episode=start_ep,
                         n_parallel=n_parallel,
                         ep_tracker=_shutdown_save["current_ep"])
    except (KeyboardInterrupt, SystemExit):
        ep_done = _shutdown_save["current_ep"][0]
        total_ep = start_ep + ep_done
        print(f"\n[!] Interrupted at episode {total_ep}. Saving...")
        _save_model(agents[0], model_dir, model_name, model_ext, use_dqn,
                    episode=total_ep)
        renderer.quit()
        return

    _save_model(agents[0], model_dir, model_name, model_ext, use_dqn,
                episode=start_ep + num)
    _print_summary(wins, num)

    # Auto-demo
    demo_eps = args.demo_episodes
    print(f"\n[*] Launching visual demo ({demo_eps} episodes) ...")
    agents[0].epsilon = cfg.EPSILON_MIN
    _run_loop(agents, encoder, renderer, demo_eps,
              learn=False, fast=False, model_dir=model_dir,
              model_name=model_name, model_ext=model_ext,
              use_dqn=use_dqn)

    renderer.quit()


# ===========================================================================
# Core training / demo loop
# ===========================================================================

def _run_loop(agents, encoder, renderer: Renderer, num_episodes: int,
              learn: bool, fast: bool, model_dir: str,
              model_name: str, model_ext: str,
              use_dqn: bool = False,
              start_episode: int = 0,
              n_parallel: int = 1,
              ep_tracker: list | None = None) -> dict:
    """Run episodes.  Returns win tally.
    
    start_episode: cumulative episode offset (for resumed training).
    n_parallel: number of parallel game instances (only used for DQN training).
    """
    use_parallel = learn and n_parallel > 1 and use_dqn

    # Create game manager(s)
    gms = [GameManager() for _ in range(n_parallel if use_parallel else 1)]
    gm = gms[0]  # primary (rendered) game

    wins = {0: 0, 1: 0, None: 0}
    t_start = time.perf_counter()
    last_report = t_start

    ep_rewards: list[float] = []
    epsilon_history: list[float] = []
    model_size_history: list[int] = []
    win_history: list[int | None] = []

    # Expert opponent for mixed training
    expert_agent = None
    expert_ratio = getattr(cfg, 'EXPERT_MIX_RATIO', 0.0)
    if learn and use_dqn and expert_ratio > 0:
        from ai.expert_agent import ExpertAgent
        expert_agent = ExpertAgent(1)
        print(f"[*] Expert opponent mixed in at {expert_ratio*100:.0f}% of episodes")

    mode = "training" + (" (fast)" if fast else "") if learn else "demo"
    agent_type = "DQN" if use_dqn else "Q-table"
    par_str = f" x{n_parallel} parallel" if use_parallel else ""
    resume_str = f" (resuming from ep {start_episode})" if start_episode > 0 else ""
    print(f"[*] {mode} [{agent_type}{par_str}]: "
          f"ep {start_episode+1}-{start_episode+num_episodes}{resume_str}  "
          f"(eps_decay={cfg.EPSILON_DECAY}, timeout={cfg.ROUND_TIMEOUT})")

    for ep in range(1, num_episodes + 1):
        if ep_tracker is not None:
            ep_tracker[0] = ep
        now = time.perf_counter()
        elapsed = now - t_start
        cumulative_ep = start_episode + ep
        eps_sec = ep / max(elapsed, 0.001)
        stats = {"ep": cumulative_ep, "total": start_episode + num_episodes,
                 "eps_sec": eps_sec}

        # Decide if this episode uses expert opponent
        vs_expert = (expert_agent is not None and
                     random.random() < expert_ratio)
        if vs_expert:
            # Tank 0 = DQN (learns), Tank 1 = Expert (doesn't learn)
            ep_agents = [agents[0], expert_agent]
        else:
            ep_agents = agents

        # Curriculum learning (use cumulative episode for resumed training)
        if learn:
            for g in gms:
                _apply_curriculum(g, cumulative_ep)

        if use_parallel:
            # Run all N games simultaneously, one shared agent
            results = run_parallel_episodes(
                gms, agents[0], encoder, renderer,
                fast=fast, stats=stats, use_dqn=use_dqn)
            # Use first game's winner for tracking
            winner = results[0][0]
            total_r = sum(r for _, r in results)
            if ep_rewards is not None:
                ep_rewards.append(total_r)
        else:
            winner = run_episode(gm, ep_agents, encoder, renderer,
                                 learn=learn, fast=fast, stats=stats,
                                 ep_rewards=ep_rewards if learn else None,
                                 use_dqn=use_dqn)

        wins[winner] = wins.get(winner, 0) + 1

        if learn:
            epsilon_history.append(agents[0].epsilon)
            if use_dqn:
                model_size_history.append(agents[0].buffer_size())
            else:
                model_size_history.append(
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

            if use_dqn:
                size_str = f"buf={agents[0].buffer_size()}"
            else:
                size_str = f"Q={len(agents[0].q_table_a) + len(agents[0].q_table_b)}"

            c_ep = start_episode + ep
            if c_ep <= cfg.CURRICULUM_PHASE_1_END:
                phase = "P1:shoot"
            elif c_ep <= cfg.CURRICULUM_PHASE_2_END:
                phase = "P2:walls"
            elif c_ep <= cfg.CURRICULUM_PHASE_3_END:
                phase = "P3:pwrup"
            elif c_ep <= cfg.CURRICULUM_PHASE_4_END:
                phase = "P4:mines"
            else:
                phase = "P5:full"

            total_target = start_episode + num_episodes
            print(f"  Ep {c_ep:>5}/{total_target}  |  "
                  f"Wins B:{wins[0]} R:{wins[1]} D:{wins.get(None,0)}  |  "
                  f"eps={agents[0].epsilon:.4f}  |  "
                  f"{size_str}  |  {phase}  |  "
                  f"{avg_rate:.1f} ep/s (recent {recent_rate:.1f})")

        # Periodic save every 50 episodes
        if learn and ep % 50 == 0:
            _save_model(agents[0], model_dir, model_name, model_ext, use_dqn,
                        episode=cumulative_ep)

    if learn and num_episodes > 0:
        size_label = "Replay buffer" if use_dqn else "Q-table Size"
        _generate_training_plots(win_history, ep_rewards,
                                 epsilon_history, model_size_history,
                                 size_label=size_label)

    return wins


# ===========================================================================
# Training graphs
# ===========================================================================

def _generate_training_plots(win_history: list,
                             ep_rewards: list[float],
                             epsilon_history: list[float],
                             model_size_history: list[int],
                             size_label: str = "Q-table Size") -> None:
    """Save training progress plots to training_results.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed -- skipping training graphs.")
        return

    num_eps = len(win_history)
    if num_eps < 2:
        print("[!] Too few episodes for meaningful graphs, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tank Duel AI -- Training Results", fontsize=14, fontweight="bold")
    episodes = list(range(1, num_eps + 1))

    # 1. Win rate (50-episode rolling window)
    ax = axes[0][0]
    window = 50
    win_rate_blue = []
    win_rate_red = []
    for i in range(num_eps):
        start = max(0, i - window + 1)
        chunk = win_history[start:i + 1]
        n = len(chunk)
        win_rate_blue.append(sum(1 for w in chunk if w == 0) / n * 100)
        win_rate_red.append(sum(1 for w in chunk if w == 1) / n * 100)
    ax.plot(episodes, win_rate_blue, label="Blue win%", color="#3C78DC", linewidth=1)
    ax.plot(episodes, win_rate_red, label="Red win%", color="#DC3C3C", linewidth=1)
    ax.set_title("Win Rate (50-ep rolling window)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average reward per episode
    ax = axes[0][1]
    ax.plot(episodes, ep_rewards, color="#2ca02c", linewidth=0.5, alpha=0.4)
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

    # 3. Epsilon decay
    ax = axes[1][0]
    ax.plot(episodes, epsilon_history, color="#ff7f0e", linewidth=1.5)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    # 4. Model size
    ax = axes[1][1]
    ax.plot(episodes, model_size_history, color="#9467bd", linewidth=1.5)
    ax.set_title(size_label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entries")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "training_results.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] Training graphs saved to {out_path}")


# ===========================================================================
# Helpers
# ===========================================================================

def _apply_curriculum(gm: GameManager, cumulative_ep: int) -> None:
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


def _load_model(agent, model_dir: str, model_name: str,
                model_ext: str, use_dqn: bool,
                force_epsilon: float | None = None) -> dict:
    """Load model if it exists.  Returns metadata dict with 'episode' key.
    
    If force_epsilon is set (e.g. for demo mode), overrides saved epsilon.
    Otherwise keeps the saved epsilon for training resume.
    """
    meta = {"episode": 0, "epsilon": agent.epsilon}
    path = os.path.join(model_dir, model_name + model_ext)
    if os.path.exists(path):
        result = agent.load(path)
        if isinstance(result, dict):
            meta.update(result)
        label = "DQN" if use_dqn else "Q-table"
        print(f"[*] Loaded {label} from {path} "
              f"(ep={meta.get('episode', '?')}, eps={agent.epsilon:.4f})")
        if force_epsilon is not None:
            agent.epsilon = force_epsilon
        return meta
    # Try legacy paths
    legacy = os.path.join(model_dir, "agent_shared.pkl")
    if not use_dqn and os.path.exists(legacy):
        result = agent.load(legacy)
        if isinstance(result, dict):
            meta.update(result)
        print(f"[*] Loaded Q-table from legacy {legacy}")
        if force_epsilon is not None:
            agent.epsilon = force_epsilon
        return meta
    print(f"[!] No saved model at {path}, starting fresh")
    return meta


def _save_model(agent, model_dir: str, model_name: str,
                model_ext: str, use_dqn: bool,
                episode: int = 0) -> None:
    path = os.path.join(model_dir, model_name + model_ext)
    if hasattr(agent, 'save') and 'episode' in agent.save.__code__.co_varnames:
        agent.save(path, episode=episode)
    else:
        agent.save(path)
    label = "DQN model" if use_dqn else "Shared Q-table"
    print(f"[*] {label} saved to {path} (episode {episode})")


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
