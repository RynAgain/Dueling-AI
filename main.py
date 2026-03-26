# ---------------------------------------------------------------------------
# main.py -- Entry point for the Tank Duel AI game
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import os
import time
import random
import pygame
import config as cfg
from game.game_manager import GameManager, EVT_ROUND_OVER
from ai.reward import compute_reward
from rendering.renderer import Renderer


# ===========================================================================
# Run one full episode
# ===========================================================================

def run_episode(gm: GameManager, agents, encoder, renderer: Renderer,
                learn: bool = True, fast: bool = False,
                stats: dict | None = None,
                ep_rewards: list[float] | None = None) -> int | None:
    """Play one episode.  Returns winner id or None."""
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
    _cr = compute_reward
    _reset_enc = getattr(encoder, 'reset', None)

    while not gm.episode_over() and rounds_played < max_rounds:
        gm.new_round()
        if _reset_enc:
            _reset_enc()
        rounds_played += 1

        tanks = gm.tanks
        bullets = gm.bullets
        arena = gm.arena
        walls = arena.walls

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise SystemExit("Window closed")

            s0 = _encode(tanks, bullets, walls, arena, 0)
            s1 = _encode(tanks, bullets, walls, arena, 1)
            a0 = _choose0(s0)
            a1 = _choose1(s1)
            events = _step(a0, a1)

            if learn:
                sn0 = _encode(tanks, bullets, walls, arena, 0)
                sn1 = _encode(tanks, bullets, walls, arena, 1)
                r0 = _cr(events, 0)
                r1 = _cr(events, 1)
                round_done = events[EVT_ROUND_OVER]
                _learn0(s0, a0, r0, sn0, done=round_done)
                _learn1(s1, a1, r1, sn1, done=round_done)
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
                          stats: dict | None = None) -> list[tuple[int | None, float]]:
    """Run all N games until every game finishes its current episode."""
    n = len(game_managers)
    max_rounds = cfg.MAX_ROUNDS_PER_EPISODE
    _encode = encoder.encode
    _choose = agent.choose_action
    _learn = agent.learn
    _cr = compute_reward
    _reset_enc = getattr(encoder, 'reset', None)

    rounds = [0] * n
    rewards = [0.0] * n
    finished = [False] * n
    winners: list[int | None] = [None] * n

    for gm in game_managers:
        gm.new_episode()

    for i, gm in enumerate(game_managers):
        gm.new_round()
        if _reset_enc:
            _reset_enc()
        rounds[i] = 1

    while not all(finished):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit("Window closed")

        for i, gm in enumerate(game_managers):
            if finished[i]:
                continue

            tanks = gm.tanks
            walls = gm.arena.walls

            s0 = _encode(tanks, gm.bullets, walls, gm.arena, 0)
            s1 = _encode(tanks, gm.bullets, walls, gm.arena, 1)
            a0 = _choose(s0)
            a1 = _choose(s1)
            events = gm.step(a0, a1)

            sn0 = _encode(tanks, gm.bullets, walls, gm.arena, 0)
            sn1 = _encode(tanks, gm.bullets, walls, gm.arena, 1)
            r0 = _cr(events, 0)
            r1 = _cr(events, 1)
            rd = events[EVT_ROUND_OVER]
            _learn(s0, a0, r0, sn0, done=rd)
            _learn(s1, a1, r1, sn1, done=rd)
            rewards[i] += r0 + r1

            if events[EVT_ROUND_OVER]:
                if gm.episode_over() or rounds[i] >= max_rounds:
                    finished[i] = True
                    winners[i] = gm.get_episode_winner()
                else:
                    gm.new_round()
                    if _reset_enc:
                        _reset_enc()
                    rounds[i] += 1

        if not fast and not finished[0]:
            renderer.draw_frame(game_managers[0], [agent, agent], stats=stats)
            renderer.tick()

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
    parser.add_argument("--parallel", type=int, default=2,
                        help="Number of parallel game instances (default 2)")
    parser.add_argument("--expert", action="store_true",
                        help="Run rule-based expert agent (no learning)")
    parser.add_argument("--population", action="store_true",
                        help="Use Population-Based Training")
    parser.add_argument("--pop-size", type=int, default=None,
                        help=f"Population size (default {cfg.PBT_POPULATION_SIZE})")
    parser.add_argument("--generations", type=int, default=40,
                        help="Number of PBT generations (default 40)")
    args = parser.parse_args()

    model_dir = args.model_dir

    # -- Population-Based Training mode --------------------------------------
    if args.population:
        _run_population_mode(args)
        return

    # -- Expert mode ---------------------------------------------------------
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
                         learn=False, fast=False, model_dir=model_dir)
        _print_summary(wins, num)
        renderer.quit()
        return

    # -- DQN agent -----------------------------------------------------------
    try:
        from ai.dqn_agent import DQNAgent
        from ai.state_encoder_continuous import ContinuousStateEncoder
    except ImportError as e:
        print(f"[!] DQN requires PyTorch: {e}")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return

    shared_agent = DQNAgent(0)
    agents = [shared_agent, shared_agent]
    encoder = ContinuousStateEncoder()
    model_name = "dqn_shared"
    model_ext = ".pt"
    print(f"[*] DQN agent ({shared_agent.network_param_count()} params, "
          f"{cfg.NUM_ACTIONS} actions, {cfg.DQN_STATE_DIM}-dim state)")

    renderer = Renderer()

    # -- Demo mode -----------------------------------------------------------
    if args.demo:
        _load_model(agents[0], model_dir, model_name, model_ext,
                    force_epsilon=cfg.EPSILON_MIN)
        num = args.episodes if args.episodes != 2000 else 5
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=False, fast=False, model_dir=model_dir)
        _print_summary(wins, num)
        renderer.quit()
        return

    # -- Training mode (auto-resume) -----------------------------------------
    start_ep = 0
    model_path = os.path.join(model_dir, model_name + model_ext)
    if os.path.exists(model_path):
        meta = _load_model(agents[0], model_dir, model_name, model_ext)
        start_ep = meta.get("episode", 0)
        if start_ep > 0:
            print(f"[*] Resuming from episode {start_ep}, "
                  f"epsilon={agents[0].epsilon:.4f}")

    num = args.episodes
    n_parallel = args.parallel

    _shutdown_save = {
        "agent": agents[0], "model_dir": model_dir,
        "model_name": model_name, "model_ext": model_ext,
        "start_ep": start_ep, "current_ep": [0],
    }

    try:
        wins = _run_loop(agents, encoder, renderer, num,
                         learn=True, fast=args.fast, model_dir=model_dir,
                         start_episode=start_ep, n_parallel=n_parallel,
                         ep_tracker=_shutdown_save["current_ep"])
    except (KeyboardInterrupt, SystemExit):
        ep_done = _shutdown_save["current_ep"][0]
        total_ep = start_ep + ep_done
        print(f"\n[!] Interrupted at episode {total_ep}. Saving...")
        _save_model(agents[0], model_dir, model_name, model_ext,
                    episode=total_ep)
        renderer.quit()
        return

    _save_model(agents[0], model_dir, model_name, model_ext,
                episode=start_ep + num)
    _print_summary(wins, num)

    # Auto-demo
    demo_eps = args.demo_episodes
    print(f"\n[*] Launching visual demo ({demo_eps} episodes) ...")
    agents[0].epsilon = cfg.EPSILON_MIN
    _run_loop(agents, encoder, renderer, demo_eps,
              learn=False, fast=False, model_dir=model_dir)

    renderer.quit()


# ===========================================================================
# Core training / demo loop
# ===========================================================================

def _run_loop(agents, encoder, renderer: Renderer, num_episodes: int,
              learn: bool = True, fast: bool = False, model_dir: str = "saved_models",
              start_episode: int = 0, n_parallel: int = 1,
              ep_tracker: list | None = None) -> dict:
    """Run episodes.  Returns win tally."""
    use_parallel = learn and n_parallel > 1

    gms = [GameManager() for _ in range(n_parallel if use_parallel else 1)]
    gm = gms[0]

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
    if learn and expert_ratio > 0:
        from ai.expert_agent import ExpertAgent
        expert_agent = ExpertAgent(1)
        print(f"[*] Expert opponent mixed in at {expert_ratio*100:.0f}% of episodes")

    mode = "training" + (" (fast)" if fast else "") if learn else "demo"
    par_str = f" x{n_parallel} parallel" if use_parallel else ""
    resume_str = f" (resuming from ep {start_episode})" if start_episode > 0 else ""
    print(f"[*] {mode} [DQN{par_str}]: "
          f"ep {start_episode+1}-{start_episode+num_episodes}{resume_str}  "
          f"(eps_decay={agents[0].epsilon_decay}, timeout={cfg.ROUND_TIMEOUT})")

    for ep in range(1, num_episodes + 1):
        if ep_tracker is not None:
            ep_tracker[0] = ep
        now = time.perf_counter()
        elapsed = now - t_start
        cumulative_ep = start_episode + ep
        eps_sec = ep / max(elapsed, 0.001)
        stats = {"ep": cumulative_ep, "total": start_episode + num_episodes,
                 "eps_sec": eps_sec}

        # Expert opponent mixing
        vs_expert = (expert_agent is not None and
                     random.random() < expert_ratio)
        ep_agents = [agents[0], expert_agent] if vs_expert else agents

        # Curriculum: 2 phases
        if learn:
            for g in gms:
                _apply_curriculum(g, cumulative_ep)

        if use_parallel:
            results = run_parallel_episodes(
                gms, agents[0], encoder, renderer,
                fast=fast, stats=stats)
            winner = results[0][0]
            total_r = sum(r for _, r in results)
            ep_rewards.append(total_r)
        else:
            winner = run_episode(gm, ep_agents, encoder, renderer,
                                 learn=learn, fast=fast, stats=stats,
                                 ep_rewards=ep_rewards if learn else None)

        wins[winner] = wins.get(winner, 0) + 1

        if learn:
            epsilon_history.append(agents[0].epsilon)
            model_size_history.append(agents[0].buffer_size())
            win_history.append(winner)

        # Progress every 50 episodes
        if ep % 50 == 0 or ep == num_episodes:
            now2 = time.perf_counter()
            elapsed2 = now2 - t_start
            avg_rate = ep / max(elapsed2, 0.001)
            recent = now2 - last_report
            recent_rate = 50 / max(recent, 0.001)
            last_report = now2

            c_ep = start_episode + ep
            phase = "P1:Combat" if c_ep <= cfg.CURRICULUM_PHASE_1_END else "P2:Walls"

            total_target = start_episode + num_episodes
            print(f"  Ep {c_ep:>5}/{total_target}  |  "
                  f"Wins B:{wins[0]} R:{wins[1]} D:{wins.get(None,0)}  |  "
                  f"eps={agents[0].epsilon:.4f}  |  "
                  f"buf={agents[0].buffer_size()}  |  {phase}  |  "
                  f"{avg_rate:.1f} ep/s (recent {recent_rate:.1f})")

        # Periodic save every 50 episodes
        if learn and ep % 50 == 0:
            _save_model(agents[0], model_dir, "dqn_shared", ".pt",
                        episode=cumulative_ep)

    if learn and num_episodes > 0:
        _generate_training_plots(win_history, ep_rewards,
                                 epsilon_history, model_size_history)

    return wins


# ===========================================================================
# Training graphs
# ===========================================================================

def _generate_training_plots(win_history: list,
                             ep_rewards: list[float],
                             epsilon_history: list[float],
                             model_size_history: list[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed -- skipping training graphs.")
        return

    num_eps = len(win_history)
    if num_eps < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tank Duel AI -- Training Results", fontsize=14, fontweight="bold")
    episodes = list(range(1, num_eps + 1))
    window = 50

    # 1. Win rate
    ax = axes[0][0]
    wr_blue = []
    wr_red = []
    for i in range(num_eps):
        start = max(0, i - window + 1)
        chunk = win_history[start:i + 1]
        n = len(chunk)
        wr_blue.append(sum(1 for w in chunk if w == 0) / n * 100)
        wr_red.append(sum(1 for w in chunk if w == 1) / n * 100)
    ax.plot(episodes, wr_blue, label="Blue win%", color="#3C78DC", linewidth=1)
    ax.plot(episodes, wr_red, label="Red win%", color="#DC3C3C", linewidth=1)
    ax.set_title("Win Rate (50-ep window)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Reward
    ax = axes[0][1]
    ax.plot(episodes, ep_rewards, color="#2ca02c", linewidth=0.5, alpha=0.4)
    if num_eps >= window:
        smoothed = []
        for i in range(num_eps):
            s = max(0, i - window + 1)
            smoothed.append(sum(ep_rewards[s:i + 1]) / (i - s + 1))
        ax.plot(episodes, smoothed, color="#2ca02c", linewidth=1.5,
                label=f"Smoothed ({window}-ep)")
        ax.legend()
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # 3. Epsilon
    ax = axes[1][0]
    ax.plot(episodes, epsilon_history, color="#ff7f0e", linewidth=1.5)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    # 4. Buffer size
    ax = axes[1][1]
    ax.plot(episodes, model_size_history, color="#9467bd", linewidth=1.5)
    ax.set_title("Replay Buffer")
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
    """2 phases: no walls (pure combat) -> full walls."""
    if cumulative_ep <= cfg.CURRICULUM_PHASE_1_END:
        gm.set_wall_count(0)
    else:
        gm.set_wall_count(None)


def _load_model(agent, model_dir: str, model_name: str,
                model_ext: str,
                force_epsilon: float | None = None) -> dict:
    meta = {"episode": 0, "epsilon": agent.epsilon}
    path = os.path.join(model_dir, model_name + model_ext)
    if os.path.exists(path):
        result = agent.load(path)
        if isinstance(result, dict):
            meta.update(result)
        print(f"[*] Loaded DQN from {path} "
              f"(ep={meta.get('episode', '?')}, eps={agent.epsilon:.4f})")
        if force_epsilon is not None:
            agent.epsilon = force_epsilon
        return meta
    print(f"[!] No saved model at {path}, starting fresh")
    return meta


def _save_model(agent, model_dir: str, model_name: str,
                model_ext: str, episode: int = 0) -> None:
    path = os.path.join(model_dir, model_name + model_ext)
    agent.save(path, episode=episode)
    print(f"[*] DQN saved to {path} (episode {episode})")


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


# ===========================================================================
# Population-Based Training mode
# ===========================================================================

def _run_population_mode(args) -> None:
    """Run PBT: a population of DQN agents compete in round-robin tournaments."""
    from ai.population import Population
    from ai.state_encoder_continuous import ContinuousStateEncoder

    try:
        from ai.dqn_agent import DQNAgent
    except ImportError as e:
        print(f"[!] PBT requires PyTorch: {e}")
        return

    model_dir = args.model_dir
    pop_size = args.pop_size or cfg.PBT_POPULATION_SIZE
    num_generations = args.generations
    fast = args.fast

    encoder = ContinuousStateEncoder()
    renderer = Renderer()

    pop = Population(size=pop_size, encoder=encoder)

    model_path = os.path.join(model_dir, "dqn_shared.pt")
    start_ep = 0
    if os.path.exists(model_path):
        meta = pop.load_best_into_all(model_path)
        start_ep = meta.get("episode", 0)
        pop.cumulative_episode = start_ep
        print(f"[*] Population seeded from {model_path} (ep={start_ep})")

    param_count = pop.agents[0].network_param_count()
    print(f"[*] PBT: {pop_size} agents x {param_count} params, "
          f"{num_generations} generations")

    t_start = time.perf_counter()
    last_rankings = None
    gen = 0

    try:
        for gen in range(1, num_generations + 1):
            t_gen = time.perf_counter()
            print(f"\n  Generation {gen}/{num_generations} "
                  f"(cumulative ep ~{pop.cumulative_episode})")

            rankings = pop.run_tournament(fast=fast, renderer=renderer)
            last_rankings = rankings
            pop.print_fitness(rankings)
            pop.evolve(rankings)

            best_path = os.path.join(model_dir, "dqn_shared.pt")
            pop.save_best(best_path, rankings,
                          episode=pop.cumulative_episode)

            elapsed = time.perf_counter() - t_gen
            total_elapsed = time.perf_counter() - t_start
            best_fit = rankings[0]
            print(f"    gen time: {elapsed:.1f}s  |  "
                  f"total: {total_elapsed:.0f}s  |  "
                  f"best=A{best_fit.agent_idx} "
                  f"Elo={best_fit.elo:.0f} "
                  f"eps={pop.agents[best_fit.agent_idx].epsilon:.4f}")

    except (KeyboardInterrupt, SystemExit):
        print(f"\n[!] Interrupted at generation {gen}. Saving...")
        if last_rankings:
            best_path = os.path.join(model_dir, "dqn_shared.pt")
            pop.save_best(best_path, last_rankings,
                          episode=pop.cumulative_episode)
        renderer.quit()
        return

    pop.save_all(model_dir, episode=pop.cumulative_episode)
    if last_rankings:
        best_path = os.path.join(model_dir, "dqn_shared.pt")
        pop.save_best(best_path, last_rankings,
                      episode=pop.cumulative_episode)

    total_elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  PBT COMPLETE -- {num_generations} generations in {total_elapsed:.0f}s")
    print(f"  Total episodes: ~{pop.cumulative_episode}")
    if last_rankings:
        best_fit = last_rankings[0]
        print(f"  Champion: A{best_fit.agent_idx}  "
              f"Elo={best_fit.elo:.0f}  "
              f"K/D={best_fit.kd_ratio:.1f}  "
              f"WR={best_fit.win_rate*100:.0f}%")
    print(f"{'='*60}")

    if last_rankings:
        best = pop.best_agent(last_rankings)
        best.epsilon = cfg.EPSILON_MIN
        agents = [best, best]
        demo_eps = args.demo_episodes
        print(f"\n[*] Demo ({demo_eps} episodes) with champion ...")
        _run_loop(agents, encoder, renderer, demo_eps,
                  learn=False, fast=False, model_dir=model_dir)

    renderer.quit()


if __name__ == "__main__":
    main()
