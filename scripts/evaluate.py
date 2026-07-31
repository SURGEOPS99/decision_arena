import numpy as np
from arena.envs.sb3_wrapper import SB3MarketArenaWrapper
from arena.agents.rl_agent import PPOAgent
from arena.agents.heuristics import TrendFollowingHeuristic, RandomAgent
from arena.utils.metrics import update_elo, calculate_sharpe_ratio

def run_evaluation(model_path="models/ppo_arena_v1.zip", eval_episodes=50):
    opponents = {
        "agent_1": TrendFollowingHeuristic(drawdown_cutoff=0.12),
        "agent_2": RandomAgent()
    }
    
    env = SB3MarketArenaWrapper(ego_agent_id="agent_0", opponent_map=opponents)
    ppo_agent = PPOAgent(model_path, device="cpu")

    # Tracking containers
    agent_names = ["PPO_Agent_0", "Trend_Agent_1", "Random_Agent_2"]
    wealth_history = {name: [] for name in agent_names}
    win_counts = {name: 0 for name in agent_names}
    elo_ratings = {name: 1500.0 for name in agent_names}

    print(f"Running Tournament across {eval_episodes} episodes...\n")

    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = ppo_agent.act(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

        wealths = info.get("all_wealth", {})
        ep_results = {
            "PPO_Agent_0": wealths.get("agent_0", 1000.0),
            "Trend_Agent_1": wealths.get("agent_1", 1000.0),
            "Random_Agent_2": wealths.get("agent_2", 1000.0)
        }

        # Store wealth
        for name, val in ep_results.items():
            wealth_history[name].append(val)

        # Record winner
        winner = max(ep_results, key=ep_results.get)
        win_counts[winner] += 1

        # Pairwise ELO updates based on final wealth comparison
        keys = list(ep_results.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a1, a2 = keys[i], keys[j]
                score_a1 = 1.0 if ep_results[a1] > ep_results[a2] else (0.5 if ep_results[a1] == ep_results[a2] else 0.0)
                
                # Update ratings
                new_r_a1 = update_elo(elo_ratings[a1], elo_ratings[a2], score_a1)
                new_r_a2 = update_elo(elo_ratings[a2], elo_ratings[a1], 1.0 - score_a1)
                elo_ratings[a1], elo_ratings[a2] = new_r_a1, new_r_a2

    # Print tournament leaderboard
    print("========================= ELO LEADERBOARD & BENCHMARK =========================")
    print(f"{'Agent':16s} | {'Mean Wealth':12s} | {'Win Rate':10s} | {'Final ELO':10s}")
    print("-------------------------------------------------------------------------------")
    for name in agent_names:
        mean_w = np.mean(wealth_history[name])
        std_w = np.std(wealth_history[name])
        win_pct = (win_counts[name] / eval_episodes) * 100.0
        elo = elo_ratings[name]
        print(f"{name:16s} | ${mean_w:7.2f} ± ${std_w:5.2f} | {win_pct:8.1f}% | {elo:10.1f}")
    print("===============================================================================")

if __name__ == "__main__":
    run_evaluation()