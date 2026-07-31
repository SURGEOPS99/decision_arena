import numpy as np
from arena.envs.sb3_wrapper import SB3MarketArenaWrapper
from arena.agents.rl_agent import PPOAgent
from arena.agents.offline_rl import TD3_BCAgent
from arena.agents.heuristics import TrendFollowingHeuristic, RandomAgent
from arena.utils.guardrails import GuardrailedAgent, RiskGuardrail
from arena.utils.metrics import update_elo

def run_guardrail_benchmark(eval_episodes=50):
    # Raw Agents
    raw_ppo = PPOAgent("models/ppo_arena_v1.zip", device="cpu")
    raw_td3 = TD3_BCAgent(device="cpu")
    raw_td3.load("models/td3_bc_arena_v1.pt")

    # Guardrailed Variants
    guardrailed_ppo = GuardrailedAgent(raw_ppo)
    guardrailed_td3 = GuardrailedAgent(raw_td3)

    opponents = {
        "agent_1": TrendFollowingHeuristic(drawdown_cutoff=0.12),
        "agent_2": RandomAgent()
    }

    env = SB3MarketArenaWrapper(ego_agent_id="agent_0", opponent_map=opponents)

    candidate_agents = {
        "Raw_PPO": raw_ppo,
        "Guardrailed_PPO": guardrailed_ppo,
        "Raw_TD3_BC": raw_td3,
        "Guardrailed_TD3_BC": guardrailed_td3
    }

    results = {}

    print(f"Benchmarking Policies with & without Guardrails ({eval_episodes} episodes per policy)...\n")

    for agent_name, agent_instance in candidate_agents.items():
        wealths = []
        drawdowns = []

        for ep in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            max_ep_dd = 0.0

            while not done:
                action = agent_instance.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                max_ep_dd = max(max_ep_dd, info.get("drawdown", 0.0))

            all_w = info.get("all_wealth", {})
            wealths.append(all_w.get("agent_0", 1000.0))
            drawdowns.append(max_ep_dd)

        results[agent_name] = {
            "mean_wealth": np.mean(wealths),
            "std_wealth": np.std(wealths),
            "max_drawdown": np.max(drawdowns),
            "avg_drawdown": np.mean(drawdowns)
        }

    print("=========================== GUARDRAIL BENCHMARK LEADERBOARD ===========================")
    print(f"{'Policy Variant':20s} | {'Mean Wealth':14s} | {'Max Drawdown':14s} | {'Avg Drawdown':12s}")
    print("---------------------------------------------------------------------------------------")
    for name, stats in results.items():
        mw = stats["mean_wealth"]
        sw = stats["std_wealth"]
        mdd = stats["max_drawdown"] * 100.0
        add = stats["avg_drawdown"] * 100.0
        print(f"{name:20s} | ${mw:7.2f} ± ${sw:5.2f} | {mdd:12.1f}% | {add:10.1f}%")
    print("=======================================================================================")

if __name__ == "__main__":
    run_guardrail_benchmark()