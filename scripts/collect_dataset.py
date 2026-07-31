import os
import numpy as np
import pandas as pd
from arena.envs.market_arena import MarketArenaEnv
from arena.agents.rl_agent import PPOAgent
from arena.agents.heuristics import TrendFollowingHeuristic, RandomAgent

def collect_trajectories(model_path="models/ppo_arena_v1.zip", num_episodes=100, output_file="data/arena_trajectories.parquet"):
    os.makedirs("data", exist_ok=True)
    
    env = MarketArenaEnv(num_agents=3, max_steps=200)
    ppo_agent = PPOAgent(model_path, device="cpu")
    
    agents_map = {
        "agent_0": ppo_agent,
        "agent_1": TrendFollowingHeuristic(drawdown_cutoff=0.12),
        "agent_2": RandomAgent()
    }

    records = []

    print(f"Collecting trajectory data across {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs_dict, info_dict = env.reset()
        done = False
        step = 0

        while not done:
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = agents_map[agent_id].act(obs_dict[agent_id])

            next_obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = env.step(actions)
            done = any(term_dict.values()) or any(trunc_dict.values())

            # Log step transition for each agent
            for agent_id in obs_dict.keys():
                records.append({
                    "episode": ep,
                    "step": step,
                    "agent_id": agent_id,
                    "obs": obs_dict[agent_id].tolist(),
                    "action": float(actions[agent_id][0]),
                    "reward": float(rewards_dict.get(agent_id, 0.0)),
                    "next_obs": next_obs_dict[agent_id].tolist(),
                    "wealth": float(info_dict[agent_id].get("wealth", 1000.0)),
                    "drawdown": float(info_dict[agent_id].get("drawdown", 0.0)),
                    "done": done
                })

            obs_dict = next_obs_dict
            step += 1

    df = pd.DataFrame(records)
    df.to_parquet(output_file, index=False)
    print(f"Dataset saved to {output_file} ({len(df)} transition steps).")

if __name__ == "__main__":
    collect_trajectories()