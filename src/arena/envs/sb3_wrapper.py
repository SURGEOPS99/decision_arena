import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arena.envs.market_arena import MarketArenaEnv
from arena.agents.heuristics import RandomAgent, TrendFollowingHeuristic

class SB3MarketArenaWrapper(gym.Env):
    """
    Gymnasium wrapper around MarketArenaEnv.
    Focuses training on `ego_agent_id` while driving opponents with heuristics.
    """
    metadata = {"render_modes": []}

    def __init__(self, ego_agent_id="agent_0", opponent_map=None, max_steps=200):
        super().__init__()
        self.ego_agent_id = ego_agent_id
        self.env = MarketArenaEnv(num_agents=3, max_steps=max_steps)
        
        # Define default opponents if none provided
        if opponent_map is None:
            self.opponent_map = {
                "agent_1": TrendFollowingHeuristic(drawdown_cutoff=0.12),
                "agent_2": RandomAgent()
            }
        else:
            self.opponent_map = opponent_map

        # Expose ego agent's observation and action spaces to SB3
        self.observation_space = self.env.observation_spaces[self.ego_agent_id]
        self.action_space = self.env.action_spaces[self.ego_agent_id]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        self.current_obs_dict = obs_dict
        ego_obs = obs_dict[self.ego_agent_id]
        ego_info = info_dict.get(self.ego_agent_id, {})
        
        return ego_obs, ego_info

    def step(self, action):
        # 1. Format action dictionary for all active agents
        actions_dict = {}
        
        # Ego agent action from SB3
        actions_dict[self.ego_agent_id] = np.array(action, dtype=np.float32)
        
        # Opponent actions from baseline heuristics
        for opp_id, opp_agent in self.opponent_map.items():
            if opp_id in self.env.agents:
                opp_obs = self.current_obs_dict[opp_id]
                actions_dict[opp_id] = opp_agent.act(opp_obs)

        # 2. Step underlying PettingZoo environment
        obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict)
        self.current_obs_dict = obs_dict

        # 3. Extract ego agent metrics for SB3
        ego_obs = obs_dict[self.ego_agent_id]
        ego_reward = rewards_dict.get(self.ego_agent_id, 0.0)
        ego_term = term_dict.get(self.ego_agent_id, False)
        ego_trunc = trunc_dict.get(self.ego_agent_id, False)
        ego_info = info_dict.get(self.ego_agent_id, {})

        # Include tournament snapshot in info for evaluation logging
        ego_info["all_wealth"] = {a: info_dict[a]["wealth"] for a in info_dict if "wealth" in info_dict[a]}

        return ego_obs, ego_reward, ego_term, ego_trunc, ego_info