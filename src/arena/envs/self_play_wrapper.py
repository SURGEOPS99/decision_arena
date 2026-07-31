import glob
import numpy as np
import gymnasium as gym
from arena.envs.market_arena import MarketArenaEnv
from arena.agents.rl_agent import PPOAgent
from arena.agents.heuristics import RandomAgent

class SelfPlayArenaWrapper(gym.Env):
    """
    Gymnasium wrapper where opponents are randomly sampled from previously 
    saved PPO checkpoint versions in `models/checkpoints/`.
    """
    def __init__(self, checkpoint_dir="models/checkpoints", max_steps=200):
        super().__init__()
        self.env = MarketArenaEnv(num_agents=3, max_steps=max_steps)
        self.checkpoint_dir = checkpoint_dir
        
        self.observation_space = self.env.observation_spaces["agent_0"]
        self.action_space = self.env.action_spaces["agent_0"]
        self.opponent_agents = {}

    def _sample_opponents(self):
        checkpoints = glob.glob(f"{self.checkpoint_dir}/*.zip")
        opps = {}
        
        for opp_id in ["agent_1", "agent_2"]:
            if checkpoints and np.random.rand() > 0.3:
                # 70% chance to fight a historical PPO checkpoint
                cp = np.random.choice(checkpoints)
                opps[opp_id] = PPOAgent(cp, device="cpu")
            else:
                # 30% chance to fight baseline random noise for regularization
                opps[opp_id] = RandomAgent()
                
        return opps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.opponent_agents = self._sample_opponents()
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        return obs_dict["agent_0"], info_dict.get("agent_0", {})

    def step(self, action):
        actions_dict = {"agent_0": np.array(action, dtype=np.float32)}
        
        for opp_id, opp_agent in self.opponent_agents.items():
            if opp_id in self.env.agents:
                actions_dict[opp_id] = opp_agent.act(self.current_obs_dict[opp_id])

        obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict)
        self.current_obs_dict = obs_dict

        ego_info = info_dict.get("agent_0", {})
        ego_info["all_wealth"] = {a: info_dict[a]["wealth"] for a in info_dict if "wealth" in info_dict[a]}

        return (
            obs_dict["agent_0"],
            rewards_dict.get("agent_0", 0.0),
            term_dict.get("agent_0", False),
            trunc_dict.get("agent_0", False),
            ego_info
        )