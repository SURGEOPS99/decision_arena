import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from arena.agents.heuristics import RandomAgent, TrendFollowingHeuristic
from arena.utils.data_loader import HistoricalDataFeed

class MarketArenaEnv(ParallelEnv):
    metadata = {"name": "market_arena_v0.1_real"}

    def __init__(self, num_agents=3, max_steps=200, initial_budget=1000.0, data_feed: HistoricalDataFeed = None):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.data_feed = data_feed
        
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.step_count = 0
        self.peak_wealth = {a: self.initial_budget for a in self.agents}

        if self.data_feed is not None:
            self.episode_window = self.data_feed.sample_episode_window(max_steps=self.max_steps)
            self.use_real_data = True
            current_row = self.episode_window.iloc[0]
            self.price = float(current_row["Close"])
            self.start_price = max(1e-8, self.price)  # Store initial price for relative normalization
            self.volatility = float(current_row["Volatility"])
            self.drift = float(current_row["Trend"])
            self.liquidity = float(current_row["Liquidity"])
        else:
            self.use_real_data = False
            self.price = 100.0
            self.start_price = 100.0
            self.volatility = 0.02
            self.drift = 0.001
            self.liquidity = 1.0

        self.state_data = {
            a: {"cash": self.initial_budget, "holdings": 0.0, "wealth": self.initial_budget, "drawdown": 0.0}
            for a in self.agents
        }
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _get_obs(self, agent_id):
        data = self.state_data[agent_id]
        avg_wealth = np.mean([d["wealth"] for d in self.state_data.values()])
        rel_rank = data["wealth"] / (avg_wealth + 1e-8)
        time_left = 1.0 - (self.step_count / self.max_steps)

        # Feature 0: Relative Price Normalization (Price / Start_Price) bounded near 1.0
        rel_price = self.price / self.start_price

        return np.array([
            rel_price,
            self.volatility,
            self.drift,
            self.liquidity,
            data["cash"] / self.initial_budget,
            (data["holdings"] * self.price) / self.initial_budget,  # Hold value normalized to budget
            data["wealth"] / self.initial_budget,
            data["drawdown"],
            rel_rank,
            time_left
        ], dtype=np.float32)

    def step(self, actions):
        self.step_count += 1
        
        if self.use_real_data and self.step_count < len(self.episode_window):
            current_row = self.episode_window.iloc[self.step_count]
            self.price = float(current_row["Close"])
            self.volatility = float(current_row["Volatility"])
            self.drift = float(current_row["Trend"])
            self.liquidity = float(current_row["Liquidity"])
        else:
            price_return = np.random.normal(self.drift, self.volatility)
            self.price = max(1.0, self.price * (1.0 + price_return))

        rewards, terminations, truncations, infos = {}, {}, {}, {}
        env_done = self.step_count >= self.max_steps or (self.use_real_data and self.step_count >= len(self.episode_window) - 1)

        for agent in self.agents:
            act = float(actions[agent][0]) if agent in actions else 0.0
            prev_wealth = self.state_data[agent]["wealth"]
            
            effective_act = act * self.liquidity
            
            # FRACTIONAL EXECUTION ENGINE:
            if effective_act > 0 and self.state_data[agent]["cash"] > 0:
                # Buy: Allocate fraction of available cash
                trade_value = min(self.state_data[agent]["cash"], self.state_data[agent]["cash"] * effective_act)
                fractional_units = trade_value / self.price
                self.state_data[agent]["cash"] -= trade_value
                self.state_data[agent]["holdings"] += fractional_units

            elif effective_act < 0 and self.state_data[agent]["holdings"] > 0:
                # Sell: Liquidate fraction of current holdings
                units_to_sell = self.state_data[agent]["holdings"] * abs(effective_act)
                trade_value = units_to_sell * self.price
                self.state_data[agent]["cash"] += trade_value
                self.state_data[agent]["holdings"] -= units_to_sell

            curr_wealth = self.state_data[agent]["cash"] + (self.state_data[agent]["holdings"] * self.price)
            self.state_data[agent]["wealth"] = curr_wealth
            self.peak_wealth[agent] = max(self.peak_wealth[agent], curr_wealth)
            drawdown = (self.peak_wealth[agent] - curr_wealth) / self.peak_wealth[agent]
            self.state_data[agent]["drawdown"] = drawdown

            pct_return = (curr_wealth - prev_wealth) / prev_wealth
            pnl_reward = 100.0 * pct_return
            drawdown_penalty = 5.0 * max(0.0, drawdown - 0.15)
            
            rewards[agent] = float(pnl_reward - drawdown_penalty)
            terminations[agent] = False
            truncations[agent] = env_done
            infos[agent] = {"wealth": curr_wealth, "drawdown": drawdown}

        if env_done:
            self.agents = []

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        return observations, rewards, terminations, truncations, infos


class SB3MarketArenaWrapper(gym.Env):
    def __init__(self, ego_agent_id="agent_0", opponent_map=None, max_steps=200, data_feed=None):
        super().__init__()
        self.ego_agent_id = ego_agent_id
        self.env = MarketArenaEnv(num_agents=3, max_steps=max_steps, data_feed=data_feed)
        self.opponent_map = opponent_map or {
            "agent_1": TrendFollowingHeuristic(drawdown_cutoff=0.12),
            "agent_2": RandomAgent()
        }

        self.observation_space = self.env.observation_spaces[self.ego_agent_id]
        self.action_space = self.env.action_spaces[self.ego_agent_id]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        return obs_dict[self.ego_agent_id], info_dict.get(self.ego_agent_id, {})

    def step(self, action):
        actions_dict = {self.ego_agent_id: np.array(action, dtype=np.float32)}
        for opp_id, opp_agent in self.opponent_map.items():
            if opp_id in self.env.agents:
                actions_dict[opp_id] = opp_agent.act(self.current_obs_dict[opp_id])

        obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict)
        self.current_obs_dict = obs_dict

        ego_info = info_dict.get(self.ego_agent_id, {})
        ego_info["all_wealth"] = {a: info_dict[a]["wealth"] for a in info_dict if "wealth" in info_dict[a]}

        return (
            obs_dict[self.ego_agent_id],
            rewards_dict.get(self.ego_agent_id, 0.0),
            term_dict.get(self.ego_agent_id, False),
            trunc_dict.get(self.ego_agent_id, False),
            ego_info
        )