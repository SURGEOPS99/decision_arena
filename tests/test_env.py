import os
import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


# =====================================================================
# 1. Import Sanity Checks
# =====================================================================
def test_package_imports():
    """Verify all modules can be imported without ImportError after reorganization."""
    try:
        from arena.envs.market_arena import MarketArenaEnv
        from arena.envs.sb3_wrapper import SB3MarketArenaWrapper
        from arena.agents.base import BaseAgent
        from arena.agents.heuristics import RandomAgent, TrendFollowingHeuristic
        from arena.agents.rl_agent import PPOAgent
        from arena.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown, update_elo
        from arena.utils.callbacks import DecisionArenaMetricsCallback
    except ImportError as e:
        pytest.fail(f"Module import failed! Check package structure or pythonpath. Error: {e}")


# =====================================================================
# 2. PettingZoo Core Environment Tests
# =====================================================================
def test_market_arena_reset_and_step():
    """Verify PettingZoo ParallelEnv lifecycle: reset, action stepping, observation shapes."""
    from arena.envs.market_arena import MarketArenaEnv

    num_agents = 3
    max_steps = 10
    env = MarketArenaEnv(num_agents=num_agents, max_steps=max_steps)

    # Test Reset
    obs_dict, info_dict = env.reset(seed=42)
    assert len(obs_dict) == num_agents
    for agent_id, obs in obs_dict.items():
        assert obs.shape == (10,)
        assert not np.isnan(obs).any()

    # Test Step
    actions = {agent: np.array([0.5], dtype=np.float32) for agent in env.agents}
    next_obs, rewards, term, trunc, infos = env.step(actions)

    assert len(next_obs) == num_agents
    assert len(rewards) == num_agents
    for agent_id in env.possible_agents:
        assert isinstance(rewards[agent_id], float)
        assert "wealth" in infos[agent_id]
        assert "drawdown" in infos[agent_id]

    # Run to episode truncation
    step_count = 1
    while env.agents:
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        _, _, _, trunc, _ = env.step(actions)
        step_count += 1

    assert step_count == max_steps


# =====================================================================
# 3. Gymnasium SB3 Adapter Wrapper Tests
# =====================================================================
def test_sb3_wrapper_compliance():
    """Verify Gymnasium adapter complies with single-agent Gymnasium API for SB3."""
    from arena.envs.sb3_wrapper import SB3MarketArenaWrapper

    wrapper = SB3MarketArenaWrapper(ego_agent_id="agent_0", max_steps=15)

    # Space checks
    assert isinstance(wrapper.observation_space, gym.spaces.Box)
    assert isinstance(wrapper.action_space, gym.spaces.Box)
    assert wrapper.observation_space.shape == (10,)
    assert wrapper.action_space.shape == (1,)

    # Reset check
    obs, info = wrapper.reset(seed=42)
    assert obs.shape == (10,)
    assert not np.isnan(obs).any()

    # Step check
    action = wrapper.action_space.sample()
    next_obs, reward, terminated, truncated, info = wrapper.step(action)

    assert next_obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "all_wealth" in info
    assert len(info["all_wealth"]) == 3


# =====================================================================
# 4. Agent Interface Tests
# =====================================================================
def test_heuristic_agents_actions():
    """Verify baseline heuristic agents return valid bounding action arrays."""
    from arena.agents.heuristics import RandomAgent, TrendFollowingHeuristic

    dummy_obs = np.array([1.0, 0.02, 0.001, 1.0, 1.0, 0.0, 1.0, 0.05, 1.0, 0.9], dtype=np.float32)

    # Random Agent
    rand_agent = RandomAgent()
    action = rand_agent.act(dummy_obs)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0

    # Trend Follower (Normal condition)
    trend_agent = TrendFollowingHeuristic(drawdown_cutoff=0.12)
    action = trend_agent.act(dummy_obs)
    assert action.shape == (1,)
    assert action[0] == 0.5  # Positive trend -> Buy action

    # Trend Follower (Drawdown breach condition)
    high_drawdown_obs = dummy_obs.copy()
    high_drawdown_obs[7] = 0.15  # Drawdown = 15% (> cutoff 12%)
    action_circuit_break = trend_agent.act(high_drawdown_obs)
    assert action_circuit_break[0] == -1.0  # Liquidate action


def test_ppo_agent_wrapper(tmp_path):
    """Verify PPOAgent wrapper successfully loads a saved SB3 model and acts."""
    from arena.envs.sb3_wrapper import SB3MarketArenaWrapper
    from arena.agents.rl_agent import PPOAgent

    # 1. Create and save a minimal dummy PPO model
    dummy_env = SB3MarketArenaWrapper(max_steps=10)
    dummy_model = PPO("MlpPolicy", dummy_env, n_steps=64, batch_size=32)
    save_file = tmp_path / "dummy_ppo.zip"
    dummy_model.save(str(save_file))

    # 2. Test PPOAgent wrapper loading
    agent = PPOAgent(str(save_file))
    obs, _ = dummy_env.reset()
    action = agent.act(obs)

    assert action.shape == (1,)
    assert not np.isnan(action).any()


# =====================================================================
# 5. Metrics Utility Tests
# =====================================================================
def test_metrics_utility_functions():
    """Verify financial and competitive evaluation metrics functions."""
    from arena.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown, update_elo

    # Sharpe Ratio
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01], dtype=np.float32)
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)

    # Zero variance returns
    zero_var_returns = np.zeros(5)
    assert calculate_sharpe_ratio(zero_var_returns) == 0.0

    # Max Drawdown
    wealth_curve = [1000.0, 1100.0, 990.0, 1050.0, 880.0, 950.0]  # Peak 1100 -> Trough 880 = 20% DD
    mdd = calculate_max_drawdown(wealth_curve)
    assert pytest.approx(mdd, 0.01) == 0.20

    # ELO Update
    new_r_a = update_elo(r_a=1500.0, r_b=1500.0, actual_score_a=1.0)
    assert new_r_a > 1500.0  # Winner rating increases