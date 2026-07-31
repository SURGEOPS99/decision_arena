import numpy as np
from arena.utils.data_loader import HistoricalDataFeed
from arena.envs.market_arena import SB3MarketArenaWrapper
from arena.agents.rl_agent import PPOAgent
from arena.agents.offline_rl import TD3_BCAgent
from arena.utils.guardrails import GuardrailedAgent

def benchmark_on_assets(tickers=None, eval_episodes=30):
    if tickers is None:
        tickers = ["NVDA", "BTC-USD"]

    raw_ppo = PPOAgent("models/ppo_arena_v1.zip", device="cpu")
    raw_td3 = TD3_BCAgent(device="cpu")
    raw_td3.load("models/td3_bc_arena_v1.pt")

    guardrailed_ppo = GuardrailedAgent(raw_ppo)
    guardrailed_td3 = GuardrailedAgent(raw_td3)

    candidates = {
        "Raw_PPO": raw_ppo,
        "Guardrailed_PPO": guardrailed_ppo,
        "Raw_TD3_BC": raw_td3,
        "Guardrailed_TD3_BC": guardrailed_td3
    }

    for ticker in tickers:
        print(f"\n[+] Fetching real market data for {ticker} via yfinance...")
        data_feed = HistoricalDataFeed(
            ticker=ticker,
            source="yfinance",
            start_date="2021-01-01",
            end_date="2025-01-01"
        )
        env = SB3MarketArenaWrapper(ego_agent_id="agent_0", max_steps=200, data_feed=data_feed)

        print(f"\n================ BENCHMARKING ON REAL CANDLES ({ticker}) ================")
        print(f"{'Policy Variant':20s} | {'Mean Wealth':14s} | {'Max Drawdown':14s}")
        print("-------------------------------------------------------------------------")

        for name, agent in candidates.items():
            wealths = []
            drawdowns = []

            for ep in range(eval_episodes):
                obs, _ = env.reset()
                done = False
                max_dd = 0.0

                while not done:
                    action = agent.act(obs)
                    obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    max_dd = max(max_dd, info.get("drawdown", 0.0))

                all_w = info.get("all_wealth", {})
                wealths.append(all_w.get("agent_0", 1000.0))
                drawdowns.append(max_dd)

            mw = np.mean(wealths)
            sw = np.std(wealths)
            mdd = np.max(drawdowns) * 100.0
            print(f"{name:20s} | ${mw:7.2f} ± ${sw:5.2f} | {mdd:12.1f}%")

        print("=========================================================================")

if __name__ == "__main__":
    benchmark_on_assets(tickers=["NVDA", "BTC-USD"], eval_episodes=30)