Here is a complete, production-ready `README.md` for the **ARIA v0.1 / `decision_arena**` project. You can copy and paste this directly into your `README.md` file.

---

# ARIA v0.1: Autonomous Risk-Aware Reinforcement Learning Agent & Benchmark

**ARIA v0.1** (*Autonomous Risk-Aware Interactive Agent*) is a modular multi-agent reinforcement learning (MARL) environment, offline RL pipeline, and execution safety framework. Designed for trading under volatile, non-stationary market dynamics, ARIA combines a PettingZoo Dec-POMDP market simulator with deterministic execution guardrails to prevent catastrophic portfolio drawdowns.

---

## Key Features

* **Dec-POMDP Market Environment (`MarketArenaEnv`)**: Built on `PettingZoo.ParallelEnv` supporting continuous action spaces, fractional position sizing, relative price normalization, and liquidity constraints.
* **Dynamic Regime Switching**: Simulates Markovian regime transitions between low-volatility Bull runs and high-volatility Bear flash crashes.
* **Offline RL Pipeline (TD3+BC)**: Custom PyTorch implementation of Twin Delayed DDPG with Behavior Cloning, trained on 60,000 transition samples logged in Apache Parquet format.
* **Deterministic Risk Guardrails**: Execution safety interceptor enforcing dynamic drawdown scaling ($\le 15\%$) and volatility dampening under market stress.
* **Real Asset Candle Engine**: Integrated `yfinance` data loader feeding real daily OHLCV historical time series (`SPY`, `NVDA`, `BTC-USD`) into the environment.
* **Tournament Evaluation Framework**: Head-to-head multi-agent benchmarking utilizing pairwise ELO ratings, Sharpe ratios, and max drawdown tracking.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HISTORICAL DATA FEED LAYER                            │
│                 (Yahoo Finance API / CSV Historical OHLCV Feeds)                │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PettingZoo Dec-POMDP Simulator                             │
│         (Fractional Sizing, Regime Switching, & Relative Normalization)         │
└───────────────────┬─────────────────────────────────────────┬───────────────────┘
                    │                                         │
                    ▼                                         ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│          Online PPO Policy            │ │         Offline TD3+BC Policy         │
│         (Stable-Baselines3)           │ │         (PyTorch / CUDA Engine)       │
└───────────────────┬───────────────────┘ └───────────────────┬───────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Deterministic Risk Guardrails                           │
│              (Volatility Dampening & Drawdown Circuit Breakers)                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Real-Market Execution & Benchmark                        │
└─────────────────────────────────────────────────────────────────────────────────┘

```

---

## Project Directory Structure

```text
decision_arena/
├── data/
│   └── arena_trajectories.parquet   # 60,000-sample offline transition dataset
├── models/
│   ├── ppo_arena_v1.zip             # Trained Online PPO actor-critic policy
│   └── td3_bc_arena_v1.pt           # Trained Offline TD3+BC PyTorch model
├── scripts/
│   ├── collect_dataset.py           # Logs trajectory dataset to Parquet
│   ├── evaluate.py                  # Head-to-head tournament runner with ELO
│   ├── evaluate_guardrails.py       # Raw vs. Guardrailed policy benchmark
│   ├── evaluate_offline.py          # Offline TD3+BC evaluation script
│   ├── evaluate_real_data.py        # Real candle evaluation (SPY, NVDA, BTC)
│   ├── train.py                     # Online PPO training loop (SB3)
│   └── train_offline.py             # Offline PyTorch TD3+BC training loop
├── src/
│   └── arena/
│       ├── agents/
│       │   ├── base.py              # Base agent interface
│       │   ├── heuristics.py        # Trend Following & Random baselines
│       │   ├── offline_rl.py        # PyTorch TD3+BC implementation
│       │   └── rl_agent.py          # SB3 PPO wrapper
│       ├── envs/
│       │   ├── market_arena.py      # PettingZoo Dec-POMDP core simulator
│       │   ├── sb3_wrapper.py       # Gymnasium single-agent wrapper
│       │   └── self_play_wrapper.py # Self-play league environment
│       └── utils/
│           ├── callbacks.py         # TensorBoard logger
│           ├── data_loader.py       # OHLCV feature-engineering engine
│           ├── dataset.py           # PyTorch Parquet DataLoader
│           ├── guardrails.py        # Risk interceptor safety layer
│           ├── metrics.py           # ELO & financial metrics engine
│           └── tracking.py          # Metric aggregator
├── tests/
│   └── test_env.py                  # Pytest verification suite
├── .gitignore
├── LICENSE                          # MIT License
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/decision_arena.git
cd decision_arena

```


2. **Set up a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## Dec-POMDP Environment Specification

Each agent receives a 10-dimensional observation vector $s_t \in \mathbb{R}^{10}$ at step $t$:

| Index | Feature | Description |
| --- | --- | --- |
| `0` | **Relative Price** | Current Asset Price / Window Initial Price |
| `1` | **Volatility ($\sigma_t$)** | Rolling 20-period standard deviation of returns |
| `2` | **Drift ($\mu_t$)** | Rolling 10-period average return trend |
| `3` | **Liquidity** | Volume-based execution factor ($0.1 \le \lambda \le 2.0$) |
| `4` | **Normalized Cash** | Cash Balance / Initial Budget |
| `5` | **Normalized Holdings** | Position Value / Initial Budget |
| `6` | **Normalized Wealth** | Portfolio Wealth / Initial Budget |
| `7` | **Drawdown ($D_t$)** | Current Peak-to-Trough Drawdown ($0.0 \le D_t \le 1.0$) |
| `8` | **Relative Rank** | Wealth / Average Portfolio Wealth of competitors |
| `9` | **Time Remaining** | Normalized episode time remaining ($1.0 \rightarrow 0.0$) |

### Action Space & Execution

The action space is continuous: $a_t \in [-1.0, 1.0]$.

* $a_t > 0$: Purchase asset using a fraction ($a_t \cdot \text{cash}$) of available balance.
* $a_t < 0$: Liquidate a fraction ($\vert{}a_t\vert{} \cdot \text{holdings}$) of current position.
* $a_t = 0$: Hold cash/position steady.

---

## Risk Guardrail Interceptor

The `RiskGuardrail` layer wraps raw neural network outputs $a_t$ before execution in the environment:

$$\tilde{a}_t = \text{Guardrail}(a_t, \sigma_t, D_t)$$

1. **Circuit Breaker**: If current drawdown $D_t \ge 14\%$, force immediate liquidation ($\tilde{a}_t = -1.0$).
2. **Drawdown Soft Cap**: If drawdown $D_t > 8\%$, scale buy orders down proportionally to remaining headroom.
3. **Volatility Dampening**: If rolling volatility $\sigma_t > 0.03$, scale action magnitude down to prevent overexposure during market shocks.

---

## Quickstart Usage Guide

### 1. Run Real Market Benchmark (`SPY`, `NVDA`, `BTC-USD`)

Benchmark trained PPO and TD3+BC policies on historical candlestick data:

```bash
PYTHONPATH=src python3 scripts/evaluate_real_data.py

```

### 2. Evaluate Policy Guardrails Under Regime Shifts

Compare raw vs. guardrailed variants across synthetic flash crash environments:

```bash
PYTHONPATH=src python3 scripts/evaluate_guardrails.py

```

### 3. Train Online PPO Agent

Train a new PPO policy for 100,000 steps using Stable-Baselines3:

```bash
PYTHONPATH=src python3 scripts/train.py

```

### 4. Collect Offline Parquet Trajectories & Train TD3+BC

Log 60,000 transition steps from heuristic and random interactions, then optimize the offline PyTorch policy:

```bash
# Generate trajectory dataset
PYTHONPATH=src python3 scripts/collect_dataset.py

# Train offline TD3+BC model
PYTHONPATH=src python3 scripts/train_offline.py

```

---

## Empirical Benchmark Results

### Historical Asset Candle Benchmark (2021–2025)

| Asset | Policy Variant | Mean Wealth | Max Drawdown | Risk Profile |
| --- | --- | --- | --- | --- |
| **`SPY`** | Guardrailed TD3+BC | **$1039.82 ± $107** | **14.5%** | Tail Risk Protected ($\le 15\%$) |
| **`NVDA`** | Guardrailed PPO | **$1497.73 ± $621** | **31.4%** | Growth Compounder |
| **`BTC-USD`** | Guardrailed TD3+BC | **$1185.97 ± $327** | **34.3%** | Drawdown Halved ($60.4\% \rightarrow 34.3\%$) |

---

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).