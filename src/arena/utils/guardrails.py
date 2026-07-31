import numpy as np
from arena.agents.base import BaseAgent

class RiskGuardrail:
    """
    Execution boundary layer that intercepts raw policy outputs and enforces 
    deterministic risk controls based on drawdown and market volatility.
    """
    def __init__(self, max_drawdown_limit=0.14, vol_threshold=0.03):
        self.max_drawdown_limit = max_drawdown_limit
        self.vol_threshold = vol_threshold

    def apply(self, raw_action: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Observation Vector Map:
        obs[1] = Volatility (sigma_t)
        obs[7] = Current Drawdown (D_t)
        """
        action = float(raw_action[0])
        volatility = float(obs[1])
        drawdown = float(obs[7])

        # Rule 1: Circuit Breaker - Immediate liquidation if max drawdown limit is breached
        if drawdown >= self.max_drawdown_limit:
            return np.array([-1.0], dtype=np.float32)

        # Rule 2: Drawdown Soft Cap - Dynamically scale down new buys as drawdown grows past 8%
        if drawdown > 0.08 and action > 0:
            headroom = max(0.0, (self.max_drawdown_limit - drawdown) / (self.max_drawdown_limit - 0.08))
            action = action * headroom

        # Rule 3: Volatility Dampening - Scale down position sizing during high-volatility regimes
        if volatility > self.vol_threshold:
            vol_penalty = max(0.2, 1.0 - (volatility - self.vol_threshold) * 20.0)
            action = action * vol_penalty

        return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


class GuardrailedAgent(BaseAgent):
    """Wraps any BaseAgent (PPO, TD3+BC, or Heuristic) with the RiskGuardrail execution layer."""
    def __init__(self, base_agent: BaseAgent, guardrail: RiskGuardrail = None):
        self.agent = base_agent
        self.guardrail = guardrail or RiskGuardrail()

    def act(self, obs: np.ndarray) -> np.ndarray:
        raw_action = self.agent.act(obs)
        return self.guardrail.apply(raw_action, obs)