import numpy as np
from arena.agents.base import BaseAgent

class BaseAgent:
    def act(self, obs):
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def act(self, obs):
        return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

class TrendFollowingHeuristic(BaseAgent):
    """Buys when trend is positive, liquidates if drawdown exceeds limit."""
    def __init__(self, drawdown_cutoff=0.12):
        self.drawdown_cutoff = drawdown_cutoff

    def act(self, obs):
        drawdown = obs[7]
        trend = obs[2]
        
        # Risk control circuit breaker
        if drawdown > self.drawdown_cutoff:
            return np.array([-1.0], dtype=np.float32) # Sell everything
            
        if trend > 0:
            return np.array([0.5], dtype=np.float32)  # Moderate buy
        else:
            return np.array([-0.5], dtype=np.float32) # Moderate sell