import numpy as np
from stable_baselines3 import PPO
from arena.agents.base import BaseAgent

class PPOAgent(BaseAgent):
    """Wraps a trained SB3 PPO model into the standard BaseAgent interface."""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = PPO.load(model_path, device=device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action