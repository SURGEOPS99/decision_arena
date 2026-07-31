import numpy as np

class BaseAgent:
    """Abstract base interface for all decision arena agents."""
    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement act(obs)")