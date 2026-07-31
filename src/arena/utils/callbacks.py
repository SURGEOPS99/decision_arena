import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DecisionArenaMetricsCallback(BaseCallback):
    """
    Logs decision arena performance metrics to TensorBoard during PPO training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_wealths = []
        self.episode_drawdowns = []

    def _on_step(self) -> bool:
        # Check if episode ended in infos
        for info in self.training_env.get_attr("state_data"):
            pass # Access state if needed
        
        infos = self.locals.get("infos", [])
        for info in infos:
            if "wealth" in info:
                self.episode_wealths.append(info["wealth"])
            if "drawdown" in info:
                self.episode_drawdowns.append(info["drawdown"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_wealths) > 0:
            mean_wealth = np.mean(self.episode_wealths)
            max_dd = np.max(self.episode_drawdowns) if len(self.episode_drawdowns) > 0 else 0.0
            
            self.logger.record("arena/mean_final_wealth", mean_wealth)
            self.logger.record("arena/max_drawdown", max_dd)
            
            self.episode_wealths.clear()
            self.episode_drawdowns.clear()