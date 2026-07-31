import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DecisionArenaMetricsCallback(BaseCallback):
    """Logs customized arena performance metrics to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_wealths = []
        self.episode_drawdowns = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "wealth" in info:
                self.episode_wealths.append(info["wealth"])
            if "drawdown" in info:
                self.episode_drawdowns.append(info["drawdown"])
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_wealths:
            self.logger.record("arena/mean_final_wealth", np.mean(self.episode_wealths))
            self.logger.record("arena/max_drawdown", np.max(self.episode_drawdowns))
            self.episode_wealths.clear()
            self.episode_drawdowns.clear()