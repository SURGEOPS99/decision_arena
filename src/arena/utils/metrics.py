import numpy as np

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculates annualized Sharpe ratio from step returns."""
    std_dev = np.std(returns)
    if std_dev == 0:
        return 0.0
    return float((np.mean(returns) - risk_free_rate) / std_dev * np.sqrt(252))

def calculate_max_drawdown(wealth_history: list) -> float:
    """Calculates peak-to-trough max drawdown percentage."""
    wealth_arr = np.array(wealth_history)
    peaks = np.maximum.accumulate(wealth_arr)
    drawdowns = (peaks - wealth_arr) / peaks
    return float(np.max(drawdowns))

def update_elo(r_a: float, r_b: float, actual_score_a: float, k_factor: float = 32.0):
    """Standard ELO rating update formula for 1v1 agent match."""
    expected_a = 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))
    new_r_a = r_a + k_factor * (actual_score_a - expected_a)
    return new_r_a