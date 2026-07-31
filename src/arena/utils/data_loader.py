import os
import numpy as np
import pandas as pd
import yfinance as yf

class HistoricalDataFeed:
    """
    Data feed loader for ARIA v0.1 that parses historical OHLCV candles
    from yfinance or local CSV files, calculates rolling features,
    and samples trajectory windows for MarketArenaEnv episodes.
    """
    def __init__(self, ticker="SPY", source="yfinance", csv_path=None, start_date="2021-01-01", end_date="2025-01-01", interval="1d"):
        self.ticker = ticker
        self.source = source
        self.csv_path = csv_path
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        
        self.data = self._load_and_preprocess()

    def _load_and_preprocess(self) -> pd.DataFrame:
        if self.source == "csv" and self.csv_path and os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            df.columns = [c.capitalize() for c in df.columns]
        else:
            # Fetch single ticker from Yahoo Finance
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval, progress=False)
            
            # Handle MultiIndex columns returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                if isinstance(self.ticker, str) and self.ticker in df.columns.levels[1]:
                    df = df.xs(self.ticker, level=1, axis=1)
                else:
                    df.columns = df.columns.get_level_values(0)
            
            df.reset_index(inplace=True)

        # Force 1D array flattening to avoid shape broadcasting errors
        close_prices = df["Close"].values.astype(np.float32).flatten()
        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-8)
        returns = np.insert(returns, 0, 0.0)

        df["Return"] = returns
        df["Volatility"] = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0.01).values
        df["Trend"] = pd.Series(returns).rolling(window=10, min_periods=1).mean().fillna(0.0).values
        
        if "Volume" in df.columns and not df["Volume"].isnull().all():
            vol_series = df["Volume"].values.astype(np.float32).flatten()
            rolling_vol = pd.Series(vol_series).rolling(window=20, min_periods=1).mean().values
            df["Liquidity"] = np.clip(vol_series / (rolling_vol + 1e-8), 0.1, 2.0)
        else:
            df["Liquidity"] = 1.0

        return df

    def sample_episode_window(self, max_steps=200):
        """Samples a continuous sub-window of max_steps length for an episode."""
        num_records = len(self.data)
        if num_records <= max_steps:
            start_idx = 0
            end_idx = num_records
        else:
            start_idx = np.random.randint(0, num_records - max_steps)
            end_idx = start_idx + max_steps

        return self.data.iloc[start_idx:end_idx].reset_index(drop=True)