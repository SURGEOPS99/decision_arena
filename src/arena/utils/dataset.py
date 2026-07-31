import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class OfflineArenaDataset(Dataset):
    """
    PyTorch Dataset wrapper for reading parquet offline trajectory logs.
    Normalizes state features for stable offline critic learning.
    """
    def __init__(self, parquet_path="data/arena_trajectories.parquet", agent_filter=None):
        df = pd.read_parquet(parquet_path)
        
        if agent_filter:
            df = df[df["agent_id"] == agent_filter].reset_index(drop=True)

        raw_obs = np.array(df["obs"].tolist(), dtype=np.float32)
        raw_next_obs = np.array(df["next_obs"].tolist(), dtype=np.float32)

        # State normalization (Mean 0, Std 1)
        self.mean = np.mean(raw_obs, axis=0)
        self.std = np.std(raw_obs, axis=0) + 1e-6

        norm_obs = (raw_obs - self.mean) / self.std
        norm_next_obs = (raw_next_obs - self.mean) / self.std

        self.obs = torch.tensor(norm_obs, dtype=torch.float32)
        self.actions = torch.tensor(df["action"].values, dtype=torch.float32).unsqueeze(1)
        self.rewards = torch.tensor(df["reward"].values, dtype=torch.float32).unsqueeze(1)
        self.next_obs = torch.tensor(norm_next_obs, dtype=torch.float32)
        self.dones = torch.tensor(df["done"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx]
        )