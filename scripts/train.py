import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from arena.envs.market_arena import SB3MarketArenaWrapper
from arena.utils.tracking import DecisionArenaMetricsCallback

def main():
    save_path = "models/ppo_arena_v1"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    vec_env = make_vec_env(
        lambda: SB3MarketArenaWrapper(ego_agent_id="agent_0", max_steps=200),
        n_envs=4,
        vec_env_cls=DummyVecEnv
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        device="cpu",
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.03,
        tensorboard_log="./tb_logs/",
        verbose=1
    )

    print("Starting Training...")
    model.learn(total_timesteps=50_000, callback=DecisionArenaMetricsCallback())
    model.save(save_path)
    print(f"Model saved at {save_path}.zip")

if __name__ == "__main__":
    main()