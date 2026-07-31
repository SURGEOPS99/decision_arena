import os
import torch
from torch.utils.data import DataLoader
from arena.utils.dataset import OfflineArenaDataset
from arena.agents.offline_rl import TD3_BCAgent

def train_offline(batch_size=256, epochs=20, save_path="models/td3_bc_arena_v1.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading offline dataset on {device}...")
    dataset = OfflineArenaDataset("data/arena_trajectories.parquet")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    agent = TD3_BCAgent(state_dim=10, action_dim=1, device=device)
    agent.set_normalization(dataset.mean, dataset.std)

    print(f"Starting Offline TD3+BC Training ({len(dataset)} samples, {epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for state, action, reward, next_state, done in dataloader:
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            c_loss, a_loss = agent.train_step(state, action, reward, next_state, done)
            total_critic_loss += c_loss
            total_actor_loss += a_loss

        avg_c_loss = total_critic_loss / len(dataloader)
        avg_a_loss = total_actor_loss / (len(dataloader) / 2)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Critic Loss: {avg_c_loss:.4f} | Actor Loss: {avg_a_loss:.4f}")

    agent.save(save_path)
    print(f"\nModel successfully saved to {save_path}")

if __name__ == "__main__":
    train_offline()