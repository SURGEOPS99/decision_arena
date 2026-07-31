import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from arena.agents.base import BaseAgent

class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


class TD3_BCAgent(BaseAgent):
    """
    Offline RL Agent using Twin Delayed DDPG with Behavior Cloning (TD3+BC).
    """
    def __init__(self, state_dim=10, action_dim=1, alpha=2.5, gamma=0.99, tau=0.005, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.total_it = 0
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.std = np.ones(state_dim, dtype=np.float32)

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Apply normalization matching dataset stats
        norm_obs = (obs - self.mean) / self.std
        state = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        return action

    def train_step(self, state, action, reward, next_state, done):
        self.total_it += 1

        with torch.no_grad():
            # Target action smoothing
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            # Target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * self.gamma * target_Q

        # Critic update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        # Delayed policy updates (every 2 steps)
        if self.total_it % 2 == 0:
            pi = self.actor(state)
            q_val = self.critic.Q1(state, pi)
            
            # TD3+BC Actor Loss: -lambda * Q + BC_Loss
            lmbda = self.alpha / (q_val.abs().mean().detach() + 1e-6)
            actor_loss = -lmbda * q_val.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            actor_loss_val = actor_loss.item()

        return critic_loss.item(), actor_loss_val

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "mean": self.mean,
            "std": self.std
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]