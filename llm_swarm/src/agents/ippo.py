"""Independent PPO trainer for parallel multi-agent environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class IPPOConfig:
    """Hyper-parameters for IPPO training."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    learning_rate: float = 3e-4
    rollout_steps: int = 1024
    update_epochs: int = 10
    minibatch_size: int = 256

    hidden_size: int = 256
    action_std_init: float = 0.6


class ActorCritic(nn.Module):
    """Shared actor-critic network for homogeneous agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        action_std_init: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        log_std = np.log(action_std_init)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action mean and value."""
        feat = self.encoder(obs)
        mean = self.actor_mean(feat)
        value = self.critic(feat).squeeze(-1)
        return mean, value

    def get_dist_and_value(
        self,
        obs: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        """Build Gaussian policy and value estimate."""
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist, value


class RolloutBuffer:
    """Simple rollout buffer for on-policy updates."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.logprobs: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add one parallel step of transitions."""
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def clear(self) -> None:
        """Clear all buffered transitions."""
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


class IPPOTrainer:
    """Independent PPO with a shared policy across homogeneous agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: IPPOConfig,
        device: str = "cpu",
    ) -> None:
        self.cfg = config
        self.device = torch.device(device)

        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=config.hidden_size,
            action_std_init=config.action_std_init,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer()

    def act(
        self,
        obs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions from current policy.

        Args:
            obs: Batched observations with shape [n_agents, obs_dim].

        Returns:
            actions: Shape [n_agents, action_dim], clipped to [-1, 1].
            logprobs: Shape [n_agents].
            values: Shape [n_agents].
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = self.model.get_dist_and_value(obs_t)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)

        action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
        return action_np, logprob.cpu().numpy(), value.cpu().numpy()

    def value(self, obs: np.ndarray) -> np.ndarray:
        """Estimate value for batched observations."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, value = self.model.get_dist_and_value(obs_t)
        return value.cpu().numpy()

    def update(self, last_values: np.ndarray) -> dict[str, float]:
        """Run PPO updates from collected rollouts."""
        obs = np.asarray(self.buffer.obs, dtype=np.float32)
        actions = np.asarray(self.buffer.actions, dtype=np.float32)
        old_logprobs = np.asarray(self.buffer.logprobs, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros_like(last_values, dtype=np.float32)

        for t in reversed(range(rewards.shape[0])):
            next_values = last_values if t == rewards.shape[0] - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values * non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        b_obs = obs.reshape(-1, obs.shape[-1])
        b_actions = actions.reshape(-1, actions.shape[-1])
        b_old_logprobs = old_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        batch_size = b_obs.shape[0]
        idxs = np.arange(batch_size)

        actor_loss_meter = 0.0
        critic_loss_meter = 0.0
        entropy_meter = 0.0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb = idxs[start:end]

                mb_obs = torch.as_tensor(
                    b_obs[mb], dtype=torch.float32, device=self.device
                )
                mb_actions = torch.as_tensor(
                    b_actions[mb], dtype=torch.float32, device=self.device
                )
                mb_old_logprobs = torch.as_tensor(
                    b_old_logprobs[mb], dtype=torch.float32, device=self.device
                )
                mb_adv = torch.as_tensor(
                    b_advantages[mb], dtype=torch.float32, device=self.device
                )
                mb_returns = torch.as_tensor(
                    b_returns[mb], dtype=torch.float32, device=self.device
                )
                mb_old_values = torch.as_tensor(
                    b_values[mb], dtype=torch.float32, device=self.device
                )

                dist, new_values = self.model.get_dist_and_value(mb_obs)
                new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                )
                actor_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.cfg.clip_coef,
                    self.cfg.clip_coef,
                )
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = (
                    actor_loss
                    + self.cfg.vf_coef * critic_loss
                    - self.cfg.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                actor_loss_meter += float(actor_loss.item())
                critic_loss_meter += float(critic_loss.item())
                entropy_meter += float(entropy.item())

        self.buffer.clear()
        updates = max(1, self.cfg.update_epochs * (batch_size // self.cfg.minibatch_size + 1))

        return {
            "actor_loss": actor_loss_meter / updates,
            "critic_loss": critic_loss_meter / updates,
            "entropy": entropy_meter / updates,
        }
