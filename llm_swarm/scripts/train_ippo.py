"""Train an IPPO baseline on cooperative transport."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Support running as a script from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.ippo import IPPOConfig, IPPOTrainer
from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.scene_config import RandomLevel


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train IPPO for cooperative transport.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["fixed", "mild", "moderate", "full"],
    )
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=2400)
    parser.add_argument("--force-max", type=float, default=500.0)
    parser.add_argument("--stuck-patience", type=int, default=600)
    parser.add_argument("--stuck-move-eps", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/ippo_transport.pt")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    """Main training loop."""
    args = parse_args()
    set_seed(args.seed)

    level = RandomLevel(args.level)

    env = TransportParallelEnv(
        config=None,
        random_level=level,
        max_steps=args.max_episode_steps,
        force_max=args.force_max,
        stuck_patience=args.stuck_patience,
        stuck_move_eps=args.stuck_move_eps,
    )

    agent_order = list(env.possible_agents)
    obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).shape[0]

    cfg = IPPOConfig(
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    trainer = IPPOTrainer(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device=args.device)

    observations, _ = env.reset(seed=args.seed)

    global_steps = 0
    update_idx = 0
    ep_return = 0.0
    ep_len = 0
    recent_returns: list[float] = []

    while global_steps < args.total_steps:
        for _ in range(cfg.rollout_steps):
            if not observations:
                observations, _ = env.reset()

            obs_batch = np.stack([observations[a] for a in agent_order], axis=0)
            actions, logprobs, values = trainer.act(obs_batch)

            action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}
            next_obs, rewards, terms, truncs, infos = env.step(action_dict)

            reward_batch = np.asarray([rewards[a] for a in agent_order], dtype=np.float32)
            done_batch = np.asarray(
                [float(terms[a] or truncs[a]) for a in agent_order],
                dtype=np.float32,
            )

            trainer.buffer.add(
                obs=obs_batch,
                actions=actions,
                logprobs=logprobs,
                values=values,
                rewards=reward_batch,
                dones=done_batch,
            )

            shared_reward = float(np.mean(reward_batch))
            ep_return += shared_reward
            ep_len += 1

            global_steps += 1
            done = bool(all(terms[a] or truncs[a] for a in agent_order))

            if done:
                recent_returns.append(ep_return)
                if len(recent_returns) > 100:
                    recent_returns.pop(0)
                ep_return = 0.0
                ep_len = 0
                observations, _ = env.reset()
            else:
                observations = next_obs

            if global_steps >= args.total_steps:
                break

        if observations:
            last_obs_batch = np.stack([observations[a] for a in agent_order], axis=0)
            last_values = trainer.value(last_obs_batch)
        else:
            last_values = np.zeros(len(agent_order), dtype=np.float32)

        stats = trainer.update(last_values=last_values)
        update_idx += 1

        if update_idx % args.log_interval == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            print(
                f"update={update_idx} steps={global_steps} "
                f"avg_return_100={avg_return:.3f} "
                f"actor_loss={stats['actor_loss']:.4f} "
                f"critic_loss={stats['critic_loss']:.4f} "
                f"entropy={stats['entropy']:.4f}"
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "config": cfg.__dict__,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "agent_order": agent_order,
        },
        save_path,
    )
    print(f"Training complete. Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
