"""Evaluate an IPPO checkpoint on the cooperative transport environment."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Support running as a script from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.ippo import ActorCritic
from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.scene_config import RandomLevel


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained IPPO checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["fixed", "mild", "moderate", "full"],
        help="Randomization level for evaluation environment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2400,
        help="Per-episode truncation limit.",
    )
    parser.add_argument("--force-max", type=float, default=500.0)
    parser.add_argument("--stuck-patience", type=int, default=600)
    parser.add_argument("--stuck-move-eps", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Enable Pygame replay.")
    parser.add_argument("--fps", type=int, default=60, help="Replay FPS when --render is enabled.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from Gaussian policy instead of actor mean.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_actions(
    model: ActorCritic,
    obs_batch: np.ndarray,
    device: torch.device,
    stochastic: bool,
) -> np.ndarray:
    """Choose actions using actor mean or stochastic policy sampling."""
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        if stochastic:
            dist, _ = model.get_dist_and_value(obs_t)
            action_t = dist.sample()
        else:
            action_t, _ = model.forward(obs_t)
    action = torch.clamp(action_t, -1.0, 1.0)
    return action.cpu().numpy()


def main() -> None:
    """Run evaluation episodes and print summary metrics."""
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    obs_dim = int(checkpoint["obs_dim"])
    action_dim = int(checkpoint["action_dim"])
    cfg = checkpoint.get("config", {})

    hidden_size = int(cfg.get("hidden_size", 256))
    action_std_init = float(cfg.get("action_std_init", 0.6))

    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        action_std_init=action_std_init,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

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

    renderer: Any = None
    clock: Any = None
    if args.render:
        import pygame

        from src.sim.renderer import Renderer

        pygame.init()
        screen = pygame.display.set_mode((env.world.width, env.world.height))
        pygame.display.set_caption("IPPO Evaluation Replay")
        renderer = Renderer(screen, env.world)
        clock = pygame.time.Clock()

    success_count = 0
    invalid_count = 0
    returns: list[float] = []
    ep_steps: list[int] = []
    final_dists: list[float] = []

    for ep in range(args.episodes):
        observations, infos = env.reset(seed=args.seed + ep)
        episode_return = 0.0
        steps = 0
        manual_quit = False

        while observations:
            if args.render:
                assert renderer is not None and clock is not None
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        manual_quit = True
                        break
                if manual_quit:
                    break

            obs_batch = np.stack([observations[a] for a in agent_order], axis=0)
            actions = choose_actions(
                model,
                obs_batch,
                device,
                stochastic=args.stochastic,
            )
            action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}

            observations, rewards, terms, truncs, infos = env.step(action_dict)

            if args.render:
                renderer.draw()
                pygame.display.flip()
                clock.tick(args.fps)

            reward_batch = np.asarray([rewards[a] for a in agent_order], dtype=np.float32)
            episode_return += float(np.mean(reward_batch))
            steps += 1

            done = bool(all(terms[a] or truncs[a] for a in agent_order))
            if done:
                break

        if manual_quit:
            print("[Info] Replay window closed by user. Stop evaluation early.")
            break

        last_info = infos.get(agent_order[0], {}) if infos else {}
        success = bool(last_info.get("success", False))
        invalid = bool(last_info.get("invalid_state", False))
        final_dist = float(last_info.get("distance_to_goal", np.nan))

        success_count += int(success)
        invalid_count += int(invalid)
        returns.append(episode_return)
        ep_steps.append(steps)
        final_dists.append(final_dist)

        print(
            f"episode={ep + 1}/{args.episodes} "
            f"success={int(success)} invalid={int(invalid)} "
            f"return={episode_return:.3f} steps={steps} final_dist={final_dist:.2f}"
        )

    success_rate = success_count / max(1, args.episodes)
    invalid_rate = invalid_count / max(1, args.episodes)

    print("\n===== Evaluation Summary =====")
    print(f"checkpoint:   {ckpt_path}")
    print(f"episodes:     {args.episodes}")
    print(f"success_rate: {success_rate:.3f}")
    print(f"invalid_rate: {invalid_rate:.3f}")
    print(f"avg_return:   {float(np.mean(returns)):.3f}")
    print(f"avg_steps:    {float(np.mean(ep_steps)):.1f}")
    print(f"avg_final_dist: {float(np.nanmean(final_dists)):.2f}")

    if args.render:
        import pygame

        pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
