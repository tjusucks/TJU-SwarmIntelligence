"""Replay a trained IPPO checkpoint with the Pygame renderer."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

# Support running as a script from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.ippo import ActorCritic, IndependentPolicies
from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.renderer import Renderer
from src.sim.scene_config import RandomLevel, SceneGenerator


def parse_args() -> argparse.Namespace:
    """Parse replay arguments."""
    parser = argparse.ArgumentParser(description="Replay a trained IPPO checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--level",
        type=str,
        default="fixed",
        choices=["fixed", "mild", "moderate", "full"],
        help="Randomization level.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base replay seed.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1200,
        help="Per-episode truncation limit.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
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


def policy_actions(
    model: ActorCritic | IndependentPolicies,
    obs_batch: np.ndarray,
    device: torch.device,
    stochastic: bool,
) -> np.ndarray:
    """Return policy actions in [-1, 1]."""
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        if stochastic:
            dist, _ = model.get_dist_and_value(obs_t)
            action_t = dist.sample()
        else:
            mean_t, _ = model.forward(obs_t)
            action_t = mean_t
    return torch.clamp(action_t, -1.0, 1.0).cpu().numpy()


def main() -> None:
    """Run visual replay episodes."""
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    obs_dim = int(checkpoint["obs_dim"])
    action_dim = int(checkpoint["action_dim"])
    cfg = checkpoint.get("config", {})

    hidden_size = int(cfg.get("hidden_size", 256))
    action_std_init = float(cfg.get("action_std_init", 0.6))
    log_std_min = float(cfg.get("log_std_min", -1.2))
    log_std_max = float(cfg.get("log_std_max", 0.8))

    is_independent = any(k.startswith("models.") for k in checkpoint["model"].keys())
    if is_independent:
        if "agent_order" in checkpoint:
            agent_order = checkpoint["agent_order"]
        else:
            agent_ids_set = set()
            for k in checkpoint["model"].keys():
                if k.startswith("models."):
                    parts = k.split(".")
                    agent_ids_set.add(parts[1])
            agent_order = sorted(list(agent_ids_set))
        model = IndependentPolicies(
            agent_ids=agent_order,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_std_init=action_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)
    else:
        model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_std_init=action_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    goal_orientation_matching = checkpoint.get("goal_orientation_matching", False)
    goal_angle_tolerance = checkpoint.get("goal_angle_tolerance", 0.2)
    random_goal_theta = checkpoint.get("random_goal_theta", False)

    # Read environment setup parameters from checkpoint config if available
    train_args = checkpoint.get("train_args", {})
    stage = train_args.get("stage", "none")
    no_obstacles = train_args.get("no_obstacles", False)
    random_init_theta = train_args.get("random_init_theta", False)
    init_theta_min = train_args.get("init_theta_min", -np.pi)
    init_theta_max = train_args.get("init_theta_max", np.pi)
    stage3_gap_height = train_args.get("stage3_gap_height", 200.0)
    stage3_wall_width = train_args.get("stage3_wall_width", 42)
    stage4_gap_span = train_args.get("stage4_gap_span", 165.0)
    stage4_wall_width = train_args.get("stage4_wall_width", 34)

    # Fallback to checking the filename if stage is none (for older checkpoints)
    if stage == "none":
        ckpt_name = ckpt_path.name.lower()
        if "stage1" in ckpt_name:
            stage = "1"
        elif "stage2" in ckpt_name:
            stage = "2"
        elif "stage3" in ckpt_name:
            stage = "3"
        elif "stage4" in ckpt_name:
            stage = "4"

    # Automatically map stage to the correct default level, obstacles, and theta config
    level_name = args.level
    if stage == "1":
        level_name = "fixed"
        no_obstacles = True
        random_init_theta = False
    elif stage == "2":
        level_name = "mild"
        no_obstacles = True
        random_init_theta = True
    elif stage in ("3", "4"):
        level_name = "fixed"
        no_obstacles = False
        random_init_theta = True

    level = RandomLevel(level_name)
    config = SceneGenerator(level=level).generate(
        seed=args.seed,
        random_init_theta=random_init_theta,
        init_theta_min=init_theta_min,
        init_theta_max=init_theta_max,
        random_goal_theta=random_goal_theta,
    )
    env = TransportParallelEnv(
        config=config,
        random_level=level,
        max_steps=args.max_episode_steps,
        goal_orientation_matching=goal_orientation_matching,
        goal_angle_tolerance=goal_angle_tolerance,
        random_goal_theta=random_goal_theta,
        no_obstacles=no_obstacles,
        random_init_theta=random_init_theta,
        init_theta_min=init_theta_min,
        init_theta_max=init_theta_max,
        curriculum_stage=stage,
        stage3_gap_height=stage3_gap_height,
        stage3_wall_width=stage3_wall_width,
        stage4_gap_span=stage4_gap_span,
        stage4_wall_width=stage4_wall_width,
    )

    agent_order = list(env.possible_agents)

    pygame.init()
    screen = pygame.display.set_mode((env.world.width, env.world.height))
    pygame.display.set_caption("IPPO Replay")
    clock = pygame.time.Clock()
    renderer = Renderer(screen, env.world)

    paused = False
    ep_idx = 0
    observations, infos = env.reset(seed=args.seed)
    ep_return = 0.0

    while ep_idx < args.episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    observations, infos = env.reset(seed=args.seed + ep_idx)
                    ep_return = 0.0

        if not paused and observations:
            obs_batch = np.stack([observations[a] for a in agent_order], axis=0)
            actions = policy_actions(
                model, obs_batch, device, stochastic=args.stochastic
            )
            action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}

            observations, rewards, terms, truncs, infos = env.step(action_dict)
            reward_batch = np.asarray(
                [rewards[a] for a in agent_order], dtype=np.float32
            )
            ep_return += float(np.mean(reward_batch))

            done = bool(all(terms[a] or truncs[a] for a in agent_order))
            if done:
                info = infos.get(agent_order[0], {}) if infos else {}
                success = int(bool(info.get("success", False)))
                invalid = int(bool(info.get("invalid_state", False)))
                final_dist = float(info.get("distance_to_goal", np.nan))
                print(
                    f"episode={ep_idx + 1}/{args.episodes} success={success} "
                    f"invalid={invalid} return={ep_return:.3f} "
                    f"steps={int(info.get('step', 0))} final_dist={final_dist:.2f}"
                )

                ep_idx += 1
                if ep_idx >= args.episodes:
                    break

                observations, infos = env.reset(seed=args.seed + ep_idx)
                ep_return = 0.0

        renderer.draw()
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
