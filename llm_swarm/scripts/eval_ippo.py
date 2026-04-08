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
    parser.add_argument("--stage", type=str, default="none", choices=["none", "1", "2", "3"])
    parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["fixed", "mild", "moderate", "full"],
        help="Randomization level for evaluation environment.",
    )
    parser.add_argument("--cargo-preset", type=str, default="L", choices=["L", "T", "U"])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=3600,
        help="Per-episode truncation limit.",
    )
    parser.add_argument("--force-max", type=float, default=500.0)
    parser.add_argument("--stuck-patience", type=int, default=600)
    parser.add_argument("--stuck-move-eps", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=180.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--blocked-penalty-weight", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=12.0)
    parser.add_argument("--stuck-penalty", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=0.0)
    parser.add_argument("--stagnation-penalty-weight", type=float, default=0.02)
    parser.add_argument("--away-penalty-weight", type=float, default=0.0)
    parser.add_argument("--heading-reward-weight", type=float, default=0.0)
    parser.add_argument("--action-penalty-weight", type=float, default=0.0)
    parser.add_argument("--clearance-penalty-weight", type=float, default=2.0)
    parser.add_argument("--clearance-safe-distance", type=float, default=90.0)
    parser.add_argument("--clearance-penalty-power", type=float, default=2.2)
    parser.add_argument("--preclearance-penalty-weight", type=float, default=0.4)
    parser.add_argument("--avoid-alert-distance", type=float, default=180.0)
    parser.add_argument("--avoid-blend-gain", type=float, default=0.8)
    parser.add_argument("--avoid-torque-gain", type=float, default=140.0)
    parser.add_argument("--clearance-margin-reward-weight", type=float, default=0.08)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-speed-threshold", type=float, default=22.0)
    parser.add_argument("--low-speed-far-goal-radius", type=float, default=160.0)
    parser.add_argument("--omega-penalty-weight", type=float, default=0.0)
    parser.add_argument("--velocity-penalty-weight", type=float, default=0.0)
    parser.add_argument("--recovery-push-gain", type=float, default=0.45)
    parser.add_argument("--recovery-torque-gain", type=float, default=180.0)
    parser.add_argument("--recovery-stuck-steps", type=int, default=25)
    parser.add_argument("--unblock-reward-weight", type=float, default=0.25)
    parser.add_argument("--clearance-improve-reward-weight", type=float, default=0.2)
    parser.add_argument("--near-goal-radius", type=float, default=120.0)
    parser.add_argument("--near-goal-speed-penalty-weight", type=float, default=0.0)
    parser.add_argument("--route-cell-size", type=int, default=40)
    parser.add_argument("--route-inflate-margin", type=float, default=70.0)
    parser.add_argument("--route-guidance-gain", type=float, default=0.7)
    parser.add_argument("--residual-force-scale", type=float, default=0.6)
    parser.add_argument("--route-waypoint-tolerance", type=float, default=50.0)
    parser.add_argument("--route-progress-weight", type=float, default=0.0)
    parser.add_argument("--route-deviation-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-mode",
        type=str,
        default="object_wrench",
        choices=["robot_residual", "object_wrench"],
    )
    parser.add_argument("--object-wrench-residual-scale-xy", type=float, default=0.5)
    parser.add_argument("--object-wrench-residual-scale-tau", type=float, default=0.5)
    parser.add_argument("--route-torque-gain", type=float, default=220.0)
    parser.add_argument("--route-linear-damping-gain", type=float, default=8.0)
    parser.add_argument("--route-angular-damping-gain", type=float, default=24.0)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--random-init-theta", action="store_true")
    parser.add_argument("--init-theta-min", type=float, default=-np.pi)
    parser.add_argument("--init-theta-max", type=float, default=np.pi)
    parser.add_argument("--stage3-gap-height", type=float, default=200.0)
    parser.add_argument("--stage3-wall-width", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Enable Pygame replay.")
    parser.add_argument("--fps", type=int, default=120, help="Replay FPS when --render is enabled.")
    parser.add_argument(
        "--render-steps-per-frame",
        type=int,
        default=3,
        help="Number of environment steps advanced before each rendered frame.",
    )
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
    log_std_min = float(cfg.get("log_std_min", -1.2))
    log_std_max = float(cfg.get("log_std_max", 0.8))

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

    level_name = args.level
    no_obstacles = args.no_obstacles
    random_init_theta = args.random_init_theta
    if args.stage == "1":
        level_name = "fixed"
        no_obstacles = True
        random_init_theta = False
    elif args.stage == "2":
        level_name = "mild"
        no_obstacles = True
        random_init_theta = True
    elif args.stage == "3":
        level_name = "fixed"
        no_obstacles = False
        random_init_theta = True

    level = RandomLevel(level_name)
    env = TransportParallelEnv(
        config=None,
        random_level=level,
        max_steps=args.max_episode_steps,
        force_max=args.force_max,
        stuck_patience=args.stuck_patience,
        stuck_move_eps=args.stuck_move_eps,
        fixed_num_agents=args.num_agents,
        fixed_cargo_preset=args.cargo_preset,
        progress_weight=args.progress_weight,
        step_penalty=args.step_penalty,
        blocked_penalty_weight=args.blocked_penalty_weight,
        success_bonus=args.success_bonus,
        stuck_penalty=args.stuck_penalty,
        timeout_penalty=args.timeout_penalty,
        stagnation_penalty_weight=args.stagnation_penalty_weight,
        away_penalty_weight=args.away_penalty_weight,
        heading_reward_weight=args.heading_reward_weight,
        action_penalty_weight=args.action_penalty_weight,
        clearance_penalty_weight=args.clearance_penalty_weight,
        clearance_safe_distance=args.clearance_safe_distance,
        clearance_penalty_power=args.clearance_penalty_power,
        preclearance_penalty_weight=args.preclearance_penalty_weight,
        avoid_alert_distance=args.avoid_alert_distance,
        avoid_blend_gain=args.avoid_blend_gain,
        avoid_torque_gain=args.avoid_torque_gain,
        clearance_margin_reward_weight=args.clearance_margin_reward_weight,
        low_speed_penalty_weight=args.low_speed_penalty_weight,
        low_speed_threshold=args.low_speed_threshold,
        low_speed_far_goal_radius=args.low_speed_far_goal_radius,
        omega_penalty_weight=args.omega_penalty_weight,
        velocity_penalty_weight=args.velocity_penalty_weight,
        recovery_push_gain=args.recovery_push_gain,
        recovery_torque_gain=args.recovery_torque_gain,
        recovery_stuck_steps=args.recovery_stuck_steps,
        unblock_reward_weight=args.unblock_reward_weight,
        clearance_improve_reward_weight=args.clearance_improve_reward_weight,
        near_goal_radius=args.near_goal_radius,
        near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
        route_cell_size=args.route_cell_size,
        route_inflate_margin=args.route_inflate_margin,
        route_guidance_gain=args.route_guidance_gain,
        residual_force_scale=args.residual_force_scale,
        route_waypoint_tolerance=args.route_waypoint_tolerance,
        route_progress_weight=args.route_progress_weight,
        route_deviation_penalty_weight=args.route_deviation_penalty_weight,
        action_mode=args.action_mode,
        object_wrench_residual_scale_xy=args.object_wrench_residual_scale_xy,
        object_wrench_residual_scale_tau=args.object_wrench_residual_scale_tau,
        route_torque_gain=args.route_torque_gain,
        route_linear_damping_gain=args.route_linear_damping_gain,
        route_angular_damping_gain=args.route_angular_damping_gain,
        no_obstacles=no_obstacles,
        random_init_theta=random_init_theta,
        init_theta_min=args.init_theta_min,
        init_theta_max=args.init_theta_max,
        curriculum_stage=args.stage,
        stage3_gap_height=args.stage3_gap_height,
        stage3_wall_width=args.stage3_wall_width,
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

            steps_per_frame = 1
            if args.render:
                steps_per_frame = max(1, int(args.render_steps_per_frame))

            done = False
            for _ in range(steps_per_frame):
                if not observations:
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

                reward_batch = np.asarray([rewards[a] for a in agent_order], dtype=np.float32)
                episode_return += float(np.mean(reward_batch))
                steps += 1

                done = bool(all(terms[a] or truncs[a] for a in agent_order))
                if done:
                    break

            if args.render:
                renderer.draw()
                pygame.display.flip()
                clock.tick(args.fps)

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
