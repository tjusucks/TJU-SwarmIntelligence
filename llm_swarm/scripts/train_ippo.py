"""Train an IPPO baseline on cooperative transport."""

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

from src.agents.ippo import IPPOConfig, IPPOTrainer
from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.scene_config import RandomLevel


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train IPPO for cooperative transport.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=str, default="none", choices=["none", "1", "2", "3"])
    parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["fixed", "mild", "moderate", "full"],
    )
    parser.add_argument("--cargo-preset", type=str, default="L", choices=["L", "T", "U"])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=2400)
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--action-std-init", type=float, default=0.35)
    parser.add_argument("--log-std-min", type=float, default=-1.2)
    parser.add_argument("--log-std-max", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-log-ratio", type=float, default=10.0)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--render-train",
        action="store_true",
        help="Enable live visualization during training.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=180,
        help="Render FPS when --render-train is enabled.",
    )
    parser.add_argument(
        "--render-steps-per-frame",
        type=int,
        default=4,
        help="Only draw one frame every N environment steps for faster playback.",
    )
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
    obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).shape[0]

    cfg = IPPOConfig(
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        action_std_init=args.action_std_init,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        target_kl=args.target_kl,
        max_log_ratio=args.max_log_ratio,
        minibatch_size=args.minibatch_size,
    )
    trainer = IPPOTrainer(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device=args.device)

    observations, _ = env.reset(seed=args.seed)

    renderer: Any = None
    clock: Any = None
    train_render_enabled = bool(args.render_train)
    render_stride = max(1, int(args.render_steps_per_frame))
    if train_render_enabled:
        import pygame

        from src.sim.renderer import Renderer

        pygame.init()
        screen = pygame.display.set_mode((env.world.width, env.world.height))
        pygame.display.set_caption("IPPO Training Replay")
        renderer = Renderer(screen, env.world)
        clock = pygame.time.Clock()

    global_steps = 0
    update_idx = 0
    ep_return = 0.0
    ep_len = 0
    recent_returns: list[float] = []
    recent_ep_lens: list[int] = []
    total_completed_episodes = 0

    while global_steps < args.total_steps:
        for _ in range(cfg.rollout_steps):
            if not observations:
                observations, _ = env.reset()

            obs_batch = np.stack([observations[a] for a in agent_order], axis=0)
            actions, logprobs, values = trainer.act(obs_batch)

            action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}
            next_obs, rewards, terms, truncs, infos = env.step(action_dict)

            if train_render_enabled and (global_steps % render_stride == 0):
                assert renderer is not None and clock is not None
                import pygame

                keep_running = True
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        keep_running = False
                        break
                if keep_running:
                    renderer.draw()
                    pygame.display.flip()
                    clock.tick(max(1, int(args.render_fps)))
                else:
                    train_render_enabled = False

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
                recent_ep_lens.append(ep_len)
                if len(recent_returns) > 100:
                    recent_returns.pop(0)
                if len(recent_ep_lens) > 100:
                    recent_ep_lens.pop(0)
                total_completed_episodes += 1
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
            avg_len = float(np.mean(recent_ep_lens)) if recent_ep_lens else 0.0
            avg_return_per_step = avg_return / max(1.0, avg_len)
            print(
                f"update={update_idx} steps={global_steps} "
                f"episodes={total_completed_episodes} "
                f"avg_return_100={avg_return:.3f} "
                f"avg_len_100={avg_len:.1f} "
                f"avg_return_per_step_100={avg_return_per_step:.4f} "
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

    if args.render_train:
        import pygame

        pygame.quit()


if __name__ == "__main__":
    main()
