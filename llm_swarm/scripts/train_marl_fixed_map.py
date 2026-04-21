"""Train MARL on a fixed-map cooperative transport setup.

This script is a refactored pipeline for fixed-map experiments:
- Global observations for every agent.
- Direct multi-agent MARL training via shared-policy PPO.
- Optional lightweight communication channel (not hard-coded semantics).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.ippo import IPPOConfig, IPPOTrainer
from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.scene_config import RandomLevel, SceneConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored MARL training on one fixed map.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--map-mode",
        type=str,
        default="fixed",
        choices=["fixed", "fixed_four", "random_curriculum", "random_level"],
        help="Map mode: fixed | fixed_four | random curriculum (easy->hard) | random fixed difficulty.",
    )
    parser.add_argument(
        "--random-level",
        type=str,
        default="moderate",
        choices=["mild", "moderate", "full"],
        help="Used when --map-mode random_level.",
    )
    parser.add_argument(
        "--reachability-max-retries",
        type=int,
        default=30,
        help="Maximum reset retries to find a reachable map.",
    )
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--cargo-preset", type=str, default="L", choices=["L", "T", "U"])
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Maximum environment steps. <=0 means no step limit.",
    )
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Stop training after this many scene episodes. 0 means disabled.",
    )
    parser.add_argument("--force-max", type=float, default=500.0)

    parser.add_argument("--progress-weight", type=float, default=12.0)
    parser.add_argument("--step-penalty", type=float, default=0.001)
    parser.add_argument("--blocked-penalty-weight", type=float, default=0.035)
    parser.add_argument("--single-contact-penalty", type=float, default=0.0)
    parser.add_argument("--persistent-contact-penalty-weight", type=float, default=0.0015)
    parser.add_argument("--contact-ratio-threshold", type=float, default=0.01)
    parser.add_argument("--ineffective-action-penalty-weight", type=float, default=0.008)
    parser.add_argument("--ineffective-action-threshold", type=float, default=0.25)
    parser.add_argument("--ineffective-motion-threshold", type=float, default=1.5)
    parser.add_argument("--ineffective-blocked-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--contact-progress-compensation-weight", type=float, default=0.0)
    parser.add_argument("--contact-progress-ref", type=float, default=0.0014)
    parser.add_argument("--success-bonus", type=float, default=20.0)
    parser.add_argument("--stuck-penalty", type=float, default=0.1)
    parser.add_argument("--timeout-penalty", type=float, default=0.0)
    parser.add_argument("--stagnation-penalty-weight", type=float, default=0.025)
    parser.add_argument("--stagnation-window", type=int, default=20)
    parser.add_argument("--stagnation-progress-threshold", type=float, default=0.2)

    parser.add_argument("--heading-reward-weight", type=float, default=0.8)
    parser.add_argument("--rot-jam-reward-weight", type=float, default=1.5)
    parser.add_argument("--effective-rot-reward", type=float, default=0.8)
    parser.add_argument("--effective-rot-theta-threshold", type=float, default=0.04)
    parser.add_argument("--rotation-no-penalty-theta-threshold", type=float, default=0.03)
    parser.add_argument("--jam-state-motion-threshold", type=float, default=1.5)
    parser.add_argument("--jam-penalty-motion-threshold", type=float, default=1.2)
    parser.add_argument("--action-penalty-weight", type=float, default=0.0001)
    parser.add_argument("--clearance-penalty-weight", type=float, default=0.0)
    parser.add_argument("--clearance-safe-distance", type=float, default=90.0)
    parser.add_argument("--unblock-reward-weight", type=float, default=0.8)
    parser.add_argument("--clearance-improve-reward-weight", type=float, default=0.6)
    parser.add_argument("--milestone-reward", type=float, default=1.0)

    parser.add_argument("--jam-penalty-weight", type=float, default=0.0)
    parser.add_argument("--jam-distance-threshold", type=float, default=100.0)
    parser.add_argument("--turn-escape-reward-weight", type=float, default=0.0)
    parser.add_argument("--recovery-push-gain", type=float, default=0.65)
    parser.add_argument("--recovery-torque-gain", type=float, default=260.0)
    parser.add_argument("--recovery-stuck-steps", type=int, default=12)
    parser.add_argument("--recover-reward-weight", type=float, default=1.5)
    parser.add_argument("--recover-jam-steps", type=int, default=8)
    parser.add_argument("--fail-fast-low-progress-window", type=int, default=50)
    parser.add_argument("--fail-fast-low-progress-threshold", type=float, default=0.1)
    parser.add_argument("--fail-fast-high-blocked-steps", type=int, default=30)
    parser.add_argument("--fail-fast-blocked-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--fail-fast-ineffective-steps", type=int, default=25)
    parser.add_argument("--fail-fast-return-threshold", type=float, default=-1500.0)
    parser.add_argument("--fail-fast-penalty", type=float, default=0.8)
    parser.add_argument(
        "--enable-escape-burst",
        action="store_true",
        help="Enable heuristic burst escape control. Off by default for policy-driven learning.",
    )

    parser.add_argument("--action-mode", type=str, default="object_wrench", choices=["robot_residual", "object_wrench"])
    parser.add_argument("--object-wrench-residual-scale-xy", type=float, default=0.5)
    parser.add_argument("--object-wrench-residual-scale-tau", type=float, default=0.5)

    parser.add_argument("--comm-mode", type=str, default="none", choices=["none", "broadcast_action"])
    parser.add_argument("--comm-scale", type=float, default=1.0)

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

    parser.add_argument("--random-init-theta", action="store_true")
    parser.add_argument("--init-theta-min", type=float, default=-np.pi)
    parser.add_argument("--init-theta-max", type=float, default=np.pi)

    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-fps", type=int, default=120)
    parser.add_argument("--render-steps-per-frame", type=int, default=3)

    parser.add_argument("--save-path", type=str, default="checkpoints/marl_fixed_map.pt")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip post-training test (only applies to --map-mode fixed_four).",
    )
    parser.add_argument(
        "--test-episodes-per-map",
        type=int,
        default=6,
        help="Maximum test attempts per map in fixed_four mode.",
    )
    parser.add_argument(
        "--test-video-dir",
        type=str,
        default="artifacts/four_map_test_videos",
        help="Directory for successful test videos in fixed_four mode.",
    )
    parser.add_argument("--test-fps", type=int, default=30)
    parser.add_argument(
        "--test-video-frame-stride",
        type=int,
        default=2,
        help="Record one frame every N env steps during test video capture.",
    )
    parser.add_argument(
        "--test-show-window",
        action="store_true",
        help="Show a pygame window while testing fixed_four maps.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fixed_map_config(cargo_preset: str, num_agents: int) -> SceneConfig:
    """Single fixed map for direct MARL trial.

    The map is intentionally hard: three staggered narrow gates force
    repeated up/down routing (S-shape) before reaching the goal.
    """
    # Three offset gates (all passable) in zigzag order:
    # - Gate A at x=280 with opening y in [110, 270]
    # - Gate B at x=520 with opening y in [500, 660]
    # - Gate C at x=760 with opening y in [250, 410]
    # This creates a long S-shaped route and increases deadlock pressure
    # compared with the old two-gate map, while staying reachable.
    obstacles = [
        (280, 0, 36, 110),
        (280, 270, 36, 530),
        (520, 0, 36, 500),
        (520, 660, 36, 140),
        (760, 0, 36, 250),
        (760, 410, 36, 390),
    ]
    cfg = SceneConfig(
        seed=42,
        width=1000,
        height=800,
        cargo_preset=cargo_preset,
        cargo_x=140.0,
        cargo_y=700.0,
        cargo_theta=0.0,
        goal_x=900.0,
        goal_y=110.0,
        obstacles=obstacles,
        num_robots=int(num_agents),
    )
    return cfg


def build_four_fixed_map_configs(
    cargo_preset: str,
    num_agents: int,
) -> list[tuple[str, SceneConfig]]:
    """Build four fixed maps for cyclic training and testing."""

    map_defs = [
        {
            "name": "map1_s_zigzag",
            "cargo": (140.0, 700.0),
            "goal": (900.0, 110.0),
            "obstacles": [
                (280, 0, 36, 110),
                (280, 270, 36, 530),
                (520, 0, 36, 500),
                (520, 660, 36, 140),
                (760, 0, 36, 250),
                (760, 410, 36, 390),
            ],
        },
        {
            "name": "map2_reverse_s",
            "cargo": (120.0, 120.0),
            "goal": (900.0, 700.0),
            "obstacles": [
                (260, 0, 36, 520),
                (260, 690, 36, 110),
                (500, 0, 36, 180),
                (500, 340, 36, 460),
                (740, 0, 36, 470),
                (740, 630, 36, 170),
            ],
        },
        {
            "name": "map3_cross_corridor",
            "cargo": (120.0, 680.0),
            "goal": (900.0, 120.0),
            "obstacles": [
                (0, 220, 700, 34),
                (860, 220, 140, 34),
                (0, 540, 120, 34),
                (280, 540, 720, 34),
                (520, 0, 34, 300),
                (520, 460, 34, 340),
            ],
        },
        {
            "name": "map4_multi_gate",
            "cargo": (100.0, 700.0),
            "goal": (940.0, 100.0),
            "obstacles": [
                (220, 0, 32, 200),
                (220, 360, 32, 440),
                (430, 0, 32, 520),
                (430, 680, 32, 120),
                (640, 0, 32, 140),
                (640, 300, 32, 500),
                (850, 0, 32, 450),
                (850, 610, 32, 190),
            ],
        },
    ]

    out: list[tuple[str, SceneConfig]] = []
    for i, item in enumerate(map_defs):
        cfg = SceneConfig(
            seed=42 + i,
            width=1000,
            height=800,
            cargo_preset=cargo_preset,
            cargo_x=float(item["cargo"][0]),
            cargo_y=float(item["cargo"][1]),
            cargo_theta=0.0,
            goal_x=float(item["goal"][0]),
            goal_y=float(item["goal"][1]),
            obstacles=list(item["obstacles"]),
            num_robots=int(num_agents),
        )
        out.append((str(item["name"]), cfg))
    return out


def build_transport_env(
    args: argparse.Namespace,
    config: SceneConfig | None,
    random_level: RandomLevel,
) -> TransportParallelEnv:
    """Build env with the same reward/control settings used in training."""
    return TransportParallelEnv(
        config=config,
        random_level=random_level,
        max_steps=args.max_episode_steps,
        force_max=args.force_max,
        fixed_num_agents=args.num_agents,
        fixed_cargo_preset=args.cargo_preset,
        progress_weight=args.progress_weight,
        step_penalty=args.step_penalty,
        blocked_penalty_weight=args.blocked_penalty_weight,
        single_contact_penalty=args.single_contact_penalty,
        persistent_contact_penalty_weight=args.persistent_contact_penalty_weight,
        contact_ratio_threshold=args.contact_ratio_threshold,
        ineffective_action_penalty_weight=args.ineffective_action_penalty_weight,
        ineffective_action_threshold=args.ineffective_action_threshold,
        ineffective_motion_threshold=args.ineffective_motion_threshold,
        ineffective_blocked_ratio_threshold=args.ineffective_blocked_ratio_threshold,
        contact_progress_compensation_weight=args.contact_progress_compensation_weight,
        contact_progress_ref=args.contact_progress_ref,
        success_bonus=args.success_bonus,
        stuck_penalty=args.stuck_penalty,
        timeout_penalty=args.timeout_penalty,
        stagnation_penalty_weight=args.stagnation_penalty_weight,
        stagnation_window=args.stagnation_window,
        stagnation_progress_threshold=args.stagnation_progress_threshold,
        heading_reward_weight=args.heading_reward_weight,
        rot_jam_reward_weight=args.rot_jam_reward_weight,
        effective_rot_reward=args.effective_rot_reward,
        effective_rot_theta_threshold=args.effective_rot_theta_threshold,
        rotation_no_penalty_theta_threshold=args.rotation_no_penalty_theta_threshold,
        jam_state_motion_threshold=args.jam_state_motion_threshold,
        jam_penalty_motion_threshold=args.jam_penalty_motion_threshold,
        action_penalty_weight=args.action_penalty_weight,
        clearance_penalty_weight=args.clearance_penalty_weight,
        clearance_safe_distance=args.clearance_safe_distance,
        milestone_reward=args.milestone_reward,
        jam_penalty_weight=args.jam_penalty_weight,
        jam_distance_threshold=args.jam_distance_threshold,
        turn_escape_reward_weight=args.turn_escape_reward_weight,
        recovery_push_gain=args.recovery_push_gain,
        recovery_torque_gain=args.recovery_torque_gain,
        recovery_stuck_steps=args.recovery_stuck_steps,
        recover_reward_weight=args.recover_reward_weight,
        recover_jam_steps=args.recover_jam_steps,
        fail_fast_low_progress_window=args.fail_fast_low_progress_window,
        fail_fast_low_progress_threshold=args.fail_fast_low_progress_threshold,
        fail_fast_high_blocked_steps=args.fail_fast_high_blocked_steps,
        fail_fast_blocked_ratio_threshold=args.fail_fast_blocked_ratio_threshold,
        fail_fast_ineffective_steps=args.fail_fast_ineffective_steps,
        fail_fast_return_threshold=args.fail_fast_return_threshold,
        fail_fast_penalty=args.fail_fast_penalty,
        escape_burst_enabled=args.enable_escape_burst,
        unblock_reward_weight=args.unblock_reward_weight,
        clearance_improve_reward_weight=args.clearance_improve_reward_weight,
        action_mode=args.action_mode,
        object_wrench_residual_scale_xy=args.object_wrench_residual_scale_xy,
        object_wrench_residual_scale_tau=args.object_wrench_residual_scale_tau,
        random_init_theta=args.random_init_theta,
        init_theta_min=args.init_theta_min,
        init_theta_max=args.init_theta_max,
    )


def policy_mean_actions(trainer: IPPOTrainer, policy_obs: np.ndarray) -> np.ndarray:
    """Deterministic policy action for evaluation."""
    obs_t = torch.as_tensor(policy_obs, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        mean_t, _ = trainer.model.forward(obs_t)
    return torch.clamp(mean_t, -1.0, 1.0).cpu().numpy()


def open_video_writer(
    out_path: Path,
    fps: int,
) -> tuple[Any | None, str | None]:
    """Open a streaming MP4 writer."""
    try:
        import imageio.v2 as imageio

        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(out_path),
            fps=max(1, int(fps)),
            codec="libx264",
            macro_block_size=1,
        )
        return writer, None
    except Exception as e:
        return None, str(e)


def close_video_writer(writer: Any | None) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        pass


def test_fixed_four_maps(
    args: argparse.Namespace,
    trainer: IPPOTrainer,
    fixed_maps: list[tuple[str, SceneConfig]],
    comm_dim: int,
) -> None:
    """Evaluate on four fixed maps and save successful episode videos."""
    import pygame
    from src.sim.renderer import Renderer

    if len(fixed_maps) == 0:
        return

    video_dir = Path(args.test_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    show_window = bool(args.test_show_window)
    clock: Any = pygame.time.Clock() if show_window else None
    window: Any = None

    print(
        f"[Test] Start fixed-four evaluation: maps={len(fixed_maps)} "
        f"attempts_per_map={args.test_episodes_per_map}"
    )

    success_count = 0
    video_count = 0

    for map_idx, (map_name, map_cfg) in enumerate(fixed_maps, start=1):
        env = build_transport_env(
            args=args,
            config=copy.deepcopy(map_cfg),
            random_level=RandomLevel.FIXED,
        )
        agent_order = list(env.possible_agents)

        if show_window:
            if window is None or window.get_size() != (env.world.width, env.world.height):
                window = pygame.display.set_mode((env.world.width, env.world.height))
            pygame.display.set_caption(f"Fixed-Four Test: {map_name}")
            screen = window
        else:
            screen = pygame.Surface((env.world.width, env.world.height))
        renderer = Renderer(screen, env.world)

        map_success = False
        saved_path: Path | None = None
        next_seed = int(args.seed + 10000 + map_idx * 1000)
        frame_stride = max(1, int(args.test_video_frame_stride))
        writer_warned = False

        for attempt in range(1, max(1, int(args.test_episodes_per_map)) + 1):
            env._base_config = copy.deepcopy(map_cfg)
            observations, _infos, _used_seed, next_seed = reset_reachable(
                env=env,
                seed_start=next_seed,
                max_retries=max(1, int(args.reachability_max_retries)),
            )
            comm_state = np.zeros(comm_dim, dtype=np.float32)

            tmp_video = video_dir / f".tmp_{map_idx:02d}_{map_name}_attempt{attempt}.mp4"
            writer, writer_err = open_video_writer(tmp_video, fps=max(1, int(args.test_fps)))
            if writer is None and not writer_warned:
                print(f"[Test] map={map_name} video writer unavailable: {writer_err}")
                writer_warned = True

            episode_success = False
            for step_idx in range(int(args.max_episode_steps)):
                if show_window:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            show_window = False

                policy_obs = build_policy_obs(
                    observations=observations,
                    agent_order=agent_order,
                    comm_mode=args.comm_mode,
                    comm_state=comm_state,
                    comm_scale=float(args.comm_scale),
                )
                actions = policy_mean_actions(trainer, policy_obs)
                action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}

                observations, rewards, terms, truncs, infos = env.step(action_dict)
                comm_state = update_comm_state(actions, args.comm_mode)

                renderer.draw()
                draw_route_overlay(env, renderer)
                if writer is not None and (step_idx % frame_stride == 0):
                    frame = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)).copy()
                    try:
                        writer.append_data(frame)
                    except Exception as e_append:
                        close_video_writer(writer)
                        writer = None
                        if not writer_warned:
                            print(f"[Test] map={map_name} video append failed: {e_append}")
                            writer_warned = True

                if show_window:
                    pygame.display.flip()
                    if clock is not None:
                        clock.tick(max(1, int(args.test_fps)))

                done = bool(all(terms[a] or truncs[a] for a in agent_order))
                if done:
                    info0 = infos.get(agent_order[0], {}) if infos else {}
                    episode_success = bool(info0.get("success", False))
                    break

            close_video_writer(writer)

            if episode_success:
                map_success = True
                out_path = video_dir / f"{map_idx:02d}_{map_name}_success.mp4"
                if tmp_video.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.exists():
                        out_path.unlink()
                    tmp_video.replace(out_path)
                    saved_path = out_path
                    print(
                        f"[Test] map={map_name} success on attempt={attempt}; "
                        f"video={saved_path}"
                    )
                else:
                    print(
                        f"[Test] map={map_name} success on attempt={attempt}, "
                        "but video was not recorded (writer unavailable)."
                    )
                break

            if tmp_video.exists():
                tmp_video.unlink()

        if not map_success:
            print(f"[Test] map={map_name} failed within {args.test_episodes_per_map} attempts.")
        elif saved_path is None:
            print(f"[Test] map={map_name} succeeded but no video file was produced.")
            success_count += 1
        else:
            success_count += 1
            video_count += 1

        env.close()

    pygame.quit()
    print(
        f"[Test] Summary: success_maps={success_count}/{len(fixed_maps)} "
        f"saved_videos={video_count}/{len(fixed_maps)} dir={video_dir}"
    )


def draw_team_hud(
    screen: Any,
    font: Any,
    global_steps: int,
    total_steps: int,
    ep_score: float,
    comm_mode: str,
    map_level: str,
    map_seed: int,
    obstacle_count: int,
) -> None:
    import pygame

    total_steps_text = "inf" if total_steps <= 0 else str(total_steps)

    lines = [
        f"fixed-map MARL",
        f"step: {global_steps}/{total_steps_text}",
        f"episode_score: {ep_score:+.3f}",
        f"comm_mode: {comm_mode}",
        f"map_level: {map_level}",
        f"map_seed: {map_seed}",
        f"obstacles: {obstacle_count}",
    ]
    line_h = 22
    box_w = 320
    box_h = 12 + line_h * len(lines)
    overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 145))
    screen.blit(overlay, (8, 8))
    for i, txt in enumerate(lines):
        s = font.render(txt, True, (240, 240, 240))
        screen.blit(s, (14, 14 + i * line_h))


def draw_route_overlay(env: TransportParallelEnv, renderer: Any) -> None:
    """Draw planned A* route polyline on top of the scene."""
    import pygame

    pts = getattr(env, "_route_waypoints", [])
    if not pts:
        return

    route_pts = [(int(x), int(y)) for x, y in pts]
    if len(route_pts) >= 2:
        pygame.draw.lines(renderer.screen, (80, 235, 255), False, route_pts, 3)
    elif len(route_pts) == 1:
        ox, oy = int(env.world.obj.x), int(env.world.obj.y)
        gx, gy = route_pts[0]
        pygame.draw.line(renderer.screen, (80, 235, 255), (ox, oy), (gx, gy), 2)

    # Sparse dots for waypoint readability.
    for x, y in route_pts[:: max(1, len(route_pts) // 12)]:
        pygame.draw.circle(renderer.screen, (80, 235, 255), (x, y), 4)


def build_policy_obs(
    observations: dict[str, np.ndarray],
    agent_order: list[str],
    comm_mode: str,
    comm_state: np.ndarray,
    comm_scale: float,
) -> np.ndarray:
    base = np.stack([observations[a] for a in agent_order], axis=0)
    if comm_mode == "none":
        return base

    shared = np.tile((comm_state * comm_scale)[None, :], (len(agent_order), 1))
    return np.concatenate([base, shared.astype(np.float32)], axis=1)


def update_comm_state(actions: np.ndarray, comm_mode: str) -> np.ndarray:
    if comm_mode == "none":
        return np.zeros(0, dtype=np.float32)
    return np.mean(actions, axis=0).astype(np.float32)


def curriculum_level(episode_idx: int, max_scenes: int) -> RandomLevel:
    if max_scenes > 0:
        p = float(episode_idx) / float(max(1, max_scenes - 1))
    else:
        p = min(1.0, float(episode_idx) / 60.0)
    if p < 0.34:
        return RandomLevel.MILD
    if p < 0.67:
        return RandomLevel.MODERATE
    return RandomLevel.FULL


def set_env_random_level(env: TransportParallelEnv, level: RandomLevel) -> None:
    env._random_level = level
    env._generator.level = level


def reset_reachable(
    env: TransportParallelEnv,
    seed_start: int,
    max_retries: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict], int, int]:
    for k in range(max_retries):
        seed = seed_start + k
        obs, infos = env.reset(seed=seed)
        waypoints = getattr(env, "_route_waypoints", [])
        if len(waypoints) >= 2:
            return obs, infos, seed, seed + 1
    raise RuntimeError(f"Failed to sample reachable map after {max_retries} retries.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    fixed_cfg = build_fixed_map_config(args.cargo_preset, args.num_agents)
    fixed_four_maps = build_four_fixed_map_configs(args.cargo_preset, args.num_agents)
    fixed_four_idx = 0

    use_fixed_map = args.map_mode == "fixed"
    use_fixed_four = args.map_mode == "fixed_four"
    random_level_mode = args.map_mode == "random_level"

    active_cfg: SceneConfig | None
    if use_fixed_four:
        active_cfg = copy.deepcopy(fixed_four_maps[fixed_four_idx][1])
    elif use_fixed_map:
        active_cfg = fixed_cfg
    else:
        active_cfg = None

    if use_fixed_map or use_fixed_four:
        init_level = RandomLevel.FIXED
    elif random_level_mode:
        init_level = RandomLevel(args.random_level)
    else:
        init_level = RandomLevel.MILD

    env = build_transport_env(
        args=args,
        config=active_cfg,
        random_level=init_level,
    )

    agent_order = list(env.possible_agents)
    base_obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).shape[0]
    comm_dim = action_dim if args.comm_mode == "broadcast_action" else 0
    policy_obs_dim = int(base_obs_dim + comm_dim)

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
    trainer = IPPOTrainer(obs_dim=policy_obs_dim, action_dim=action_dim, config=cfg, device=args.device)

    next_reset_seed = int(args.seed)
    current_map_level = "fixed"
    if use_fixed_four:
        current_map_level = fixed_four_maps[fixed_four_idx][0]
    current_map_seed = int(args.seed)
    current_obstacle_count = 0
    if not use_fixed_map and not use_fixed_four:
        if random_level_mode:
            lvl = RandomLevel(args.random_level)
        else:
            lvl = curriculum_level(total_episodes := 0, args.max_scenes)
        set_env_random_level(env, lvl)
        current_map_level = lvl.value

    if use_fixed_four:
        env._base_config = copy.deepcopy(fixed_four_maps[fixed_four_idx][1])

    observations, _infos, current_map_seed, next_reset_seed = reset_reachable(
        env=env,
        seed_start=next_reset_seed,
        max_retries=max(1, int(args.reachability_max_retries)),
    )
    if use_fixed_four:
        current_map_seed = fixed_four_idx
    current_obstacle_count = len(getattr(env.world, "obstacles", []))
    comm_state = np.zeros(comm_dim, dtype=np.float32)

    renderer: Any = None
    clock: Any = None
    hud_font: Any = None
    render_stride = max(1, int(args.render_steps_per_frame))
    if args.render_train:
        import pygame
        from src.sim.renderer import Renderer

        pygame.init()
        screen = pygame.display.set_mode((env.world.width, env.world.height))
        pygame.display.set_caption("MARL Fixed Map Training")
        renderer = Renderer(screen, env.world)
        clock = pygame.time.Clock()
        hud_font = pygame.font.Font(None, 24)

    global_steps = 0
    update_idx = 0
    ep_return = 0.0
    recent_returns: list[float] = []
    recent_lens: list[int] = []
    ep_len = 0
    total_episodes = 0

    step_limited = args.total_steps > 0

    while True:
        if step_limited and global_steps >= args.total_steps:
            break
        if args.max_scenes > 0 and total_episodes >= args.max_scenes:
            break
        for _ in range(cfg.rollout_steps):
            if not observations:
                observations, _ = env.reset()
                comm_state = np.zeros(comm_dim, dtype=np.float32)

            policy_obs = build_policy_obs(
                observations=observations,
                agent_order=agent_order,
                comm_mode=args.comm_mode,
                comm_state=comm_state,
                comm_scale=float(args.comm_scale),
            )
            actions, logprobs, values = trainer.act(policy_obs)
            action_dict = {agent_order[i]: actions[i] for i in range(len(agent_order))}

            next_obs, rewards, terms, truncs, _infos = env.step(action_dict)
            reward_batch = np.asarray([rewards[a] for a in agent_order], dtype=np.float32)
            done_batch = np.asarray([float(terms[a] or truncs[a]) for a in agent_order], dtype=np.float32)

            trainer.buffer.add(
                obs=policy_obs,
                actions=actions,
                logprobs=logprobs,
                values=values,
                rewards=reward_batch,
                dones=done_batch,
            )

            shared_r = float(np.mean(reward_batch))
            ep_return += shared_r
            ep_len += 1
            global_steps += 1

            comm_state = update_comm_state(actions, args.comm_mode)

            done = bool(all(terms[a] or truncs[a] for a in agent_order))
            if done:
                recent_returns.append(ep_return)
                recent_lens.append(ep_len)
                if len(recent_returns) > 100:
                    recent_returns.pop(0)
                if len(recent_lens) > 100:
                    recent_lens.pop(0)
                ep_return = 0.0
                ep_len = 0
                total_episodes += 1

                if use_fixed_four:
                    fixed_four_idx = (fixed_four_idx + 1) % len(fixed_four_maps)
                    current_map_level = fixed_four_maps[fixed_four_idx][0]
                    current_map_seed = fixed_four_idx
                    env._base_config = copy.deepcopy(fixed_four_maps[fixed_four_idx][1])
                elif not use_fixed_map:
                    if random_level_mode:
                        lvl = RandomLevel(args.random_level)
                    else:
                        lvl = curriculum_level(total_episodes, args.max_scenes)
                    set_env_random_level(env, lvl)
                    current_map_level = lvl.value

                observations, _infos, current_map_seed, next_reset_seed = reset_reachable(
                    env=env,
                    seed_start=next_reset_seed,
                    max_retries=max(1, int(args.reachability_max_retries)),
                )
                if use_fixed_four:
                    current_map_seed = fixed_four_idx
                current_obstacle_count = len(getattr(env.world, "obstacles", []))
                comm_state = np.zeros(comm_dim, dtype=np.float32)
                if args.max_scenes > 0 and total_episodes >= args.max_scenes:
                    break
            else:
                observations = next_obs

            if args.render_train and renderer is not None and clock is not None and hud_font is not None:
                if global_steps % render_stride == 0:
                    import pygame

                    keep_running = True
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            keep_running = False
                            break
                    if keep_running:
                        renderer.draw()
                        draw_route_overlay(env, renderer)
                        draw_team_hud(
                            screen=renderer.screen,
                            font=hud_font,
                            global_steps=global_steps,
                            total_steps=args.total_steps,
                            ep_score=ep_return,
                            comm_mode=args.comm_mode,
                            map_level=current_map_level,
                            map_seed=current_map_seed,
                            obstacle_count=current_obstacle_count,
                        )
                        pygame.display.flip()
                        clock.tick(max(1, int(args.render_fps)))
                    else:
                        args.render_train = False

            if step_limited and global_steps >= args.total_steps:
                break

        if args.max_scenes > 0 and total_episodes >= args.max_scenes:
            break

        if observations:
            last_policy_obs = build_policy_obs(
                observations=observations,
                agent_order=agent_order,
                comm_mode=args.comm_mode,
                comm_state=comm_state,
                comm_scale=float(args.comm_scale),
            )
            last_values = trainer.value(last_policy_obs)
        else:
            last_values = np.zeros(len(agent_order), dtype=np.float32)

        _stats = trainer.update(last_values=last_values)
        update_idx += 1

        if update_idx % args.log_interval == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_len = float(np.mean(recent_lens)) if recent_lens else 0.0
            avg_ret_step = avg_return / max(1.0, avg_len)
            print(
                f"update={update_idx} steps={global_steps} episodes={total_episodes} "
                f"avg_return_100={avg_return:.3f} avg_len_100={avg_len:.1f} "
                f"avg_return_per_step_100={avg_ret_step:.4f} comm_mode={args.comm_mode} "
                f"map_mode={args.map_mode} map_level={current_map_level} "
                f"map_seed={current_map_seed} obstacles={current_obstacle_count}"
            )

    if args.max_scenes > 0 and total_episodes >= args.max_scenes:
        print(f"Reached scene limit: episodes={total_episodes}/{args.max_scenes}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_four_payload = [
        {
            "name": name,
            "width": cfg.width,
            "height": cfg.height,
            "obstacles": cfg.obstacles,
            "cargo": [cfg.cargo_x, cfg.cargo_y, cfg.cargo_theta],
            "goal": [cfg.goal_x, cfg.goal_y],
        }
        for name, cfg in fixed_four_maps
    ]
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "config": cfg.__dict__,
            "obs_dim": policy_obs_dim,
            "base_obs_dim": base_obs_dim,
            "action_dim": action_dim,
            "comm_mode": args.comm_mode,
            "comm_dim": comm_dim,
            "map_mode": args.map_mode,
            "agent_order": agent_order,
            "fixed_map": {
                "width": fixed_cfg.width,
                "height": fixed_cfg.height,
                "obstacles": fixed_cfg.obstacles,
                "cargo": [fixed_cfg.cargo_x, fixed_cfg.cargo_y, fixed_cfg.cargo_theta],
                "goal": [fixed_cfg.goal_x, fixed_cfg.goal_y],
            }
            if use_fixed_map
            else None,
            "fixed_maps": fixed_four_payload if use_fixed_four else None,
        },
        save_path,
    )
    print(f"Training complete. Checkpoint saved to: {save_path}")

    if use_fixed_four and not args.skip_test:
        test_fixed_four_maps(
            args=args,
            trainer=trainer,
            fixed_maps=fixed_four_maps,
            comm_dim=comm_dim,
        )

    if args.render_train:
        import pygame

        pygame.quit()


if __name__ == "__main__":
    main()
