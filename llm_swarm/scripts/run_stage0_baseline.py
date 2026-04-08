"""Stage-0 baselines: object-level PD and obstacle-aware PD heuristic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

# Support running as a script from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.force_allocator import allocate_wrench_to_robots
from src.planning.path_planner import plan_path
from src.sim.scene_config import RandomLevel, SceneConfig, SceneGenerator
from src.sim.world import World


def _wrap_angle(rad: float) -> float:
    return float((rad + np.pi) % (2 * np.pi) - np.pi)


def _point_aabb_distance(x: float, y: float, rect: tuple[int, int, int, int]) -> float:
    ox, oy, ow, oh = rect
    cx = float(np.clip(x, ox, ox + ow))
    cy = float(np.clip(y, oy, oy + oh))
    return float(np.hypot(x - cx, y - cy))


def _clearance_to_obstacles(
    x: float,
    y: float,
    width: int,
    height: int,
    obstacles: list[tuple[int, int, int, int]],
) -> float:
    wall_clear = min(x, y, width - x, height - y)
    obs_clear = float("inf")
    for obs in obstacles:
        obs_clear = min(obs_clear, _point_aabb_distance(x, y, obs))
    return float(min(wall_clear, obs_clear))


@dataclass
class Stage0Config:
    force_max: float = 500.0
    kp_pos: float = 5.5
    kd_v: float = 18.0
    ktheta: float = 260.0
    komega: float = 28.0
    waypoint_tolerance: float = 55.0
    cell_size: int = 40
    inflate_margin: float = 70.0
    avoid_torque_gain: float = 1.8
    front_slow_margin: float = 120.0


class Stage0Controller:
    """Object-level baseline controller with optional obstacle heuristics."""

    def __init__(self, world: World, cfg: Stage0Config, mode: str) -> None:
        self.world = world
        self.cfg = cfg
        self.mode = mode
        self.route: list[tuple[float, float]] = []
        self.route_idx = 0
        self._build_route()

    def _build_route(self) -> None:
        obj = self.world.obj
        self.route = plan_path(
            width=self.world.width,
            height=self.world.height,
            obstacles=self.world.obstacles,
            start_xy=(obj.x, obj.y),
            goal_xy=(obj.goal_x, obj.goal_y),
            cell_size=self.cfg.cell_size,
            inflate_margin=self.cfg.inflate_margin,
        )
        if len(self.route) == 0:
            self.route = [(obj.goal_x, obj.goal_y)]
        self.route_idx = 0

    def _current_waypoint(self) -> tuple[float, float]:
        if self.route_idx >= len(self.route):
            obj = self.world.obj
            return (obj.goal_x, obj.goal_y)
        return self.route[self.route_idx]

    def _advance_waypoint(self) -> None:
        obj = self.world.obj
        while self.route_idx < len(self.route):
            wx, wy = self.route[self.route_idx]
            if float(np.hypot(obj.x - wx, obj.y - wy)) > self.cfg.waypoint_tolerance:
                break
            self.route_idx += 1

    def _route_direction(self) -> np.ndarray:
        obj = self.world.obj
        wx, wy = self._current_waypoint()
        d = np.array([wx - obj.x, wy - obj.y], dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return d / n

    def _base_wrench(self) -> tuple[float, float, float]:
        obj = self.world.obj
        wp = np.array(self._current_waypoint(), dtype=np.float32)
        pos = np.array([obj.x, obj.y], dtype=np.float32)
        err = wp - pos
        fx = self.cfg.kp_pos * float(err[0]) - self.cfg.kd_v * float(obj.vx)
        fy = self.cfg.kp_pos * float(err[1]) - self.cfg.kd_v * float(obj.vy)

        total_force_max = self.cfg.force_max * max(1, len(self.world.robots)) * 0.9
        f_norm = float(np.hypot(fx, fy))
        if f_norm > total_force_max and f_norm > 1e-8:
            scale = total_force_max / f_norm
            fx *= scale
            fy *= scale

        route_dir = self._route_direction()
        if float(np.linalg.norm(route_dir)) > 1e-6:
            theta_ref = float(np.arctan2(route_dir[1], route_dir[0]))
        else:
            theta_ref = float(obj.theta)
        theta_err = _wrap_angle(theta_ref - obj.theta)
        tau = self.cfg.ktheta * theta_err - self.cfg.komega * obj.omega

        tau_max = self.cfg.force_max * 140.0
        tau = float(np.clip(tau, -tau_max, tau_max))

        return fx, fy, float(tau)

    def _apply_avoid_heuristic(self, fx: float, fy: float, tau: float) -> tuple[float, float, float]:
        obj = self.world.obj
        route_dir = self._route_direction()
        if float(np.linalg.norm(route_dir)) < 1e-6:
            return fx, fy, tau

        normal = np.array([-route_dir[1], route_dir[0]], dtype=np.float32)
        probe_lat = 90.0
        probe_front = 120.0

        left_p = np.array([obj.x, obj.y], dtype=np.float32) + normal * probe_lat
        right_p = np.array([obj.x, obj.y], dtype=np.float32) - normal * probe_lat
        front_p = np.array([obj.x, obj.y], dtype=np.float32) + route_dir * probe_front

        left_clear = _clearance_to_obstacles(
            float(left_p[0]),
            float(left_p[1]),
            self.world.width,
            self.world.height,
            self.world.obstacles,
        )
        right_clear = _clearance_to_obstacles(
            float(right_p[0]),
            float(right_p[1]),
            self.world.width,
            self.world.height,
            self.world.obstacles,
        )
        front_clear = _clearance_to_obstacles(
            float(front_p[0]),
            float(front_p[1]),
            self.world.width,
            self.world.height,
            self.world.obstacles,
        )

        # If left is tighter than right, apply positive torque to steer right.
        clearance_delta = right_clear - left_clear
        tau += self.cfg.avoid_torque_gain * clearance_delta

        # Slow down if frontal space is narrow.
        slow_scale = float(np.clip(front_clear / self.cfg.front_slow_margin, 0.25, 1.0))
        fx *= slow_scale
        fy *= slow_scale
        return fx, fy, tau

    def step(self) -> None:
        self._advance_waypoint()
        fx, fy, tau = self._base_wrench()
        if self.mode == "pd_avoid":
            fx, fy, tau = self._apply_avoid_heuristic(fx, fy, tau)

        attached_ids = [i for i, r in enumerate(self.world.robots) if r.attached]
        rel_points = np.asarray(
            [self.world.obj.attach_points_local[self.world.robots[i]._attach_idx] for i in attached_ids],
            dtype=np.float32,
        )

        alloc = allocate_wrench_to_robots(
            rel_points=rel_points,
            desired_fx=fx,
            desired_fy=fy,
            desired_tau=tau,
            force_max=self.cfg.force_max,
            ridge_lambda=5e-3,
        )
        attached_slot = {rid: slot for slot, rid in enumerate(attached_ids)}

        for rid, r in enumerate(self.world.robots):
            if not r.attached:
                r.cmd_fx = 0.0
                r.cmd_fy = 0.0
                continue
            slot = attached_slot[rid]
            r.cmd_fx = float(alloc[slot, 0])
            r.cmd_fy = float(alloc[slot, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-0 non-learning baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["fixed", "mild", "moderate", "full"],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--cargo-preset", type=str, default="L", choices=["L", "T", "U"])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--force-max", type=float, default=500.0)
    parser.add_argument("--controller", type=str, default="pd_avoid", choices=["pd", "pd_avoid"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = RandomLevel(args.level)
    generator = SceneGenerator(level=level)

    success_count = 0
    invalid_count = 0
    final_dists: list[float] = []
    steps_used: list[int] = []

    renderer = None
    clock = None
    screen = None
    if args.render:
        import pygame

        pygame.init()

    for ep in range(args.episodes):
        cfg: SceneConfig = generator.generate(seed=args.seed + ep)
        cfg.cargo_preset = args.cargo_preset
        cfg.num_robots = args.num_agents
        if args.no_obstacles:
            cfg.obstacles = []

        world = World(config=cfg)
        world.external_control = True
        world.reset(config=cfg)

        controller_cfg = Stage0Config(force_max=args.force_max)
        controller = Stage0Controller(world, controller_cfg, mode=args.controller)

        if args.render:
            import pygame

            if screen is None:
                from src.sim.renderer import Renderer

                screen = pygame.display.set_mode((world.width, world.height))
                pygame.display.set_caption("Stage-0 Baseline")
                renderer = Renderer(screen, world)
                clock = pygame.time.Clock()

        done = False
        step = 0
        while not done and step < args.max_steps:
            if args.render:
                assert renderer is not None and clock is not None
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            controller.step()
            world.step(args.dt)
            step += 1

            if args.render:
                renderer.draw()
                pygame.display.flip()
                clock.tick(args.fps)

            done = bool(world.success or world.invalid_state)

        dist = float(np.hypot(world.obj.x - world.obj.goal_x, world.obj.y - world.obj.goal_y))
        success_count += int(world.success)
        invalid_count += int(world.invalid_state)
        final_dists.append(dist)
        steps_used.append(step)

        print(
            f"episode={ep + 1}/{args.episodes} "
            f"success={int(world.success)} invalid={int(world.invalid_state)} "
            f"steps={step} final_dist={dist:.2f}"
        )

    if args.render:
        import pygame

        pygame.quit()

    episodes = max(1, args.episodes)
    print("\n===== Stage-0 Summary =====")
    print(f"controller:    {args.controller}")
    print(f"success_rate:  {success_count / episodes:.3f}")
    print(f"invalid_rate:  {invalid_count / episodes:.3f}")
    print(f"avg_steps:     {float(np.mean(steps_used)):.1f}")
    print(f"avg_final_dist:{float(np.mean(final_dists)):.2f}")


if __name__ == "__main__":
    main()
