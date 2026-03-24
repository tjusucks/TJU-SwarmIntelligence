"""Simulation world: manages robots, object, obstacles, and physics updates.

Physics model: rigid attachment + per-robot collision blocking.
- Robots are rigidly attached to the object's preset attach points.
- Each frame, we predict where each robot would land after the object moves.
- If a robot's predicted position collides with a wall or world boundary,
  that robot is marked as BLOCKED and its force contribution is zeroed out.
- Only unblocked robots contribute force, so the object steers naturally
  around obstacles driven by whichever robots still have room to push.
"""

from __future__ import annotations

import numpy as np

from src.sim.robot import Robot
from src.sim.scene_config import CARGO_PRESETS, SceneConfig
from src.sim.transport_object import TransportObject

ROBOT_COLORS = [
    (60, 160, 240),  # Blue.
    (60, 200, 120),  # Green.
    (240, 160, 60),  # Orange.
    (200, 80, 200),  # Purple.
    (240, 80, 80),  # Red.
    (80, 200, 200),  # Cyan.
]


class World:
    """Simulation world driven by a ``SceneConfig``.

    Args:
        config: Scene configuration.  If ``None``, uses default baseline.
    """

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.t: float = 0.0

        # Flags.
        self.external_control: bool = False
        self.success: bool = False

        # Populated by reset().
        self.obj: TransportObject
        self.robots: list[Robot]
        self.obstacles: list[tuple[int, int, int, int]]

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, config: SceneConfig | None = None) -> None:
        """Reset the world, optionally with a new config."""
        if config is not None:
            self.config = config
            self.width = config.width
            self.height = config.height

        self.t = 0.0
        self.success = False
        self.obstacles = list(self.config.obstacles)

        self._create_cargo()
        self._create_robots()
        self._setup_attachments()

        # Controller (only used in demo mode, not MARL).
        if not self.external_control:
            from src.sim.controller import SimpleController

            self.controller = SimpleController(
                self.robots,
                self.obj,
                self.obstacles,
            )

    def _create_cargo(self) -> None:
        """Instantiate the transport object from config."""
        preset_name = self.config.cargo_preset
        preset = CARGO_PRESETS[preset_name]

        self.obj = TransportObject(
            x=self.config.cargo_x,
            y=self.config.cargo_y,
            parts=preset["parts"],
            attach_points=preset["attach_points"],
            mass=self.config.cargo_mass or preset["mass"],
            inertia=self.config.cargo_inertia or preset["inertia"],
            linear_damping=self.config.linear_damping,
            angular_damping=self.config.angular_damping,
        )
        self.obj.goal_x = self.config.goal_x
        self.obj.goal_y = self.config.goal_y

    def _create_robots(self) -> None:
        """Spawn robots from config or auto-generate around cargo."""
        n = self.config.num_robots
        spawns = self.config.robot_spawns

        if spawns is None:
            spawns = self._auto_spawn_positions(n)

        self.robots = []
        for i, (x, y, th) in enumerate(spawns[:n]):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            self.robots.append(Robot(i, x, y, th, color=color))

    def _auto_spawn_positions(
        self,
        n: int,
    ) -> list[tuple[float, float, float]]:
        """Generate spawn positions evenly around the cargo center."""
        cx, cy = self.config.cargo_x, self.config.cargo_y
        radius = 80.0
        positions = []
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            heading = angle + np.pi  # Face toward cargo.
            positions.append((float(x), float(y), float(heading)))
        return positions

    def _setup_attachments(self) -> None:
        """Rigidly attach robots to preset attach points on the object."""
        n_attach = len(self.obj.attach_points_local)
        attach_map = self.config.attach_map

        if attach_map is None:
            # Default: assign robot i to attach point i (mod available).
            attach_map = {i: i % n_attach for i in range(len(self.robots))}

        for rid, aidx in attach_map.items():
            if rid >= len(self.robots):
                continue
            r = self.robots[rid]
            r.attached = True
            r._attach_idx = aidx
            r.blocked = False
            ap = self.obj.get_attach_point_world(aidx)
            r.x, r.y = float(ap[0]), float(ap[1])
            r._theta_offset = r.theta - self.obj.theta

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _point_in_obstacle(self, x: float, y: float, radius: float) -> bool:
        """Return True if a circle (x, y, radius) overlaps any obstacle or wall."""
        # World boundary check.
        if (
            x - radius < 0
            or x + radius > self.width
            or y - radius < 0
            or y + radius > self.height
        ):
            return True
        # Obstacle AABB vs circle.
        for ox, oy, ow, oh in self.obstacles:
            cx = np.clip(x, ox, ox + ow)
            cy = np.clip(y, oy, oy + oh)
            if np.hypot(x - cx, y - cy) < radius + 2:
                return True
        return False

    def _predict_robot_pos(
        self,
        r: Robot,
        delta_x: float,
        delta_y: float,
        delta_theta: float,
    ) -> tuple[float, float]:
        """Predict where robot r would end up after object displacement."""
        local = self.obj.attach_points_local[r._attach_idx]
        new_theta = self.obj.theta + delta_theta
        c, s = np.cos(new_theta), np.sin(new_theta)
        rot = np.array([[c, -s], [s, c]])
        ap = np.array([self.obj.x + delta_x, self.obj.y + delta_y]) + rot @ local
        return float(ap[0]), float(ap[1])

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance simulation by one time step."""
        if self.success:
            return

        self.t += dt

        # 1. Controller outputs cmd_fx / cmd_fy per robot.
        if not self.external_control:
            self.controller.update(dt)

        # 2. Compute tentative object velocity from ALL robots.
        net_fx, net_fy, net_torque = 0.0, 0.0, 0.0
        for r in self.robots:
            if not r.attached:
                continue
            fx, fy = r.cmd_fx, r.cmd_fy
            net_fx += fx
            net_fy += fy
            ap = self.obj.get_attach_point_world(r._attach_idx)
            rx, ry = ap[0] - self.obj.x, ap[1] - self.obj.y
            net_torque += rx * fy - ry * fx

        tent_vx = (
            self.obj.vx + (net_fx / self.obj.mass) * dt
        ) * self.obj.linear_damping
        tent_vy = (
            self.obj.vy + (net_fy / self.obj.mass) * dt
        ) * self.obj.linear_damping
        tent_omega = (
            self.obj.omega + (net_torque / self.obj.inertia) * dt
        ) * self.obj.angular_damping

        dx = tent_vx * dt
        dy = tent_vy * dt
        dtheta = tent_omega * dt

        # 3. Check each robot's predicted position - mark blocked if colliding.
        for r in self.robots:
            if not r.attached:
                r.blocked = False
                continue
            px, py = self._predict_robot_pos(r, dx, dy, dtheta)
            r.blocked = self._point_in_obstacle(px, py, Robot.RADIUS)

        # 4. Re-accumulate force from UNBLOCKED robots only.
        net_fx, net_fy, net_torque = 0.0, 0.0, 0.0
        active = 0
        for r in self.robots:
            if not r.attached or r.blocked:
                continue
            active += 1
            fx, fy = r.cmd_fx, r.cmd_fy
            net_fx += fx
            net_fy += fy
            ap = self.obj.get_attach_point_world(r._attach_idx)
            rx, ry = ap[0] - self.obj.x, ap[1] - self.obj.y
            net_torque += rx * fy - ry * fx

        # 5. Integrate object physics with force from unblocked robots only.
        if active > 0:
            self.obj.vx += (net_fx / self.obj.mass) * dt
            self.obj.vy += (net_fy / self.obj.mass) * dt
            self.obj.omega += (net_torque / self.obj.inertia) * dt

        self.obj.vx *= self.obj.linear_damping
        self.obj.vy *= self.obj.linear_damping
        self.obj.omega *= self.obj.angular_damping

        self.obj.x += self.obj.vx * dt
        self.obj.y += self.obj.vy * dt
        self.obj.theta += self.obj.omega * dt
        self.obj.theta = (self.obj.theta + np.pi) % (2 * np.pi) - np.pi

        # Clamp object inside world bounds.
        self.obj.x = np.clip(self.obj.x, 60, self.width - 60)
        self.obj.y = np.clip(self.obj.y, 60, self.height - 60)

        # 6. Snap all attached robots to updated attach points.
        for r in self.robots:
            if r.attached:
                ap = self.obj.get_attach_point_world(r._attach_idx)
                r.x, r.y = float(ap[0]), float(ap[1])
                r.theta = self.obj.theta + r._theta_offset

        # 7. Success check.
        if self.obj.reached_goal():
            self.success = True
