"""
Simulation world: manages robots, object, obstacles, and physics updates

Physics model: rigid attachment + per-robot collision blocking.
- Robots are rigidly attached to the object's preset attach points.
- Each frame, we predict where each robot would land after the object moves.
- If a robot's predicted position collides with a wall or world boundary,
  that robot is marked as BLOCKED and its force contribution is zeroed out.
- Only unblocked robots contribute force, so the object steers naturally
  around obstacles driven by whichever robots still have room to push.
"""

import numpy as np

from src.sim.controller import SimpleController
from src.sim.robot import Robot
from src.sim.transport_object import TransportObject

# Obstacle list: (x, y, width, height)
OBSTACLES = [
    (150, 100, 30, 200),  # Left vertical wall
    (300, 300, 200, 30),  # Center horizontal wall
    (600, 150, 30, 180),  # Right upper wall
    (500, 500, 150, 30),  # Right lower wall
    (200, 550, 30, 150),  # Bottom-left vertical wall
]

ROBOT_COLORS = [
    (60, 160, 240),  # Blue
    (60, 200, 120),  # Green
    (240, 160, 60),  # Orange
    (200, 80, 200),  # Purple
    (240, 80, 80),  # Red
    (80, 200, 200),  # Cyan
]


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.t = 0.0
        self.reset()

    def reset(self):
        self.t = 0.0

        # Transport object (placed on the left side)
        self.obj = TransportObject(x=250, y=420)
        self.obj.goal_x = 750
        self.obj.goal_y = 380

        # 4 robots spawned around the object
        spawn_positions = [
            (170, 400, 0.0),
            (250, 330, np.pi / 2),
            (330, 420, np.pi),
            (250, 510, -np.pi / 2),
        ]
        self.robots = []
        for i, (x, y, th) in enumerate(spawn_positions):
            r = Robot(i, x, y, th, color=ROBOT_COLORS[i])
            self.robots.append(r)

        self.obstacles = OBSTACLES
        self.controller = SimpleController(self.robots, self.obj, self.obstacles)
        self._setup_attachments()
        self.success = False

    def _setup_attachments(self):
        """Rigidly attach robots to preset attach points on the object."""
        attach_map = {0: 0, 1: 2, 2: 1, 3: 3}
        for rid, aidx in attach_map.items():
            r = self.robots[rid]
            r.attached = True
            r._attach_idx = aidx
            r.blocked = False
            ap = self.obj.get_attach_point_world(aidx)
            r.x, r.y = ap[0], ap[1]
            r._theta_offset = r.theta - self.obj.theta

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _point_in_obstacle(self, x: float, y: float, radius: float) -> bool:
        """Return True if a circle (x, y, radius) overlaps any obstacle or wall."""
        # World boundary check
        if (
            x - radius < 0
            or x + radius > self.width
            or y - radius < 0
            or y + radius > self.height
        ):
            return True
        # Obstacle AABB vs circle
        for ox, oy, ow, oh in self.obstacles:
            cx = np.clip(x, ox, ox + ow)
            cy = np.clip(y, oy, oy + oh)
            if np.hypot(x - cx, y - cy) < radius + 2:
                return True
        return False

    def _predict_robot_pos(self, r, delta_x: float, delta_y: float, delta_theta: float):
        """
        Predict where robot r would end up if the object moved by
        (delta_x, delta_y, delta_theta). Returns (px, py).
        """
        local = self.obj.attach_points_local[r._attach_idx]
        new_theta = self.obj.theta + delta_theta
        c, s = np.cos(new_theta), np.sin(new_theta)
        R = np.array([[c, -s], [s, c]])
        ap = np.array([self.obj.x + delta_x, self.obj.y + delta_y]) + R @ local
        return ap[0], ap[1]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, dt: float):
        if self.success:
            return

        self.t += dt

        # 1. Controller outputs cmd_fx / cmd_fy per robot
        self.controller.update(dt)

        # 2. Compute tentative object velocity from ALL robots
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

        # 3. Check each robot's predicted position - mark blocked if colliding
        for r in self.robots:
            if not r.attached:
                r.blocked = False
                continue
            px, py = self._predict_robot_pos(r, dx, dy, dtheta)
            r.blocked = self._point_in_obstacle(px, py, Robot.RADIUS)

        # 4. Re-accumulate force from UNBLOCKED robots only
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

        # 5. Integrate object physics with force from unblocked robots only
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

        # Clamp object inside world bounds
        self.obj.x = np.clip(self.obj.x, 60, self.width - 60)
        self.obj.y = np.clip(self.obj.y, 60, self.height - 60)

        # 6. Snap all attached robots to updated attach points
        for r in self.robots:
            if r.attached:
                ap = self.obj.get_attach_point_world(r._attach_idx)
                r.x, r.y = ap[0], ap[1]
                r.theta = self.obj.theta + r._theta_offset

        # 7. Success check
        if self.obj.reached_goal():
            self.success = True
