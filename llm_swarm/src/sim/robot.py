"""
Wheeled robot model (differential drive)
"""

import numpy as np


class Robot:
    RADIUS = 18  # Robot radius (pixels)
    MAX_SPEED = 120  # Max linear speed px/s
    MAX_OMEGA = 2.5  # Max angular speed rad/s

    def __init__(
        self,
        robot_id: int,
        x: float,
        y: float,
        theta: float = 0.0,
        color=(60, 160, 240),
    ):
        self.id = robot_id
        self.x = x
        self.y = y
        self.theta = theta  # Heading angle (radians)
        self.color = color

        self.vx = 0.0  # Current velocity (world frame)
        self.vy = 0.0
        self.omega = 0.0  # Angular velocity

        # Velocity commands (kept for compatibility)
        self.cmd_v = 0.0
        self.cmd_omega = 0.0

        # Force commands (used by rigid-constraint physics model)
        self.cmd_fx = 0.0
        self.cmd_fy = 0.0

        # Attachment state
        self.attached = False
        self.attach_offset = np.zeros(2)  # Attach point offset relative to robot

        # Sensing radius (used by planning algorithms)
        self.sense_radius = 150

    def set_velocity(self, v: float, omega: float):
        """Set velocity command (linear speed, angular speed)"""
        self.cmd_v = np.clip(v, -self.MAX_SPEED, self.MAX_SPEED)
        self.cmd_omega = np.clip(omega, -self.MAX_OMEGA, self.MAX_OMEGA)

    def step(self, dt: float, obstacles: list, world_w: int, world_h: int):
        """Update robot state"""
        # Differential drive kinematics
        self.theta += self.cmd_omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        dx = self.cmd_v * np.cos(self.theta) * dt
        dy = self.cmd_v * np.sin(self.theta) * dt

        new_x = self.x + dx
        new_y = self.y + dy

        # World boundary constraint
        new_x = np.clip(new_x, self.RADIUS, world_w - self.RADIUS)
        new_y = np.clip(new_y, self.RADIUS, world_h - self.RADIUS)

        # Obstacle collision check (circle vs rectangle)
        if not self._collides_with_obstacles(new_x, new_y, obstacles):
            self.x = new_x
            self.y = new_y

        self.vx = self.cmd_v * np.cos(self.theta)
        self.vy = self.cmd_v * np.sin(self.theta)

    def _collides_with_obstacles(self, x, y, obstacles):
        for obs in obstacles:
            # obs = (ox, oy, ow, oh)
            ox, oy, ow, oh = obs
            # Nearest point on rectangle
            cx = np.clip(x, ox, ox + ow)
            cy = np.clip(y, oy, oy + oh)
            dist = np.hypot(x - cx, y - cy)
            if dist < self.RADIUS + 2:
                return True
        return False

    @property
    def pos(self):
        return np.array([self.x, self.y])
