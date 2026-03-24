"""
Irregular transport object model (L-shape)
Supports multiple robot attachment points
"""

import numpy as np


class TransportObject:
    """
    L-shaped object composed of two rectangles:
      ##########
      ##
      ##
    Center of mass is offset from geometric center,
    simulating a real irregular object.
    """

    def __init__(self, x: float, y: float):
        self.x = x  # Center of mass position
        self.y = y
        self.theta = 0.0  # Heading angle

        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0

        self.mass = 5.0  # Mass (simulation units)
        self.inertia = 8000.0  # Moment of inertia

        # Linear/angular damping (simulates ground friction)
        self.linear_damping = 0.85
        self.angular_damping = 0.80

        # L-shape: two rectangles defined in local frame
        # (local_x, local_y, width, height)
        self.parts = [
            (-50, -30, 100, 40),  # Horizontal bar
            (-50, 10, 40, 60),  # Vertical bar
        ]

        # Preset attach points in local frame for robot attachment
        self.attach_points_local = np.array(
            [
                [-60, 0],  # Left side
                [60, -10],  # Right side
                [0, -40],  # Top
                [-30, 70],  # Bottom
                [40, 40],  # Bottom-right
                [-60, 50],  # Bottom-left
            ],
            dtype=float,
        )

        self.goal_x = None
        self.goal_y = None

    def apply_force(self, fx: float, fy: float, px: float, py: float, dt: float):
        """
        Apply force (fx, fy) at world position (px, py).
        Updates linear and angular velocity.
        """
        # Linear acceleration
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt

        # Torque = r x F
        rx = px - self.x
        ry = py - self.y
        torque = rx * fy - ry * fx
        self.omega += (torque / self.inertia) * dt

    def step(self, dt: float):
        """Integrate position and apply damping"""
        self.vx *= self.linear_damping
        self.vy *= self.linear_damping
        self.omega *= self.angular_damping

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.theta += self.omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_attach_point_world(self, idx: int) -> np.ndarray:
        """Transform local attach point idx to world coordinates"""
        lp = self.attach_points_local[idx]
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        return np.array([self.x, self.y]) + R @ lp

    def get_parts_world(self):
        """Return world-frame corner vertices of each part (for rendering and collision)"""
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        parts_world = []
        for lx, ly, w, h in self.parts:
            corners_local = np.array(
                [
                    [lx, ly],
                    [lx + w, ly],
                    [lx + w, ly + h],
                    [lx, ly + h],
                ]
            )
            corners_world = np.array([self.x, self.y]) + (R @ corners_local.T).T
            parts_world.append(corners_world)
        return parts_world

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def reached_goal(self, tol=30.0) -> bool:
        if self.goal_x is None:
            return False
        return np.hypot(self.x - self.goal_x, self.y - self.goal_y) < tol
