"""Irregular transport object model with configurable geometry.

Supports multiple cargo shapes (L, T, U) and robot attachment points.
Shape data is injected via constructor arguments so that the class itself
contains no hardcoded geometry — all defaults come from ``SceneConfig``.
"""

from __future__ import annotations

import numpy as np

# Default L-shape geometry (used when no explicit parts are provided).
_DEFAULT_PARTS = [(-50, -30, 100, 40), (-50, 10, 40, 60)]
_DEFAULT_ATTACH_POINTS = [
    [-60, 0],
    [60, -10],
    [0, -40],
    [-30, 70],
    [40, 40],
    [-60, 50],
]


class TransportObject:
    """Rigid cargo body with configurable shape and attachment points.

    Args:
        x: Initial center-of-mass x position.
        y: Initial center-of-mass y position.
        parts: List of (local_x, local_y, width, height) rectangles
            defining the cargo shape in the local frame.
        attach_points: List of [x, y] attachment point coordinates
            in the local frame.
        mass: Cargo mass in simulation units.
        inertia: Moment of inertia.
        linear_damping: Velocity damping factor per frame.
        angular_damping: Angular velocity damping factor per frame.
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float = 0.0,
        parts: list[tuple[int, int, int, int]] | None = None,
        attach_points: list[list[float]] | None = None,
        mass: float = 5.0,
        inertia: float = 8000.0,
        linear_damping: float = 0.85,
        angular_damping: float = 0.80,
    ) -> None:
        self.x = x
        self.y = y
        self.theta: float = float(theta)

        self.vx: float = 0.0
        self.vy: float = 0.0
        self.omega: float = 0.0

        self.mass = mass
        self.inertia = inertia
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping

        self.parts: list[tuple[int, int, int, int]] = (
            list(parts) if parts is not None else list(_DEFAULT_PARTS)
        )
        self.attach_points_local: np.ndarray = np.array(
            attach_points if attach_points is not None else _DEFAULT_ATTACH_POINTS,
            dtype=float,
        )

        self.goal_x: float | None = None
        self.goal_y: float | None = None

    def apply_force(
        self,
        fx: float,
        fy: float,
        px: float,
        py: float,
        dt: float,
    ) -> None:
        """Apply force (fx, fy) at world position (px, py)."""
        # Linear acceleration.
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt

        # Torque = r × F.
        rx = px - self.x
        ry = py - self.y
        torque = rx * fy - ry * fx
        self.omega += (torque / self.inertia) * dt

    def step(self, dt: float) -> None:
        """Integrate position and apply damping."""
        self.vx *= self.linear_damping
        self.vy *= self.linear_damping
        self.omega *= self.angular_damping

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.theta += self.omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_attach_point_world(self, idx: int) -> np.ndarray:
        """Transform local attach point *idx* to world coordinates."""
        lp = self.attach_points_local[idx]
        c, s = np.cos(self.theta), np.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        return np.array([self.x, self.y]) + rot @ lp

    def get_parts_world(self) -> list[np.ndarray]:
        """Return world-frame corner vertices of each rectangular part."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        parts_world: list[np.ndarray] = []
        for lx, ly, w, h in self.parts:
            corners_local = np.array(
                [
                    [lx, ly],
                    [lx + w, ly],
                    [lx + w, ly + h],
                    [lx, ly + h],
                ]
            )
            corners_world = np.array([self.x, self.y]) + (rot @ corners_local.T).T
            parts_world.append(corners_world)
        return parts_world

    @property
    def pos(self) -> np.ndarray:
        """Center-of-mass position as a numpy array."""
        return np.array([self.x, self.y])

    def reached_goal(self, tol: float = 30.0) -> bool:
        """Check whether the cargo is within *tol* pixels of the goal."""
        if self.goal_x is None:
            return False
        return float(np.hypot(self.x - self.goal_x, self.y - self.goal_y)) < tol
