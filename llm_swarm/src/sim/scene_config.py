"""Scene configuration and seeded generation.

Provides ``SceneConfig`` — the single source of truth for every mutable
scene parameter — and ``SceneGenerator`` which produces reproducible
configs from a seed with controllable randomisation levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Cargo shape presets
# ---------------------------------------------------------------------------


class CargoPreset(Enum):
    """Available cargo geometry presets."""

    L = "L"
    T = "T"
    U = "U"


CARGO_PRESETS: dict[str, dict] = {
    "L": {
        "parts": [(-50, -30, 100, 40), (-50, 10, 40, 60)],
        "attach_points": [
            [-60, 0],  # Left side.
            [60, -10],  # Right side.
            [0, -40],  # Top.
            [-30, 70],  # Bottom.
            [40, 40],  # Bottom-right.
            [-60, 50],  # Bottom-left.
        ],
        "mass": 5.0,
        "inertia": 8000.0,
    },
    "T": {
        "parts": [(-60, -20, 120, 30), (-20, 10, 40, 60)],
        "attach_points": [
            [-70, 0],  # Left.
            [70, 0],  # Right.
            [0, 70],  # Bottom.
            [-20, -30],  # Top-left.
            [20, -30],  # Top-right.
        ],
        "mass": 6.0,
        "inertia": 9500.0,
    },
    "U": {
        "parts": [(-50, -30, 20, 80), (-50, -30, 100, 20), (30, -30, 20, 80)],
        "attach_points": [
            [-60, 0],  # Left.
            [60, 0],  # Right.
            [-60, 50],  # Bottom-left.
            [60, 50],  # Bottom-right.
            [0, -40],  # Top-center.
        ],
        "mass": 7.0,
        "inertia": 12000.0,
    },
}

# Default obstacle layout (the original hardcoded scene).
DEFAULT_OBSTACLES: list[tuple[int, int, int, int]] = [
    (150, 100, 30, 200),
    (300, 300, 200, 30),
    (600, 150, 30, 180),
    (500, 500, 150, 30),
    (200, 550, 30, 150),
]


# ---------------------------------------------------------------------------
# SceneConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SceneConfig:
    """Single source of truth for all mutable scene parameters.

    Every field has a sensible default matching the original hardcoded
    baseline scene, so ``SceneConfig()`` reproduces the legacy behaviour
    exactly.
    """

    # -- Seed ---------------------------------------------------------------
    seed: int = 42

    # -- World dimensions ---------------------------------------------------
    width: int = 1000
    height: int = 800

    # -- Cargo --------------------------------------------------------------
    cargo_preset: str = "L"
    cargo_x: float = 250.0
    cargo_y: float = 420.0
    cargo_theta: float = 0.0

    # -- Goal ---------------------------------------------------------------
    goal_x: float = 750.0
    goal_y: float = 380.0
    goal_theta: float | None = None

    # -- Obstacles ----------------------------------------------------------
    obstacles: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_OBSTACLES),
    )

    # -- Robots -------------------------------------------------------------
    num_robots: int = 4
    robot_spawns: list[tuple[float, float, float]] | None = None
    attach_map: dict[int, int] | None = None

    # -- Physics overrides (None = use preset default) ----------------------
    cargo_mass: float | None = None
    cargo_inertia: float | None = None
    linear_damping: float = 0.85
    angular_damping: float = 0.80
    # Fraction of tangential force retained when blocked by obstacle/wall.
    # 1.0 means full wall sliding; smaller values damp sliding.
    wall_slide_gain: float = 0.90
    # Fraction of normal (into-contact) force retained when blocked.
    # 0.0 removes all normal component; >0 keeps some pushing authority.
    blocked_normal_force_gain: float = 0.20


# ---------------------------------------------------------------------------
# Randomisation levels
# ---------------------------------------------------------------------------


class RandomLevel(Enum):
    """How aggressively the generator perturbs the scene."""

    FIXED = "fixed"  # Exact baseline, seed is ignored.
    MILD = "mild"  # Small perturbations only (positions ±30 px).
    MODERATE = "moderate"  # Position jitter + obstacle count varies.
    FULL = "full"  # Full random (positions, shapes, counts).


# ---------------------------------------------------------------------------
# SceneGenerator
# ---------------------------------------------------------------------------


class SceneGenerator:
    """Produce reproducible ``SceneConfig`` instances from a seed.

    Args:
        level: Randomisation aggressiveness.
        width: World width in pixels.
        height: World height in pixels.
    """

    # Safe margins so nothing spawns too close to walls.
    _MARGIN = 80
    # Min distance between cargo start and goal.
    _MIN_GOAL_DIST = 300

    def __init__(
        self,
        level: RandomLevel = RandomLevel.FIXED,
        width: int = 1000,
        height: int = 800,
    ) -> None:
        self.level = level
        self.width = width
        self.height = height

    def generate(
        self,
        seed: int = 42,
        random_init_theta: bool = False,
        init_theta_min: float = -np.pi,
        init_theta_max: float = np.pi,
        random_goal_theta: bool = False,
    ) -> SceneConfig:
        """Generate a ``SceneConfig`` for the given seed.

        Args:
            seed: Integer seed for reproducibility.
            random_init_theta: Whether to randomize cargo starting angle.
            init_theta_min: Minimum starting angle.
            init_theta_max: Maximum starting angle.
            random_goal_theta: Whether to randomize target goal orientation.

        Returns:
            A fully populated ``SceneConfig``.
        """
        if self.level == RandomLevel.FIXED:
            cfg = SceneConfig(seed=seed, width=self.width, height=self.height)
            if random_init_theta:
                rng = np.random.default_rng(seed)
                cfg.cargo_theta = float(rng.uniform(init_theta_min, init_theta_max))
            if random_goal_theta:
                rng = np.random.default_rng(seed + 1)
                cfg.goal_theta = float(rng.uniform(-np.pi, np.pi))
            return cfg

        rng = np.random.default_rng(seed)

        # Rejection sampling loop to ensure valid initial state
        max_attempts = 100
        for _ in range(max_attempts):
            cargo_preset = self._pick_cargo(rng)
            cargo_x, cargo_y = self._random_cargo_pos(rng)
            goal_x, goal_y = self._random_goal_pos(rng, cargo_x, cargo_y)
            obstacles = self._random_obstacles(rng)
            num_robots = self._pick_num_robots(rng)

            cargo_theta = float(rng.uniform(init_theta_min, init_theta_max)) if random_init_theta else 0.0
            goal_theta = float(rng.uniform(-np.pi, np.pi)) if random_goal_theta else 0.0

            if self._is_valid_scene(
                cargo_preset,
                cargo_x,
                cargo_y,
                goal_x,
                goal_y,
                obstacles,
                num_robots,
                cargo_theta=cargo_theta,
                goal_theta=goal_theta,
            ):
                return SceneConfig(
                    seed=seed,
                    width=self.width,
                    height=self.height,
                    cargo_preset=cargo_preset,
                    cargo_x=cargo_x,
                    cargo_y=cargo_y,
                    cargo_theta=cargo_theta,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    goal_theta=goal_theta,
                    obstacles=obstacles,
                    num_robots=num_robots,
                )

        import sys

        print(
            f"[Warning] Failed to generate a valid scene for seed {seed} "
            f"after {max_attempts} attempts. Using safe default fallback.",
            file=sys.stderr,
        )
        return SceneConfig(seed=seed, width=self.width, height=self.height)

    # -- Private helpers ----------------------------------------------------

    def _pick_cargo(self, rng: np.random.Generator) -> str:
        """Select cargo shape preset."""
        if self.level == RandomLevel.MILD:
            return "L"  # Keep default shape in mild mode.
        return rng.choice(list(CARGO_PRESETS.keys()))

    def _pick_num_robots(self, rng: np.random.Generator) -> int:
        """Select number of robots."""
        if self.level in (RandomLevel.MILD, RandomLevel.MODERATE):
            return 4  # Keep default count in mild/moderate.
        return int(rng.choice([3, 4, 5]))

    def _random_cargo_pos(self, rng: np.random.Generator) -> tuple[float, float]:
        """Generate cargo start position."""
        base_x, base_y = 250.0, 420.0

        if self.level == RandomLevel.MILD:
            # Small jitter around the default position.
            jitter = 30.0
            x = base_x + rng.uniform(-jitter, jitter)
            y = base_y + rng.uniform(-jitter, jitter)
        else:
            # Left third of the map.
            x = rng.uniform(self._MARGIN, self.width * 0.35)
            y = rng.uniform(self._MARGIN, self.height - self._MARGIN)

        return float(x), float(y)

    def _random_goal_pos(
        self,
        rng: np.random.Generator,
        cargo_x: float,
        cargo_y: float,
    ) -> tuple[float, float]:
        """Generate goal position ensuring minimum distance from cargo."""
        base_gx, base_gy = 750.0, 380.0

        if self.level == RandomLevel.MILD:
            jitter = 30.0
            gx = base_gx + rng.uniform(-jitter, jitter)
            gy = base_gy + rng.uniform(-jitter, jitter)
            return float(gx), float(gy)

        # Right half of the map, must be far enough from cargo.
        for _ in range(50):
            gx = rng.uniform(self.width * 0.55, self.width - self._MARGIN)
            gy = rng.uniform(self._MARGIN, self.height - self._MARGIN)
            if np.hypot(gx - cargo_x, gy - cargo_y) >= self._MIN_GOAL_DIST:
                return float(gx), float(gy)

        # Fallback: right side center.
        return float(self.width * 0.75), float(self.height * 0.5)

    def _random_obstacles(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int, int]]:
        """Generate obstacle layout."""
        if self.level == RandomLevel.MILD:
            # Jitter existing obstacles by a few pixels.
            jittered = []
            for ox, oy, ow, oh in DEFAULT_OBSTACLES:
                dx = int(rng.integers(-15, 16))
                dy = int(rng.integers(-15, 16))
                jittered.append((ox + dx, oy + dy, ow, oh))
            return jittered

        # Moderate / Full: generate fresh obstacles.
        if self.level == RandomLevel.MODERATE:
            n = int(rng.integers(4, 7))
        else:
            n = int(rng.integers(3, 9))

        obstacles: list[tuple[int, int, int, int]] = []
        for _ in range(n):
            # Randomly choose horizontal or vertical wall.
            if rng.random() < 0.5:
                # Horizontal wall.
                w = int(rng.integers(80, 220))
                h = int(rng.integers(20, 40))
            else:
                # Vertical wall.
                w = int(rng.integers(20, 40))
                h = int(rng.integers(80, 220))

            ox = int(rng.integers(self._MARGIN, self.width - self._MARGIN - w))
            oy = int(rng.integers(self._MARGIN, self.height - self._MARGIN - h))
            obstacles.append((ox, oy, w, h))

        return obstacles

    def _poly_intersects_aabb(
        self, poly: np.ndarray, rect: tuple[int, int, int, int]
    ) -> bool:
        """SAT collision test between an arbitrary polygon and an AABB."""
        ox, oy, w, h = rect
        aabb_corners = np.array(
            [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]]
        )

        edges = np.diff(np.vstack([poly, poly[0]]), axis=0)
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])

        axes = np.vstack(([[1.0, 0.0], [0.0, 1.0]], normals))

        for ax in axes:
            mag = np.hypot(ax[0], ax[1])
            if mag < 1e-8:
                continue
            ax = ax / mag

            p_aabb = aabb_corners @ ax
            p_poly = poly @ ax

            if np.max(p_aabb) < np.min(p_poly) or np.max(p_poly) < np.min(p_aabb):
                return False
        return True

    def _is_valid_scene(
        self,
        cargo_preset: str,
        cargo_x: float,
        cargo_y: float,
        goal_x: float,
        goal_y: float,
        obstacles: list[tuple[int, int, int, int]],
        num_robots: int,
        cargo_theta: float = 0.0,
        goal_theta: float = 0.0,
    ) -> bool:
        """
        Check if the generated scene has any overlapping colliders or bounds violations.
        """
        cargo_parts = CARGO_PRESETS[cargo_preset]["parts"]
        padding = 5.0  # Small safety margin

        # 1. Check Cargo vs Boundaries and Obstacles at START POSE
        c, s = np.cos(cargo_theta), np.sin(cargo_theta)
        rot = np.array([[c, -s], [s, c]])
        for lx, ly, w, h in cargo_parts:
            corners_local = np.array(
                [[lx, ly], [lx + w, ly], [lx + w, ly + h], [lx, ly + h]]
            )
            corners_world = np.array([cargo_x, cargo_y]) + (rot @ corners_local.T).T

            # Boundary check
            if (
                np.any(corners_world[:, 0] < padding)
                or np.any(corners_world[:, 0] > self.width - padding)
                or np.any(corners_world[:, 1] < padding)
                or np.any(corners_world[:, 1] > self.height - padding)
            ):
                return False

            # Obstacle check
            for obs in obstacles:
                if self._poly_intersects_aabb(corners_world, obs):
                    return False

        # 2. Check Cargo vs Boundaries and Obstacles at GOAL POSE
        cg, sg = np.cos(goal_theta), np.sin(goal_theta)
        rot_g = np.array([[cg, -sg], [sg, cg]])
        for lx, ly, w, h in cargo_parts:
            corners_local = np.array(
                [[lx, ly], [lx + w, ly], [lx + w, ly + h], [lx, ly + h]]
            )
            corners_world = np.array([goal_x, goal_y]) + (rot_g @ corners_local.T).T

            # Boundary check
            if (
                np.any(corners_world[:, 0] < padding)
                or np.any(corners_world[:, 0] > self.width - padding)
                or np.any(corners_world[:, 1] < padding)
                or np.any(corners_world[:, 1] > self.height - padding)
            ):
                return False

            # Obstacle check
            for obs in obstacles:
                if self._poly_intersects_aabb(corners_world, obs):
                    return False

        # 3. Check Attached Robots vs Boundaries and Obstacles
        attach_points = CARGO_PRESETS[cargo_preset]["attach_points"]
        n_attach = len(attach_points)
        robot_radius = 18.0 + 2.0  # Robot.RADIUS + margin

        for i in range(num_robots):
            idx = i % n_attach
            rx, ry = attach_points[idx]
            r_world = np.array([cargo_x, cargo_y]) + rot @ np.array([rx, ry])
            r_world_x = r_world[0]
            r_world_y = r_world[1]

            # Boundary check
            if (
                r_world_x - robot_radius < 0
                or r_world_x + robot_radius > self.width
                or r_world_y - robot_radius < 0
                or r_world_y + robot_radius > self.height
            ):
                return False

            # Obstacle check (circle vs AABB)
            for ox, oy, ow, oh in obstacles:
                cx = max(ox, min(r_world_x, ox + ow))
                cy = max(oy, min(r_world_y, oy + oh))
                if np.hypot(r_world_x - cx, r_world_y - cy) < robot_radius:
                    return False

        # 4. Check Goal vs Obstacles
        goal_radius = 30.0
        for ox, oy, ow, oh in obstacles:
            cx = max(ox, min(goal_x, ox + ow))
            cy = max(oy, min(goal_y, oy + oh))
            if np.hypot(goal_x - cx, goal_y - cy) < goal_radius:
                return False

        return True
