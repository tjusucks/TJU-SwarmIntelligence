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

    def generate(self, seed: int = 42) -> SceneConfig:
        """Generate a ``SceneConfig`` for the given seed.

        Args:
            seed: Integer seed for reproducibility.

        Returns:
            A fully populated ``SceneConfig``.
        """
        if self.level == RandomLevel.FIXED:
            cfg = SceneConfig(seed=seed, width=self.width, height=self.height)
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

            if self._is_valid_scene(
                cargo_preset, cargo_x, cargo_y, goal_x, goal_y, obstacles, num_robots
            ):
                return SceneConfig(
                    seed=seed,
                    width=self.width,
                    height=self.height,
                    cargo_preset=cargo_preset,
                    cargo_x=cargo_x,
                    cargo_y=cargo_y,
                    goal_x=goal_x,
                    goal_y=goal_y,
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

    def _is_valid_scene(
        self,
        cargo_preset: str,
        cargo_x: float,
        cargo_y: float,
        goal_x: float,
        goal_y: float,
        obstacles: list[tuple[int, int, int, int]],
        num_robots: int,
    ) -> bool:
        """
        Check if the generated scene has any overlapping colliders or bounds violations.
        """
        cargo_parts = CARGO_PRESETS[cargo_preset]["parts"]
        padding = 5.0  # Small safety margin

        # 1. Check Cargo vs Boundaries and Obstacles
        for lx, ly, w, h in cargo_parts:
            # AABB of this part in world space (initial rotation is 0)
            c_left = cargo_x + lx - padding
            c_right = cargo_x + lx + w + padding
            c_top = cargo_y + ly - padding
            c_bottom = cargo_y + ly + h + padding

            # Boundary check
            if (
                c_left < 0
                or c_right > self.width
                or c_top < 0
                or c_bottom > self.height
            ):
                return False

            # Obstacle check
            for ox, oy, ow, oh in obstacles:
                o_left, o_right = ox, ox + ow
                o_top, o_bottom = oy, oy + oh

                # AABB intersection test
                if not (
                    c_right <= o_left
                    or c_left >= o_right
                    or c_bottom <= o_top
                    or c_top >= o_bottom
                ):
                    return False

        # 2. Check Attached Robots vs Boundaries and Obstacles
        attach_points = CARGO_PRESETS[cargo_preset]["attach_points"]
        n_attach = len(attach_points)
        robot_radius = 18.0 + 2.0  # Robot.RADIUS + margin

        for i in range(num_robots):
            idx = i % n_attach
            rx, ry = attach_points[idx]
            r_world_x = cargo_x + rx
            r_world_y = cargo_y + ry

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

        # 3. Check Goal vs Obstacles
        goal_radius = 30.0
        for ox, oy, ow, oh in obstacles:
            cx = max(ox, min(goal_x, ox + ow))
            cy = max(oy, min(goal_y, oy + oh))
            if np.hypot(goal_x - cx, goal_y - cy) < goal_radius:
                return False

        return True
