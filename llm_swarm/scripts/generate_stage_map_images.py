"""Generate stage map snapshots for presentation slides."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Support running as a script from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame

from src.envs.transport_parallel_env import TransportParallelEnv
from src.sim.renderer import Renderer
from src.sim.scene_config import RandomLevel


@dataclass(frozen=True)
class StageProfile:
    stage: str
    title: str
    subtitle: str
    level: RandomLevel
    no_obstacles: bool
    random_init_theta: bool
    stage3_gap_height: float = 220.0
    stage3_wall_width: int = 42
    stage4_gap_span: float = 165.0
    stage4_wall_width: int = 34


PROFILES: list[StageProfile] = [
    StageProfile(
        stage="1",
        title="Stage 1",
        subtitle="Fixed scene, no obstacles, fixed initial orientation",
        level=RandomLevel.FIXED,
        no_obstacles=True,
        random_init_theta=False,
    ),
    StageProfile(
        stage="2",
        title="Stage 2",
        subtitle="No obstacles, random initial orientation",
        level=RandomLevel.MILD,
        no_obstacles=True,
        random_init_theta=True,
    ),
    StageProfile(
        stage="3",
        title="Stage 3",
        subtitle="Fixed obstacles, random orientation, wider passage training",
        level=RandomLevel.FIXED,
        no_obstacles=False,
        random_init_theta=True,
        stage3_gap_height=240.0,
    ),
    StageProfile(
        stage="4",
        title="Stage 4",
        subtitle="Narrower, more complex passages and obstacles",
        level=RandomLevel.FIXED,
        no_obstacles=False,
        random_init_theta=True,
        stage4_gap_span=160.0,
        stage4_wall_width=36,
    ),
    StageProfile(
        stage="5",
        title="Stage 5",
        subtitle="Random obstacle template (gate / corridor / pillars / chicane)",
        level=RandomLevel.FIXED,
        no_obstacles=False,
        random_init_theta=True,
    ),
    StageProfile(
        stage="6",
        title="Stage 6",
        subtitle="Harder templates, denser obstacles, opposite-corner start/goal",
        level=RandomLevel.FIXED,
        no_obstacles=False,
        random_init_theta=True,
    ),
]


def build_env(
    profile: StageProfile,
    seed: int,
    cargo_preset: str,
    goal_orientation_matching: bool,
    goal_theta: float | None,
    random_goal_theta: bool,
) -> TransportParallelEnv:
    """Create an environment snapshot for one curriculum stage."""
    env = TransportParallelEnv(
        config=None,
        random_level=profile.level,
        max_steps=1,
        fixed_num_agents=4,
        fixed_cargo_preset=cargo_preset,
        curriculum_stage=profile.stage,
        no_obstacles=profile.no_obstacles,
        random_init_theta=profile.random_init_theta,
        stage3_gap_height=profile.stage3_gap_height,
        stage3_wall_width=profile.stage3_wall_width,
        stage4_gap_span=profile.stage4_gap_span,
        stage4_wall_width=profile.stage4_wall_width,
        action_mode="object_wrench",
        goal_orientation_matching=goal_orientation_matching,
        random_goal_theta=random_goal_theta,
    )
    env.reset(seed=seed)
    # Override the goal_theta after reset so the renderer ghost shows the
    # exact angle the user asked for (random_goal_theta is overridden when
    # an explicit goal_theta is provided).
    if goal_theta is not None:
        env.world.obj.goal_theta = float(goal_theta)
    return env


def draw_title_bar(screen: pygame.Surface, title: str, subtitle: str) -> None:
    """Draw a top title band to make the image slide-ready."""
    w, _ = screen.get_size()
    band_h = 72
    overlay = pygame.Surface((w, band_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    font_title = pygame.font.SysFont("arial", 30, bold=True)
    font_sub = pygame.font.SysFont("arial", 20)

    t1 = font_title.render(title, True, (245, 245, 245))
    t2 = font_sub.render(subtitle, True, (210, 220, 230))
    screen.blit(t1, (16, 10))
    screen.blit(t2, (16, 42))


def save_stage_image(
    out_dir: Path,
    profile: StageProfile,
    seed: int,
    cargo_preset: str,
    goal_orientation_matching: bool,
    goal_theta: float | None,
    random_goal_theta: bool,
    suffix: str = "",
) -> Path:
    """Render and save one stage map image."""
    env = build_env(
        profile=profile,
        seed=seed,
        cargo_preset=cargo_preset,
        goal_orientation_matching=goal_orientation_matching,
        goal_theta=goal_theta,
        random_goal_theta=random_goal_theta,
    )
    w, h = env.world.width, env.world.height
    screen = pygame.display.set_mode((w, h))

    renderer = Renderer(screen, env.world)
    renderer.draw()
    # Append cargo shape to the subtitle so the image is self-describing.
    subtitle = f"{profile.subtitle}  ·  Cargo {cargo_preset}"
    draw_title_bar(screen, profile.title, subtitle)

    fname = f"stage_{profile.stage}_{cargo_preset}{suffix}.png"
    out_path = out_dir / fname
    pygame.image.save(screen, str(out_path))
    env.close()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage map images for PPT.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/stage_maps",
        help="Output directory for generated PNG files.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for deterministic stage snapshots.",
    )
    parser.add_argument(
        "--samples-per-stage",
        type=int,
        default=1,
        help="How many randomized snapshots to generate per stage "
        "(useful for stage 5/6 which pick random templates).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to specific stages, e.g. --stages 5 6. "
        "Default = all stages defined in PROFILES.",
    )
    parser.add_argument(
        "--cargo-presets",
        type=str,
        nargs="*",
        default=["L"],
        choices=["L", "T", "U"],
        help="Cargo shapes to render. Pass multiple to render one image per "
        "shape per stage, e.g. --cargo-presets L T U.",
    )
    parser.add_argument(
        "--goal-theta-deg",
        type=float,
        default=None,
        help="Fixed target goal angle in degrees. When set, every snapshot "
        "draws the goal pose ghost at this angle. Overrides "
        "--random-goal-theta.",
    )
    parser.add_argument(
        "--random-goal-theta",
        action="store_true",
        help="Sample a random target goal angle per snapshot so the goal "
        "ghost shows a varied orientation.",
    )
    parser.add_argument(
        "--no-goal-orientation",
        action="store_true",
        help="Disable goal-orientation rendering even if --goal-theta-deg / "
        "--random-goal-theta is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure headless rendering works on servers without X display.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    selected = (
        [p for p in PROFILES if p.stage in set(args.stages)]
        if args.stages
        else PROFILES
    )
    samples = max(1, int(args.samples_per_stage))

    # Resolve goal-orientation behavior.
    explicit_theta_rad: float | None = None
    if args.goal_theta_deg is not None:
        explicit_theta_rad = float(np.deg2rad(args.goal_theta_deg))
    goal_orientation_matching = (
        not args.no_goal_orientation
        and (explicit_theta_rad is not None or args.random_goal_theta)
    )
    # If user gave explicit angle, never randomize on top of it.
    random_goal_theta = bool(args.random_goal_theta and explicit_theta_rad is None)

    try:
        for stage_idx, profile in enumerate(selected):
            for preset_idx, preset in enumerate(args.cargo_presets):
                for k in range(samples):
                    seed = (
                        args.base_seed
                        + stage_idx * 1000
                        + preset_idx * 100
                        + k
                    )
                    suffix = "" if samples == 1 else f"_s{k}"
                    out = save_stage_image(
                        out_dir=out_dir,
                        profile=profile,
                        seed=seed,
                        cargo_preset=preset,
                        goal_orientation_matching=goal_orientation_matching,
                        goal_theta=explicit_theta_rad,
                        random_goal_theta=random_goal_theta,
                        suffix=suffix,
                    )
                    print(f"Generated: {out}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
