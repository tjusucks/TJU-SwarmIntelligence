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
]


def build_env(profile: StageProfile, seed: int) -> TransportParallelEnv:
    """Create an environment snapshot for one curriculum stage."""
    env = TransportParallelEnv(
        config=None,
        random_level=profile.level,
        max_steps=1,
        fixed_num_agents=4,
        fixed_cargo_preset="L",
        curriculum_stage=profile.stage,
        no_obstacles=profile.no_obstacles,
        random_init_theta=profile.random_init_theta,
        stage3_gap_height=profile.stage3_gap_height,
        stage3_wall_width=profile.stage3_wall_width,
        stage4_gap_span=profile.stage4_gap_span,
        stage4_wall_width=profile.stage4_wall_width,
        action_mode="object_wrench",
    )
    env.reset(seed=seed)
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


def save_stage_image(out_dir: Path, profile: StageProfile, seed: int) -> Path:
    """Render and save one stage map image."""
    env = build_env(profile, seed)
    w, h = env.world.width, env.world.height
    screen = pygame.display.set_mode((w, h))

    renderer = Renderer(screen, env.world)
    renderer.draw()
    draw_title_bar(screen, profile.title, profile.subtitle)

    out_path = out_dir / f"stage_{profile.stage}.png"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure headless rendering works on servers without X display.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        for idx, profile in enumerate(PROFILES):
            out = save_stage_image(out_dir, profile, seed=args.base_seed + idx)
            print(f"Generated: {out}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
