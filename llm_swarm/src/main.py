"""Multi-Robot Cooperative Transport Simulation — Entry Point.

Run:  uv run python -m src.main [--seed SEED] [--level fixed|mild|moderate|full]
"""

from __future__ import annotations

import argparse
import sys

import pygame

from src.sim.renderer import Renderer
from src.sim.scene_config import RandomLevel, SceneGenerator
from src.sim.world import World

FPS = 60


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Robot Cooperative Transport Simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scenario seed (default: 42).",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="fixed",
        choices=["fixed", "mild", "moderate", "full"],
        help="Randomisation level (default: fixed).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the interactive Pygame simulation."""
    args = parse_args()

    # Generate scene config from seed + level.
    level = RandomLevel(args.level)
    generator = SceneGenerator(level=level)
    config = generator.generate(seed=args.seed)
    current_seed = args.seed

    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Multi-Robot Cooperative Transport")
    clock = pygame.time.Clock()

    world = World(config=config)
    renderer = Renderer(screen, world)

    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    # Reset with same config.
                    world.reset()
                if event.key == pygame.K_n:
                    # New random scene (increment seed).
                    current_seed += 1
                    new_config = generator.generate(seed=current_seed)
                    world.reset(config=new_config)

        if not paused:
            world.step(dt=1.0 / FPS)

        renderer.draw()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
