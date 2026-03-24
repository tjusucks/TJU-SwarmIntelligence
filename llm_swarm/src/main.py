"""
Multi-Robot Cooperative Transport Simulation - Entry Point
Run: python main.py
Dependencies: pip install pygame numpy
"""

import sys

import pygame

from src.sim.renderer import Renderer
from src.sim.world import World

FPS = 60
WINDOW_W, WINDOW_H = 1000, 800


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Multi-Robot Cooperative Transport")
    clock = pygame.time.Clock()

    world = World(width=WINDOW_W, height=WINDOW_H)
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
                    world.reset()

        if not paused:
            world.step(dt=1.0 / FPS)

        renderer.draw()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
