"""
Renderer: handles all drawing logic
"""

import numpy as np
import pygame

BG_COLOR = (18, 22, 30)
OBSTACLE_COLOR = (80, 88, 105)
OBJ_COLOR = (230, 190, 80)
OBJ_OUTLINE = (255, 220, 100)
GOAL_COLOR = (80, 220, 120)
ATTACH_COLOR = (255, 255, 255)
SPRING_COLOR = (180, 180, 255)
GRID_COLOR = (28, 32, 42)
TEXT_COLOR = (200, 210, 220)
SUCCESS_COLOR = (60, 230, 130)


class Renderer:
    def __init__(self, screen: pygame.Surface, world):
        self.screen = screen
        self.world = world
        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_md = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 36, bold=True)

    def draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_goal()
        self._draw_obstacles()
        self._draw_object()
        self._draw_springs()
        self._draw_robots()
        self._draw_hud()
        if self.world.success:
            self._draw_success()

    # -- Grid -----------------------------------------------------------
    def _draw_grid(self):
        w, h = self.world.width, self.world.height
        step = 50
        for x in range(0, w, step):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, h))
        for y in range(0, h, step):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (w, y))

    # -- Goal zone ------------------------------------------------------
    def _draw_goal(self):
        obj = self.world.obj
        if obj.goal_x is None:
            return
        gx, gy = int(obj.goal_x), int(obj.goal_y)
        for r in range(40, 0, -8):
            alpha = int(60 * (1 - r / 40))
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*GOAL_COLOR, alpha), (r, r), r)
            self.screen.blit(s, (gx - r, gy - r))
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), 12, 2)
        label = self.font_sm.render("GOAL", True, GOAL_COLOR)
        self.screen.blit(label, (gx - label.get_width() // 2, gy + 16))

    # -- Obstacles ------------------------------------------------------
    def _draw_obstacles(self):
        for ox, oy, ow, oh in self.world.obstacles:
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, (ox, oy, ow, oh))
            pygame.draw.rect(self.screen, (120, 130, 150), (ox, oy, ow, oh), 1)

    # -- Transport object (L-shape) ------------------------------------
    def _draw_object(self):
        obj = self.world.obj
        parts = obj.get_parts_world()
        for corners in parts:
            pts = [(int(p[0]), int(p[1])) for p in corners]
            pygame.draw.polygon(self.screen, OBJ_COLOR, pts)
            pygame.draw.polygon(self.screen, OBJ_OUTLINE, pts, 2)

        # Center of mass marker
        cx, cy = int(obj.x), int(obj.y)
        pygame.draw.circle(self.screen, OBJ_OUTLINE, (cx, cy), 5)

        # Heading arrow
        arrow_len = 30
        ex = cx + int(arrow_len * np.cos(obj.theta))
        ey = cy + int(arrow_len * np.sin(obj.theta))
        pygame.draw.line(self.screen, OBJ_OUTLINE, (cx, cy), (ex, ey), 2)

        # Attach points
        for i in range(len(obj.attach_points_local)):
            ap = obj.get_attach_point_world(i)
            pygame.draw.circle(
                self.screen, ATTACH_COLOR, (int(ap[0]), int(ap[1])), 4, 1
            )

    # -- Spring lines ---------------------------------------------------
    def _draw_springs(self):
        obj = self.world.obj
        for r in self.world.robots:
            if r.attached:
                ap = obj.get_attach_point_world(r._attach_idx)
                rx, ry = int(r.x), int(r.y)
                ax, ay = int(ap[0]), int(ap[1])
                pygame.draw.line(self.screen, SPRING_COLOR, (rx, ry), (ax, ay), 1)

    # -- Robots ---------------------------------------------------------
    def _draw_robots(self):
        for r in self.world.robots:
            cx, cy = int(r.x), int(r.y)
            blocked = getattr(r, "blocked", False)

            # Sensing radius - only when not blocked
            if not blocked:
                s = pygame.Surface(
                    (r.sense_radius * 2, r.sense_radius * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    s, (*r.color, 15), (r.sense_radius, r.sense_radius), r.sense_radius
                )
                self.screen.blit(s, (cx - r.sense_radius, cy - r.sense_radius))

            # Body: gray when blocked
            body_color = (100, 100, 110) if blocked else r.color
            pygame.draw.circle(self.screen, body_color, (cx, cy), r.RADIUS)

            # Outline: red + thicker when blocked
            outline_color = (240, 60, 60) if blocked else (255, 255, 255)
            outline_width = 3 if blocked else 2
            pygame.draw.circle(
                self.screen, outline_color, (cx, cy), r.RADIUS, outline_width
            )

            # Heading indicator
            ex = cx + int(r.RADIUS * np.cos(r.theta))
            ey = cy + int(r.RADIUS * np.sin(r.theta))
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (ex, ey), 3)

            # ID label - show X when blocked
            status = " X" if blocked else ""
            label = self.font_sm.render(f"R{r.id}{status}", True, (255, 255, 255))
            self.screen.blit(label, (cx - label.get_width() // 2, cy - 8))

    # -- HUD ------------------------------------------------------------
    def _draw_hud(self):
        obj = self.world.obj
        dist = 0.0
        if obj.goal_x is not None:
            dist = np.hypot(obj.x - obj.goal_x, obj.y - obj.goal_y)

        lines = [
            f"Time:     {self.world.t:.1f}s",
            f"Obj pos:  ({obj.x:.0f}, {obj.y:.0f})",
            f"To goal:  {dist:.0f} px",
            f"Heading:  {np.degrees(obj.theta):.1f} deg",
            "",
            "SPACE: pause / resume",
            "R:     reset",
        ]
        for i, line in enumerate(lines):
            surf = self.font_sm.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (10, 10 + i * 18))

    # -- Success screen -------------------------------------------------
    def _draw_success(self):
        w, h = self.world.width, self.world.height
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        msg = self.font_lg.render("Transport Complete!", True, SUCCESS_COLOR)
        sub = self.font_md.render(
            f"Time: {self.world.t:.1f}s  |  Press R to reset", True, TEXT_COLOR
        )
        self.screen.blit(msg, (w // 2 - msg.get_width() // 2, h // 2 - 40))
        self.screen.blit(sub, (w // 2 - sub.get_width() // 2, h // 2 + 20))
