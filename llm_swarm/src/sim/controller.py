"""
Simple cooperative controller
Each attached robot outputs a force vector (cmd_fx, cmd_fy) pointing toward the goal.
The world sums these forces and applies them to the object.
Replace this module with more advanced swarm intelligence algorithms later.
"""

import numpy as np

FORCE_MAX = 300.0  # Max force per robot (simulation units)


class SimpleController:
    def __init__(self, robots, obj, obstacles):
        self.robots = robots
        self.obj = obj
        self.obstacles = obstacles

    def update(self, dt: float):
        goal = np.array([self.obj.goal_x, self.obj.goal_y])
        obj_pos = self.obj.pos

        to_goal = goal - obj_pos
        dist = np.linalg.norm(to_goal)

        if dist < 1e-3:
            for r in self.robots:
                r.cmd_fx = 0.0
                r.cmd_fy = 0.0
            return

        # Unit vector toward goal
        direction = to_goal / dist

        # Scale force: full force when far, taper off near goal
        scale = min(1.0, dist / 150.0)

        for r in self.robots:
            if not r.attached:
                r.cmd_fx = 0.0
                r.cmd_fy = 0.0
                continue

            # All attached robots push in the goal direction
            r.cmd_fx = direction[0] * FORCE_MAX * scale
            r.cmd_fy = direction[1] * FORCE_MAX * scale
