"""PettingZoo parallel environment for cooperative transport."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.sim.scene_config import RandomLevel, SceneConfig, SceneGenerator
from src.sim.world import World


class TransportParallelEnv(ParallelEnv):
    """Parallel MARL environment with global-map observations.

    Each robot is an agent with a 2D continuous action representing normalized
    force components in range [-1, 1]. The environment maps actions to force
    commands and advances the shared world by one simulation step.
    """

    metadata = {"name": "transport_parallel_v0", "render_modes": ["none"]}

    def __init__(
        self,
        config: SceneConfig | None = None,
        random_level: RandomLevel = RandomLevel.FIXED,
        max_steps: int = 2400,
        dt: float = 1.0 / 60.0,
        force_max: float = 500.0,
        max_obstacles: int = 12,
        stuck_patience: int = 600,
        stuck_move_eps: float = 0.5,
        fixed_num_agents: int | None = 4,
    ) -> None:
        self._base_config = config
        self._random_level = random_level
        self._generator = SceneGenerator(level=random_level)

        self.max_steps = max_steps
        self.dt = dt
        self.force_max = force_max
        self.max_obstacles = max_obstacles
        self.stuck_patience = stuck_patience
        self.stuck_move_eps = stuck_move_eps
        self.fixed_num_agents = fixed_num_agents

        self._step_count = 0
        self._episode_seed = 42
        self._prev_distance = 0.0
        self._stuck_steps = 0
        self._last_obj_pos = np.zeros(2, dtype=np.float32)

        init_config = config or self._generator.generate(seed=self._episode_seed)
        self.world = World(config=init_config)
        self.world.external_control = True
        self.world.reset(config=init_config)

        if self.fixed_num_agents is not None:
            if len(self.world.robots) != self.fixed_num_agents:
                fixed_cfg = init_config
                fixed_cfg.num_robots = int(self.fixed_num_agents)
                self.world.reset(config=fixed_cfg)

        self.possible_agents = [
            f"robot_{i}" for i in range(len(self.world.robots))
        ]
        self.agents = list(self.possible_agents)

        self._obs_dim = self._compute_obs_dim()
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def observation_space(self, agent: str) -> spaces.Box:
        """Return observation space for an agent."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Box:
        """Return action space for an agent."""
        return self._action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset environment and return initial observations."""
        if seed is not None:
            self._episode_seed = seed

        if self._base_config is not None:
            config = self._base_config
        else:
            config = self._generator.generate(seed=self._episode_seed)
            self._episode_seed += 1

        if self.fixed_num_agents is not None:
            config.num_robots = int(self.fixed_num_agents)

        self.world.external_control = True
        self.world.reset(config=config)
        self._step_count = 0
        self.agents = list(self.possible_agents)

        self._prev_distance = self._distance_to_goal()
        self._stuck_steps = 0
        self._last_obj_pos = np.array(
            [self.world.obj.x, self.world.obj.y],
            dtype=np.float32,
        )
        observations = self._collect_obs()
        infos = {agent: self._build_info() for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Apply actions and advance one world step."""
        if not self.agents:
            return {}, {}, {}, {}, {}

        for idx, agent in enumerate(self.possible_agents):
            robot = self.world.robots[idx]
            action = np.asarray(actions.get(agent, np.zeros(2, dtype=np.float32)))
            action = np.clip(action, -1.0, 1.0)
            robot.cmd_fx = float(action[0] * self.force_max)
            robot.cmd_fy = float(action[1] * self.force_max)

        prev_pos = np.array([self.world.obj.x, self.world.obj.y], dtype=np.float32)
        self.world.step(self.dt)
        self._step_count += 1

        curr_pos = np.array([self.world.obj.x, self.world.obj.y], dtype=np.float32)
        move_dist = float(np.linalg.norm(curr_pos - prev_pos))
        self._last_obj_pos = curr_pos

        curr_dist = self._distance_to_goal()
        progress = (self._prev_distance - curr_dist) / self._map_diagonal()
        self._prev_distance = curr_dist

        blocked_ratio = (
            sum(float(getattr(r, "blocked", False)) for r in self.world.robots)
            / max(1, len(self.world.robots))
        )
        all_blocked = blocked_ratio >= 0.999

        if move_dist < self.stuck_move_eps:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        stuck_failure = bool(
            self._stuck_steps >= self.stuck_patience
            and (all_blocked or self.world.invalid_state)
        )

        reward = progress - 0.001 - 0.002 * blocked_ratio

        if self.world.success:
            reward += 5.0
        if stuck_failure:
            reward -= 2.0

        terminated = bool(self.world.success or stuck_failure)
        truncated = bool(self._step_count >= self.max_steps)

        rewards = {agent: float(reward) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: self._build_info() for agent in self.agents}

        observations = self._collect_obs()

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """No-op render for training environment."""
        return

    def close(self) -> None:
        """Close resources."""
        return

    def _compute_obs_dim(self) -> int:
        base = 10
        per_robot = 7
        per_obstacle = 5
        own = 1
        return base + per_robot * len(self.world.robots) + per_obstacle * self.max_obstacles + own

    def _collect_obs(self) -> dict[str, np.ndarray]:
        global_obs = self._global_obs_vector()
        obs = {}
        n = max(1, len(self.world.robots))
        for i, agent in enumerate(self.agents):
            own_idx = np.array([float(i) / float(n)], dtype=np.float32)
            obs[agent] = np.concatenate([global_obs, own_idx], dtype=np.float32)
        return obs

    def _global_obs_vector(self) -> np.ndarray:
        obj = self.world.obj
        w = float(self.world.width)
        h = float(self.world.height)
        diag = self._map_diagonal()

        vec: list[float] = [
            obj.x / w,
            obj.y / h,
            float(np.sin(obj.theta)),
            float(np.cos(obj.theta)),
            obj.vx / 200.0,
            obj.vy / 200.0,
            obj.omega / 5.0,
            (obj.goal_x - obj.x) / w,
            (obj.goal_y - obj.y) / h,
            self._distance_to_goal() / diag,
        ]

        for r in self.world.robots:
            vec.extend(
                [
                    (r.x - obj.x) / w,
                    (r.y - obj.y) / h,
                    float(np.sin(r.theta)),
                    float(np.cos(r.theta)),
                    r.cmd_fx / max(1e-6, self.force_max),
                    r.cmd_fy / max(1e-6, self.force_max),
                    float(getattr(r, "blocked", False)),
                ]
            )

        for i in range(self.max_obstacles):
            if i < len(self.world.obstacles):
                ox, oy, ow, oh = self.world.obstacles[i]
                cx = ox + ow * 0.5
                cy = oy + oh * 0.5
                vec.extend([cx / w, cy / h, ow / w, oh / h, 1.0])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(vec, dtype=np.float32)

    def _distance_to_goal(self) -> float:
        obj = self.world.obj
        return float(np.hypot(obj.x - obj.goal_x, obj.y - obj.goal_y))

    def _map_diagonal(self) -> float:
        return float(np.hypot(self.world.width, self.world.height))

    def _build_info(self) -> dict:
        blocked_ratio = (
            sum(float(getattr(r, "blocked", False)) for r in self.world.robots)
            / max(1, len(self.world.robots))
        )
        stuck_failure = bool(
            self._stuck_steps >= self.stuck_patience
            and (blocked_ratio >= 0.999 or self.world.invalid_state)
        )
        return {
            "distance_to_goal": self._distance_to_goal(),
            "time": self.world.t,
            "success": self.world.success,
            "invalid_state": self.world.invalid_state,
            "stuck_steps": self._stuck_steps,
            "stuck_failure": stuck_failure,
            "step": self._step_count,
        }
