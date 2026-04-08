"""PettingZoo parallel environment for cooperative transport."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.control.force_allocator import allocate_wrench_to_robots
from src.planning.path_planner import plan_path
from src.sim.scene_config import CargoPreset, RandomLevel, SceneConfig, SceneGenerator
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
        fixed_cargo_preset: str | None = "L",
        progress_weight: float = 180.0,
        step_penalty: float = 0.0,
        blocked_penalty_weight: float = 0.0,
        success_bonus: float = 12.0,
        stuck_penalty: float = 1.0,
        timeout_penalty: float = 0.0,
        stagnation_penalty_weight: float = 0.02,
        away_penalty_weight: float = 0.0,
        heading_reward_weight: float = 0.8,
        action_penalty_weight: float = 0.01,
        clearance_penalty_weight: float = 2.0,
        clearance_safe_distance: float = 90.0,
        clearance_penalty_power: float = 2.2,
        preclearance_penalty_weight: float = 0.4,
        avoid_alert_distance: float = 180.0,
        avoid_blend_gain: float = 0.8,
        avoid_torque_gain: float = 140.0,
        clearance_margin_reward_weight: float = 0.08,
        low_speed_penalty_weight: float = 0.0,
        low_speed_threshold: float = 22.0,
        low_speed_far_goal_radius: float = 160.0,
        omega_penalty_weight: float = 0.02,
        velocity_penalty_weight: float = 0.01,
        near_goal_radius: float = 120.0,
        near_goal_speed_penalty_weight: float = 0.0,
        route_cell_size: int = 40,
        route_inflate_margin: float = 70.0,
        route_guidance_gain: float = 0.7,
        residual_force_scale: float = 0.6,
        route_waypoint_tolerance: float = 50.0,
        route_progress_weight: float = 0.0,
        route_deviation_penalty_weight: float = 0.0,
        action_mode: str = "robot_residual",
        object_wrench_residual_scale_xy: float = 0.5,
        object_wrench_residual_scale_tau: float = 0.5,
        route_torque_gain: float = 220.0,
        route_linear_damping_gain: float = 8.0,
        route_angular_damping_gain: float = 24.0,
        no_obstacles: bool = False,
        random_init_theta: bool = False,
        init_theta_min: float = -np.pi,
        init_theta_max: float = np.pi,
        curriculum_stage: str = "none",
        stage3_gap_height: float = 200.0,
        stage3_wall_width: int = 42,
        recovery_push_gain: float = 0.45,
        recovery_torque_gain: float = 180.0,
        recovery_stuck_steps: int = 25,
        unblock_reward_weight: float = 0.25,
        clearance_improve_reward_weight: float = 0.2,
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
        self.fixed_cargo_preset = fixed_cargo_preset
        self.progress_weight = progress_weight
        self.step_penalty = step_penalty
        self.blocked_penalty_weight = blocked_penalty_weight
        self.success_bonus = success_bonus
        self.stuck_penalty = stuck_penalty
        self.timeout_penalty = timeout_penalty
        self.stagnation_penalty_weight = stagnation_penalty_weight
        self.away_penalty_weight = away_penalty_weight
        self.heading_reward_weight = heading_reward_weight
        self.action_penalty_weight = action_penalty_weight
        self.clearance_penalty_weight = clearance_penalty_weight
        self.clearance_safe_distance = clearance_safe_distance
        self.clearance_penalty_power = clearance_penalty_power
        self.preclearance_penalty_weight = preclearance_penalty_weight
        self.avoid_alert_distance = avoid_alert_distance
        self.avoid_blend_gain = avoid_blend_gain
        self.avoid_torque_gain = avoid_torque_gain
        self.clearance_margin_reward_weight = clearance_margin_reward_weight
        self.low_speed_penalty_weight = low_speed_penalty_weight
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_far_goal_radius = low_speed_far_goal_radius
        self.omega_penalty_weight = omega_penalty_weight
        self.velocity_penalty_weight = velocity_penalty_weight
        self.near_goal_radius = near_goal_radius
        self.near_goal_speed_penalty_weight = near_goal_speed_penalty_weight
        self.route_cell_size = route_cell_size
        self.route_inflate_margin = route_inflate_margin
        self.route_guidance_gain = route_guidance_gain
        self.residual_force_scale = residual_force_scale
        self.route_waypoint_tolerance = route_waypoint_tolerance
        self.route_progress_weight = route_progress_weight
        self.route_deviation_penalty_weight = route_deviation_penalty_weight
        if action_mode not in {"robot_residual", "object_wrench"}:
            raise ValueError("action_mode must be 'robot_residual' or 'object_wrench'.")
        self.action_mode = action_mode
        self.object_wrench_residual_scale_xy = object_wrench_residual_scale_xy
        self.object_wrench_residual_scale_tau = object_wrench_residual_scale_tau
        self.route_torque_gain = route_torque_gain
        self.route_linear_damping_gain = route_linear_damping_gain
        self.route_angular_damping_gain = route_angular_damping_gain
        self.no_obstacles = no_obstacles
        self.random_init_theta = random_init_theta
        self.init_theta_min = float(init_theta_min)
        self.init_theta_max = float(init_theta_max)
        self.curriculum_stage = curriculum_stage
        self.stage3_gap_height = stage3_gap_height
        self.stage3_wall_width = stage3_wall_width
        self.recovery_push_gain = recovery_push_gain
        self.recovery_torque_gain = recovery_torque_gain
        self.recovery_stuck_steps = recovery_stuck_steps
        self.unblock_reward_weight = unblock_reward_weight
        self.clearance_improve_reward_weight = clearance_improve_reward_weight

        self._route_waypoints: list[tuple[float, float]] = []
        self._route_idx: int = 0
        self._prev_route_dist: float = 0.0

        self._step_count = 0
        self._episode_seed = 42
        self._prev_distance = 0.0
        self._stuck_steps = 0
        self._last_obj_pos = np.zeros(2, dtype=np.float32)
        self._prev_heading_abs_err = 0.0
        self._prev_blocked_ratio = 0.0
        self._prev_obs_clearance = 1e6

        init_config = config or self._generator.generate(seed=self._episode_seed)
        self.world = World(config=init_config)
        self.world.external_control = True
        self.world.reset(config=init_config)

        if self.fixed_num_agents is not None:
            if len(self.world.robots) != self.fixed_num_agents:
                fixed_cfg = init_config
                fixed_cfg.num_robots = int(self.fixed_num_agents)
                if self.fixed_cargo_preset is not None:
                    fixed_cfg.cargo_preset = self.fixed_cargo_preset
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
        if self.action_mode == "robot_residual":
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
        else:
            # Residual wrench action: [dFx, dFy, dTau].
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
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
        if self.fixed_cargo_preset is not None:
            if self.fixed_cargo_preset not in {p.value for p in CargoPreset}:
                raise ValueError(
                    f"Unsupported cargo preset: {self.fixed_cargo_preset}. "
                    "Expected one of: L, T, U."
                )
            config.cargo_preset = self.fixed_cargo_preset

        if self.curriculum_stage == "3":
            self._apply_stage3_layout(config)

        if self.no_obstacles:
            config.obstacles = []
        if self.random_init_theta:
            config.cargo_theta = float(np.random.uniform(self.init_theta_min, self.init_theta_max))
        else:
            config.cargo_theta = float(getattr(config, "cargo_theta", 0.0))

        self.world.external_control = True
        self.world.reset(config=config)
        self._step_count = 0
        self.agents = list(self.possible_agents)

        self._prev_distance = self._distance_to_goal()
        self._build_route_guidance()
        self._prev_route_dist = self._distance_to_current_waypoint()
        self._stuck_steps = 0
        self._last_obj_pos = np.array(
            [self.world.obj.x, self.world.obj.y],
            dtype=np.float32,
        )
        self._prev_heading_abs_err = self._heading_abs_error()
        self._prev_blocked_ratio = 0.0
        self._prev_obs_clearance = self._distance_object_to_obstacle_or_wall()
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

        action_norm_sum = 0.0
        advanced = self._advance_waypoint_if_reached()
        if advanced:
            # Avoid a spurious large negative route-progress spike when the
            # current waypoint advances before motion integration.
            self._prev_route_dist = self._distance_to_current_waypoint()

        if self.action_mode == "robot_residual":
            route_dir = self._route_direction_unit()
            base_fx = route_dir[0] * self.force_max * self.route_guidance_gain
            base_fy = route_dir[1] * self.force_max * self.route_guidance_gain

            for idx, agent in enumerate(self.possible_agents):
                robot = self.world.robots[idx]
                action = np.asarray(actions.get(agent, np.zeros(2, dtype=np.float32)))
                action = np.clip(action, -1.0, 1.0)
                residual_fx = float(action[0] * self.force_max * self.residual_force_scale)
                residual_fy = float(action[1] * self.force_max * self.residual_force_scale)
                cmd_fx = base_fx + residual_fx
                cmd_fy = base_fy + residual_fy
                cmd_norm = float(np.hypot(cmd_fx, cmd_fy))
                if cmd_norm > self.force_max:
                    scale = self.force_max / max(1e-6, cmd_norm)
                    cmd_fx *= scale
                    cmd_fy *= scale

                robot.cmd_fx = float(cmd_fx)
                robot.cmd_fy = float(cmd_fy)
                action_norm_sum += float(np.linalg.norm(action))
        else:
            base_fx, base_fy, base_tau = self._route_base_wrench()

            residuals = []
            for agent in self.possible_agents:
                a = np.asarray(actions.get(agent, np.zeros(3, dtype=np.float32)))
                a = np.clip(a, -1.0, 1.0)
                residuals.append(a)
                action_norm_sum += float(np.linalg.norm(a))
            residual_mean = np.mean(np.asarray(residuals, dtype=np.float32), axis=0)

            desired_fx = base_fx + float(
                residual_mean[0] * self.force_max * self.object_wrench_residual_scale_xy
            )
            desired_fy = base_fy + float(
                residual_mean[1] * self.force_max * self.object_wrench_residual_scale_xy
            )
            tau_scale = self.force_max * 120.0
            desired_tau = base_tau + float(
                residual_mean[2] * tau_scale * self.object_wrench_residual_scale_tau
            )

            attached_indices = [i for i, r in enumerate(self.world.robots) if r.attached]
            rel_points = np.asarray(
                [
                    self.world.obj.attach_points_local[self.world.robots[i]._attach_idx]
                    for i in attached_indices
                ],
                dtype=np.float32,
            )
            alloc = allocate_wrench_to_robots(
                rel_points=rel_points,
                desired_fx=desired_fx,
                desired_fy=desired_fy,
                desired_tau=desired_tau,
                force_max=self.force_max,
                ridge_lambda=5e-3,
            )
            attached_slot = {rid: slot for slot, rid in enumerate(attached_indices)}

            for i, robot in enumerate(self.world.robots):
                if not robot.attached:
                    robot.cmd_fx = 0.0
                    robot.cmd_fy = 0.0
                    continue
                slot = attached_slot[i]
                fx, fy = alloc[slot]
                robot.cmd_fx = float(fx)
                robot.cmd_fy = float(fy)

        prev_pos = np.array([self.world.obj.x, self.world.obj.y], dtype=np.float32)
        self.world.step(self.dt)
        self._step_count += 1

        curr_pos = np.array([self.world.obj.x, self.world.obj.y], dtype=np.float32)
        move_dist = float(np.linalg.norm(curr_pos - prev_pos))
        self._last_obj_pos = curr_pos

        curr_dist = self._distance_to_goal()
        progress = (self._prev_distance - curr_dist) / self._map_diagonal()
        self._prev_distance = curr_dist
        self._prev_route_dist = self._distance_to_current_waypoint()

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

        # Minimal reward set requested by user:
        # 1) progress reward
        # 2) collision (blocked) penalty
        # 3) small stagnation penalty
        # 4) terminal success / failure bonuses
        reward = self.progress_weight * progress
        reward -= self.blocked_penalty_weight * blocked_ratio
        if move_dist < self.stuck_move_eps:
            reward -= self.stagnation_penalty_weight

        # Nonlinear proximity penalty: approaching walls/obstacles is allowed,
        # but penalty rises sharply as clearance gets smaller.
        obs_clearance = self._distance_object_to_obstacle_or_wall()
        if obs_clearance < self.clearance_safe_distance:
            proximity = 1.0 - obs_clearance / max(1e-6, self.clearance_safe_distance)
            reward -= self.clearance_penalty_weight * (
                max(0.0, proximity) ** self.clearance_penalty_power
            )

        # Anticipatory shaping: start penalizing when entering an alert band
        # before hard-contact distance, so policy learns earlier avoidance.
        if obs_clearance < self.avoid_alert_distance:
            pre_proximity = 1.0 - obs_clearance / max(1e-6, self.avoid_alert_distance)
            reward -= self.preclearance_penalty_weight * (max(0.0, pre_proximity) ** 2)

        # Small positive bias for keeping a healthy clearance margin.
        clearance_margin = min(obs_clearance, self.avoid_alert_distance) / max(
            1e-6,
            self.avoid_alert_distance,
        )
        reward += self.clearance_margin_reward_weight * clearance_margin

        # Anti-camping shaping: when still far from goal, moving too slowly is penalized.
        # Suppress this term when most robots are blocked to avoid forcing wall pushing.
        obj_speed = float(np.hypot(self.world.obj.vx, self.world.obj.vy))
        if curr_dist > self.low_speed_far_goal_radius and obj_speed < self.low_speed_threshold:
            speed_deficit = 1.0 - obj_speed / max(1e-6, self.low_speed_threshold)
            mobility_factor = float(np.clip(1.0 - blocked_ratio, 0.0, 1.0))
            reward -= (
                self.low_speed_penalty_weight
                * (max(0.0, speed_deficit) ** 2)
                * mobility_factor
            )

        # Positive shaping for explicit correction behavior.
        blocked_delta = self._prev_blocked_ratio - blocked_ratio
        reward += self.unblock_reward_weight * blocked_delta
        clearance_delta = obs_clearance - self._prev_obs_clearance
        if clearance_delta > 0.0:
            reward += self.clearance_improve_reward_weight * (
                clearance_delta / max(1.0, self.clearance_safe_distance)
            )
        self._prev_blocked_ratio = blocked_ratio
        self._prev_obs_clearance = obs_clearance

        terminated = bool(self.world.success or stuck_failure)
        truncated = bool(self._step_count >= self.max_steps)

        if self.world.success:
            reward += self.success_bonus
        if stuck_failure:
            reward -= self.stuck_penalty
        if truncated and not self.world.success:
            reward -= self.timeout_penalty

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
        base = 17
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

        wp = self._current_waypoint()
        wp_rel = np.array([wp[0] - obj.x, wp[1] - obj.y], dtype=np.float32)
        wp_dist = float(np.linalg.norm(wp_rel))
        route_dir = self._route_direction_unit()
        vel = np.array([obj.vx, obj.vy], dtype=np.float32)
        vel_norm = float(np.linalg.norm(vel))
        align = 0.0
        if vel_norm > 1e-6:
            align = float(np.dot(vel, route_dir) / (vel_norm + 1e-8))

        vec.extend(
            [
                wp_rel[0] / w,
                wp_rel[1] / h,
                wp_dist / diag,
                align,
                self._distance_to_route() / diag,
            ]
        )

        blocked_ratio = (
            sum(float(getattr(r, "blocked", False)) for r in self.world.robots)
            / max(1, len(self.world.robots))
        )
        vec.extend(
            [
                self._distance_object_to_obstacle_or_wall() / diag,
                blocked_ratio,
            ]
        )

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
            "distance_to_waypoint": self._distance_to_current_waypoint(),
            "distance_to_route": self._distance_to_route(),
            "time": self.world.t,
            "success": self.world.success,
            "invalid_state": self.world.invalid_state,
            "stuck_steps": self._stuck_steps,
            "stuck_failure": stuck_failure,
            "step": self._step_count,
        }

    def _apply_stage3_layout(self, config: SceneConfig) -> None:
        """Apply stage-3 layout with stronger geometric diversity and mismatch."""
        rng = np.random.default_rng(int(getattr(config, "seed", 42)))
        w = float(config.width)
        h = float(config.height)

        wall_x = int(rng.uniform(0.42 * w, 0.58 * w))
        wall_w = int(self.stage3_wall_width)
        gap_h = float(np.clip(self.stage3_gap_height, 160.0, h * 0.45))
        gap_cy = float(rng.uniform(0.30 * h, 0.70 * h))
        gap_top = int(np.clip(gap_cy - 0.5 * gap_h, 80, h - 80))
        gap_bottom = int(np.clip(gap_cy + 0.5 * gap_h, 80, h - 80))

        top_h = max(0, gap_top)
        bot_y = min(int(h), gap_bottom)
        bot_h = max(0, int(h) - bot_y)

        obstacles: list[tuple[int, int, int, int]] = []
        if top_h > 0:
            obstacles.append((wall_x, 0, wall_w, top_h))
        if bot_h > 0:
            obstacles.append((wall_x, bot_y, wall_w, bot_h))

        config.obstacles = obstacles

        # Force start/goal to opposite sides of barrier and opposite sides of
        # the gap center in y, so straight-line solutions are much less common.
        config.cargo_x = float(rng.uniform(0.12 * w, 0.28 * w))
        config.goal_x = float(rng.uniform(0.72 * w, 0.88 * w))

        y_sep = float(rng.uniform(0.28 * gap_h, 0.48 * gap_h))
        if rng.random() < 0.5:
            cargo_y = gap_cy - y_sep
            goal_y = gap_cy + y_sep
        else:
            cargo_y = gap_cy + y_sep
            goal_y = gap_cy - y_sep

        jitter = float(rng.uniform(12.0, 36.0))
        cargo_y += float(rng.uniform(-jitter, jitter))
        goal_y += float(rng.uniform(-jitter, jitter))

        config.cargo_y = float(np.clip(cargo_y, 80, h - 80))
        config.goal_y = float(np.clip(goal_y, 80, h - 80))

    def _build_route_guidance(self) -> None:
        obj = self.world.obj
        self._route_waypoints = plan_path(
            width=self.world.width,
            height=self.world.height,
            obstacles=self.world.obstacles,
            start_xy=(obj.x, obj.y),
            goal_xy=(obj.goal_x, obj.goal_y),
            cell_size=self.route_cell_size,
            inflate_margin=self.route_inflate_margin,
        )
        if len(self._route_waypoints) == 0:
            self._route_waypoints = [(obj.goal_x, obj.goal_y)]
        self._route_idx = 0

    def _current_waypoint(self) -> tuple[float, float]:
        if self._route_idx >= len(self._route_waypoints):
            obj = self.world.obj
            return (obj.goal_x, obj.goal_y)
        return self._route_waypoints[self._route_idx]

    def _distance_to_current_waypoint(self) -> float:
        obj = self.world.obj
        wp = self._current_waypoint()
        return float(np.hypot(obj.x - wp[0], obj.y - wp[1]))

    def _advance_waypoint_if_reached(self) -> bool:
        advanced = False
        while self._route_idx < len(self._route_waypoints):
            if self._distance_to_current_waypoint() > self.route_waypoint_tolerance:
                break
            self._route_idx += 1
            advanced = True
        return advanced

    def _route_direction_unit(self) -> np.ndarray:
        obj = self.world.obj
        wp = self._current_waypoint()
        d = np.array([wp[0] - obj.x, wp[1] - obj.y], dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return d / n

    def _route_base_wrench(self) -> tuple[float, float, float]:
        """Compute planner-guided object-level wrench command."""
        route_dir = self._route_direction_unit()
        repel_dir, obs_clearance = self._nearest_obstacle_repulsion()
        avoid_intensity = float(
            np.clip(
                1.0 - obs_clearance / max(1.0, self.avoid_alert_distance),
                0.0,
                1.0,
            )
        )
        steer_dir = route_dir.copy()
        if float(np.hypot(repel_dir[0], repel_dir[1])) > 1e-8 and avoid_intensity > 0.0:
            steer_dir = route_dir + repel_dir * (self.avoid_blend_gain * (avoid_intensity**2))
            n = float(np.hypot(steer_dir[0], steer_dir[1]))
            if n > 1e-8:
                steer_dir = steer_dir / n
            else:
                steer_dir = route_dir
        obj = self.world.obj

        desired_vx = float(steer_dir[0] * self.force_max * self.route_guidance_gain)
        desired_vy = float(steer_dir[1] * self.force_max * self.route_guidance_gain)
        base_fx = desired_vx - self.route_linear_damping_gain * obj.vx
        base_fy = desired_vy - self.route_linear_damping_gain * obj.vy

        if float(np.hypot(steer_dir[0], steer_dir[1])) > 1e-6:
            theta_ref = float(np.arctan2(steer_dir[1], steer_dir[0]))
        else:
            theta_ref = float(obj.theta)
        theta_err = (theta_ref - obj.theta + np.pi) % (2 * np.pi) - np.pi
        base_tau = (
            self.route_torque_gain * theta_err
            - self.route_angular_damping_gain * obj.omega
        )

        if avoid_intensity > 0.0 and float(np.hypot(repel_dir[0], repel_dir[1])) > 1e-8:
            tangent = np.array([-repel_dir[1], repel_dir[0]], dtype=np.float32)
            turn_sign = float(np.sign(np.dot(route_dir, tangent)))
            if abs(turn_sign) < 1e-6:
                turn_sign = 1.0
            base_tau += self.avoid_torque_gain * turn_sign * (avoid_intensity**2)

        # Collision recovery: when close to walls/obstacles or repeatedly stuck,
        # add an outward push and a turning bias instead of sliding along walls.
        blocked_ratio = (
            sum(float(getattr(r, "blocked", False)) for r in self.world.robots)
            / max(1, len(self.world.robots))
        )
        repel_dir, obs_clearance = self._nearest_obstacle_repulsion()
        in_recovery = bool(
            blocked_ratio > 0.0
            or self._stuck_steps >= self.recovery_stuck_steps
            or obs_clearance < self.clearance_safe_distance
        )
        if in_recovery and float(np.hypot(repel_dir[0], repel_dir[1])) > 1e-8:
            intensity = 1.0 - obs_clearance / max(1.0, self.clearance_safe_distance)
            intensity = float(np.clip(intensity, 0.0, 1.0))
            push = self.recovery_push_gain * self.force_max * (0.45 + intensity)
            base_fx += float(repel_dir[0] * push)
            base_fy += float(repel_dir[1] * push)

            tangent = np.array([-repel_dir[1], repel_dir[0]], dtype=np.float32)
            turn_sign = float(np.sign(np.dot(route_dir, tangent)))
            if abs(turn_sign) < 1e-6:
                turn_sign = 1.0
            base_tau += self.recovery_torque_gain * turn_sign * (0.35 + intensity)

        return float(base_fx), float(base_fy), float(base_tau)

    def _nearest_obstacle_repulsion(self) -> tuple[np.ndarray, float]:
        """Return repulsion unit direction and nearest clearance to wall/obstacle."""
        best_d = float("inf")
        best_dir = np.zeros(2, dtype=np.float32)
        obj_center = np.array([self.world.obj.x, self.world.obj.y], dtype=np.float32)

        for poly in self.world.obj.get_parts_world():
            for vx, vy in poly:
                candidates: list[tuple[float, np.ndarray]] = [
                    (float(vx), np.array([1.0, 0.0], dtype=np.float32)),
                    (float(self.world.width - vx), np.array([-1.0, 0.0], dtype=np.float32)),
                    (float(vy), np.array([0.0, 1.0], dtype=np.float32)),
                    (float(self.world.height - vy), np.array([0.0, -1.0], dtype=np.float32)),
                ]

                for ox, oy, ow, oh in self.world.obstacles:
                    cx = float(np.clip(vx, ox, ox + ow))
                    cy = float(np.clip(vy, oy, oy + oh))
                    dv = np.array([float(vx - cx), float(vy - cy)], dtype=np.float32)
                    d = float(np.hypot(dv[0], dv[1]))
                    if d < 1e-6:
                        occ = np.array([ox + 0.5 * ow, oy + 0.5 * oh], dtype=np.float32)
                        dv = obj_center - occ
                        d = float(np.hypot(dv[0], dv[1]))
                        if d < 1e-6:
                            dv = np.array([1.0, 0.0], dtype=np.float32)
                            d = 1.0
                    candidates.append((d, dv / max(1e-6, d)))

                for d, n in candidates:
                    if d < best_d:
                        best_d = float(d)
                        best_dir = n.astype(np.float32)

        if not np.isfinite(best_d):
            return np.zeros(2, dtype=np.float32), 1e6
        nrm = float(np.hypot(best_dir[0], best_dir[1]))
        if nrm < 1e-8:
            return np.zeros(2, dtype=np.float32), float(best_d)
        return (best_dir / nrm).astype(np.float32), float(best_d)

    def _heading_abs_error(self) -> float:
        """Absolute heading error between cargo orientation and route direction."""
        route_dir = self._route_direction_unit()
        if float(np.hypot(route_dir[0], route_dir[1])) < 1e-8:
            return 0.0
        theta_ref = float(np.arctan2(route_dir[1], route_dir[0]))
        theta_err = (theta_ref - self.world.obj.theta + np.pi) % (2 * np.pi) - np.pi
        return float(abs(theta_err))

    def _distance_object_to_obstacle_or_wall(self) -> float:
        """Approximate minimum clearance from cargo footprint to obstacles/walls."""
        min_dist = float("inf")
        for poly in self.world.obj.get_parts_world():
            for vx, vy in poly:
                # Distance to world boundary.
                wall_d = min(vx, self.world.width - vx, vy, self.world.height - vy)
                min_dist = min(min_dist, float(wall_d))

                # Distance to obstacle AABBs.
                for ox, oy, ow, oh in self.world.obstacles:
                    cx = float(np.clip(vx, ox, ox + ow))
                    cy = float(np.clip(vy, oy, oy + oh))
                    d = float(np.hypot(vx - cx, vy - cy))
                    min_dist = min(min_dist, d)

        if not np.isfinite(min_dist):
            return 1e6
        return max(0.0, float(min_dist))

    def _distance_to_route(self) -> float:
        if len(self._route_waypoints) == 0:
            return 0.0
        obj = self.world.obj
        p = np.array([obj.x, obj.y], dtype=np.float32)
        pts = np.asarray(self._route_waypoints, dtype=np.float32)
        dists = np.linalg.norm(pts - p, axis=1)
        return float(np.min(dists))
