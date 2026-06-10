import torch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDisc(D1HMoEBase):
    def _init_buffers(self):
        super()._init_buffers()
        self.step_contact_timer = torch.zeros(self.num_envs, device=self.device)
        self.step_jam_time = torch.zeros(self.num_envs, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.step_contact_timer[env_ids] = 0.0
        self.step_jam_time[env_ids] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_step_contact_state()

    def _update_terrain_curriculum(self, env_ids):
        """Use a stair-specific curriculum instead of pure full-terrain distance."""
        if not self.init_done:
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up_distance = getattr(self.cfg.terrain, "curriculum_move_up_distance", 3.0)
        move_down_expected_factor = getattr(self.cfg.terrain, "curriculum_move_down_expected_factor", 0.30)
        move_down_min_distance = getattr(self.cfg.terrain, "curriculum_move_down_min_distance", 1.0)
        success_reward_threshold = getattr(self.cfg.terrain, "curriculum_success_reward_threshold", 0.85)
        success_down_threshold = getattr(self.cfg.terrain, "curriculum_success_down_threshold", 0.15)
        success_min_distance = getattr(self.cfg.terrain, "curriculum_success_min_distance", 1.8)
        success_min_episode_time = getattr(self.cfg.terrain, "curriculum_success_min_episode_time", 8.0)
        max_allowed_level = min(
            getattr(self.cfg.terrain, "curriculum_max_terrain_level", self.max_terrain_level - 1),
            self.max_terrain_level - 1,
        )

        expected_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
        episode_time = self.episode_length_buf[env_ids].float() * self.dt
        if hasattr(self, "episode_sums") and "step_success" in self.episode_sums:
            step_success = self.episode_sums["step_success"][env_ids] / self.max_episode_length_s
        else:
            step_success = torch.zeros_like(distance)

        move_up_by_success = (
            (step_success > success_reward_threshold)
            & (distance > success_min_distance)
            & (episode_time > success_min_episode_time)
        )
        move_up_by_distance = distance > move_up_distance
        move_up = move_up_by_success | move_up_by_distance
        move_down_distance = torch.clamp(expected_distance * move_down_expected_factor, min=move_down_min_distance)
        move_down = (step_success < success_down_threshold) & (distance < move_down_distance) & ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, max_allowed_level)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_step_lift_context(self):
        zeros = torch.zeros(self.num_envs, device=self.device)

        if not getattr(self.cfg.terrain, "measure_heights", False):
            return None
        if not hasattr(self, "measured_heights") or not torch.is_tensor(self.measured_heights):
            return None
        if self.measured_heights.ndim != 2:
            return None
        if not hasattr(self, "height_points") or not hasattr(self, "commands"):
            return None

        points = self.height_points[0]
        px = points[:, 0]
        py = points[:, 1]

        front_x_min = getattr(self.cfg.rewards, "step_clearance_front_x_min", 0.20)
        front_x_max = getattr(self.cfg.rewards, "step_clearance_front_x_max", 0.80)
        center_x_abs = getattr(self.cfg.rewards, "step_clearance_center_x_abs", 0.15)
        y_abs = getattr(self.cfg.rewards, "step_clearance_y_abs", 0.35)

        front_mask = (
            (px >= front_x_min)
            & (px <= front_x_max)
            & (torch.abs(py) <= y_abs)
        )
        center_mask = (
            (torch.abs(px) <= center_x_abs)
            & (torch.abs(py) <= y_abs)
        )

        if front_mask.sum().item() == 0 or center_mask.sum().item() == 0:
            return None

        front_height = self.measured_heights[:, front_mask].max(dim=1).values
        center_height = self.measured_heights[:, center_mask].mean(dim=1)

        max_obstacle_height = getattr(self.cfg.rewards, "step_clearance_max_obstacle_height", 0.20)
        obstacle_height = torch.clamp(front_height - center_height, min=0.0, max=max_obstacle_height)

        trigger_height = getattr(self.cfg.rewards, "step_clearance_trigger_height", 0.03)
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        active = (obstacle_height > trigger_height) & (self.commands[:, 0] > min_cmd_x)

        if hasattr(self, "rigid_body_states") and hasattr(self, "feet_indices"):
            foot_z = self.rigid_body_states[:, self.feet_indices, 2]
            foot_clearance = foot_z - center_height.unsqueeze(1)
        elif hasattr(self, "feet_body_frame_height") and torch.is_tensor(self.feet_body_frame_height):
            foot_clearance = self.feet_body_frame_height
        else:
            return None

        return active, obstacle_height, foot_clearance, zeros

    def _get_stair_height_context(self):
        """Estimate whether the base has actually migrated onto the higher stair."""
        zeros = torch.zeros(self.num_envs, device=self.device)

        if not getattr(self.cfg.terrain, "measure_heights", False):
            return None
        if not hasattr(self, "measured_heights") or not torch.is_tensor(self.measured_heights):
            return None
        if self.measured_heights.ndim != 2:
            return None
        if not hasattr(self, "height_points") or not hasattr(self, "commands"):
            return None

        points = self.height_points[0]
        px = points[:, 0]
        py = points[:, 1]

        front_x_min = getattr(self.cfg.rewards, "step_clearance_front_x_min", 0.20)
        front_x_max = getattr(self.cfg.rewards, "step_clearance_front_x_max", 0.80)
        center_x_abs = getattr(self.cfg.rewards, "step_clearance_center_x_abs", 0.15)
        rear_x_min = getattr(self.cfg.rewards, "step_success_rear_x_min", -0.75)
        rear_x_max = getattr(self.cfg.rewards, "step_success_rear_x_max", -0.20)
        y_abs = getattr(self.cfg.rewards, "step_clearance_y_abs", 0.35)

        front_mask = (
            (px >= front_x_min)
            & (px <= front_x_max)
            & (torch.abs(py) <= y_abs)
        )
        center_mask = (
            (torch.abs(px) <= center_x_abs)
            & (torch.abs(py) <= y_abs)
        )
        rear_mask = (
            (px >= rear_x_min)
            & (px <= rear_x_max)
            & (torch.abs(py) <= y_abs)
        )

        if front_mask.sum().item() == 0 or center_mask.sum().item() == 0 or rear_mask.sum().item() == 0:
            return None

        front_height = self.measured_heights[:, front_mask].max(dim=1).values
        center_height = self.measured_heights[:, center_mask].mean(dim=1)
        rear_height = self.measured_heights[:, rear_mask].mean(dim=1)

        max_obstacle_height = getattr(self.cfg.rewards, "step_clearance_max_obstacle_height", 0.20)
        obstacle_height = torch.clamp(front_height - center_height, min=0.0, max=max_obstacle_height)
        climbed_height = torch.clamp(center_height - rear_height, min=0.0, max=max_obstacle_height)

        trigger_height = getattr(self.cfg.rewards, "step_clearance_trigger_height", 0.03)
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        active = ((obstacle_height > trigger_height) | (climbed_height > trigger_height)) & (self.commands[:, 0] > min_cmd_x)

        return active, obstacle_height, climbed_height, center_height, rear_height, zeros

    def _get_stair_posture_gate(self):
        """Gate stair shaping to attempts that are still physically meaningful."""
        upright_score = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        upright_min = getattr(self.cfg.rewards, "stair_gate_upright_min", 0.70)
        upright_gate = torch.clamp((upright_score - upright_min) / (1.0 - upright_min), 0.0, 1.0)

        base_height = self._get_base_heights()
        min_height = getattr(self.cfg.rewards, "stair_gate_base_height_min", 0.28)
        full_height = getattr(self.cfg.rewards, "stair_gate_base_height_full", 0.40)
        height_gate = torch.clamp((base_height - min_height) / max(full_height - min_height, 1e-6), 0.0, 1.0)

        return upright_gate * height_gate

    def _get_stair_reward_gate(self):
        """Gate positive stair rewards to upright, non-colliding attempts."""
        posture_gate = self._get_stair_posture_gate()

        contact_gate = torch.ones(self.num_envs, device=self.device)
        if hasattr(self, "contact_forces") and hasattr(self, "penalised_contact_indices"):
            if len(self.penalised_contact_indices) > 0:
                force_threshold = getattr(self.cfg.rewards, "stair_gate_bad_contact_force", 5.0)
                bad_contact = torch.any(
                    torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > force_threshold,
                    dim=1,
                )
                contact_gate = (~bad_contact).float()

        return posture_gate * contact_gate

    def _get_foot_contact_norm(self):
        if hasattr(self, "force_sensor_tensor") and torch.is_tensor(self.force_sensor_tensor):
            return torch.norm(self.force_sensor_tensor[:, :, :3], dim=-1)
        if not hasattr(self, "contact_forces") or not hasattr(self, "feet_indices"):
            return None
        return torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

    def _get_step_blocking_signal(self):
        context = self._get_step_lift_context()
        contact_norm = self._get_foot_contact_norm()
        if context is None or contact_norm is None:
            return None

        active, obstacle_height, foot_clearance, zeros = context
        margin = getattr(self.cfg.rewards, "step_block_clearance_margin", 0.04)
        target_clearance = torch.clamp(obstacle_height + margin, min=0.04)
        max_clearance = torch.clamp(foot_clearance.max(dim=1).values, min=0.0)
        low_clearance = torch.clamp(
            (target_clearance - max_clearance) / torch.clamp(target_clearance, min=0.04),
            0.0,
            1.0,
        )
        max_contact = contact_norm.max(dim=1).values
        return active, obstacle_height, foot_clearance, max_contact, low_clearance, zeros

    def _update_step_contact_state(self):
        signal = self._get_step_blocking_signal()
        if signal is None:
            self.step_contact_timer.zero_()
            self.step_jam_time.zero_()
            return

        active, _, _, max_contact, low_clearance, _ = signal
        contact_force = getattr(self.cfg.rewards, "step_contact_force_threshold", 80.0)
        jam_force = getattr(self.cfg.rewards, "step_jam_force_threshold", 280.0)
        jam_clearance = getattr(self.cfg.rewards, "step_jam_clearance_ratio", 0.45)
        jam_speed = getattr(self.cfg.rewards, "step_jam_min_speed", 0.08)
        contact_memory = getattr(self.cfg.rewards, "step_contact_memory_time", 0.25)

        step_contact = active & (max_contact > contact_force)
        jammed = active & (max_contact > jam_force) & (low_clearance > jam_clearance) & (self.base_lin_vel[:, 0] < jam_speed)

        self.step_contact_timer = torch.where(
            step_contact,
            torch.full_like(self.step_contact_timer, contact_memory),
            torch.clamp(self.step_contact_timer - self.dt, min=0.0),
        )
        self.step_jam_time = torch.where(jammed, self.step_jam_time + self.dt, torch.zeros_like(self.step_jam_time))

    def _reward_step_clearance(self):
        """Reward useful wheel/foot clearance when a front up-step is detected."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        clearance_margin = getattr(self.cfg.rewards, "step_clearance_margin", 0.04)
        sigma = getattr(self.cfg.rewards, "step_clearance_sigma", 0.04)

        target_clearance = torch.clamp(obstacle_height + clearance_margin, min=0.04).unsqueeze(1)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)

        per_foot_progress = torch.clamp(positive_clearance / target_clearance, 0.0, 1.0)
        lead_progress = per_foot_progress.max(dim=1).values
        support_progress = per_foot_progress.min(dim=1).values
        single_leg_lead = lead_progress * (1.0 - support_progress)
        reward = 0.70 * lead_progress + 0.30 * single_leg_lead

        return reward * active.float() * self._get_stair_reward_gate()

    def _reward_step_lift(self):
        """Reward reaching a useful lift height while the front step is active."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_lift_min_height", 0.05)
        margin = getattr(self.cfg.rewards, "step_lift_margin", 0.06)
        sigma = getattr(self.cfg.rewards, "step_lift_sigma", 0.05)
        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)

        lift_error = torch.clamp(target_lift.unsqueeze(1) - positive_clearance, min=0.0)
        per_foot_lift = torch.exp(-torch.square(lift_error / sigma))
        lead_lift = per_foot_lift.max(dim=1).values
        support_lift = per_foot_lift.min(dim=1).values
        single_leg_lift = torch.clamp(lead_lift - support_lift, min=0.0)
        reward = 0.75 * lead_lift + 0.25 * single_leg_lift

        return reward * active.float() * self._get_stair_reward_gate()

    def _reward_step_pre_lift(self):
        """Small bonus for using height scan to lift before contact."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        contact_norm = self._get_foot_contact_norm()
        if contact_norm is None:
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_pre_lift_min_height", 0.06)
        margin = getattr(self.cfg.rewards, "step_pre_lift_margin", 0.07)
        sigma = getattr(self.cfg.rewards, "step_pre_lift_sigma", 0.05)
        max_lift_contact = getattr(self.cfg.rewards, "step_pre_lift_max_contact_force", 35.0)

        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)
        lift_error = torch.clamp(target_lift.unsqueeze(1) - positive_clearance, min=0.0)
        lift_score = torch.exp(-torch.square(lift_error / sigma))

        # A real pre-lift should happen with low foot contact; if the stair edge
        # is pushing the wheel up, contact force is usually high.
        low_contact_score = torch.clamp(
            (max_lift_contact - contact_norm) / max(max_lift_contact, 1e-6),
            0.0,
            1.0,
        )
        reward = (lift_score * low_contact_score).max(dim=1).values

        return reward * active.float() * self._get_stair_reward_gate()

    def _reward_step_reactive_lift(self):
        """Reward lifting and unloading shortly after contacting the stair."""

        signal = self._get_step_blocking_signal()
        if signal is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, max_contact, _, zeros = signal
        recent_contact = self.step_contact_timer > 0.0
        if not torch.any(active & recent_contact):
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_reactive_lift_min_height", 0.08)
        margin = getattr(self.cfg.rewards, "step_reactive_lift_margin", 0.07)
        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)
        per_foot_progress = torch.clamp(positive_clearance / torch.clamp(target_lift.unsqueeze(1), min=0.04), 0.0, 1.0)
        lead_progress = per_foot_progress.max(dim=1).values
        support_progress = per_foot_progress.min(dim=1).values
        single_leg_lead = lead_progress * (1.0 - support_progress)
        lift_progress = 0.75 * lead_progress + 0.25 * single_leg_lead

        unload_low = getattr(self.cfg.rewards, "step_reactive_unload_force_low", 80.0)
        unload_high = getattr(self.cfg.rewards, "step_reactive_unload_force_high", 300.0)
        unload_score = 1.0 - torch.clamp((max_contact - unload_low) / max(unload_high - unload_low, 1e-6), 0.0, 1.0)

        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        cmd_x = torch.clamp(self.commands[:, 0], min=min_cmd_x)
        forward_score = torch.clamp(self.base_lin_vel[:, 0] / cmd_x, 0.0, 1.0)

        reward = 0.70 * lift_progress + 0.05 * unload_score + 0.25 * forward_score
        return reward * active.float() * recent_contact.float() * self._get_stair_reward_gate()

    def _reward_step_bump(self):
        """Penalize sustained jamming, not the first probing contact."""

        grace_time = getattr(self.cfg.rewards, "step_jam_grace_time", 0.12)
        time_scale = getattr(self.cfg.rewards, "step_jam_time_scale", 0.20)
        jam_score = torch.clamp((self.step_jam_time - grace_time) / max(time_scale, 1e-6), 0.0, 1.0)
        return jam_score * self._get_stair_posture_gate()

    def _reward_step_progress(self):
        """Reward forward progress when a front step is detected."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        cmd_x = torch.clamp(self.commands[:, 0], min=min_cmd_x)
        progress = torch.clamp(self.base_lin_vel[:, 0] / cmd_x, 0.0, 1.0)
        return progress * active.float() * self._get_stair_reward_gate()

    def _reward_step_up(self):
        """Reward actual terrain-height migration, not just lifting a foot."""

        context = self._get_stair_height_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, climbed_height, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_height = getattr(self.cfg.rewards, "step_success_min_height", 0.025)
        height_scale = torch.clamp(torch.maximum(obstacle_height, climbed_height), min=min_height)
        up_progress = torch.clamp(climbed_height / height_scale, 0.0, 1.0)

        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        cmd_x = torch.clamp(self.commands[:, 0], min=min_cmd_x)
        forward_score = torch.clamp(self.base_lin_vel[:, 0] / cmd_x, 0.0, 1.0)

        return up_progress * (0.5 + 0.5 * forward_score) * active.float() * self._get_stair_reward_gate()

    def _reward_step_success(self):
        """Reward a completed stair transition with a dense, visible success signal."""

        context = self._get_stair_height_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, climbed_height, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_height = getattr(self.cfg.rewards, "step_success_min_height", 0.025)
        start_ratio = getattr(self.cfg.rewards, "step_success_start_ratio", 0.35)
        complete_ratio = getattr(self.cfg.rewards, "step_success_complete_ratio", 0.70)
        height_scale = torch.clamp(torch.maximum(obstacle_height, climbed_height), min=min_height)
        height_ratio = torch.clamp(climbed_height / height_scale, 0.0, 1.0)
        height_complete = torch.clamp((height_ratio - start_ratio) / max(complete_ratio - start_ratio, 1e-6), 0.0, 1.0)

        base_height = self._get_base_heights()
        min_base_height = getattr(self.cfg.rewards, "step_success_min_base_height", 0.30)
        full_base_height = getattr(self.cfg.rewards, "step_success_full_base_height", 0.40)
        base_score = torch.clamp(
            (base_height - min_base_height) / max(full_base_height - min_base_height, 1e-6),
            0.0,
            1.0,
        )

        min_speed = getattr(self.cfg.rewards, "step_success_min_speed", 0.12)
        speed_score = torch.clamp(self.base_lin_vel[:, 0] / max(min_speed, 1e-6), 0.0, 1.0)
        recovery_score = 0.5 + 0.5 * speed_score

        success_score = (0.35 * height_ratio + 0.65 * height_complete) * base_score * recovery_score
        return success_score * active.float() * self._get_stair_reward_gate()

    def _reward_step_stall(self):
        """Penalize stopping at the step edge instead of attempting to climb."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_speed = getattr(self.cfg.rewards, "step_stall_min_speed", 0.08)
        stalled = self.base_lin_vel[:, 0] < min_speed
        return (active & stalled).float() * self._get_stair_reward_gate()


class D1HMoEDiscCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        # First make the stair skill learnable; broaden this only after
        # step_success and terrain_level climb reliably.
        curriculum = False
        max_curriculum_x = 0.55
        max_curriculum_x_back = 0.0
        max_curriculum_y = 0.0
        max_curriculum_yaw = 0.0
        resampling_time = 10.0
        heading_command = True
        zero_command_ratio = 0.0
        startup_freeze_time = 0.0

        class ranges:
            # Values below 0.2 are zeroed by the base command sampler.
            lin_vel_x = [0.28, 0.38]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class terrain(D1HMoEBaseCfg.terrain):
        # Stage 1: only clean up-stairs, starting at the easiest row.
        curriculum = True
        max_init_terrain_level = 0
        # Terrain order: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles].
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        step_height = [0.035, 0.11]
        step_width_range = [0.40, 0.55]
        slope = [0.0, 0.02]
        slope_treshold = 0.3
        curriculum_move_up_distance = 4.5
        curriculum_move_down_expected_factor = 0.25
        curriculum_move_down_min_distance = 0.8
        curriculum_success_reward_threshold = 0.85
        curriculum_success_down_threshold = 0.15
        curriculum_success_min_distance = 1.8
        curriculum_success_min_episode_time = 8.0
        curriculum_max_terrain_level = 5

    class domain_rand(D1HMoEBaseCfg.domain_rand):
        # Stair-up is a fine contact skill. Remove early noise sources that make
        # residual credit assignment look random.
        randomize_friction = True
        friction_range = [0.8, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 0.0]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        randomize_base_com = False
        added_com_range = [0.0, 0.0]
        push_robots = False
        disturbance = False
        randomize_motor = False
        motor_strength_range = [1.0, 1.0]
        randomize_lag_timesteps = False
        lag_timesteps = 2

    class rewards(D1HMoEBaseCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.07
        distance_sigma = 0.08
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.98
        soft_torque_limit = 0.98
        base_height_target = 0.45
        base_height_scale = 0.05
        base_height_deadband = 0.01

        # Front height scan window used only by the stair rewards.
        step_clearance_front_x_min = 0.05
        step_clearance_front_x_max = 0.75
        step_clearance_center_x_abs = 0.15
        step_clearance_y_abs = 0.45

        # Clearance target = detected step height + margin.
        step_clearance_trigger_height = 0.025
        step_clearance_margin = 0.08
        step_clearance_max_obstacle_height = 0.24
        step_clearance_sigma = 0.08
        step_clearance_min_cmd_x = 0.08
        step_lift_min_height = 0.08
        step_lift_margin = 0.09
        step_lift_sigma = 0.07
        step_stall_min_speed = 0.08
        step_pre_lift_min_height = 0.08
        step_pre_lift_margin = 0.09
        step_pre_lift_sigma = 0.05
        step_pre_lift_max_contact_force = 80.0
        step_contact_force_threshold = 80.0
        step_contact_memory_time = 0.25
        step_block_clearance_margin = 0.04
        step_reactive_lift_min_height = 0.08
        step_reactive_lift_margin = 0.07
        step_reactive_unload_force_low = 80.0
        step_reactive_unload_force_high = 300.0
        step_jam_force_threshold = 280.0
        step_jam_clearance_ratio = 0.45
        step_jam_min_speed = 0.08
        step_jam_grace_time = 0.12
        step_jam_time_scale = 0.20
        stair_gate_upright_min = 0.70
        stair_gate_base_height_min = 0.28
        stair_gate_base_height_full = 0.40
        stair_gate_bad_contact_force = 5.0
        step_success_rear_x_min = -0.75
        step_success_rear_x_max = -0.20
        step_success_min_height = 0.025
        step_success_start_ratio = 0.35
        step_success_complete_ratio = 0.70
        step_success_min_base_height = 0.30
        step_success_full_base_height = 0.40
        step_success_min_speed = 0.10

        class scales(D1HMoEBaseCfg.rewards.scales):
            # Disabled legacy aggregate tracker; this expert uses axis-specific tracking below.
            tracking_lin_vel = 0.0
            # Keep only a weak forward guardrail. Y/yaw rewards were mostly free
            # bonuses with zero commands, so they hide the real stair signal.
            tracking_lin_vel_x = 6.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 0.0
            heading = -2.0

            # Stability guardrails. They should prevent garbage motion, not dominate climbing.
            orientation = -8.0
            upward = 0.0
            ang_vel_xy = -0.05
            base_height = -1.0
            lin_vel_z = -0.3

            # Failure/contact penalties.
            termination = -400.0
            collision = -12.0
            collision_hard = -60.0
            collision_head = 0.0

            # Remove tiny regularizers that only add noise to the scalar reward.
            torques = 0.0
            powers = 0.0
            dof_acc = 0.0
            action_rate = -0.02
            action_smoothness = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0

            # Zero-command rewards are disabled because this expert never samples zero commands.
            stand_still = 0.0
            zero_base_vel = 0.0
            zero_yaw_rate = 0.0
            zero_wheel_vel = 0.0

            # Air-time is not a stair-success signal for this wheel-legged robot.
            feet_air_time = 0.0
            feet_contact_forces = 0.0
            feet_stumble = 0.0
            stumble = 0.0
            no_jump = 0.0

            # Disable weak geometry priors until the stair skill exists.
            body_pos_to_feet_x = 0.0
            body_feet_distance_x = 0.0
            body_feet_distance_y = 0.0
            body_symmetry_y = 0.0
            body_symmetry_z = 0.0

            # Main stair-up objective. Clearance/lift remain auxiliary; the
            # curriculum follows step_success.
            step_clearance = 3.0
            step_lift = 3.0
            step_pre_lift = 0.0
            step_reactive_lift = 0.0
            step_progress = 0.0
            step_up = 20.0
            step_success = 120.0
            step_stall = -4.0
            step_bump = -80.0

    class normalization(D1HMoEBaseCfg.normalization):
        # Keep exploration broad enough for stair actions, but prevent unbounded
        # residual samples from destroying the frozen base policy.
        clip_actions = 1.3

    class costs(D1HMoEBaseCfg.costs):
        class scales(D1HMoEBaseCfg.costs.scales):
            # Stage 1 uses reward/termination safety only. These costs were
            # near-zero but still drove the constrained loss.
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

        class d_values(D1HMoEBaseCfg.costs.d_values):
            # Keep the original zero-budget interpretation explicit.
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class algorithm(D1HMoEBaseCfgPPO.algorithm):
        entropy_coef = 0.001
        residual_l2_coef = 0.25
        learning_rate = 3.0e-4
        learning_rate_min = 5.0e-5
        learning_rate_max = 6.0e-3
        schedule = "adaptive"
        desired_kl = 0.01
        gamma = 0.995
        lam = 0.95
        clip_param = 0.2
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        value_loss_coef = 1.0
        cost_value_loss_coef = 0.05
        cost_viol_loss_coef = 0.05

    class policy(D1HMoEBaseCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        barlow_actor_hidden_dims = [512, 256, 128]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 0.45

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = "d1h_moe_disc"
        max_iterations = 20000
        num_steps_per_env = 32
        save_interval = 200
