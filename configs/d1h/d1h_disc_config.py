import torch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDisc(D1HMoEBase):
    def _update_terrain_curriculum(self, env_ids):
        """Use a stair-specific curriculum instead of pure full-terrain distance."""
        if not self.init_done:
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up_distance = getattr(self.cfg.terrain, "curriculum_move_up_distance", 3.0)
        move_down_expected_factor = getattr(self.cfg.terrain, "curriculum_move_down_expected_factor", 0.30)
        move_down_min_distance = getattr(self.cfg.terrain, "curriculum_move_down_min_distance", 1.0)

        expected_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
        move_up = distance > move_up_distance
        move_down_distance = torch.clamp(expected_distance * move_down_expected_factor, min=move_down_min_distance)
        move_down = (distance < move_down_distance) & ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
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

    def _get_stair_reward_gate(self):
        """Gate stair rewards to upright, non-colliding attempts."""
        upright_score = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        upright_min = getattr(self.cfg.rewards, "stair_gate_upright_min", 0.70)
        upright_gate = torch.clamp((upright_score - upright_min) / (1.0 - upright_min), 0.0, 1.0)

        base_height = self._get_base_heights()
        min_height = getattr(self.cfg.rewards, "stair_gate_base_height_min", 0.28)
        full_height = getattr(self.cfg.rewards, "stair_gate_base_height_full", 0.40)
        height_gate = torch.clamp((base_height - min_height) / max(full_height - min_height, 1e-6), 0.0, 1.0)

        contact_gate = torch.ones(self.num_envs, device=self.device)
        if hasattr(self, "contact_forces") and hasattr(self, "penalised_contact_indices"):
            if len(self.penalised_contact_indices) > 0:
                force_threshold = getattr(self.cfg.rewards, "stair_gate_bad_contact_force", 5.0)
                bad_contact = torch.any(
                    torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > force_threshold,
                    dim=1,
                )
                contact_gate = (~bad_contact).float()

        return upright_gate * height_gate * contact_gate

    def _get_foot_contact_norm(self):
        if not hasattr(self, "contact_forces") or not hasattr(self, "feet_indices"):
            return None
        return torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

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

        # Mean progress keeps both wheels involved; max progress gives early signal
        # when only one side has discovered the lift motion.
        per_foot_progress = torch.clamp(positive_clearance / target_clearance, 0.0, 1.0)
        mean_progress = per_foot_progress.mean(dim=1)
        max_progress = per_foot_progress.max(dim=1).values
        min_clearance = positive_clearance.min(dim=1).values
        clearance_error = torch.clamp(target_clearance.squeeze(1) - min_clearance, min=0.0)
        both_clear_bonus = torch.exp(-torch.square(clearance_error / sigma))
        reward = 0.55 * mean_progress + 0.30 * max_progress + 0.15 * both_clear_bonus

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
        reward = 0.70 * per_foot_lift.mean(dim=1) + 0.30 * per_foot_lift.max(dim=1).values

        return reward * active.float() * self._get_stair_reward_gate()

    def _reward_step_pre_lift(self):
        """Reward lifting before impact, not being lifted by the stair edge."""

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

    def _reward_step_bump(self):
        """Penalize hitting the stair with low clearance and high foot contact."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        contact_norm = self._get_foot_contact_norm()
        if contact_norm is None:
            return zeros

        margin = getattr(self.cfg.rewards, "step_bump_clearance_margin", 0.04)
        force_threshold = getattr(self.cfg.rewards, "step_bump_force_threshold", 500.0)
        force_scale = getattr(self.cfg.rewards, "step_bump_force_scale", 800.0)

        target_clearance = torch.clamp(obstacle_height + margin, min=0.04)
        max_clearance = torch.clamp(foot_clearance.max(dim=1).values, min=0.0)
        low_clearance = torch.clamp(
            (target_clearance - max_clearance) / torch.clamp(target_clearance, min=0.04),
            0.0,
            1.0,
        )
        max_contact = contact_norm.max(dim=1).values
        impact = torch.clamp((max_contact - force_threshold) / max(force_scale, 1e-6), 0.0, 1.0)

        return low_clearance * impact * active.float() * self._get_stair_reward_gate()

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
        # Keep command curriculum, but make the initial task a straight stair climb.
        curriculum = True
        max_curriculum_x = 0.8
        max_curriculum_x_back = 0.0
        max_curriculum_y = 0.0
        max_curriculum_yaw = 0.0
        resampling_time = 10.0
        heading_command = True
        zero_command_ratio = 0.0
        startup_freeze_time = 0.0

        class ranges:
            # Values below 0.2 are zeroed by the base command sampler.
            lin_vel_x = [0.25, 0.45]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class terrain(D1HMoEBaseCfg.terrain):
        # Terrain curriculum still provides gradual generalization across levels.
        curriculum = True
        max_init_terrain_level = 1
        # Terrain order: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles].
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        step_height = [0.04, 0.20]
        step_width_range = [0.30, 0.55]
        slope = [0.0, 0.08]
        slope_treshold = 0.3
        curriculum_move_up_distance = 3.0
        curriculum_move_down_expected_factor = 0.30
        curriculum_move_down_min_distance = 1.0

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
        step_stall_min_speed = 0.12
        step_pre_lift_min_height = 0.06
        step_pre_lift_margin = 0.07
        step_pre_lift_sigma = 0.05
        step_pre_lift_max_contact_force = 35.0
        step_bump_clearance_margin = 0.04
        step_bump_force_threshold = 500.0
        step_bump_force_scale = 800.0
        stair_gate_upright_min = 0.70
        stair_gate_base_height_min = 0.28
        stair_gate_base_height_full = 0.40
        stair_gate_bad_contact_force = 5.0

        class scales(D1HMoEBaseCfg.rewards.scales):
            # Disabled legacy aggregate tracker; this expert uses axis-specific tracking below.
            tracking_lin_vel = 0.0
            # Keep forward tracking useful but below the stair-specific rewards.
            tracking_lin_vel_x = 14.0
            # Reward holding the lateral velocity near zero.
            tracking_lin_vel_y = 7.0
            # Mild yaw stabilization; heading is fixed to zero.
            tracking_ang_vel = 12.0
            heading = -5.0

            # Stability guardrails. They should prevent garbage motion, not dominate climbing.
            orientation = -16.0
            upward = 2.0
            ang_vel_xy = -0.12
            base_height = -3.0
            lin_vel_z = -0.5

            # Failure/contact penalties.
            termination = -600.0
            collision = -14.0
            collision_hard = -35.0
            collision_head = 0.0

            # Effort and smoothness are intentionally light during skill acquisition.
            torques = 0.0
            powers = -2.0e-5
            dof_acc = -1.5e-7
            action_rate = -0.045
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

            # Body geometry priors are weak; strong values can block the step-up posture.
            body_pos_to_feet_x = 0.2
            body_feet_distance_x = -0.3
            body_feet_distance_y = -1.0
            body_symmetry_y = 0.1
            body_symmetry_z = 0.0

            # Main stair-up objective.
            step_clearance = 32.0
            step_lift = 20.0
            step_pre_lift = 16.0
            step_progress = 12.0
            step_stall = -18.0
            step_bump = -8.0

    class normalization(D1HMoEBaseCfg.normalization):
        # Keep exploration broad enough for stair actions, but prevent unbounded
        # residual samples from destroying the frozen base policy.
        clip_actions = 2.5

    class costs(D1HMoEBaseCfg.costs):
        class scales(D1HMoEBaseCfg.costs.scales):
            # Keep constraints visible, but less suppressive than the base setting.
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1

        class d_values(D1HMoEBaseCfg.costs.d_values):
            # Keep the original zero-budget interpretation explicit.
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class algorithm(D1HMoEBaseCfgPPO.algorithm):
        entropy_coef = 0.0
        residual_l2_coef = 0.035
        learning_rate = 1.0e-3
        schedule = "adaptive"
        desired_kl = 0.02
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
