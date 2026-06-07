import torch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDisc(D1HMoEBase):
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

    def _reward_step_clearance(self):
        """
        前方存在上台阶高度差时，鼓励轮/足抬高。

        说明：
        1. 这个 reward 使用 measured_heights，属于训练时特权信息。
        2. measured_heights 不进入 actor 输入，只用于 reward 计算。
        3. 只有当前方地形明显高于当前地面，并且命令为向前走时，这个 reward 才激活。
        """

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        clearance_margin = getattr(self.cfg.rewards, "step_clearance_margin", 0.04)
        sigma = getattr(self.cfg.rewards, "step_clearance_sigma", 0.04)

        # 目标抬高高度 = 前方台阶高度 + 余量。
        target_clearance = obstacle_height + clearance_margin
        max_foot_clearance = torch.clamp(foot_clearance.max(dim=1).values, min=0.0)

        # 原来的 exp(-error^2/sigma^2) 在没抬到目标前太稀疏。
        # 这里加入线性进度，让“开始抬”本身也有奖励。
        progress = torch.clamp(max_foot_clearance / torch.clamp(target_clearance, min=0.04), 0.0, 1.0)
        clearance_error = torch.clamp(target_clearance - max_foot_clearance, min=0.0)
        success_bonus = torch.exp(-torch.square(clearance_error / sigma))
        reward = 0.75 * progress + 0.25 * success_bonus

        return reward * active.float()

    def _reward_step_lift(self):
        """Reward reaching a useful lift height when a front step is detected."""

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
        max_lift = torch.clamp(foot_clearance.max(dim=1).values, min=0.0)

        lift_error = torch.clamp(target_lift - max_lift, min=0.0)
        reward = torch.exp(-torch.square(lift_error / sigma))

        return reward * active.float()


class D1HMoEDiscCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        curriculum = True

        max_curriculum_x = 1.2
        max_curriculum_x_back = 0.2
        max_curriculum_y = 0.10
        max_curriculum_yaw = 0.30

        zero_command_ratio = 0.02

        class ranges:
            lin_vel_x = [0.05, 0.5]
            lin_vel_y = [-0.05, 0.05]
            ang_vel_yaw = [-0.05, 0.05]
            heading = [-0.14, 0.14]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        terrain_proportions = [0.0, 0.02, 0.9, 0.05, 0.03]
        step_height = [0.05, 0.2]
        step_width_range = [0.25, 0.7]
        slope = [0.0, 0.15]

        # 越小，越容易把陡峭边缘修正成竖直面，台阶边缘更硬。
        slope_treshold = 0.3

    class rewards(D1HMoEBaseCfg.rewards):
        # step clearance reward parameters
        # 前方检测区域
        step_clearance_front_x_min = 0.1
        step_clearance_front_x_max = 0.80
        step_clearance_center_x_abs = 0.15
        step_clearance_y_abs = 0.35

        # 触发和目标高度
        step_clearance_trigger_height = 0.03
        step_clearance_margin = 0.04
        step_clearance_max_obstacle_height = 0.20
        step_clearance_sigma = 0.06
        step_clearance_min_cmd_x = 0.03
        step_lift_min_height = 0.05
        step_lift_margin = 0.06
        step_lift_sigma = 0.05

        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel_x = 23.0
            tracking_lin_vel_y = 5.0
            tracking_ang_vel = 18.0
            orientation = -18.0
            upward = 4.0
            collision = -10.0
            collision_hard = -15.0
            action_rate = -0.03
            stand_still = -0.2
            zero_base_vel = -1.0
            zero_yaw_rate = -1.0
            zero_wheel_vel = -0.02
            feet_air_time = 0.2
            lin_vel_z = -1.0
            body_pos_to_feet_x = 0.3
            body_feet_distance_x = -1.0
            body_feet_distance_y = -5.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.0

            # 前方有上台阶高度差时，给密集的抬脚进度奖励。
            step_clearance = 45.0
            # 专门鼓励至少一个轮/足达到越障所需高度。
            step_lift = 25.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class algorithm(D1HMoEBaseCfgPPO.algorithm):
        # 防止 residual 训练时动作 std 被 entropy bonus 推大。
        entropy_coef = 0.0

    class policy(D1HMoEBaseCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        barlow_actor_hidden_dims = [256, 128, 64]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.8

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_disc'
