import torch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDisc(D1HMoEBase):
    def _reward_step_clearance(self):
        """
        前方存在上台阶高度差时，鼓励轮/足抬高。

        说明：
        1. 这个 reward 使用 measured_heights，属于训练时特权信息。
        2. measured_heights 不进入 actor 输入，只用于 reward 计算。
        3. 只有当前方地形明显高于当前地面，并且命令为向前走时，这个 reward 才激活。
        """

        if not getattr(self.cfg.terrain, "measure_heights", False):
            return torch.zeros(self.num_envs, device=self.device)

        if not hasattr(self, "measured_heights"):
            return torch.zeros(self.num_envs, device=self.device)

        if not torch.is_tensor(self.measured_heights):
            return torch.zeros(self.num_envs, device=self.device)

        if self.measured_heights.ndim != 2:
            return torch.zeros(self.num_envs, device=self.device)

        if not hasattr(self, "height_points"):
            return torch.zeros(self.num_envs, device=self.device)

        if not hasattr(self, "commands"):
            return torch.zeros(self.num_envs, device=self.device)

        # height_points: [num_envs, num_height_points, 3]
        # 这里取第 0 个 env 的局部采样点坐标，因为所有 env 的采样点布局相同。
        points = self.height_points[0]
        px = points[:, 0]
        py = points[:, 1]

        # 从 cfg.rewards 读取参数，后面只需要在 config 里调这些数。
        front_x_min = getattr(self.cfg.rewards, "step_clearance_front_x_min", 0.20)
        front_x_max = getattr(self.cfg.rewards, "step_clearance_front_x_max", 0.80)
        center_x_abs = getattr(self.cfg.rewards, "step_clearance_center_x_abs", 0.15)
        y_abs = getattr(self.cfg.rewards, "step_clearance_y_abs", 0.35)

        trigger_height = getattr(self.cfg.rewards, "step_clearance_trigger_height", 0.03)
        clearance_margin = getattr(self.cfg.rewards, "step_clearance_margin", 0.04)
        max_obstacle_height = getattr(self.cfg.rewards, "step_clearance_max_obstacle_height", 0.20)
        sigma = getattr(self.cfg.rewards, "step_clearance_sigma", 0.04)
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)

        # 前方区域：机器人前方 0.20~0.80 m，左右宽度 |y| <= 0.35 m。
        front_mask = (
            (px >= front_x_min)
            & (px <= front_x_max)
            & (torch.abs(py) <= y_abs)
        )

        # 当前脚下/车身中心附近区域，用于估计当前地面高度。
        center_mask = (
            (torch.abs(px) <= center_x_abs)
            & (torch.abs(py) <= y_abs)
        )

        if front_mask.sum().item() == 0 or center_mask.sum().item() == 0:
            return torch.zeros(self.num_envs, device=self.device)

        front_height = self.measured_heights[:, front_mask].max(dim=1).values
        center_height = self.measured_heights[:, center_mask].mean(dim=1)

        # 前方相对当前地面的高度差。只关心上台阶，不奖励下台阶。
        obstacle_height = torch.clamp(
            front_height - center_height,
            min=0.0,
            max=max_obstacle_height,
        )

        # 只有前方确实有坎，并且命令向前走时才激活。
        need_step = obstacle_height > trigger_height
        forward_cmd = self.commands[:, 0] > min_cmd_x
        active = need_step & forward_cmd

        if not torch.any(active):
            return torch.zeros(self.num_envs, device=self.device)

        # 优先用 rigid_body_states 计算轮/足的世界 z 坐标。
        # feet_indices 在 D1HMoEBase._create_envs() 里已经创建。
        if hasattr(self, "rigid_body_states") and hasattr(self, "feet_indices"):
            foot_z = self.rigid_body_states[:, self.feet_indices, 2]
            foot_clearance = foot_z - center_height.unsqueeze(1)
            max_foot_clearance = foot_clearance.max(dim=1).values

        # 如果某些版本里没有 rigid_body_states，则退化使用 feet_body_frame_height。
        # 这个分支主要是防止属性不存在导致训练直接报错。
        elif hasattr(self, "feet_body_frame_height") and torch.is_tensor(self.feet_body_frame_height):
            max_foot_clearance = self.feet_body_frame_height.max(dim=1).values

        else:
            return torch.zeros(self.num_envs, device=self.device)

        # 目标抬高高度 = 前方台阶高度 + 余量。
        target_clearance = obstacle_height + clearance_margin

        # 没达到目标时有误差，达到后误差为 0。
        clearance_error = torch.clamp(
            target_clearance - max_foot_clearance,
            min=0.0,
        )

        reward = torch.exp(-torch.square(clearance_error / sigma))

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
        terrain_proportions = [0.0, 0.02, 0.8, 0.15, 0.03]
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
        step_clearance_sigma = 0.03
        step_clearance_min_cmd_x = 0.03

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
            feet_air_time = 2.0
            lin_vel_z = -1.0
            body_pos_to_feet_x = 0.3
            body_feet_distance_x = -1.0
            body_feet_distance_y = -5.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.0

            # 新增：前方有上台阶高度差时，鼓励轮/足抬高。
            # 第一版不要太大，先看 rew_step_clearance 是否正常出现。
            step_clearance = 30.0


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