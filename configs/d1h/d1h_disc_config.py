from .d1h_base_config import D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDiscCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        max_curriculum = 1.2
        max_curriculum_x = 1.0
        max_curriculum_x_back = 0.8
        max_curriculum_y = 0.25
        max_curriculum_yaw = 0.6
        zero_command_ratio = 0.1

        class ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.35, 0.35]
            heading = [-3.14, 3.14]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        terrain_proportions = [0.0, 0.05, 0.55, 0.3, 0.1]
        step_height = [0.05, 0.2]
        step_width_range = [0.18, 0.5]
        slope = [0.0, 0.15]

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel_x = 18.0
            tracking_lin_vel_y = 8.0
            tracking_ang_vel = 16.0
            orientation = -14.0
            upward = 4.0
            collision = -14.0
            collision_hard = -20.0
            action_rate = -0.03
            zero_base_vel = -8.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_disc'