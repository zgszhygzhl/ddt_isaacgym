from .d1h_base_config import D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEPrecCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        zero_command_ratio = 0.65
        max_curriculum = 0.8
        max_curriculum_x = 0.5
        max_curriculum_x_back = 0.5
        max_curriculum_y = 0.2
        max_curriculum_yaw = 0.4

        class ranges:
            lin_vel_x = [-0.2, 0.2]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.15, 0.15]
            heading = [-3.14, 3.14]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        step_height = [0.0, 0.0]
        slope = [0.0, 0.05]

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel_x = 16.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 14.0
            stand_still = -7.0
            zero_base_vel = -28.0
            zero_wheel_vel = -0.12
            zero_yaw_rate = -35.0
            body_symmetry_y = 0.8
            body_symmetry_z = 0.5
            action_rate = -0.08


class D1HMoEPrecCfgPPO(D1HMoEBaseCfgPPO):
    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_prec'