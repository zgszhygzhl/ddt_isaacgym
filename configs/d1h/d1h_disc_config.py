from .d1h_base_config import D1HMoEBaseCfg, D1HMoEBaseCfgPPO


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
            ang_vel_yaw = [-0.25, 0.25]
            heading = [-3.14, 3.14]


    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        terrain_proportions = [0.0, 0.05, 0.6, 0.3, 0.05]
        step_height = [0.05, 0.2]
        step_width_range = [0.25, 0.5]
        slope = [0.0, 0.15]

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel_x = 14.0
            tracking_lin_vel_y = 5.0
            tracking_ang_vel = 12.0
            orientation = -14.0
            upward = 4.0
            collision = -14.0
            collision_hard = -20.0
            action_rate = -0.03
            stand_still = -0.2
            zero_base_vel = -1.0
            zero_yaw_rate = -1.0
            zero_wheel_vel = -0.02
            feet_air_time = 1.0
            lin_vel_z = -1.0
            body_pos_to_feet_x = 0.3
            body_feet_distance_x = -1.0
            body_feet_distance_y = -5.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class policy(D1HMoEBaseCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        barlow_actor_hidden_dims = [256, 128, 64]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.3

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_disc'