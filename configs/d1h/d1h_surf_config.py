from .d1h_base_config import D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoESurfCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        max_curriculum = 1.8
        max_curriculum_x = 1.5
        max_curriculum_x_back = 1.0
        max_curriculum_y = 0.35
        max_curriculum_yaw = 0.8
        zero_command_ratio = 0.18

        class ranges:
            lin_vel_x = [-0.6, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.4, 0.4]
            heading = [-3.14, 3.14]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        terrain_proportions = [0.55, 0.45, 0.0, 0.0, 0.0]
        step_height = [0.0, 0.0]
        slope = [0.0, 0.25]

    class domain_rand(D1HMoEBaseCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.08, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 0.2]
        randomize_base_mass = True
        added_mass_range = [-1.5, 4.0]
        randomize_base_com = True
        added_com_range = [-0.12, 0.12]
        randomize_motor = True
        randomize_kpkd = True
        randomize_lag_timesteps = True
        disturbance = True
        push_robots = True

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel_x = 24.0
            tracking_lin_vel_y = 12.0
            tracking_ang_vel = 22.0
            orientation = -14.0
            ang_vel_xy = -0.2
            powers = -4e-5
            action_rate = -0.06
            upward = 4.0


class D1HMoESurfCfgPPO(D1HMoEBaseCfgPPO):
    class policy(D1HMoEBaseCfgPPO.policy):
        barlow_actor_hidden_dims = [256, 128, 64]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_surf'