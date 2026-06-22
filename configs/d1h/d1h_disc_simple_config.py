# configs/d1h/d1h_disc_simple_config.py

from .d1h_base_config import (
    D1HMoEBase,
    D1HMoEBaseCfg,
    D1HMoEBaseCfgPPO,
)

from .d1h_y1_style_mixin import D1HY1StyleMixin


class D1HMoEDiscSimple(D1HY1StyleMixin, D1HMoEBase):
    """
    Simple stair-climb task for D1H.

    This task uses:
        - D1H observation/action/asset/control logic from D1HMoEBase
        - y1v0h-style command sampling/smoothing/reward functions from D1HY1StyleMixin
        - y1v0h climb reward/terrain/domain-rand/policy settings below
    """
    pass


class D1HMoEDiscSimpleCfg(D1HMoEBaseCfg):
    class env(D1HMoEBaseCfg.env):
        # Keep D1H observation dimension from D1HMoEBaseCfg:
        # n_proprio=36, n_scan=187, history_len=10.
        num_envs = 4096

    class init_state(D1HMoEBaseCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0, 0.0, 0.0, 1]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        random_ori_probability = 0.0
        random_dof_pos_probability = 0.0

        reset_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
        }

        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
        }

        desired_feet_distance = 0.38
        feet_distance_range = [0.32, 0.50]

    class control(D1HMoEBaseCfg.control):
        control_type = "P"

        stiffness = {
            "hip": 40.0,
            "thigh": 40.0,
            "calf": 40.0,
            "foot": 10.0,
        }

        damping = {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 1.0,
            "foot": 0.5,
        }

        action_scale = 0.5
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands(D1HMoEBaseCfg.commands):
        # y1v0h command proportions:
        # 1.x
        # 2.y
        # 3.xy_mix
        # 4.spot_turn
        # 5.x_rotation
        # 6.y_rotation
        # 7.xy_mix_rotation
        # 8.stand_still, effectively everything after sum(first 7)
        commands_proportion = [
            0.45,
            0.10,
            0.10,
            0.10,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]

        curriculum = True
        max_curriculum = 1.0

        max_curriculum_x = 1.0
        max_curriculum_y = 1.0
        min_curriculum_x = -1.0
        min_curriculum_y = -1.0
        max_curriculum_z = 1.0

        # Keep D1H compatibility fields.
        max_curriculum_x_back = 1.0
        max_curriculum_yaw = 1.0

        num_commands = 4
        resampling_time = 10.0

        # Match y1v0h climb config.
        heading_command = False
        global_reference = False

        flip_same_sign_probability = 0.2

        max_lin_vel_x_change_rate = 0.5
        max_lin_vel_y_change_rate = 0.3
        max_ang_vel_change_rate = 0.5

        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        # Disable D1H extra zero-command injection.
        # y1v0h already has stand-still probability via commands_proportion.
        zero_command_ratio = 0.0
        zero_lin_vel_threshold = 0.05
        zero_yaw_threshold = 0.02
        startup_freeze_time = 0.0

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class asset(D1HMoEBaseCfg.asset):
        # Keep D1H asset/URDF.
        file = "{ROOT_DIR}/resources/d1h/urdf/robot.urdf"
        foot_name = "foot"
        name = "d1h"

        # Match y1v0h reward style:
        # collision penalizes calf/thigh, base contact terminates.
        penalize_contacts_on = ["calf", "thigh"]
        penalize_contact_head_on = ["base"]
        terminate_after_contacts_on = ["base"]

        use_force_sensor_contacts = True
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class rewards(D1HMoEBaseCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9

        # y1v0h climb values.
        base_height_target = 0.5
        tracking_sigma = 0.07

        # Kept for D1H parent functions that may still read it.
        distance_sigma = 0.08

        max_contact_force = 500.0
        only_positive_rewards = False

        class scales(D1HMoEBaseCfg.rewards.scales):
            # ---------------- y1v0h climb reward scales ----------------

            torques = 0.0
            powers = -2e-5
            termination = -100.0

            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 5.0

            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_vel = 0.0
            dof_acc = -2.5e-7

            base_height = -20.0
            feet_air_time = 0.0
            collision = -10.0
            stumble = 0.0

            action_rate = -0.1
            action_smoothness = 0.0

            stand_still = -1.0
            foot_clearance = -0.0
            orientation = -10.0
            no_gait = 5.0

            body_pos_to_feet_x = 1.0
            body_feet_distance_x = -50.0
            body_feet_distance_y = -100.0
            body_symmetry_y = 0.3
            body_symmetry_z = 0.9

            heading = 10.0
            upward = 1.0

            # ---------------- explicitly disable inherited D1H extras ----------------
            # These exist in D1H base/disc configs and should not affect this
            # y1v0h-style reproduction run.

            zero_base_vel = 0.0
            zero_wheel_vel = 0.0
            zero_yaw_rate = 0.0
            feet_stumble = 0.0
            collision_hard = 0.0
            collision_head = 0.0
            no_jump = 0.0

            # Disable all complex stair-feedforward / stair-shaping rewards if
            # they exist in your local inherited configs.
            step_clearance = 0.0
            stair_ff_tracking = 0.0
            step_reactive_lift = 0.0
            step_leg_imbalance = 0.0
            step_bump = 0.0
            step_drive = 0.0
            step_progress = 0.0
            step_up = 0.0
            step_success = 0.0
            step_stall = 0.0
            opposite_base_vel = 0.0
            stair_lateral_vel = 0.0
            stair_yaw_swing = 0.0
            stair_roll_pitch_rate = 0.0

    class domain_rand(D1HMoEBaseCfg.domain_rand):
        # y1v0h climb domain randomization.

        randomize_friction = True
        friction_range = [0.2, 2.75]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]

        randomize_base_com = True
        added_com_range = [-0.1, 0.1]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

    class depth(D1HMoEBaseCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0.0, 0.03]
        angle = [-5, 5]

        update_interval = 1

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class costs(D1HMoEBaseCfg.costs):
        # y1v0h climb uses 6 costs.

        num_costs = 6

        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            acc_smoothness = 0.1
            feet_contact_forces = 0.8
            stumble = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            acc_smoothness = 0.0
            feet_contact_forces = 0.0
            stumble = 0.0

    class cost:
        # Some older configs use cfg.cost.num_costs, while your current D1H
        # base mainly uses cfg.costs.num_costs. Keep both for compatibility.
        num_costs = 6

    class terrain(D1HMoEBaseCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = "trimesh"
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False

        # y1v0h terrain proportions:
        # [smooth slope, rough slope, stairs up, stairs down, discrete,
        #  stepping stones, gap, obstacles crossing, high platform]
        terrain_proportions = [
            0.1,
            0.0,
            0.8,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        # y1v0h field.
        stairs_max_height = 0.15

        # D1H terrain generator is likely to read step_height, so keep this
        # compatibility field while matching y1v0h max height.
        step_height = [0.0, 0.15]

        step_width_range = [0.40, 0.55]
        step_width = 0.40

        max_init_terrain_level = 0
        slope_treshold = 0.75
        slope = [0.0, 0.02]


class D1HMoEDiscSimpleCfgPPO(D1HMoEBaseCfgPPO):
    class algorithm(D1HMoEBaseCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4

        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(D1HMoEBaseCfgPPO.policy):
        # y1v0h climb policy size.

        init_noise_std = 1.0
        continue_from_last_std = True

        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        priv_encoder_dims = []
        activation = "elu"

        rnn_type = "lstm"
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 6

        teacher_act = True
        imi_flag = True

        # ActorCriticBarlowTwins compatibility.
        barlow_actor_hidden_dims = [512, 256, 128]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]

    class runner(D1HMoEBaseCfgPPO.runner):
        run_name = "d1h_moe_disc_simple"
        experiment_name = "d1h_moe_disc_simple"

        policy_class_name = "ActorCriticBarlowTwins"
        runner_class_name = "OnConstraintPolicyRunner"
        algorithm_class_name = "NP3O"

        save_interval = 200
        max_iterations = 40000
        num_steps_per_env = 24

        record_video = True
        video_interval = 500
        video_duration = 3.0
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 640
        video_tile_height = 360
        video_width = 1280
        video_height = 720

        resume = False
        resume_path = ""