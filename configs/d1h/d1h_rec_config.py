import torch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float
from isaacgym import gymapi, gymtorch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoERecovery(D1HMoEBase):
    def _init_buffers(self):
        super()._init_buffers()
        self.recovery_level = int(getattr(self.cfg.env, "recovery_start_level", 0))
        self.recovered_time = torch.zeros(self.num_envs, device=self.device)
        self.episode_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_recent_success = torch.zeros(0, dtype=torch.float, device=self.device)
        self.hard_contact_time = torch.zeros(self.num_envs, device=self.device)
        self.env_recovery_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _get_recovery_curriculum_params(self, level=None):
        level = int(self.recovery_level if level is None else level)

        if level <= 0:
            return dict(
                roll=0.10,
                pitch=0.10,
                yaw=0.15,
                joint_noise=0.04,
                lin_vel=0.08,
                ang_vel=0.40,
                push=False,
                disturbance=False,
                max_push_vel_xy=0.0,
                disturbance_range=0.0,
            )
        if level == 1:
            return dict(
                roll=0.25,
                pitch=0.25,
                yaw=0.35,
                joint_noise=0.10,
                lin_vel=0.15,
                ang_vel=0.80,
                push=False,
                disturbance=False,
                max_push_vel_xy=0.0,
                disturbance_range=0.0,
            )
        if level == 2:
            return dict(
                roll=0.55,
                pitch=0.55,
                yaw=0.70,
                joint_noise=0.18,
                lin_vel=0.30,
                ang_vel=1.80,
                push=True,
                disturbance=False,
                max_push_vel_xy=0.35,
                disturbance_range=0.0,
            )
        if level == 3:
            return dict(
                roll=0.85,
                pitch=0.85,
                yaw=1.10,
                joint_noise=0.24,
                lin_vel=0.45,
                ang_vel=3.00,
                push=True,
                disturbance=True,
                max_push_vel_xy=0.70,
                disturbance_range=20.0,
            )
        return dict(
            roll=1.00,
            pitch=1.00,
            yaw=1.30,
            joint_noise=0.28,
            lin_vel=0.50,
            ang_vel=3.20,
            push=True,
            disturbance=True,
            max_push_vel_xy=0.80,
            disturbance_range=25.0,
        )

    def _sample_env_recovery_levels(self, env_ids):
        max_level = int(self.recovery_level)
        if max_level <= 0:
            self.env_recovery_levels[env_ids] = 0
            return

        self.env_recovery_levels[env_ids] = torch.randint(
            0,
            max_level + 1,
            (len(env_ids),),
            device=self.device,
        )

    def _get_folded_dof_pos(self, env_ids):
        n = len(env_ids)
        q = self.default_dof_pos.clone()
        if q.ndim == 1 or q.shape[0] == 1:
            q = q.reshape(1, -1).repeat(n, 1)
        else:
            q = q[env_ids].clone()

        folded_angles = {
            "FL_hip_joint": 0.2,
            "FR_hip_joint": -0.2,
            "FL_thigh_joint": 1.3,
            "FR_thigh_joint": 1.3,
            "FL_calf_joint": -2.75,
            "FR_calf_joint": -2.75,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
        }

        for name, value in folded_angles.items():
            if name in self.dof_names:
                q[:, self.dof_names.index(name)] = value
            else:
                print(f"[D1HMoERecovery] Warning: joint {name} not found in dof_names")

        if hasattr(self, "dof_pos_limits"):
            lower = self.dof_pos_limits[:, 0].unsqueeze(0)
            upper = self.dof_pos_limits[:, 1].unsqueeze(0)
            q = torch.max(torch.min(q, upper), lower)
        return q

    def _reset_dofs(self, env_ids):
        n = len(env_ids)
        folded_q = self._get_folded_dof_pos(env_ids)
        joint_noise = torch.zeros(n, 1, device=self.device)
        for level in torch.unique(self.env_recovery_levels[env_ids]).tolist():
            level_mask = self.env_recovery_levels[env_ids] == int(level)
            params = self._get_recovery_curriculum_params(level)
            joint_noise[level_mask] = float(params["joint_noise"])

        self.dof_pos[env_ids] = folded_q + torch_rand_float(
            -1.0,
            1.0,
            (n, self.num_dof),
            device=self.device,
        ) * joint_noise
        dof_vel = torch_rand_float(-0.1, 0.1, (n, self.num_dof), device=self.device)
        dof_vel[self.env_recovery_levels[env_ids] <= 0] = 0.0
        self.dof_vel[env_ids] = dof_vel

        if hasattr(self, "dof_pos_limits"):
            lower = self.dof_pos_limits[:, 0].unsqueeze(0)
            upper = self.dof_pos_limits[:, 1].unsqueeze(0)
            self.dof_pos[env_ids] = torch.max(torch.min(self.dof_pos[env_ids], upper), lower)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        n = len(env_ids)

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        random_roll = torch.zeros(n, device=self.device)
        random_pitch = torch.zeros(n, device=self.device)
        random_yaw = torch.zeros(n, device=self.device)
        lin_vel = torch.zeros(n, 3, device=self.device)
        ang_vel = torch.zeros(n, 3, device=self.device)

        for level in torch.unique(self.env_recovery_levels[env_ids]).tolist():
            level_mask = self.env_recovery_levels[env_ids] == int(level)
            count = int(level_mask.sum().item())
            params = self._get_recovery_curriculum_params(level)
            random_roll[level_mask] = torch_rand_float(
                -float(params["roll"]),
                float(params["roll"]),
                (count, 1),
                device=self.device,
            ).squeeze(1)
            random_pitch[level_mask] = torch_rand_float(
                -float(params["pitch"]),
                float(params["pitch"]),
                (count, 1),
                device=self.device,
            ).squeeze(1)
            random_yaw[level_mask] = torch_rand_float(
                -float(params["yaw"]),
                float(params["yaw"]),
                (count, 1),
                device=self.device,
            ).squeeze(1)
            lin_vel[level_mask] = torch_rand_float(
                -float(params["lin_vel"]),
                float(params["lin_vel"]),
                (count, 3),
                device=self.device,
            )
            ang_vel[level_mask] = torch_rand_float(
                -float(params["ang_vel"]),
                float(params["ang_vel"]),
                (count, 3),
                device=self.device,
            )
        self.root_states[env_ids, 7:10] = lin_vel
        self.root_states[env_ids, 10:13] = ang_vel

        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            random_roll,
            random_pitch,
            random_yaw,
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % self.cfg.domain_rand.push_interval == 0:
            self._push_recovery_envs()
        if self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0:
            self._disturbance_recovery_envs()
        self._update_recovered_time()

    def _push_recovery_envs(self):
        for level in torch.unique(self.env_recovery_levels).tolist():
            params = self._get_recovery_curriculum_params(level)
            if not params["push"]:
                continue
            env_mask = self.env_recovery_levels == int(level)
            max_push_vel_xy = float(params["max_push_vel_xy"])
            self.root_states[env_mask, 7:9] = torch_rand_float(
                -max_push_vel_xy,
                max_push_vel_xy,
                (int(env_mask.sum().item()), 2),
                device=self.device,
            )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_recovery_envs(self):
        self.disturbance.zero_()
        any_disturbance = False
        for level in torch.unique(self.env_recovery_levels).tolist():
            params = self._get_recovery_curriculum_params(level)
            if not params["disturbance"]:
                continue
            env_mask = self.env_recovery_levels == int(level)
            disturbance_range = float(params["disturbance_range"])
            if disturbance_range <= 0.0:
                continue
            count = int(env_mask.sum().item())
            self.disturbance[env_mask, 0, :] = torch_rand_float(
                -disturbance_range,
                disturbance_range,
                (count, 3),
                device=self.device,
            )
            any_disturbance = True
        if any_disturbance:
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                forceTensor=gymtorch.unwrap_tensor(self.disturbance),
                space=gymapi.CoordinateSpace.LOCAL_SPACE,
            )

    def _is_recovered(self):
        upright_score = -self.projected_gravity[:, 2]
        base_height = self._get_base_heights()
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        lin_vel_z = torch.abs(self.base_lin_vel[:, 2])

        upright = upright_score > 0.85
        height_ok = base_height > self.cfg.env.recovery_target_height
        low_ang = ang_vel_xy < 0.6
        low_lin = lin_vel_xy < 0.35
        low_lin_z = lin_vel_z < 0.35
        return upright & height_ok & low_ang & low_lin & low_lin_z

    def _update_recovered_time(self):
        recovered_now = self._is_recovered()
        self.recovered_time = torch.where(
            recovered_now,
            self.recovered_time + self.dt,
            torch.zeros_like(self.recovered_time),
        )
        success_hold = self.recovered_time >= self.cfg.env.recovery_success_hold_time
        self.episode_recovered |= success_hold

    def _maybe_update_recovery_curriculum(self):
        if not getattr(self.cfg.env, "recovery_curriculum", True):
            return

        min_items = int(getattr(self.cfg.env, "recovery_min_success_episodes", 256))
        if self.recovery_recent_success.numel() < min_items:
            return

        success_rate = self.recovery_recent_success.mean().item()
        threshold = float(getattr(self.cfg.env, "recovery_success_threshold", 0.80))
        max_level = int(getattr(self.cfg.env, "recovery_max_level", 4))

        if success_rate > threshold and self.recovery_level < max_level:
            self.recovery_level += 1
            self.recovery_recent_success = torch.zeros(0, dtype=torch.float, device=self.device)
            print(f"[D1HMoERecovery] Promote recovery_level to {self.recovery_level}")

    def check_termination(self):
        episode_time = self.episode_length_buf.float() * self.dt
        base_height = self._get_base_heights()

        bad_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        hard_contact = torch.any(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
            > self.cfg.env.recovery_hard_contact_force,
            dim=1,
        )
        past_grace_time = episode_time >= self.cfg.env.contact_termination_grace_time
        past_hard_grace_time = episode_time >= self.cfg.env.hard_contact_termination_grace_time
        recovered_now = self._is_recovered()

        self.bad_contact_time = torch.where(
            bad_contact & past_grace_time & recovered_now,
            self.bad_contact_time + self.dt,
            torch.zeros_like(self.bad_contact_time),
        )
        self.hard_contact_time = torch.where(
            hard_contact & past_hard_grace_time,
            self.hard_contact_time + self.dt,
            torch.zeros_like(self.hard_contact_time),
        )

        contact_reset = self.bad_contact_time >= self.cfg.env.contact_termination_duration
        hard_contact_reset = self.hard_contact_time >= self.cfg.env.hard_contact_termination_duration
        low_height_reset = base_height < self.cfg.env.recovery_min_height_for_reset
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = contact_reset | hard_contact_reset | low_height_reset | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        success_rate = None
        if hasattr(self, "episode_recovered"):
            finished_env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(finished_env_ids) > 0:
                success = self.episode_recovered[finished_env_ids].float().detach()
                self.recovery_recent_success = torch.cat([self.recovery_recent_success, success])
                max_items = int(getattr(self.cfg.env, "recovery_min_success_episodes", 256))
                if self.recovery_recent_success.numel() > max_items:
                    self.recovery_recent_success = self.recovery_recent_success[-max_items:]

                if self.recovery_recent_success.numel() > 0:
                    success_rate = self.recovery_recent_success.mean()
                self._maybe_update_recovery_curriculum()

        self._sample_env_recovery_levels(env_ids)
        super().reset_idx(env_ids)

        if hasattr(self, "episode_recovered"):
            self.recovered_time[env_ids] = 0.0
            self.episode_recovered[env_ids] = False
        self.hard_contact_time[env_ids] = 0.0

        if "episode" in self.extras:
            self.extras["episode"]["recovery_level"] = torch.as_tensor(
                self.recovery_level,
                device=self.device,
                dtype=torch.float,
            )
            self.extras["episode"]["recovery_recovered_frac"] = self._is_recovered().float().mean()
            self.extras["episode"]["recovery_recent_success_rate"] = (
                success_rate
                if success_rate is not None
                else torch.tensor(0.0, device=self.device)
            )
            self.extras["episode"]["recovery_mean_recovered_time"] = self.recovered_time.mean()
            self.extras["episode"]["recovery_sampled_level_mean"] = self.env_recovery_levels.float().mean()

    def _reward_recovery_height(self):
        h = self._get_base_heights()
        h_min = 0.15
        h_target = self.cfg.env.recovery_target_height
        return torch.clamp((h - h_min) / (h_target - h_min), 0.0, 1.0)

    def _reward_recovery_upright(self):
        upright_score = -self.projected_gravity[:, 2]
        return torch.clamp((upright_score - 0.3) / (0.9 - 0.3), 0.0, 1.0)

    def _reward_recovery_success(self):
        return self._is_recovered().float()

    def _reward_stable_after_recovery(self):
        recovered = self._is_recovered()
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        stable_score = torch.exp(-2.0 * ang_vel_xy) * torch.exp(-2.0 * lin_vel_xy)
        return recovered.float() * stable_score


class D1HMoERecCfg(D1HMoEBaseCfg):
    class env(D1HMoEBaseCfg.env):
        episode_length_s = 6.0

        recovery_curriculum = True
        recovery_start_level = 0
        recovery_max_level = 4
        recovery_success_threshold = 0.70
        recovery_min_success_episodes = 1024
        recovery_success_hold_time = 0.35

        recovery_target_height = 0.38
        recovery_min_height_for_reset = 0.03
        contact_termination_grace_time = 1.0
        contact_termination_duration = 0.20
        hard_contact_termination_grace_time = 0.80
        hard_contact_termination_duration = 0.25
        recovery_hard_contact_force = 250.0
        min_base_height_for_reset = 0.03

    class init_state(D1HMoEBaseCfg.init_state):
        pos = [0.0, 0.0, 0.16]
        reset_joint_angles = {
            "FL_hip_joint": 0.2,
            "FR_hip_joint": -0.2,
            "FL_thigh_joint": 1.3,
            "FR_thigh_joint": 1.3,
            "FL_calf_joint": -2.75,
            "FR_calf_joint": -2.75,
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

    class commands(D1HMoEBaseCfg.commands):
        curriculum = False
        zero_command_ratio = 1.0
        startup_freeze_time = 3.0

        max_curriculum = 0.2
        max_curriculum_x = 0.15
        max_curriculum_x_back = 0.05
        max_curriculum_y = 0.03
        max_curriculum_yaw = 0.05

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = False
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        step_height = [0.0, 0.0]
        slope = [0.0, 0.0]

    class domain_rand(D1HMoEBaseCfg.domain_rand):
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.0
        disturbance = False
        disturbance_range = [-0.0, 0.0]
        disturbance_interval = 4
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        randomize_friction = True
        friction_range = [0.6, 1.3]

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 0.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            stand_still = -0.8
            zero_yaw_rate = 0.0
            heading = 0.0
            lin_vel_z = 0.0
            dof_pos_limits = 0.0
            body_pos_to_feet_x = 0.0

            recovery_height = 8.0
            recovery_upright = 8.0
            recovery_success = 15.0
            stable_after_recovery = 6.0

            orientation = -3.0
            ang_vel_xy = -0.8
            base_height = 0.0
            upward = 0.0

            zero_base_vel = -2.0
            zero_wheel_vel = -0.05
            action_rate = -0.06
            dof_acc = -2.5e-7
            torques = -1.0e-5

            collision = -3.0
            collision_hard = -25.0

            body_feet_distance_x = -2.0
            body_feet_distance_y = -6.0
            body_symmetry_y = 0.5
            body_symmetry_z = 0.5


class D1HMoERecCfgPPO(D1HMoEBaseCfgPPO):
    class policy(D1HMoEBaseCfgPPO.policy):
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        barlow_actor_hidden_dims = [512, 256, 128]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = []

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = 'd1h_moe_rec'
        run_name = 'folded_recovery_curriculum'
        max_iterations = 8000
        num_steps_per_env = 32
        save_interval = 200
        record_video = True
        video_interval = 300
        video_duration = 5.0
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 320
        video_tile_height = 180
