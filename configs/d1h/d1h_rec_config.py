import torch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float
from isaacgym import gymtorch

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoERecovery(D1HMoEBase):
    def _init_buffers(self):
        super()._init_buffers()
        self.recovery_level = int(getattr(self.cfg.env, "recovery_start_level", 0))
        self.recovered_time = torch.zeros(self.num_envs, device=self.device)
        self.episode_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_recent_success = torch.zeros(0, dtype=torch.float, device=self.device)

    def _get_recovery_curriculum_params(self):
        level = int(self.recovery_level)

        if level <= 0:
            return dict(
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                joint_noise=0.0,
                lin_vel=0.0,
                ang_vel=0.0,
                push=False,
                disturbance=False,
                max_push_vel_xy=0.0,
                disturbance_range=0.0,
                friction_range=(0.8, 1.2),
            )
        if level == 1:
            return dict(
                roll=0.08,
                pitch=0.08,
                yaw=0.10,
                joint_noise=0.05,
                lin_vel=0.05,
                ang_vel=0.20,
                push=False,
                disturbance=False,
                max_push_vel_xy=0.0,
                disturbance_range=0.0,
                friction_range=(0.7, 1.3),
            )
        if level == 2:
            return dict(
                roll=0.25,
                pitch=0.25,
                yaw=0.30,
                joint_noise=0.12,
                lin_vel=0.15,
                ang_vel=0.80,
                push=False,
                disturbance=False,
                max_push_vel_xy=0.0,
                disturbance_range=0.0,
                friction_range=(0.6, 1.5),
            )
        if level == 3:
            return dict(
                roll=0.55,
                pitch=0.55,
                yaw=0.60,
                joint_noise=0.20,
                lin_vel=0.30,
                ang_vel=2.00,
                push=True,
                disturbance=True,
                max_push_vel_xy=0.4,
                disturbance_range=10.0,
                friction_range=(0.5, 1.6),
            )
        return dict(
            roll=0.70,
            pitch=0.70,
            yaw=1.00,
            joint_noise=0.25,
            lin_vel=0.40,
            ang_vel=3.00,
            push=True,
            disturbance=True,
            max_push_vel_xy=0.8,
            disturbance_range=25.0,
            friction_range=(0.35, 1.8),
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
        params = self._get_recovery_curriculum_params()
        n = len(env_ids)
        folded_q = self._get_folded_dof_pos(env_ids)

        if self.recovery_level <= 0:
            small_noise_mask = torch.rand(n, device=self.device) < 0.20
            noise = torch.zeros((n, self.num_dof), device=self.device)
            if small_noise_mask.any():
                noise[small_noise_mask] = torch_rand_float(
                    -0.03,
                    0.03,
                    (int(small_noise_mask.sum().item()), self.num_dof),
                    device=self.device,
                )
            self.dof_pos[env_ids] = folded_q + noise
            self.dof_vel[env_ids] = 0.0
        else:
            joint_noise = float(params["joint_noise"])
            self.dof_pos[env_ids] = folded_q + torch_rand_float(
                -joint_noise,
                joint_noise,
                (n, self.num_dof),
                device=self.device,
            )
            self.dof_vel[env_ids] = torch_rand_float(
                -0.1,
                0.1,
                (n, self.num_dof),
                device=self.device,
            )

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
        params = self._get_recovery_curriculum_params()
        n = len(env_ids)

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if self.recovery_level <= 0:
            random_roll = torch.zeros(n, device=self.device)
            random_pitch = torch.zeros(n, device=self.device)
            random_yaw = torch.zeros(n, device=self.device)
            self.root_states[env_ids, 7:13] = 0.0
        else:
            random_roll = torch_rand_float(
                -float(params["roll"]),
                float(params["roll"]),
                (n, 1),
                device=self.device,
            ).squeeze(1)
            random_pitch = torch_rand_float(
                -float(params["pitch"]),
                float(params["pitch"]),
                (n, 1),
                device=self.device,
            ).squeeze(1)
            random_yaw = torch_rand_float(
                -float(params["yaw"]),
                float(params["yaw"]),
                (n, 1),
                device=self.device,
            ).squeeze(1)
            self.root_states[env_ids, 7:10] = torch_rand_float(
                -float(params["lin_vel"]),
                float(params["lin_vel"]),
                (n, 3),
                device=self.device,
            )
            self.root_states[env_ids, 10:13] = torch_rand_float(
                -float(params["ang_vel"]),
                float(params["ang_vel"]),
                (n, 3),
                device=self.device,
            )

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
        self._update_recovered_time()

    def _is_recovered(self):
        upright_score = -self.projected_gravity[:, 2]
        base_height = self._get_base_heights()
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)

        upright = upright_score > 0.85
        height_ok = base_height > self.cfg.env.recovery_target_height
        low_ang = ang_vel_xy < 0.6
        low_lin = lin_vel_xy < 0.35
        return upright & height_ok & low_ang & low_lin

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
        past_grace_time = episode_time >= self.cfg.env.contact_termination_grace_time
        recovered_now = self._is_recovered()

        self.bad_contact_time = torch.where(
            bad_contact & past_grace_time & recovered_now,
            self.bad_contact_time + self.dt,
            torch.zeros_like(self.bad_contact_time),
        )

        contact_reset = self.bad_contact_time >= self.cfg.env.contact_termination_duration
        low_height_reset = base_height < self.cfg.env.recovery_min_height_for_reset
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = contact_reset | low_height_reset | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        success_rate = None
        if hasattr(self, "episode_recovered"):
            success = self.episode_recovered[env_ids].float().detach()
            self.recovery_recent_success = torch.cat([self.recovery_recent_success, success])
            max_items = int(getattr(self.cfg.env, "recovery_min_success_episodes", 256))
            if self.recovery_recent_success.numel() > max_items:
                self.recovery_recent_success = self.recovery_recent_success[-max_items:]

            if self.recovery_recent_success.numel() > 0:
                success_rate = self.recovery_recent_success.mean()
            self._maybe_update_recovery_curriculum()

        super().reset_idx(env_ids)

        if hasattr(self, "episode_recovered"):
            self.recovered_time[env_ids] = 0.0
            self.episode_recovered[env_ids] = False

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
        recovery_success_threshold = 0.80
        recovery_min_success_episodes = 256
        recovery_success_hold_time = 0.50

        recovery_target_height = 0.38
        recovery_min_height_for_reset = 0.03
        contact_termination_grace_time = 4.0
        contact_termination_duration = 0.35
        min_base_height_for_reset = 0.03

    class init_state(D1HMoEBaseCfg.init_state):
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
        friction_range = [0.7, 1.3]

    class rewards(D1HMoEBaseCfg.rewards):
        class scales(D1HMoEBaseCfg.rewards.scales):
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 0.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            stand_still = 0.0
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
