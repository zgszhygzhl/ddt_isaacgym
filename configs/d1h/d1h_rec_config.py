from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float
from isaacgym import gymtorch

from .d1h_rough_config import D1HRough, D1HRoughCfg, D1HRoughCfgPPO


class D1HMoERecovery(D1HRough):
    def _reset_root_states(self, env_ids):
        deterministic_reset = getattr(self.cfg.env, "deterministic_reset", False)

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if deterministic_reset:
            self.root_states[env_ids, 3:7] = self.base_init_state[3:7]
            self.root_states[env_ids, 7:13] = 0.0
        else:
            random_roll = torch_rand_float(-0.45, 0.45, (len(env_ids), 1), device=self.device).squeeze(1)
            random_pitch = torch_rand_float(-0.45, 0.45, (len(env_ids), 1), device=self.device).squeeze(1)
            random_yaw = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
            self.root_states[env_ids, 7:10] = torch_rand_float(-0.4, 0.4, (len(env_ids), 3), device=self.device)
            self.root_states[env_ids, 10:13] = torch_rand_float(-2.5, 2.5, (len(env_ids), 3), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )


class D1HMoERecCfg(D1HRoughCfg):
    class commands(D1HRoughCfg.commands):
        zero_command_ratio = 0.55
        max_curriculum = 1.0
        max_curriculum_x = 0.8
        max_curriculum_x_back = 0.6
        max_curriculum_y = 0.2
        max_curriculum_yaw = 0.5

        class ranges:
            lin_vel_x = [-0.25, 0.35]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.2, 0.2]
            heading = [-3.14, 3.14]

    class terrain(D1HRoughCfg.terrain):
        curriculum = True
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        step_height = [0.0, 0.0]
        slope = [0.0, 0.12]

    class domain_rand(D1HRoughCfg.domain_rand):
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.5
        disturbance = True
        disturbance_range = [-50.0, 50.0]
        disturbance_interval = 4
        randomize_base_mass = True
        added_mass_range = [-2.0, 4.5]
        randomize_base_com = True
        added_com_range = [-0.15, 0.15]
        randomize_friction = True
        friction_range = [0.2, 1.8]

    class rewards(D1HRoughCfg.rewards):
        class scales(D1HRoughCfg.rewards.scales):
            tracking_lin_vel_x = 10.0
            tracking_lin_vel_y = 5.0
            tracking_ang_vel = 8.0
            orientation = -20.0
            ang_vel_xy = -0.4
            base_height = -45.0
            upward = 6.0
            collision = -25.0
            collision_hard = -40.0
            zero_base_vel = -10.0
            action_rate = -0.04


class D1HMoERecCfgPPO(D1HRoughCfgPPO):
    class runner(D1HRoughCfgPPO.runner):
        experiment_name = 'd1h_moe_rec'