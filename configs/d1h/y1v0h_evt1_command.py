from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from utils.math import wrap_to_pi
from global_config import ROOT_DIR
from .. import LeggedRobot
import numpy as np
import os
import random

class Y1v0hEvt1Command(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.commands_given = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.odemetry_vel = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 初始化每个环境的feet_distance
        self.env_feet_distances = torch.ones(self.num_envs, device=self.device) * self.cfg.init_state.desired_feet_distance
        self.last_commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_head_contact_force = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.try_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2] += 0.2
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        random_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.init_state.random_ori_probability
        self.root_states[env_ids[random_mask], 3:7] = random_quat(torch_rand_float(0, 1, (random_mask.sum(), 4), device=self.device))
        self.root_states[env_ids[~random_mask], 3:7] = self.base_init_state[3:7]

        self.root_states[env_ids, 2:3] += torch_rand_float(0, 0.2, (len(env_ids), 1), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.commands_given[env_ids] = torch.zeros_like(self.commands[env_ids])
        self.odemetry_vel[env_ids] = torch.zeros_like(self.commands[env_ids])
        
        
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        
        if len(env_ids) > 0:
            # 基于概率决定哪些agent进行resample
            resample_prob = torch.clamp(torch.tensor(self.cfg.commands.resampling_time/10), 0.1, 1.0)
            # 为每个到达resampling时间的agent生成随机数
            random_probs = torch.rand(len(env_ids), device=self.device)
            # 只有概率小于resample_probability的agent才进行resample
            resample_env_ids = env_ids[random_probs <= resample_prob]
            if len(resample_env_ids) > 0:
                self._resample_commands(resample_env_ids)
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(1.0*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        else:
            self.commands[:, 3] += self.commands[:, 2]*self.dt
            self.commands[:, 3] = wrap_to_pi(self.commands[:, 3])

        # self.compute_given_commands()

        if self.cfg.terrain.measure_heights:
            # if self.global_counter % self.cfg.depth.update_interval == 0:
            self.measured_heights = self._get_heights()
            
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def compute_given_commands(self):

        self.odemetry_vel[:,:2] = self.base_lin_vel[:, :2]
        self.odemetry_vel[:,2] = self.base_ang_vel[:, 2]
        
        max_change_rates = torch.tensor([
            self.cfg.commands.max_lin_vel_x_change_rate,
            self.cfg.commands.max_lin_vel_y_change_rate, 
            self.cfg.commands.max_ang_vel_change_rate,
            0.0  # heading命令不需要缓冲
        ], device=self.device)
        
        # 计算每步最大允许的变化量
        max_change_per_step = max_change_rates * self.dt
        
        # 计算命令差值
        command_diff = self.commands[:, :3] - self.odemetry_vel[:, :3]
        
        # 根据命令和实际速度的大小关系动态调整x方向的最大变化率
        # max_change_rates[0] = torch.where(
        #     abs(self.commands[:, 0]) > abs(self.odemetry_vel[:, 0]),
        #     self.cfg.commands.max_lin_vel_x_change_rate,
        #     self.cfg.commands.max_lin_vel_x_change_rate * 2
        # )
            
        # 对每个命令维度进行处理
        for i in range(3):  # 只对前3个命令（x, y, ang_vel）进行缓冲
            diff_magnitude = torch.abs(command_diff[:, i])
            max_allowed_change = max_change_per_step[i]
            
            # 检测制动情况：目标速度接近0
            # 如果目标命令为0（制动），则增加最大允许变化率（2倍）
            is_braking = (torch.abs(self.commands[:, i]) < 0.1) & (torch.abs(self.commands[:, i]) <= torch.abs(self.odemetry_vel[:, i])) 
            
            # 动态调整max_allowed_change：制动时为2倍
            max_allowed_change = torch.where(
                is_braking,
                max_allowed_change * 2,
                max_allowed_change
            )
        
            # 如果差值大于最大变化率，则只增加最大允许的变化量
            # 否则直接设置为目标命令
            new_command = torch.where(
                diff_magnitude > max_allowed_change,
                self.commands_given[:, i] + torch.sign(command_diff[:, i]) * max_allowed_change,
                self.commands[:, i]  # 直接设置为目标命令
            )
            
            # 更新commands_given
            self.commands_given[:, i] = new_command
            
            # # 如果新命令与目标命令的差值小于commands_given与目标命令的差值，则更新commands_given
            # self.commands_given[:, i] = torch.where(
            #     abs(new_command-self.commands[:, i]) < abs(self.commands_given[:, i]-self.commands[:, i]),
            #     new_command,
            #     self.commands_given[:, i]
            # )

        # print("commands:",self.commands)
        # print("commands_given:",self.commands_given)

    def _update_command_curriculum(self, env_ids):
        if "tracking_lin_vel" not in self.reward_scales:
            if "tracking_lin_vel_x" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_x"]:
                    self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, self.cfg.commands.min_curriculum_x, 0.)
                    self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum_x)
            
            if "tracking_lin_vel_y" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_y"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_y"]:
                    self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, self.cfg.commands.min_curriculum_y, 0.)
                    self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum_y)

        elif torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 1.x 2.y 3.xy_mix 4.spot_turn 5.x_rotation 6.y_rotation 7.xy_mix_rotation 8.stand_still
        command_select = torch.rand(len(env_ids), device=self.device) 
        commands_proportion = torch.tensor(self.cfg.commands.commands_proportion, device=self.device)
        
        random_select_x = command_select < commands_proportion[0]
        random_select_y = (command_select < torch.sum(commands_proportion[:2])) & (command_select > torch.sum(commands_proportion[:1]))
        random_select_xy = (command_select >= torch.sum(commands_proportion[:2])) & (command_select < torch.sum(commands_proportion[:3]))
        random_select_spot_turn = (command_select >= torch.sum(commands_proportion[:3])) & (command_select < torch.sum(commands_proportion[:4]))
        random_select_x_rotation = (command_select >= torch.sum(commands_proportion[:4])) & (command_select < torch.sum(commands_proportion[:5]))
        random_select_y_rotation = (command_select >= torch.sum(commands_proportion[:5])) & (command_select < torch.sum(commands_proportion[:6]))
        random_select_xy_mix_rotation = (command_select >= torch.sum(commands_proportion[:6])) & (command_select < torch.sum(commands_proportion[:7]))
        random_select_stand_still = (command_select >= torch.sum(commands_proportion[:7]))
        self.commands[env_ids, :] = 0.0

        combined_x_env_ids = torch.cat([env_ids[random_select_x], env_ids[random_select_xy],  env_ids[random_select_x_rotation], env_ids[random_select_xy_mix_rotation]])
        combined_y_env_ids = torch.cat([env_ids[random_select_y], env_ids[random_select_xy],  env_ids[random_select_y_rotation], env_ids[random_select_xy_mix_rotation]])
        combined_angle_env_ids =torch.cat([env_ids[random_select_spot_turn], env_ids[random_select_x_rotation], env_ids[random_select_y_rotation], env_ids[random_select_xy_mix_rotation]])

        # 生成新的commands
        new_x_commands = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(combined_x_env_ids), 1), device=self.device).squeeze(1)
        new_y_commands = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(combined_y_env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            new_heading_commands = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(combined_angle_env_ids), 1), device=self.device).squeeze(1)
        else:
            new_ang_vel_commands = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(combined_angle_env_ids), 1), device=self.device).squeeze(1)
        
        # 应用新的commands
        self.commands[combined_x_env_ids, 0] = new_x_commands
        self.commands[combined_y_env_ids, 1] = new_y_commands
        if self.cfg.commands.heading_command:
            self.commands[combined_angle_env_ids, 3] = new_heading_commands
        else:
            self.commands[combined_angle_env_ids, 2] = new_ang_vel_commands

    def compute_observations(self):
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # 3 + 3 + 3 + 8 + 8 + 8 = 31 
        obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            self.reindex(dof_pos * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            #self.reindex_feet(self.contact_filt.float()-0.5),
                            # self.reindex(self.action_history_buf[:,-1])),dim=-1)
                            self.action_history_buf[:,-1]),dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),
                               torch.ones(
                                   6) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   8) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               #torch.ones(4) * noise_scales.contact_states * noise_level,
                               #torch.zeros(4),
                               torch.zeros(self.num_actions),
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        priv_latent = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.reindex_feet(self.contact_filt.float()-0.5),
            self.randomized_lag_tensor,
            #self.base_ang_vel  * self.obs_scales.ang_vel,
            # self.base_lin_vel * self.obs_scales.lin_vel,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.restitution_coeffs_tensor,
            self.motor_strength, 
            self.kp_factor,
            self.kd_factor), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            # print(self.root_states[:, 2].unsqueeze(1))
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)
        
    def _compute_torques(self, actions):
        self.dof_pos[:,[3, 7]]  = 0 
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions[:, :8] * self.cfg.control.action_scale
        actions_scaled[:, [0, 4]] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        # joint_pos_target = torch.clamp(joint_pos_target,self.dof_pos-1,self.dof_pos+1)

        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target- self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques[:,[3, 7]] = self.kp_factor[:,[3, 7]]  * 10*(joint_pos_target[:,[3, 7]]) - 0.5*self.kd_factor[:,[3, 7]] *(self.dof_vel[:,[3, 7]])   

        torques = torques * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis) - 使用缓冲后的命令
        lin_vel_x_error = torch.square(self.commands_given[:, 0] - self.base_lin_vel[:, 0])
        reward = torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)
        # print("commands:",self.commands_given[:, 0])
        # print("base_lin_vel:",self.base_lin_vel[:, 0])
        # print("reward_tracking_lin_vel_x:",reward)
        return reward

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis) - 使用缓冲后的命令
        lin_vel_y_error = torch.square(self.commands_given[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands_given[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_body_pos_to_feet_x(self):
        # 保证机体距离Los较小
        base_derivation = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1) 
        base_derivation_xy = torch.zeros_like(base_derivation[:,:,:2])
        
        # 对每个脚分别处理
        for i in range(base_derivation.shape[1]):
            # 传入完整的3维向量，然后只使用前两个分量
            rotated_3d = quat_rotate_inverse(self.base_quat, base_derivation[:, i, :])
            base_derivation_xy[:, i, :] = rotated_3d[:, :2]
        
        distance_x = torch.abs(torch.mean(base_derivation_xy[:,:,0], dim=1))
        reward = torch.exp(-distance_x / self.cfg.rewards.tracking_sigma)
        return reward
        
    def _reward_body_feet_distance_x(self):
        # 保证两腿距离
        foot_distance_world = self.foot_positions[:,0,:]-self.foot_positions[:,1,:]
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        # print("foot_distance_base:",foot_distance_base)
        foot_x_err = torch.abs(foot_distance_base[:,0])
        reward = foot_x_err**2
        return reward

    def _reward_body_feet_distance_y(self):
        # 保证两腿距离
        foot_distance_world = self.foot_positions[:,0,:]-self.foot_positions[:,1,:] 
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        foot_y_err = torch.abs(torch.abs(foot_distance_base[:,1])-self.cfg.init_state.desired_feet_distance)
        reward = foot_y_err**2
        return reward

    def _reward_body_symmetry_y(self):
        # 保证机体距离两足y方向位置一致，即不会向某一侧偏移（不是倾斜）
        foot_position_base_world = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_y_err = torch.abs(torch.abs(foot1_base[:, 1]) - torch.abs(foot2_base[:, 1]))
        reward = torch.exp(-symmetry_y_err / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_no_gait(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        both_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.*both_contact*(torch.abs(self.commands_given[:,1]) < 0.1)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_heading(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_err = torch.abs(wrap_to_pi(self.commands[:, 3] - heading))
        reward = torch.exp(-heading_err / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_body_symmetry_z(self):
        # 保证机体距离两足y方向位置一致，即不会向某一侧偏移（不是倾斜）
        foot_position_base_world = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_z_err = torch.abs(torch.abs(foot1_base[:, 2]) - torch.abs(foot2_base[:, 2]))
        reward = torch.exp(-symmetry_z_err / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_body_symmetry(self):
        foot_position_base_world = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_err = torch.sum(torch.abs(foot1_base[:, :3] - foot2_base[:, :3]), dim=1)
        reward = torch.exp(-symmetry_err / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        joint_pos_penalty = torch.sum(torch.abs(self.dof_pos[:, [0,1,2,4,5,6]] - self.default_dof_pos[:, [0,1,2,4,5,6]]), dim=1)
        lin_vel_penalty = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        return joint_pos_penalty*(torch.norm(self.commands_given[:, :2], dim=1) < 0.1) + lin_vel_penalty*(torch.norm(self.commands_given[:, :2], dim=1) < 0.1)

    def _reward_ang_vel_smoothness(self):
        ang_acc = torch.abs((self.base_ang_vel[:, 2]-self.last_root_vel[:, 5])/self.dt)
        return torch.exp(-ang_acc/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 10.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time) * first_contact, dim=1) 
        rew_airTime *= ((torch.abs(self.commands_given[:, 1]) > 0.1).to(dtype=torch.float) * 2 - 1)
        self.feet_air_time *= ~contact_filt
        # print("rew_airTime:",rew_airTime)
        return rew_airTime

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print("base_height",base_height)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
def random_quat(U):
    u1 = U[:,0].unsqueeze(1)
    u2 = U[:,1].unsqueeze(1)
    u3 = U[:,2].unsqueeze(1)
    q1 = torch.sqrt(1-u1)*torch.sin(2*torch.pi*u2)
    q2 = torch.sqrt(1-u1)*torch.cos(2*torch.pi*u2)
    q3 = torch.sqrt(u1)*torch.sin(2*torch.pi*u3)
    q4 = torch.sqrt(u1)*torch.cos(2*torch.pi*u3)
    Q = torch.cat([q1,q2,q3,q4],dim=-1)
    # q1 = torch.zeros(1, device="cuda:0", dtype=torch.float)
    # q2 = 0.7071*torch.ones(1, device="cuda:0", dtype=torch.float)
    # q3 = torch.zeros(1, device="cuda:0", dtype=torch.float)
    # q4 = 0.7071*torch.ones(1, device="cuda:0", dtype=torch.float)
    # Q = torch.cat([q1,q2,q3,q4],dim=-1)
    return Q