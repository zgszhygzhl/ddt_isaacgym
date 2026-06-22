from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from utils.math import wrap_to_pi
from global_config import ROOT_DIR
from .. import LeggedRobot

import os
import random
class Y1v0hEvt1(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.commands_target = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_rate = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 初始化每个环境的feet_distance
        self.env_feet_distances = torch.ones(self.num_envs, device=self.device) * self.cfg.init_state.desired_feet_distance
        self.last_commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_head_contact_force =  torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.try_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        root_name = [s for s in body_names if self.cfg.asset.root_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in ["left_leg_4", "right_leg_4","FL_foot","FR_foot"] :
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        penalized_contact_head_name = []
        if hasattr(self.cfg.asset, 'penalize_contacts_on_head'):
            for name in self.cfg.asset.penalize_contacts_on_head:
                penalized_contact_head_name.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 初始化body_com_tensor来存储所有body的质心
        # 假设每个body的质心是3维向量(x,y,z)
        self.body_com_tensor = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_com_tensor = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.attach_camera(i, env_handle, actor_handle)
            
            # 获取所有body的质心
            for j in range(len(body_props)):
                com_vec3 = body_props[j].com
                # 将Vec3转换为numpy数组或直接提取分量
                com_array = np.array([com_vec3.x, com_vec3.y, com_vec3.z])
                self.body_com_tensor[i, j, :] = torch.from_numpy(com_array).to(self.device).to(torch.float)
            # print(f"Environment {i} - All body COMs: {self.body_com_tensor[i, :, :]}")
            self.base_com_tensor[i, :] = self.body_com_tensor[i, 0, :]
            
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            friction_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].friction
            self.friction_coeffs_tensor = friction_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs_tensor = self.restitution_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            restitution_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].restitution
            self.restitution_coeffs_tensor = restitution_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [random.randint(0,self.cfg.domain_rand.lag_timesteps-1) for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False
        else:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [self.cfg.domain_rand.lag_timesteps-1 for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False

        # self.root_indices = torch.zeros(len(root_name), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(root_name)):
        #     self.root_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], root_name[i])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        self.penalised_contact_head_index = torch.zeros(len(penalized_contact_head_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_head_name)):
            self.penalised_contact_head_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_head_name[i])
        
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    def step(self, actions):
        
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        return super().step(actions)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        
        # 对每个环境随机采样desired_feet_distance
        self._resample_feet_distance(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.last_distance[env_ids] = 0.
        self.last_head_contact_force[env_ids]=0.
        self.try_times[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.contact_times_stair[env_ids] = 0.
        self.last_contact_foot[env_ids] = 0.
        self.robot_rigid_body_on_air[env_ids] = 1.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.last_contact_forces[env_ids] = torch.zeros_like(self.contact_forces[env_ids])
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["min_command_x"] = self.command_ranges["lin_vel_x"][0]
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["min_command_y"] = self.command_ranges["lin_vel_y"][0]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["min_command_angle"] = self.command_ranges["ang_vel_yaw"][0]
            self.extras["episode"]["max_command_angle"] = self.command_ranges["ang_vel_yaw"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        
        if len(env_ids) > 0:
            # 基于概率决定哪些agent进行resample
            if hasattr(self.cfg.commands, 'resample_probability'):
                resample_prob = self.cfg.commands.resample_probability
                # 为每个到达resampling时间的agent生成随机数
                random_probs = torch.rand(len(env_ids), device=self.device)
                # 只有概率小于resample_probability的agent才进行resample
                resample_env_ids = env_ids[random_probs < resample_prob]
                if len(resample_env_ids) > 0:
                    self._resample_commands(resample_env_ids)
            else:
                # 如果没有设置概率，则所有agent都进行resample（保持原有行为）
                self._resample_commands(env_ids)
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(1.0*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        else:
            self.commands[:, 3] += self.commands[:, 2]*self.dt
            self.commands[:, 3] = wrap_to_pi(self.commands[:, 3])

        if self.cfg.terrain.measure_heights:
            # if self.global_counter % self.cfg.depth.update_interval == 0:
            self.measured_heights = self._get_heights()
            
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    
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

            # Generate random rates for smooth command transitions
            # Use max_curriculum as a scaling factor, not as a dimension
            self.commands_rate = torch.rand(
                self.num_envs, self.cfg.commands.num_commands, 
                dtype=torch.float, device=self.device, requires_grad=False
            ) * self.cfg.commands.max_curriculum

        elif torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            
        # Always ensure commands_rate has reasonable values, even if curriculum doesn't trigger
        if torch.all(self.commands_rate == 0):
            self.commands_rate = torch.rand(
                self.num_envs, self.cfg.commands.num_commands, 
                dtype=torch.float, device=self.device, requires_grad=False
            ) * self.cfg.commands.max_curriculum

    def _resample_feet_distance(self, env_ids):
        """为每个环境随机采样desired_feet_distance
        
        Args:
            env_ids (torch.Tensor): 需要采样的环境ID
        """
        if hasattr(self.cfg.init_state, 'feet_distance_range'):
            # 获取配置的feet_distance范围
            min_distance = self.cfg.init_state.feet_distance_range[0]
            max_distance = self.cfg.init_state.feet_distance_range[1]
            
            # 为每个环境生成随机的feet_distance
            random_feet_distances = torch_rand_float(
                min_distance, max_distance, 
                (len(env_ids), 1), self.device
            ).squeeze(1)
            
            # 将随机采样的feet_distance存储到每个环境
            # 这里我们需要为每个环境创建一个独立的feet_distance值
            if not hasattr(self, 'env_feet_distances'):
                # 初始化时创建tensor
                self.env_feet_distances = torch.ones(self.num_envs, device=self.device) * self.cfg.init_state.desired_feet_distance
            
            # 更新指定环境的feet_distance
            self.env_feet_distances[env_ids] = random_feet_distances
            
            # 打印调试信息（可选）
            # if len(env_ids) > 0 and hasattr(self, 'common_step_counter') and self.common_step_counter % 1000 == 0:
            #     print(f"Resampled feet_distance for {len(env_ids)} envs. "
            #           f"Range: [{min_distance:.3f}, {max_distance:.3f}], "
            #           f"Sample values: {random_feet_distances[:5].cpu().numpy()}")

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
        
        # 检查x方向commands的正负号是否相同
        if len(combined_x_env_ids) > 0:
            last_x_signs = torch.sign(self.last_commands[combined_x_env_ids, 0])
            new_x_signs = torch.sign(new_x_commands)
            same_sign_mask = (last_x_signs == new_x_signs) & (last_x_signs != 0)  # 排除0值

            flip_mask = (torch.rand(len(combined_x_env_ids), device=self.device) < self.cfg.commands.flip_same_sign_probability) & same_sign_mask
            new_x_commands[flip_mask] = -new_x_commands[flip_mask]
        
        # 检查y方向commands的正负号是否相同
        if len(combined_y_env_ids) > 0:
            last_y_signs = torch.sign(self.last_commands[combined_y_env_ids, 1])
            new_y_signs = torch.sign(new_y_commands)
            same_sign_mask = (last_y_signs == new_y_signs) & (last_y_signs != 0)  # 排除0值
            
            # 有30%的概率取反相同正负号的commands
            flip_mask = (torch.rand(len(combined_y_env_ids), device=self.device) < self.cfg.commands.flip_same_sign_probability) & same_sign_mask
            new_y_commands[flip_mask] = -new_y_commands[flip_mask]
        
        # 应用新的commands
        self.commands[combined_x_env_ids, 0] = new_x_commands
        self.commands[combined_y_env_ids, 1] = new_y_commands
        if self.cfg.commands.heading_command:
            self.commands[combined_angle_env_ids, 3] = new_heading_commands
        else:
            self.commands[combined_angle_env_ids, 2] = new_ang_vel_commands
        
        # 保存当前commands作为下次比较的基准
        if not hasattr(self, 'last_commands') or self.last_commands is None:
            self.last_commands = torch.zeros_like(self.commands)
        self.last_commands[env_ids] = self.commands[env_ids].clone()

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
        # Tracking of linear velocity commands (x axis)
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        reward = torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)
        # print("commands:",self.commands[:, 0])
        # print("base_lin_vel:",self.base_lin_vel[:, 0])
        # print("reward_tracking_lin_vel_x:",reward)
        return reward

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.cfg.rewards.tracking_sigma)

    def _reward_body_pos_to_feet_x(self):
        # 保证机体距离Los较小
        base_derivation = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1) 
        distance = torch.abs(torch.mean(base_derivation[:,:,0], dim=1))
        reward = torch.exp(-distance / self.cfg.rewards.tracking_sigma)
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
        # wl2 y-distance = 2*(0.0775+0.09+0.06935) = 2*0.2368 = 0.4736

        # desired_feet_dist = getattr(self, 'env_feet_distances', None)
        # if desired_feet_dist is None:
        #     desired_feet_dist = self.cfg.init_state.desired_feet_distance
        # foot_y_err = torch.abs(torch.abs(foot_distance_base[:,1])-desired_feet_dist)
        foot_y_err = torch.abs(torch.abs(foot_distance_base[:,1])-self.cfg.init_state.desired_feet_distance)
        # print("torch.abs(foot_distance_base[:,1]):",torch.abs(foot_distance_base[:,1]))
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
        return 1.*both_contact*(torch.abs(self.commands[:,1]) < 0.1)

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
        return joint_pos_penalty*(torch.norm(self.commands[:, :2], dim=1) < 0.1) + lin_vel_penalty*(torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_ang_vel_smoothness(self):
        ang_acc = torch.abs((self.base_ang_vel[:, 2]-self.last_root_vel[:, 5])/self.dt)
        return torch.exp(-ang_acc/self.cfg.rewards.tracking_sigma)
    
    def _reward_body_vel_to_feet_x(self):
        # 保证机体距离Los较小
        base_derivation = self.foot_positions - self.root_states[:, 0:3].unsqueeze(1) 
        distance = torch.mean(base_derivation[:,:,0], dim=1)
        base_derivation_vel = torch.clip((distance- self.last_distance)/self.dt, -10.0, 10.0)
        base_derivation_vel = torch.abs(base_derivation_vel)
        self.last_distance = distance
        reward = torch.exp(-base_derivation_vel / self.cfg.rewards.tracking_sigma)
        return reward

    