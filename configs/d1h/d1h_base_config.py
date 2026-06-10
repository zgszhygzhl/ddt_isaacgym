from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import os
import random
import torch
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi

class D1HMoEBase(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4]
        self.foot_joint_indices = [3, 7]
        self.bad_contact_time = torch.zeros(self.num_envs, device=self.device)

    def _set_zero_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, :3] = 0.0
        if self.cfg.commands.heading_command:
            _, _, heading = get_euler_xyz(self.base_quat[env_ids])
            heading = torch.where(heading > torch.pi, heading - 2 * torch.pi, heading)
            self.commands[env_ids, 3] = heading

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        zero_ratio = getattr(self.cfg.commands, "zero_command_ratio", 0.0)
        if zero_ratio > 0.0:
            zero_mask = torch.rand(len(env_ids), device=self.device) < zero_ratio
            self._set_zero_commands(env_ids[zero_mask])

        startup_freeze_time = getattr(self.cfg.commands, "startup_freeze_time", 0.0)
        if startup_freeze_time > 0.0:
            startup_mask = (self.episode_length_buf[env_ids].float() * self.dt) < startup_freeze_time
            self._set_zero_commands(env_ids[startup_mask])

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        resample_ids = []

        periodic_env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        if len(periodic_env_ids) > 0:
            resample_ids.append(periodic_env_ids)

        startup_freeze_time = getattr(self.cfg.commands, "startup_freeze_time", 0.0)
        if startup_freeze_time > 0.0:
            startup_steps = int(np.ceil(startup_freeze_time / self.dt))
            startup_release_env_ids = (self.episode_length_buf == startup_steps).nonzero(as_tuple=False).flatten()
            if len(startup_release_env_ids) > 0:
                resample_ids.append(startup_release_env_ids)

        if len(resample_ids) > 0:
            env_ids = torch.unique(torch.cat(resample_ids))
            self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(1.0 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.feet_heights = self._get_feet_heights()
            self.feet_body_frame_height = self._get_feet_local_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.bad_contact_time[env_ids] = 0
    
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
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in feet_names:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False
            sensor_options.enable_constraint_solver_forces = True
            sensor_options.use_world_frame = True
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose, sensor_options)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        penalized_contact_head_names = []
        for name in self.cfg.asset.penalize_contact_head_on:
            penalized_contact_head_names.extend([s for s in body_names if name in s])
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

        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[2] += self.base_init_state[2]
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

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.penalised_contact_head_index = torch.zeros(len(penalized_contact_head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_head_names)):
            self.penalised_contact_head_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_head_names[i])

    def _reset_dofs(self, env_ids):
        """Reset DOF states.

        If deterministic_reset=True, use exact default joint angles.
        Otherwise use small randomization.
        """
        deterministic_reset = getattr(self.cfg.env, "deterministic_reset", False)

        if deterministic_reset:
            self.dof_pos[env_ids] = self.reset_dof_pos
        else:
            self.dof_pos[env_ids] = self.reset_dof_pos * torch_rand_float(
                0.9,
                1.1,
                (len(env_ids), self.num_dof),
                device=self.device,
            )

        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        deterministic_reset = getattr(self.cfg.env, "deterministic_reset", False)
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation
        if deterministic_reset:
            self.root_states[env_ids, 3:7] = self.base_init_state[3:7]
            self.root_states[env_ids, 7:13] = 0.0
        else:
            random_roll = torch_rand_float(-0.05, 0.05, (len(env_ids),1), device=self.device).squeeze(1)
            random_pitch = torch_rand_float(-0.05, 0.05, (len(env_ids),1), device=self.device).squeeze(1)
            random_yaw = torch_rand_float(-0.05, 0.05, (len(env_ids),1), device=self.device).squeeze(1)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.02, 0.02, (len(env_ids), 6), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        bad_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        episode_time = self.episode_length_buf.float() * self.dt
        past_grace_time = episode_time >= self.cfg.env.contact_termination_grace_time
        self.bad_contact_time = torch.where(
            bad_contact & past_grace_time,
            self.bad_contact_time + self.dt,
            torch.zeros_like(self.bad_contact_time),
        )
        self.reset_buf = self.bad_contact_time >= self.cfg.env.contact_termination_duration
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self._get_base_heights() < self.cfg.env.min_base_height_for_reset

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras
    
    def _compute_torques(self, actions):

        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if "tracking_lin_vel" not in self.reward_scales:
            if "tracking_lin_vel_x" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.75 * self.reward_scales["tracking_lin_vel_x"]:
                    self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum_x_back, 0.)
                    self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum_x)
            
            if "tracking_lin_vel_y" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_y"][env_ids]) / self.max_episode_length > 0.75 * self.reward_scales["tracking_lin_vel_y"]:
                    self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.2, -self.cfg.commands.max_curriculum_y, 0.)
                    self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.2, 0., self.cfg.commands.max_curriculum_y)

        elif "tracking_lin_vel" in self.reward_scales:
            if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.75 * self.reward_scales["tracking_lin_vel"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2 , -self.cfg.commands.max_curriculum_x_back, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum_x)

    #------------ reward functions----------------
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        lin_vel_x_error = torch.clamp(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 0]))/(0.25+torch.abs(self.commands[:, 0]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_x_error/tracking_sigma)
        return reward
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        lin_vel_y_error = torch.clamp(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 1]))/(0.25+torch.abs(self.commands[:, 1]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_y_error/tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 2]))/(0.25+torch.abs(self.commands[:, 2]))
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/tracking_sigma)

    def _reward_stand_still(self):
        zero_lin_cmd = torch.norm(self.commands[:, :2], dim=1) < self.cfg.commands.zero_lin_vel_threshold
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * zero_lin_cmd

    def _reward_zero_base_vel(self):
        zero_lin_cmd = torch.norm(self.commands[:, :2], dim=1) < self.cfg.commands.zero_lin_vel_threshold
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * zero_lin_cmd

    def _reward_zero_wheel_vel(self):
        zero_lin_cmd = torch.norm(self.commands[:, :2], dim=1) < self.cfg.commands.zero_lin_vel_threshold
        return torch.sum(torch.square(self.dof_vel[:, self.foot_joint_indices]), dim=1) * zero_lin_cmd

    def _reward_zero_yaw_rate(self):
        zero_lin_cmd = torch.norm(self.commands[:, :2], dim=1) < self.cfg.commands.zero_lin_vel_threshold
        zero_yaw_cmd = torch.abs(self.commands[:, 2]) < self.cfg.commands.zero_yaw_threshold
        return torch.square(self.base_ang_vel[:, 2]) * (zero_lin_cmd & zero_yaw_cmd)

    def _reward_upward(self):
        return 1 - torch.clamp(self.projected_gravity[:,2], -1, 1)

    def _reward_body_pos_to_feet_x(self):
        # keep body relative position to Los small
        base_derivation = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_derivation_xyz = torch.zeros_like(base_derivation[:,:,:])
        
        for i in range(base_derivation.shape[1]):
            base_derivation_xyz[:, i, :] = quat_rotate_inverse(self.base_quat, base_derivation[:, i, :])
        
        distance_x = torch.abs(torch.mean(base_derivation_xyz[:,:,0], dim=1))
        reward = torch.exp(-distance_x / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_body_feet_distance_x(self):
        foot_distance_world = self.feet_pos[:,0,:]-self.feet_pos[:,1,:]
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        foot_x_err = torch.abs(foot_distance_base[:,0])
        reward = foot_x_err**2
        return reward

    def _reward_body_feet_distance_y(self):
        foot_distance_world = self.feet_pos[:,0,:]-self.feet_pos[:,1,:] 
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        foot_y_err = torch.abs(torch.abs(foot_distance_base[:,1])-self.cfg.init_state.desired_feet_distance)
        reward = foot_y_err**2
        return reward

    def _reward_body_symmetry_y(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_y_err = torch.abs(torch.abs(foot1_base[:, 1]) - torch.abs(foot2_base[:, 1]))
        reward = torch.exp(-symmetry_y_err / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_body_symmetry_z(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_z_err = torch.abs(torch.abs(foot1_base[:, 2]) - torch.abs(foot2_base[:, 2]))
        reward = torch.exp(-symmetry_z_err / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_collision_head(self):
        head_contact_force = torch.norm(self.contact_forces[:, self.penalised_contact_head_index, :], dim=-1)   
        return torch.sum(1.*(head_contact_force > 10), dim=1)

    def _reward_no_jump(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 10
        on_jump = torch.sum(1.*contacts, dim=1)==0
        return 1.*on_jump

    def _reward_heading(self):
        if self.cfg.commands.heading_command:
            _, _, heading = get_euler_xyz(self.base_quat)
            heading = torch.where(heading > torch.pi, heading - 2 * torch.pi, heading) # limit heading to [-pi, pi]
            reward = torch.square(heading - self.commands[:, 3])
            return reward 
        else:
            return 0
    
    def _reward_collision_hard(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 100), dim=1)
    
    
    # # ------------ cost functions----------------
    def _cost_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0),dim=-1)
    
    def _cost_default_joint(self):
        # Penalize motion at zero commands
        non_hip_foot_indices = [i for i in range(self.num_dof) if i not in self.hip_joint_indices and i not in self.foot_joint_indices]
        return torch.sum(torch.abs(self.dof_pos[:, non_hip_foot_indices] - self.default_dof_pos[:,non_hip_foot_indices]), dim=1)
    
class D1HMoEBaseCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  2 + 1 + 4 + 1 + 1 + 8 + 8 + 8
        n_proprio = 36 # 3+3+3+3+8+8+8
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 8
        contact_termination_grace_time = 2.0
        contact_termination_duration = 0.03
        min_base_height_for_reset = 0.03
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        reset_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,

            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,

            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
        desired_feet_distance = 0.38

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.,
                     'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.commands ):
        curriculum = True 
        max_curriculum = 3.0
        max_curriculum_x = 3
        max_curriculum_x_back = 3
        max_curriculum_y = 1.0
        max_curriculum_yaw = 2.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        zero_command_ratio = 0.22
        zero_lin_vel_threshold = 0.05
        zero_yaw_threshold = 0.02
        startup_freeze_time = 0.1

        class ranges:
            lin_vel_x = [-0.6, 0.6]  # min max [m/s]
            lin_vel_y = [-0.1, 0.1]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1h/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1h"
        penalize_contacts_on = ["thigh", "calf", "base"]
        penalize_contact_head_on = ["base"]
        terminate_after_contacts_on = ["base"]
        use_force_sensor_contacts = True
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = -2e-5
            termination = -100.0
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 25.0
            tracking_lin_vel_y = 12.0
            tracking_ang_vel = 24.0
            lin_vel_z = -5.0
            orientation = -18.0  #projected_gravity 前两个分量的平方和，惩罚机身倾斜
            ang_vel_xy = -0.18   #x、y 轴角速度的平方和，惩罚前后翻、左右晃的角速度
            dof_acc = -2.5e-7
            base_height = -2.5
            feet_air_time = 0.5
            collision = -18.0
            feet_stumble = 0.0
            action_rate = -0.1
            stand_still = -2.0
            zero_base_vel = -16.0
            zero_wheel_vel = -0.05
            zero_yaw_rate = -25.0
            upward = 3.0
            heading = -6.0
            # collision_head = -100.0
            body_pos_to_feet_x = 1.0
            body_feet_distance_x = -4.0
            body_feet_distance_y = -13.0
            body_symmetry_y = 0.5
            body_symmetry_z = 0.2
            collision_hard = -30.0
        
        only_positive_rewards = False
        tracking_sigma = 0.07  # tracking reward = exp(-error^2/sigma)
        distance_sigma = 0.08  # distance reward = exp(-distance^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        base_height_scale = 0.05
        base_height_deadband = 0.01
        max_contact_force = 500.  # forces above this value are penalized
    class costs(LeggedRobotCfg.costs):
        num_costs = 3
        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            # hip_pos = 0.0
            # default_joint= 0.0

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # hip_pos = 0.0
            # default_joint = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # 只保留平地到轻微斜坡，用作 base policy 的默认滚动训练场景。
        # 在 terrain.py 中，smooth slope 会随着 curriculum 的 difficulty 从平地逐步过渡到轻微坡面。
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        step_height = [0.0, 0.0]
        step_width_range = [0.20, 0.82]
        slope = [0.0, 0.12]
        # mesh_type = 'plane'
        # curriculum = True
        # measure_heights = True


    class sim(LeggedRobotCfg.sim):
        dt = 0.0025
class D1HMoEBaseCfg_Play( D1HMoEBaseCfg ):
    class env(D1HMoEBaseCfg.env):
        num_envs = 10
        deterministic_reset = True
    class init_state(D1HMoEBaseCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0, 0.0, 0.0, 1]
        reset_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
    class terrain(D1HMoEBaseCfg.terrain):
        mesh_type = 'trimesh'
        num_rows = 1
        num_cols = 1
        curriculum = True
        max_init_terrain_level = 0
        selected = False
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.2
        step_height = [0.0, 0.0]
        step_width = 0.3
        slope = [0.05, 0.05]
        # mesh_type = 'plane'
        # curriculum = True
        # measure_heights = True


    class noise( D1HMoEBaseCfg.noise ):
        add_noise = False
    class control ( D1HMoEBaseCfg.control ):
        use_filter = True
    class domain_rand( D1HMoEBaseCfg.domain_rand ):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_friction = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False
    class commands( D1HMoEBaseCfg.commands ):
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [1.3, 1.3]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            heading = [0.0, 0.0]
            # lin_vel_x = [-.0,.0]  # min max [m/s]
            # lin_vel_y = [-.0, .0]  # min max [m/s]
            # ang_vel_yaw = [-.0, .0]  # min max [rad/s]
            # heading = [-.0, .0]
            
class D1HMoEBaseCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        barlow_actor_hidden_dims = [512, 256, 128]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 3

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1h_moe_base'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 200
        max_iterations = 40000
        num_steps_per_env = 24
        record_video = True
        video_interval = 500
        video_duration = 3.0 #秒
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 640
        video_tile_height = 360
        video_width = 1280
        video_height = 720
        resume = False
        resume_path = ''

