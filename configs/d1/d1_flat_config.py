from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# config
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot

class D1Flat(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.cfg.init_state.pos
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2] += torch_rand_float(0., 0.2, (len(env_ids), 1), device=self.device).squeeze(1) # z position within 0.2m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation
        random_roll = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        random_pitch = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self._get_base_heights() < 0

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
        # 如果使用滤波器，则对动作进行滤波
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
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]
                - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.clamp(-self.projected_gravity[:,2],0,1) * torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_base_ang_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square((self.last_root_vel[:, 3:] - self.root_states[:, 10:13]) / self.dt), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_orientation_y(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 1])

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        target = self.cfg.rewards.base_height_target

        height_scale = max(getattr(self.cfg.rewards, "base_height_scale", 0.05), 1e-6)
        deadband = max(getattr(self.cfg.rewards, "base_height_deadband", 0.01), 0.0)

        error = torch.abs(base_height - target)
        error = torch.clamp(error - deadband, min=0.0)

        return torch.clamp(-self.projected_gravity[:,2],0,1) * torch.square(error / height_scale)
    def _reward_torques(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_powers(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        #return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_powers_dist(self):
        # Penalize power dist
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.var(self.torques*self.dof_vel, dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(out_of_limits, dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_upward(self):
        # print(self.projected_gravity[:,2])
        return 1 - torch.clamp(self.projected_gravity[:,2], -1, 1)
        # return 1 - self.projected_gravity[:,2]
    
    def _reward_feet_distance(self):
        cur_footsteps_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat,
                                                                 cur_footsteps_translated[:, i, :])

        stance_length = 0.4 * torch.ones([self.num_envs, 1,], device=self.device)
        stance_width = 0.5 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_xs = torch.cat([stance_length / 2, stance_length / 2, -stance_length / 2, -stance_length / 2], dim=1)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0]).sum(dim=1)
        stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        # return stance_diff_x + stance_diff_y
        return torch.exp((-stance_diff_x - stance_diff_y)/0.05)
    
    def _reward_hip_pos(self):
        # penalty hip joint position not equal to zero
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, [0, 4, 8, 12]] - torch.zeros_like(self.default_dof_pos[:, [0, 4, 8, 12]])), dim=1)/0.05) 
        return torch.clamp(-self.projected_gravity[:,2],0,1) * reward  # torch.sum(torch.square(self.dof_pos[:, [0, 4, 8, 12]] - torch.zeros_like(self.dof_pos[:, [0, 4, 8, 12]])), dim=1)
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        # reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05) +\
        #     torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05)
        reward = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1) +\
                 torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*reward        
    
    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] < 1.
        return torch.clamp(-self.projected_gravity[:,2],0,1)*0.25 * torch.sum(contact, dim=1)
    
    # ------------ cost functions----------------
    def _cost_torque_limit(self):
        # constaint torque over limit
        #return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
        # return 1.*(torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _cost_pos_limit(self):
        # upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        # lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        # out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        # return out_limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # return 1.*(torch.sum(out_of_limits, dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(out_of_limits, dim=1)
   
    def _cost_dof_vel_limits(self):
        # return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
        # return 1.*(torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)>0.0)

        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.abs(self.dof_vel[:, [0,1,2,4,5,6,8,9,10,12,13,14]]) - self.dof_vel_limits[ [0,1,2,4,5,6,8,9,10,12,13,14]]*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    def _cost_hip_pos(self):
        # max_rad = 0.05
        # hip_err = torch.where(torch.abs(self.dof_pos[:, self.hip_joint_indices] ) < max_rad, torch.zeros_like(self.dof_pos[:, self.hip_joint_indices] ), torch.abs(self.dof_pos[:, self.hip_joint_indices]) - max_rad)
        # # print('hip_err:', hip_err)
        # return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(hip_err), dim=1)

        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # return flag * torch.mean(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0),dim=-1)
    
    def _cost_default_joint(self):
        # Penalize motion at zero commands
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.abs(self.dof_pos[:, [1,2,5,6,9,10,13,14]] - self.default_dof_pos[:,[1,2,5,6,9,10,13,14]]), dim=1)
    
class D1FlatCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 16 + 16 + 16
        n_proprio = 60 #
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 16
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.60] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

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
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True 
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            orientation = -1.0
            orientation_y = -10.0
            ang_vel_xy = -0.05
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -1.0
            feet_air_time = 0.
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            # hip_pos = 0.5
            upward = 0.5
            # feet_all_contact = -0.5
            # feet_contact_forces = -0.1
            # joint_power=-2e-5
            # powers_dist =-1.0e-5
        
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        base_height_scale = 0.05
        base_height_deadband = 0.01
        max_contact_force = 500.  # forces above this value are penalized
    class costs(LeggedRobotCfg.costs):
        num_costs = 5
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            hip_pos = 2.0
            default_joint= 0.2

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            hip_pos = 0.0
            default_joint = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.7, 0.3, 0.0, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        slope = [0, 0.6]

class D1FlatCfg_Play( D1FlatCfg ):
    class env(D1FlatCfg.env):
        num_envs = 10
    class terrain(D1FlatCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        num_rows = 5
        num_cols = 5
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0, 0, 0, 0, 0, 0, 0]
        curriculum = False
        # selected = True  # select a unique terrain type and pass all arguments
        # terrain_kwargs = {
        #     "type": "pit_terrain",  
        #     "depth": 0.5,                     
        #     "platform_size": 4.0               
        # } # Dict of arguments for selected terrain
    class noise( D1FlatCfg.noise ):
        add_noise = False
    class control ( D1FlatCfg.control ):
        use_filter = True
    class domain_rand( D1FlatCfg.domain_rand ):
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
    class commands( D1FlatCfg.commands ):
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [3.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            heading = [-0.0, 0.0]
            
class D1FlatCfgPPO( LeggedRobotCfgPPO ):
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
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 5

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 6000
        num_steps_per_env = 24
        resume = False
        resume_path = ''
