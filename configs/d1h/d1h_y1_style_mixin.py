# configs/d1h/d1h_y1_style_mixin.py

import numpy as np
import torch

from isaacgym.torch_utils import (
    torch_rand_float,
    quat_apply,
    quat_rotate_inverse,
)

from utils.math import wrap_to_pi


class D1HY1StyleMixin:
    """
    Port only y1v0h-style command sampling, command smoothing, and reward functions.

    This mixin intentionally does NOT override:
        - _create_envs()
        - compute_observations()
        - _compute_torques()
        - _reset_root_states()
        - D1H asset / URDF / contact sensor logic

    It should be used as:

        class D1HMoEDiscSimple(D1HY1StyleMixin, D1HMoEBase):
            pass

    The inheritance order matters.
    """

    # -------------------------------------------------------------------------
    # buffers
    # -------------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()

        self.commands_given = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Keep the original misspelling from y1v0h code style for consistency.
        self.odemetry_vel = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.rwd_angVelTrackPrev = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        self.commands_given[env_ids] = 0.0
        self.odemetry_vel[env_ids] = 0.0
        self.last_commands[env_ids] = 0.0
        self.rwd_angVelTrackPrev[env_ids] = 0.0

    # -------------------------------------------------------------------------
    # post physics callback / command smoothing
    # -------------------------------------------------------------------------

    def _post_physics_step_callback(self):
        """
        Keep D1H base callback first:
            - command resampling
            - heading-command processing if enabled
            - terrain height measurement
            - random push / disturbance

        Then add y1v0h-style command smoothing.

        Important:
        The uploaded y1v0h command file had compute_given_commands() commented out.
        Here it is explicitly enabled; otherwise tracking rewards would mostly
        track zero command.
        """
        super()._post_physics_step_callback()

        # D1H base callback only handles heading_command=True.
        # y1v0h integrates commanded yaw rate into commands[:, 3] when
        # heading_command=False, so we reproduce that behavior here.
        if not self.cfg.commands.heading_command:
            self.commands[:, 3] += self.commands[:, 2] * self.dt
            self.commands[:, 3] = wrap_to_pi(self.commands[:, 3])

        self.compute_given_commands()

    def compute_given_commands(self):
        """
        Smooth raw velocity commands into commands_given.

        Tracking rewards below use commands_given, not raw self.commands.
        """
        self.odemetry_vel[:, :2] = self.base_lin_vel[:, :2]
        self.odemetry_vel[:, 2] = self.base_ang_vel[:, 2]

        max_change_rates = torch.tensor(
            [
                self.cfg.commands.max_lin_vel_x_change_rate,
                self.cfg.commands.max_lin_vel_y_change_rate,
                self.cfg.commands.max_ang_vel_change_rate,
                0.0,
            ],
            device=self.device,
            dtype=torch.float,
        )

        max_change_per_step = max_change_rates * self.dt
        command_diff = self.commands[:, :3] - self.odemetry_vel[:, :3]

        for i in range(3):
            diff_magnitude = torch.abs(command_diff[:, i])
            max_allowed_change = max_change_per_step[i]

            # If target command is near zero, allow faster braking.
            is_braking = (
                (torch.abs(self.commands[:, i]) < 0.1)
                & (torch.abs(self.commands[:, i]) <= torch.abs(self.odemetry_vel[:, i]))
            )

            max_allowed_change = torch.where(
                is_braking,
                max_allowed_change * 2.0,
                max_allowed_change,
            )

            new_command = torch.where(
                diff_magnitude > max_allowed_change,
                self.commands_given[:, i]
                + torch.sign(command_diff[:, i]) * max_allowed_change,
                self.commands[:, i],
            )

            self.commands_given[:, i] = new_command

    # -------------------------------------------------------------------------
    # command curriculum / command resampling
    # -------------------------------------------------------------------------

    def _update_command_curriculum(self, env_ids):
        """
        y1v0h-style command curriculum.

        It expands x/y command ranges if the corresponding tracking reward is
        high enough.
        """
        if len(env_ids) == 0:
            return

        if "tracking_lin_vel" not in self.reward_scales:
            if "tracking_lin_vel_x" in self.reward_scales:
                if (
                    torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids])
                    / self.max_episode_length
                    > 0.8 * self.reward_scales["tracking_lin_vel_x"]
                ):
                    self.command_ranges["lin_vel_x"][0] = np.clip(
                        self.command_ranges["lin_vel_x"][0] - 0.5,
                        self.cfg.commands.min_curriculum_x,
                        0.0,
                    )
                    self.command_ranges["lin_vel_x"][1] = np.clip(
                        self.command_ranges["lin_vel_x"][1] + 0.5,
                        0.0,
                        self.cfg.commands.max_curriculum_x,
                    )

            if "tracking_lin_vel_y" in self.reward_scales:
                if (
                    torch.mean(self.episode_sums["tracking_lin_vel_y"][env_ids])
                    / self.max_episode_length
                    > 0.8 * self.reward_scales["tracking_lin_vel_y"]
                ):
                    self.command_ranges["lin_vel_y"][0] = np.clip(
                        self.command_ranges["lin_vel_y"][0] - 0.5,
                        self.cfg.commands.min_curriculum_y,
                        0.0,
                    )
                    self.command_ranges["lin_vel_y"][1] = np.clip(
                        self.command_ranges["lin_vel_y"][1] + 0.5,
                        0.0,
                        self.cfg.commands.max_curriculum_y,
                    )

        elif (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def _resample_commands(self, env_ids):
        """
        y1v0h-style command sampler.

        Command modes:
            1. x
            2. y
            3. xy_mix
            4. spot_turn
            5. x_rotation
            6. y_rotation
            7. xy_mix_rotation
            8. stand_still

        Note:
            y1v0h climb config gives 9 numbers:
                [0.45,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05]
            The original logic treats everything after sum(first 7) as stand_still.
        """
        if len(env_ids) == 0:
            return

        command_select = torch.rand(len(env_ids), device=self.device)
        commands_proportion = torch.tensor(
            self.cfg.commands.commands_proportion,
            device=self.device,
            dtype=torch.float,
        )

        random_select_x = command_select < commands_proportion[0]

        random_select_y = (
            (command_select < torch.sum(commands_proportion[:2]))
            & (command_select > torch.sum(commands_proportion[:1]))
        )

        random_select_xy = (
            (command_select >= torch.sum(commands_proportion[:2]))
            & (command_select < torch.sum(commands_proportion[:3]))
        )

        random_select_spot_turn = (
            (command_select >= torch.sum(commands_proportion[:3]))
            & (command_select < torch.sum(commands_proportion[:4]))
        )

        random_select_x_rotation = (
            (command_select >= torch.sum(commands_proportion[:4]))
            & (command_select < torch.sum(commands_proportion[:5]))
        )

        random_select_y_rotation = (
            (command_select >= torch.sum(commands_proportion[:5]))
            & (command_select < torch.sum(commands_proportion[:6]))
        )

        random_select_xy_mix_rotation = (
            (command_select >= torch.sum(commands_proportion[:6]))
            & (command_select < torch.sum(commands_proportion[:7]))
        )

        self.commands[env_ids, :] = 0.0

        combined_x_env_ids = torch.cat(
            [
                env_ids[random_select_x],
                env_ids[random_select_xy],
                env_ids[random_select_x_rotation],
                env_ids[random_select_xy_mix_rotation],
            ]
        )

        combined_y_env_ids = torch.cat(
            [
                env_ids[random_select_y],
                env_ids[random_select_xy],
                env_ids[random_select_y_rotation],
                env_ids[random_select_xy_mix_rotation],
            ]
        )

        combined_angle_env_ids = torch.cat(
            [
                env_ids[random_select_spot_turn],
                env_ids[random_select_x_rotation],
                env_ids[random_select_y_rotation],
                env_ids[random_select_xy_mix_rotation],
            ]
        )

        if len(combined_x_env_ids) > 0:
            new_x_commands = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(combined_x_env_ids), 1),
                device=self.device,
            ).squeeze(1)

            # y1v0h_evt1.py has same-sign flip logic; keep it because the
            # climb config exposes flip_same_sign_probability.
            last_x_signs = torch.sign(self.last_commands[combined_x_env_ids, 0])
            new_x_signs = torch.sign(new_x_commands)
            same_sign_mask = (last_x_signs == new_x_signs) & (last_x_signs != 0)

            flip_mask = (
                torch.rand(len(combined_x_env_ids), device=self.device)
                < self.cfg.commands.flip_same_sign_probability
            ) & same_sign_mask

            new_x_commands[flip_mask] = -new_x_commands[flip_mask]
            self.commands[combined_x_env_ids, 0] = new_x_commands

        if len(combined_y_env_ids) > 0:
            new_y_commands = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(combined_y_env_ids), 1),
                device=self.device,
            ).squeeze(1)

            last_y_signs = torch.sign(self.last_commands[combined_y_env_ids, 1])
            new_y_signs = torch.sign(new_y_commands)
            same_sign_mask = (last_y_signs == new_y_signs) & (last_y_signs != 0)

            flip_mask = (
                torch.rand(len(combined_y_env_ids), device=self.device)
                < self.cfg.commands.flip_same_sign_probability
            ) & same_sign_mask

            new_y_commands[flip_mask] = -new_y_commands[flip_mask]
            self.commands[combined_y_env_ids, 1] = new_y_commands

        if len(combined_angle_env_ids) > 0:
            if self.cfg.commands.heading_command:
                new_heading_commands = torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1],
                    (len(combined_angle_env_ids), 1),
                    device=self.device,
                ).squeeze(1)
                self.commands[combined_angle_env_ids, 3] = new_heading_commands
            else:
                new_ang_vel_commands = torch_rand_float(
                    self.command_ranges["ang_vel_yaw"][0],
                    self.command_ranges["ang_vel_yaw"][1],
                    (len(combined_angle_env_ids), 1),
                    device=self.device,
                ).squeeze(1)
                self.commands[combined_angle_env_ids, 2] = new_ang_vel_commands

        self.last_commands[env_ids] = self.commands[env_ids].clone()

    # -------------------------------------------------------------------------
    # helper
    # -------------------------------------------------------------------------

    def _y1_feet_contact_forces(self):
        """
        Use D1H force-sensor contact getter if available.
        Fall back to raw contact_forces[:, feet_indices, :].
        """
        if hasattr(self, "_get_feet_contact_forces"):
            return self._get_feet_contact_forces()
        return self.contact_forces[:, self.feet_indices, :]

    # -------------------------------------------------------------------------
    # y1v0h-style reward overrides
    # -------------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        lin_vel_x_error = torch.square(
            self.commands_given[:, 0] - self.base_lin_vel[:, 0]
        )
        return torch.exp(-lin_vel_x_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        lin_vel_y_error = torch.square(
            self.commands_given[:, 1] - self.base_lin_vel[:, 1]
        )
        return torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(
            self.commands_given[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_body_pos_to_feet_x(self):
        base_derivation = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_derivation_xy = torch.zeros_like(base_derivation[:, :, :2])

        for i in range(base_derivation.shape[1]):
            rotated_3d = quat_rotate_inverse(
                self.base_quat,
                base_derivation[:, i, :],
            )
            base_derivation_xy[:, i, :] = rotated_3d[:, :2]

        distance_x = torch.abs(torch.mean(base_derivation_xy[:, :, 0], dim=1))
        return torch.exp(-distance_x / self.cfg.rewards.tracking_sigma)

    def _reward_body_symmetry_y(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)

        foot1_base = quat_rotate_inverse(
            self.base_quat,
            foot_position_base_world[:, 0, :],
        )
        foot2_base = quat_rotate_inverse(
            self.base_quat,
            foot_position_base_world[:, 1, :],
        )

        symmetry_y_err = torch.abs(
            torch.abs(foot1_base[:, 1]) - torch.abs(foot2_base[:, 1])
        )
        return torch.exp(-symmetry_y_err / self.cfg.rewards.tracking_sigma)

    def _reward_body_symmetry_z(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)

        foot1_base = quat_rotate_inverse(
            self.base_quat,
            foot_position_base_world[:, 0, :],
        )
        foot2_base = quat_rotate_inverse(
            self.base_quat,
            foot_position_base_world[:, 1, :],
        )

        symmetry_z_err = torch.abs(
            torch.abs(foot1_base[:, 2]) - torch.abs(foot2_base[:, 2])
        )
        return torch.exp(-symmetry_z_err / self.cfg.rewards.tracking_sigma)

    def _reward_no_gait(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        both_contact = torch.sum(1.0 * contacts, dim=1) == 2
        return 1.0 * both_contact * (torch.abs(self.commands_given[:, 1]) < 0.1)

    def _reward_heading(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_err = torch.abs(wrap_to_pi(self.commands[:, 3] - heading))
        return torch.exp(-heading_err / self.cfg.rewards.tracking_sigma)

    def _reward_stand_still(self):
        # y1v0h only penalizes leg joints, not wheel accumulated positions.
        leg_joint_ids = [0, 1, 2, 4, 5, 6]

        joint_pos_penalty = torch.sum(
            torch.abs(
                self.dof_pos[:, leg_joint_ids]
                - self.default_dof_pos[:, leg_joint_ids]
            ),
            dim=1,
        )

        lin_vel_penalty = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

        zero_cmd = torch.norm(self.commands_given[:, :2], dim=1) < 0.1

        return joint_pos_penalty * zero_cmd + lin_vel_penalty * zero_cmd

    def _reward_base_height(self):
        """
        y1v0h-style base height penalty:
            square(base_height - target)

        Different from current D1H base version, which uses deadband and scaling.
        """
        if hasattr(self, "measured_heights"):
            base_height = torch.mean(
                self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
                dim=1,
            )
        else:
            base_height = self._get_base_heights()

        return torch.square(base_height - self.cfg.rewards.base_height_target)

    # -------------------------------------------------------------------------
    # extra cost functions for y1v0h 6-cost setup
    # -------------------------------------------------------------------------

    def _cost_acc_smoothness(self):
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt),
            dim=1,
        )

    def _cost_feet_contact_forces(self):
        feet_contact_forces = self._y1_feet_contact_forces()
        return torch.sum(
            (
                torch.norm(feet_contact_forces, dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _cost_stumble(self):
        feet_contact_forces = self._y1_feet_contact_forces()
        stumble = torch.any(
            torch.norm(feet_contact_forces[:, :, :2], dim=2)
            > 5.0 * torch.abs(feet_contact_forces[:, :, 2]),
            dim=1,
        )
        return stumble.float()