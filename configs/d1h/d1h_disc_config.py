import math

import torch
from utils.stair_ik_feedforward import compute_stair_ik_ff_offsets_b

from .d1h_base_config import D1HMoEBase, D1HMoEBaseCfg, D1HMoEBaseCfgPPO


class D1HMoEDisc(D1HMoEBase):
    def _init_buffers(self):
        super()._init_buffers()
        self.step_contact_timer = torch.zeros(self.num_envs, device=self.device)
        self.step_jam_time = torch.zeros(self.num_envs, device=self.device)
        self.step_imbalance_time = torch.zeros(self.num_envs, device=self.device)
        self.stair_lift_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.stair_lift_active = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.stair_lift_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stair_contact_hist = torch.zeros(self.num_envs, 2, 6, device=self.device)
        self.last_stair_ff_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_stair_trigger = torch.zeros(self.num_envs, 2, device=self.device)
        self.stair_followup_used = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.stair_ff_cooldown_until = torch.zeros(self.num_envs, 2, device=self.device)
        self.stair_ff_contact_hit_sum = torch.zeros(self.num_envs, device=self.device)
        self.stair_ff_active_sum = torch.zeros(self.num_envs, device=self.device)
        self.last_stair_ff_teacher_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_stair_ff_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.stair_ff_anneal_adaptive_iter = torch.zeros((), device=self.device)
        self.stair_ff_anneal_last_train_iter = torch.zeros((), device=self.device)
        self.stair_ff_anneal_speed_scale = torch.ones((), device=self.device)
        self.stair_ff_anneal_initialized = False

    def reset_idx(self, env_ids):
        ff_contact_hit_ratio = None
        ff_active_ratio = None
        if len(env_ids) > 0 and hasattr(self, "stair_ff_contact_hit_sum"):
            denom = torch.clamp(self.episode_length_buf[env_ids].float(), min=1.0)
            ff_contact_hit_ratio = torch.mean(self.stair_ff_contact_hit_sum[env_ids] / denom)
            ff_active_ratio = torch.mean(self.stair_ff_active_sum[env_ids] / denom)

        super().reset_idx(env_ids)

        if hasattr(self, "_terrain_forward_progress"):
            self.extras["episode"]["terrain_forward_progress"] = torch.mean(
                self._terrain_forward_progress
            )
            self.extras["episode"]["terrain_promote_rate"] = torch.mean(
                self._terrain_move_up.float()
            )
            if hasattr(self, "_terrain_move_down"):
                self.extras["episode"]["terrain_move_down_rate"] = torch.mean(
                    self._terrain_move_down.float()
                )
        if ff_contact_hit_ratio is not None:
            self.extras["episode"]["stair_ff_contact_hit_ratio"] = ff_contact_hit_ratio
            self.extras["episode"]["stair_ff_active_ratio"] = ff_active_ratio
            self._update_stair_ff_adaptive_anneal(ff_contact_hit_ratio, ff_active_ratio)
            self.extras["episode"]["stair_ff_anneal_scale"] = self._get_stair_ff_anneal_scale().detach()
            if hasattr(self, "stair_ff_anneal_speed_scale"):
                self.extras["episode"]["stair_ff_anneal_speed_scale"] = self.stair_ff_anneal_speed_scale.detach()
            if hasattr(self, "stair_ff_anneal_adaptive_iter"):
                self.extras["episode"]["stair_ff_anneal_adaptive_iter"] = self.stair_ff_anneal_adaptive_iter.detach()

        self.step_contact_timer[env_ids] = 0.0
        self.step_jam_time[env_ids] = 0.0
        self.step_imbalance_time[env_ids] = 0.0
        self.stair_lift_phase[env_ids] = 0.0
        self.stair_lift_active[env_ids] = False
        self.stair_lift_side[env_ids] = 0
        self.stair_contact_hist[env_ids] = 0.0
        self.last_stair_ff_signal[env_ids] = 0.0
        self.last_stair_trigger[env_ids] = 0.0
        self.stair_followup_used[env_ids] = False
        self.stair_ff_cooldown_until[env_ids] = 0.0
        self.stair_ff_contact_hit_sum[env_ids] = 0.0
        self.stair_ff_active_sum[env_ids] = 0.0
        if hasattr(self, "last_stair_ff_teacher_offsets"):
            self.last_stair_ff_teacher_offsets[env_ids] = 0.0
        self.last_stair_ff_offsets[env_ids] = 0.0

    def step(self, actions):
        actions = self._apply_stair_feedforward(actions)
        clipped_actions = torch.clamp(
            actions.to(self.device),
            -float(self.cfg.normalization.clip_actions),
            float(self.cfg.normalization.clip_actions),
        )
        self._stair_blended_history_action = self._get_action_history_actions(clipped_actions)
        return super().step(actions)

    def compute_observations(self):
        if hasattr(self, "_stair_blended_history_action"):
            history_action = self._stair_blended_history_action
            if hasattr(self, "reset_buf"):
                history_action = torch.where(
                    self.reset_buf.unsqueeze(1),
                    torch.zeros_like(history_action),
                    history_action,
                )
            self.action_history_buf[:, -1] = history_action
        return super().compute_observations()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_step_contact_state()
        self._update_step_imbalance_state()
        if hasattr(self, "stair_ff_vis_first_pulse"):
            self.stair_ff_vis_first_pulse = torch.clamp(
                self.stair_ff_vis_first_pulse - self.dt,
                min=0.0,
            )
        if hasattr(self, "stair_ff_vis_follow_pulse"):
            self.stair_ff_vis_follow_pulse = torch.clamp(
                self.stair_ff_vis_follow_pulse - self.dt,
                min=0.0,
            )



    def get_video_debug_state(self, env_ids):
        """Return lightweight stair feedforward state for train-video overlays.

        The runner calls this with `video_env_ids`.  Values are converted to CPU
        lists so the video code can draw them without touching CUDA tensors.
        """
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return {}

        zeros_bool = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        zeros_long = torch.zeros(env_ids_t.numel(), dtype=torch.long, device=self.device)
        zeros_float = torch.zeros(env_ids_t.numel(), dtype=torch.float, device=self.device)

        if hasattr(self, "stair_lift_active"):
            left_active = self.stair_lift_active[env_ids_t, 0]
            right_active = self.stair_lift_active[env_ids_t, 1]
            ff_active = self.stair_lift_active[env_ids_t].any(dim=1)
        else:
            left_active = right_active = ff_active = zeros_bool

        if hasattr(self, "stair_lift_phase"):
            left_phase = self.stair_lift_phase[env_ids_t, 0]
            right_phase = self.stair_lift_phase[env_ids_t, 1]
        else:
            left_phase = right_phase = zeros_float

        if hasattr(self, "stair_ff_vis_first_pulse"):
            first_pulse = self.stair_ff_vis_first_pulse[env_ids_t] > 0.0
        else:
            first_pulse = zeros_bool
        if hasattr(self, "stair_ff_vis_follow_pulse"):
            follow_pulse = self.stair_ff_vis_follow_pulse[env_ids_t] > 0.0
        else:
            follow_pulse = zeros_bool

        first_leg = self.stair_first_leg[env_ids_t] if hasattr(self, "stair_first_leg") else zeros_long
        terrain_level = self.terrain_levels[env_ids_t] if hasattr(self, "terrain_levels") else zeros_long

        return {
            "ff_active": ff_active.detach().cpu().tolist(),
            "left_active": left_active.detach().cpu().tolist(),
            "right_active": right_active.detach().cpu().tolist(),
            "left_phase": left_phase.detach().cpu().tolist(),
            "right_phase": right_phase.detach().cpu().tolist(),
            "first_pulse": first_pulse.detach().cpu().tolist(),
            "follow_pulse": follow_pulse.detach().cpu().tolist(),
            "first_leg": first_leg.detach().cpu().tolist(),
            "terrain_level": terrain_level.detach().cpu().tolist(),
        }

    def _get_stair_ff_contact_forces(self):
        if hasattr(self, "force_sensor_tensor") and torch.is_tensor(self.force_sensor_tensor):
            contact_force_vec = self.force_sensor_tensor[:, :, :3]
        elif hasattr(self, "contact_forces") and hasattr(self, "feet_indices"):
            contact_force_vec = self.contact_forces[:, self.feet_indices, :]
        else:
            return None

        if contact_force_vec.shape[1] < 2:
            return None

        force_axis = getattr(self.cfg.control, "stair_ff_contact_force_axis", "horizontal")
        if force_axis == "horizontal":
            contact_forces = torch.norm(contact_force_vec[:, :, :2], dim=-1)
        elif force_axis == "vertical":
            contact_forces = torch.abs(contact_force_vec[:, :, 2])
        else:
            contact_forces = torch.norm(contact_force_vec, dim=-1)
        return contact_forces[:, :2]

    def _get_stair_ff_trigger_arm(self):
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        cmd_x = self.commands[:, 0]
        arm_condition = cmd_x > min_cmd_x

        min_travel = getattr(self.cfg.control, "stair_ff_min_forward_travel", 0.15)
        if min_travel > 0.0:
            forward_travel = self.root_states[:, 0] - self.env_origins[:, 0]
            arm_condition = arm_condition & (forward_travel >= min_travel)

        return arm_condition

    def _get_stair_ff_gate(self):
        return self.stair_lift_active.any(dim=1).float()

    def _get_action_history_actions(self, actions):
        """Expose the action actually sent to the joint target, in policy-action units."""
        ff_offset = self._get_stair_ff_joint_target_offsets()
        action_scale = torch.full_like(actions, float(self.cfg.control.action_scale))
        action_scale[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction
        equivalent_ff = torch.where(
            torch.abs(action_scale) > 1e-6,
            ff_offset / action_scale,
            torch.zeros_like(ff_offset),
        )
        clip_actions = float(self.cfg.normalization.clip_actions)
        return torch.clamp(actions + equivalent_ff, -clip_actions, clip_actions)

    def _update_stair_ff_adaptive_anneal(self, ff_contact_hit_ratio, ff_active_ratio):
        if not getattr(self.cfg.control, "stair_ff_anneal_adaptive_enabled", False):
            return
        if ff_contact_hit_ratio is None:
            return

        progress = getattr(self, "_terrain_forward_progress", None)
        if progress is None:
            return

        progress_mean = torch.mean(progress).detach()
        promote_rate = torch.mean(getattr(self, "_terrain_move_up", torch.zeros_like(progress)).float()).detach()
        move_down_rate = torch.mean(getattr(self, "_terrain_move_down", torch.zeros_like(progress)).float()).detach()
        contact_hit = ff_contact_hit_ratio.detach()

        good_progress = float(getattr(self.cfg.control, "stair_ff_anneal_good_progress", 2.5))
        bad_progress = float(getattr(self.cfg.control, "stair_ff_anneal_bad_progress", 1.8))
        good_promote = float(getattr(self.cfg.control, "stair_ff_anneal_good_promote_rate", 0.25))
        bad_move_down = float(getattr(self.cfg.control, "stair_ff_anneal_bad_move_down_rate", 0.12))
        min_contact_hit = float(getattr(self.cfg.control, "stair_ff_anneal_min_contact_hit_ratio", 0.05))
        good_contact_hit = float(getattr(self.cfg.control, "stair_ff_anneal_good_contact_hit_ratio", 0.12))

        fast_scale = float(getattr(self.cfg.control, "stair_ff_anneal_fast_speed", 1.5))
        normal_scale = float(getattr(self.cfg.control, "stair_ff_anneal_normal_speed", 1.0))
        slow_scale = float(getattr(self.cfg.control, "stair_ff_anneal_slow_speed", 0.35))
        smoothing = float(getattr(self.cfg.control, "stair_ff_anneal_speed_smoothing", 0.3))
        smoothing = min(max(smoothing, 0.0), 1.0)

        good = ((progress_mean >= good_progress) | (promote_rate >= good_promote)) & (contact_hit >= good_contact_hit)
        bad = (progress_mean <= bad_progress) | (move_down_rate >= bad_move_down) | (contact_hit <= min_contact_hit)

        target_speed = torch.as_tensor(normal_scale, device=self.device)
        target_speed = torch.where(good, torch.as_tensor(fast_scale, device=self.device), target_speed)
        target_speed = torch.where(bad, torch.as_tensor(slow_scale, device=self.device), target_speed)
        self.stair_ff_anneal_speed_scale = (
            (1.0 - smoothing) * self.stair_ff_anneal_speed_scale + smoothing * target_speed
        ).clamp(min=0.0, max=max(fast_scale, normal_scale, 1.0))
    def _get_stair_ff_anneal_scale(self):
        if not getattr(self.cfg.control, "stair_ff_anneal_enabled", False):
            return torch.ones((), device=self.device)

        override_scale = getattr(self.cfg.control, "stair_ff_anneal_override_scale", None)
        if override_scale is not None:
            return torch.as_tensor(float(override_scale), device=self.device).clamp(0.0, 1.0)

        steps_per_iter = max(int(getattr(self.cfg.control, "stair_ff_anneal_steps_per_iter", 32)), 1)
        local_train_iter = torch.as_tensor(
            float(getattr(self, "common_step_counter", 0)) / steps_per_iter,
            device=self.device,
        )
        iter_offset = float(getattr(self.cfg.control, "stair_ff_anneal_iter_offset", 0.0))
        if getattr(self.cfg.control, "stair_ff_anneal_adaptive_enabled", False):
            if not getattr(self, "stair_ff_anneal_initialized", False):
                self.stair_ff_anneal_adaptive_iter = torch.as_tensor(iter_offset, device=self.device)
                self.stair_ff_anneal_last_train_iter = local_train_iter.detach()
                self.stair_ff_anneal_initialized = True
            delta_iter = torch.clamp(local_train_iter - self.stair_ff_anneal_last_train_iter, min=0.0, max=1.0)
            self.stair_ff_anneal_adaptive_iter = self.stair_ff_anneal_adaptive_iter + delta_iter * self.stair_ff_anneal_speed_scale
            self.stair_ff_anneal_last_train_iter = local_train_iter.detach()
            train_iter = self.stair_ff_anneal_adaptive_iter
        else:
            train_iter = local_train_iter + iter_offset
        start_iter = float(getattr(self.cfg.control, "stair_ff_anneal_start_iter", 0.0))
        iterations = float(getattr(self.cfg.control, "stair_ff_anneal_iterations", 1.0))
        end_iter = float(getattr(self.cfg.control, "stair_ff_anneal_end_iter", start_iter + iterations))
        final_scale = float(getattr(self.cfg.control, "stair_ff_anneal_final_scale", 0.0))
        mode = str(getattr(self.cfg.control, "stair_ff_anneal_mode", "cosine")).lower()

        progress = torch.clamp((train_iter - start_iter) / max(end_iter - start_iter, 1e-6), 0.0, 1.0)
        if mode == "linear":
            schedule = 1.0 - progress
        elif mode in ("cos", "cosine"):
            schedule = 0.5 * (1.0 + torch.cos(math.pi * progress))
        elif mode in ("smoothstep", "smooth_step"):
            schedule = 1.0 - progress * progress * (3.0 - 2.0 * progress)
        else:
            raise ValueError(f"Unknown stair_ff_anneal_mode: {mode}")
        return final_scale + (1.0 - final_scale) * schedule

    def _trigger_stair_lift(self, env_mask, side, episode_time, is_followup=False):
        trigger_mask = (
            env_mask
            & ~self.stair_lift_active[:, side]
            & (episode_time >= self.stair_ff_cooldown_until[:, side])
        )
        if not torch.any(trigger_mask):
            return

        if not is_followup:
            # A new first-leg trigger starts a new two-leg rescue sequence.
            self.stair_followup_used[trigger_mask] = False

        self.stair_lift_active[trigger_mask, side] = True

        # Follow-up legs often start after the robot has already contacted or
        # climbed the stair edge.  Start them slightly inside the local phase so
        # the IK feedforward enters the lifting segment faster, without making
        # the two legs lift fully simultaneously.
        if is_followup:
            phase_init = float(getattr(self.cfg.control, "stair_ff_followup_phase_init", 0.06))
            phase_init = min(max(phase_init, 0.0), 0.35)
        else:
            phase_init = 0.0
        self.stair_lift_phase[trigger_mask, side] = phase_init

        self.stair_lift_side[trigger_mask] = side
        self.last_stair_trigger[trigger_mask, side] = episode_time[trigger_mask] + self.dt

        duration = max(float(getattr(self.cfg.control, "stair_ff_duration", 0.48)), 0.0)
        cooldown = max(float(getattr(self.cfg.control, "stair_ff_cooldown", 0.18)), 0.0)

        extend_enabled = bool(getattr(self.cfg.control, "stair_ff_extend_enabled", True))
        extend_ratio = float(getattr(self.cfg.control, "stair_ff_extend_ratio", 0.50))
        extend_ratio = max(extend_ratio, 0.0)

        phase_end = 1.0 + extend_ratio if extend_enabled else 1.0

        self.stair_ff_cooldown_until[trigger_mask, side] = (
            episode_time[trigger_mask] + duration * phase_end + cooldown
        )

        if is_followup:
            self.stair_followup_used[trigger_mask, side] = True

    def _update_stair_feedforward_state(self):
        """Update the blind stair-rescue feedforward state.

        The first trigger is conservative enough to avoid ordinary ground contact:
        it requires a forward command, sufficient travelled distance, sustained
        contact force, and a mild blocked-motion condition.  Follow-up triggering
        is not based on a second impact; instead, it starts the other leg when the
        first leg reaches the middle-late phase and the robot posture is still
        acceptable.  This avoids both extremes: simultaneous leg lifting and a
        second leg that never follows.
        """
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            self.last_stair_ff_signal.zero_()
            if hasattr(self, "last_stair_ff_teacher_offsets"):
                self.last_stair_ff_teacher_offsets.zero_()
            if hasattr(self, "last_stair_ff_offsets"):
                self.last_stair_ff_offsets.zero_()
            return

        contact_forces = self._get_stair_ff_contact_forces()
        if contact_forces is None:
            self.last_stair_ff_signal.zero_()
            if hasattr(self, "last_stair_ff_teacher_offsets"):
                self.last_stair_ff_teacher_offsets.zero_()
            if hasattr(self, "last_stair_ff_offsets"):
                self.last_stair_ff_offsets.zero_()
            return

        self.stair_contact_hist = torch.cat(
            [self.stair_contact_hist[:, :, 1:].clone(), contact_forces.unsqueeze(-1)],
            dim=2,
        )
        stable_frames = int(getattr(self.cfg.control, "stair_ff_contact_stable_frames", 3))
        stable_frames = max(1, min(stable_frames, self.stair_contact_hist.shape[2]))
        recent_contact = self.stair_contact_hist[:, :, -stable_frames:]
        smooth_contact = recent_contact.mean(dim=2)

        duration = max(float(getattr(self.cfg.control, "stair_ff_duration", 0.48)), 1e-6)
        followup_phase = float(getattr(self.cfg.control, "stair_ff_followup_phase", 0.65))
        followup_phase = min(max(followup_phase, 0.0), 1.0)
        threshold = float(getattr(self.cfg.control, "stair_ff_contact_threshold", 55.0))
        episode_time = self.episode_length_buf.float() * self.dt

        trigger_arm = self._get_stair_ff_trigger_arm()
        stable_contact = (recent_contact > threshold).all(dim=2)
        raw_contact_hit = stable_contact & (smooth_contact > threshold) & trigger_arm.unsqueeze(1)

        # Blind-walking trigger: a high contact force only matters when the robot
        # is trying to go forward but its forward velocity is clearly suppressed.
        cmd_x = self.commands[:, 0]
        base_vx = self.base_lin_vel[:, 0]
        min_cmd_x = float(getattr(self.cfg.control, "stair_ff_min_cmd_x", 0.12))
        jam_speed_ratio = float(getattr(self.cfg.control, "stair_ff_jam_speed_ratio", 0.55))
        jam_abs_speed = float(getattr(self.cfg.control, "stair_ff_jam_abs_speed", 0.12))
        forward_cmd = cmd_x > min_cmd_x
        slow_by_ratio = base_vx < cmd_x * jam_speed_ratio
        slow_by_abs = base_vx < jam_abs_speed
        blocked_motion = forward_cmd & (slow_by_ratio | slow_by_abs)

        upright_score = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        min_upright = float(getattr(self.cfg.control, "stair_ff_min_upright", 0.62))
        upright_ok = upright_score > min_upright

        base_height = self._get_base_heights() if hasattr(self, "_get_base_heights") else self.root_states[:, 2]
        min_base_height = float(getattr(self.cfg.control, "stair_ff_min_base_height", 0.30))
        height_ok = base_height > min_base_height

        posture_ok = forward_cmd & upright_ok & height_ok
        contact_hit = raw_contact_hit & blocked_motion.unsqueeze(1) & posture_ok.unsqueeze(1)
        self.stair_ff_contact_hit_sum += contact_hit.any(dim=1).float()

        # First trigger is mutually exclusive: if both sides hit, lift the side
        # with the stronger contact force.  Do not lift both legs at the same time.
        both_hit = contact_hit[:, 0] & contact_hit[:, 1]
        left_stronger = smooth_contact[:, 0] >= smooth_contact[:, 1]
        no_active = ~self.stair_lift_active.any(dim=1)
        left_first = no_active & ((contact_hit[:, 0] & ~contact_hit[:, 1]) | (both_hit & left_stronger))
        right_first = no_active & ((contact_hit[:, 1] & ~contact_hit[:, 0]) | (both_hit & ~left_stronger))
        self._trigger_stair_lift(left_first, 0, episode_time)
        self._trigger_stair_lift(right_first, 1, episode_time)

        # Follow-up uses a looser posture gate than the first impact trigger.
        # When one leg has already climbed, the trailing leg may be stuck and the
        # body posture is usually worse than at the first contact.  Reusing the
        # strict first-trigger gate can suppress the exact rescue we need.
        followup_min_upright = float(
            getattr(
                self.cfg.control,
                "stair_ff_followup_min_upright",
                getattr(self.cfg.control, "stair_ff_min_upright", 0.62),
            )
        )
        followup_min_base_height = float(
            getattr(
                self.cfg.control,
                "stair_ff_followup_min_base_height",
                getattr(self.cfg.control, "stair_ff_min_base_height", 0.30),
            )
        )
        followup_upright_ok = upright_score > followup_min_upright
        followup_height_ok = base_height > followup_min_base_height
        followup_posture_ok = forward_cmd & followup_upright_ok & followup_height_ok

        # Normal phase follow-up: after the first leg reaches the middle-late
        # rescue phase, start the other leg even without a second hard impact.
        left_phase_followup = (
            self.stair_lift_active[:, 1]
            & ~self.stair_lift_active[:, 0]
            & (self.last_stair_trigger[:, 1] > 0.0)
            & ~self.stair_followup_used[:, 0]
            & (self.stair_lift_phase[:, 1] >= followup_phase)
            & followup_posture_ok
        )
        right_phase_followup = (
            self.stair_lift_active[:, 0]
            & ~self.stair_lift_active[:, 1]
            & (self.last_stair_trigger[:, 0] > 0.0)
            & ~self.stair_followup_used[:, 1]
            & (self.stair_lift_phase[:, 0] >= followup_phase)
            & followup_posture_ok
        )

        # Trailing-leg jam rescue: if the non-lifting leg hits the stair edge
        # and the base is slow, trigger it earlier than the nominal follow-up
        # phase.  This directly targets the failure mode where one leg climbs
        # while the other leg remains blocked and stalls the whole robot.
        opposite_jam_force = float(getattr(self.cfg.control, "stair_ff_opposite_jam_force", 35.0))
        opposite_jam_abs_speed = float(getattr(self.cfg.control, "stair_ff_opposite_jam_abs_speed", 0.16))
        opposite_jam_speed_ratio = float(getattr(self.cfg.control, "stair_ff_opposite_jam_speed_ratio", 0.70))
        trailing_jam_min_first_phase = float(
            getattr(self.cfg.control, "stair_ff_trailing_jam_min_first_phase", 0.70)
        )

        opposite_contact = smooth_contact > opposite_jam_force
        trailing_slow = (base_vx < opposite_jam_abs_speed) | (base_vx < cmd_x * opposite_jam_speed_ratio)

        left_trailing_jam = (
            self.stair_lift_active[:, 1]
            & ~self.stair_lift_active[:, 0]
            & (self.last_stair_trigger[:, 1] > 0.0)
            & ~self.stair_followup_used[:, 0]
            & (self.stair_lift_phase[:, 1] >= trailing_jam_min_first_phase)
            & opposite_contact[:, 0]
            & trailing_slow
            & followup_posture_ok
        )
        right_trailing_jam = (
            self.stair_lift_active[:, 0]
            & ~self.stair_lift_active[:, 1]
            & (self.last_stair_trigger[:, 0] > 0.0)
            & ~self.stair_followup_used[:, 1]
            & (self.stair_lift_phase[:, 0] >= trailing_jam_min_first_phase)
            & opposite_contact[:, 1]
            & trailing_slow
            & followup_posture_ok
        )

        left_followup = left_phase_followup | left_trailing_jam
        right_followup = right_phase_followup | right_trailing_jam

        self._trigger_stair_lift(left_followup, 0, episode_time, is_followup=True)
        self._trigger_stair_lift(right_followup, 1, episode_time, is_followup=True)

        self.stair_lift_phase = torch.where(
            self.stair_lift_active,
            self.stair_lift_phase + self.dt / duration,
            self.stair_lift_phase,
        )

        extend_enabled = bool(getattr(self.cfg.control, "stair_ff_extend_enabled", True))
        extend_ratio = float(getattr(self.cfg.control, "stair_ff_extend_ratio", 0.50))
        extend_ratio = max(extend_ratio, 0.0)

        phase_end = 1.0 + extend_ratio if extend_enabled else 1.0

        done = self.stair_lift_phase >= phase_end
        self.stair_lift_active = self.stair_lift_active & ~done
        self.stair_lift_phase = torch.where(
            done,
            torch.zeros_like(self.stair_lift_phase),
            self.stair_lift_phase,
        )

        no_active_after_update = ~self.stair_lift_active.any(dim=1)
        self.last_stair_trigger[no_active_after_update] = 0.0
        self.stair_followup_used[no_active_after_update] = False

        # Smooth bell-shaped IK activation for video/debug overlays and feedforward bookkeeping.
        phase = torch.clamp(self.stair_lift_phase, 0.0, 1.0)
        signal = 0.5 * (1.0 - torch.cos(2.0 * math.pi * phase))

        if extend_enabled and extend_ratio > 1e-6:
            ext_u = torch.clamp((self.stair_lift_phase - 1.0) / extend_ratio, 0.0, 1.0)
            release_gate = 1.0 - (ext_u * ext_u * (3.0 - 2.0 * ext_u))
            signal = torch.where(
                self.stair_lift_phase > 1.0,
                release_gate,
                signal,
            )

        self.last_stair_ff_signal = signal * self.stair_lift_active.float()
        self.stair_ff_active_sum += self.stair_lift_active.any(dim=1).float()

    def _apply_stair_feedforward(self, actions):
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            return actions

        self._update_stair_feedforward_state()
        return actions

    def _get_stair_ff_joint_target_offsets(self):
        """Return stair IK feedforward joint-target offsets in radians.

        This function does not return policy actions.  The returned value is added
        after `default_dof_pos + action_scale * actions`, so the unit is rad.
        """
        ff_offset = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            if hasattr(self, "last_stair_ff_teacher_offsets"):
                self.last_stair_ff_teacher_offsets.zero_()
            if hasattr(self, "last_stair_ff_offsets"):
                self.last_stair_ff_offsets.zero_()
            return ff_offset
        if not hasattr(self, "stair_lift_active") or not torch.any(self.stair_lift_active):
            if hasattr(self, "last_stair_ff_teacher_offsets"):
                self.last_stair_ff_teacher_offsets.zero_()
            if hasattr(self, "last_stair_ff_offsets"):
                self.last_stair_ff_offsets.zero_()
            return ff_offset

        raw_ff_offset = compute_stair_ik_ff_offsets_b(
            phase_local=self.stair_lift_phase,
            active=self.stair_lift_active,
            dof_names=self.dof_names,
            num_actions=self.num_actions,
            device=self.device,
            cfg_control=self.cfg.control,
        )

        k_ff = float(getattr(self.cfg.control, "stair_ff_k", 0.55))
        ff_gate = self._get_stair_ff_gate().unsqueeze(1)
        ff_teacher = ff_gate * k_ff * raw_ff_offset

        final_max = float(getattr(self.cfg.control, "stair_ff_final_max_offset", 0.65))
        if final_max > 0.0:
            ff_teacher = torch.clamp(ff_teacher, -final_max, final_max)

        anneal_scale = self._get_stair_ff_anneal_scale()
        ff_exec = anneal_scale * ff_teacher

        if hasattr(self, "last_stair_ff_teacher_offsets"):
            self.last_stair_ff_teacher_offsets = ff_teacher.detach()
        if hasattr(self, "last_stair_ff_offsets"):
            self.last_stair_ff_offsets = ff_exec.detach()
        return ff_exec

    def _compute_torques(self, actions):
        """Compute torques with stair feedforward injected as joint-target offsets."""
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:, 1:, :].clone(), actions_scaled.unsqueeze(1).clone()], dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes, self.randomized_lag, :] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        joint_pos_target = joint_pos_target + self._get_stair_ff_joint_target_offsets()

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:
                torques = self.p_gains * (joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = (
                    self.p_gains[self.foot_joint_indices] * actions_scaled[:, self.foot_joint_indices]
                    - self.d_gains[self.foot_joint_indices] * self.dof_vel[:, self.foot_joint_indices]
                )
            else:
                torques = self.kp_factor * self.p_gains * (joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = (
                    self.kp_factor[:, self.foot_joint_indices]
                    * self.p_gains[self.foot_joint_indices]
                    * actions_scaled[:, self.foot_joint_indices]
                    - self.kd_factor[:, self.foot_joint_indices]
                    * self.d_gains[self.foot_joint_indices]
                    * self.dof_vel[:, self.foot_joint_indices]
                )
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_terrain_curriculum(self, env_ids):
        """Stair-specific terrain curriculum based on forward progress.

        Rule:
            - move up if the robot travels far enough;
            - move down if the robot barely moves forward;
            - otherwise keep the current terrain level.

        This prevents the diagnostic 'only move up' behavior from pushing the
        robot into high stair levels before the skill is stable.
        """
        if not self.init_done:
            return

        if len(env_ids) == 0:
            return

        forward_progress = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]

        move_up_distance = float(
            getattr(self.cfg.terrain, "curriculum_move_up_distance", 2.2)
        )
        move_down_distance = float(
            getattr(self.cfg.terrain, "curriculum_move_down_distance", 0.8)
        )

        move_up = forward_progress > move_up_distance

        current_level = self.terrain_levels[env_ids]
        strict_min_level = int(getattr(self.cfg.terrain, "curriculum_move_up_strict_min_level", 3))
        strict_gate = current_level >= strict_min_level

        height_ready = torch.ones_like(move_up, dtype=torch.bool)
        height_context = self._get_stair_height_context()
        if height_context is not None:
            _, obstacle_height, climbed_height, _, _, _ = height_context
            env_obstacle = obstacle_height[env_ids]
            env_climbed = climbed_height[env_ids]
            min_height = float(getattr(self.cfg.rewards, "step_success_min_height", 0.025))
            height_scale = torch.clamp(torch.maximum(env_obstacle, env_climbed), min=min_height)
            height_ratio = torch.clamp(env_climbed / height_scale, 0.0, 1.0)
            min_height_ratio = float(getattr(self.cfg.terrain, "curriculum_move_up_min_height_ratio", 0.55))
            min_base_height = float(getattr(self.cfg.terrain, "curriculum_move_up_min_base_height", 0.38))
            base_height = self._get_base_heights()[env_ids]
            strict_height_ready = (height_ratio >= min_height_ratio) & (base_height >= min_base_height)
            height_ready = (~strict_gate) | strict_height_ready

        anneal_ready = torch.ones_like(move_up, dtype=torch.bool)
        if getattr(self.cfg.control, "stair_ff_anneal_enabled", False):
            max_ff_scale = float(getattr(self.cfg.terrain, "curriculum_move_up_max_ff_scale", 0.75))
            anneal_scale = self._get_stair_ff_anneal_scale().detach().item()
            anneal_ready = (~strict_gate) | (anneal_scale <= max_ff_scale)

        move_up = move_up & height_ready & anneal_ready

        # Downgrade when the robot cannot make useful forward progress. The middle
        # band [move_down_distance, move_up_distance] keeps the level unchanged.
        move_down = forward_progress < move_down_distance

        # Avoid contradictory update if thresholds are changed badly.
        move_down = move_down & ~move_up

        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.clip(
            self.terrain_levels[env_ids],
            min=0,
            max=self.max_terrain_level - 1,
        )

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids],
        ]

        self._terrain_forward_progress = forward_progress.detach()
        self._terrain_move_up = move_up.detach()
        self._terrain_move_down = move_down.detach()

    def _get_step_lift_context(self):
        zeros = torch.zeros(self.num_envs, device=self.device)

        if not getattr(self.cfg.terrain, "measure_heights", False):
            return None
        if not hasattr(self, "measured_heights") or not torch.is_tensor(self.measured_heights):
            return None
        if self.measured_heights.ndim != 2:
            return None
        if not hasattr(self, "height_points") or not hasattr(self, "commands"):
            return None

        points = self.height_points[0]
        px = points[:, 0]
        py = points[:, 1]

        front_x_min = getattr(self.cfg.rewards, "step_clearance_front_x_min", 0.20)
        front_x_max = getattr(self.cfg.rewards, "step_clearance_front_x_max", 0.80)
        center_x_abs = getattr(self.cfg.rewards, "step_clearance_center_x_abs", 0.15)
        y_abs = getattr(self.cfg.rewards, "step_clearance_y_abs", 0.35)

        front_mask = (
            (px >= front_x_min)
            & (px <= front_x_max)
            & (torch.abs(py) <= y_abs)
        )
        center_mask = (
            (torch.abs(px) <= center_x_abs)
            & (torch.abs(py) <= y_abs)
        )

        if front_mask.sum().item() == 0 or center_mask.sum().item() == 0:
            return None

        front_height = self.measured_heights[:, front_mask].max(dim=1).values
        center_height = self.measured_heights[:, center_mask].mean(dim=1)

        max_obstacle_height = getattr(self.cfg.rewards, "step_clearance_max_obstacle_height", 0.20)
        obstacle_height = torch.clamp(front_height - center_height, min=0.0, max=max_obstacle_height)

        trigger_height = getattr(self.cfg.rewards, "step_clearance_trigger_height", 0.03)
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        active = (obstacle_height > trigger_height) & (self.commands[:, 0] > min_cmd_x)

        if hasattr(self, "rigid_body_states") and hasattr(self, "feet_indices"):
            foot_z = self.rigid_body_states[:, self.feet_indices, 2]
            foot_clearance = foot_z - center_height.unsqueeze(1)
        elif hasattr(self, "feet_body_frame_height") and torch.is_tensor(self.feet_body_frame_height):
            foot_clearance = self.feet_body_frame_height
        else:
            return None

        return active, obstacle_height, foot_clearance, zeros

    def _get_stair_height_context(self):
        """Estimate whether the base has actually migrated onto the higher stair."""
        zeros = torch.zeros(self.num_envs, device=self.device)

        if not getattr(self.cfg.terrain, "measure_heights", False):
            return None
        if not hasattr(self, "measured_heights") or not torch.is_tensor(self.measured_heights):
            return None
        if self.measured_heights.ndim != 2:
            return None
        if not hasattr(self, "height_points") or not hasattr(self, "commands"):
            return None

        points = self.height_points[0]
        px = points[:, 0]
        py = points[:, 1]

        front_x_min = getattr(self.cfg.rewards, "step_clearance_front_x_min", 0.20)
        front_x_max = getattr(self.cfg.rewards, "step_clearance_front_x_max", 0.80)
        center_x_abs = getattr(self.cfg.rewards, "step_clearance_center_x_abs", 0.15)
        rear_x_min = getattr(self.cfg.rewards, "step_success_rear_x_min", -0.75)
        rear_x_max = getattr(self.cfg.rewards, "step_success_rear_x_max", -0.20)
        y_abs = getattr(self.cfg.rewards, "step_clearance_y_abs", 0.35)

        front_mask = (
            (px >= front_x_min)
            & (px <= front_x_max)
            & (torch.abs(py) <= y_abs)
        )
        center_mask = (
            (torch.abs(px) <= center_x_abs)
            & (torch.abs(py) <= y_abs)
        )
        rear_mask = (
            (px >= rear_x_min)
            & (px <= rear_x_max)
            & (torch.abs(py) <= y_abs)
        )

        if front_mask.sum().item() == 0 or center_mask.sum().item() == 0 or rear_mask.sum().item() == 0:
            return None

        front_height = self.measured_heights[:, front_mask].max(dim=1).values
        center_height = self.measured_heights[:, center_mask].mean(dim=1)
        rear_height = self.measured_heights[:, rear_mask].mean(dim=1)

        max_obstacle_height = getattr(self.cfg.rewards, "step_clearance_max_obstacle_height", 0.20)
        obstacle_height = torch.clamp(front_height - center_height, min=0.0, max=max_obstacle_height)
        climbed_height = torch.clamp(center_height - rear_height, min=0.0, max=max_obstacle_height)

        trigger_height = getattr(self.cfg.rewards, "step_clearance_trigger_height", 0.03)
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        active = ((obstacle_height > trigger_height) | (climbed_height > trigger_height)) & (self.commands[:, 0] > min_cmd_x)

        return active, obstacle_height, climbed_height, center_height, rear_height, zeros

    def _get_stair_posture_gate(self):
        """Gate stair shaping to attempts that are still physically meaningful."""
        upright_score = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        upright_min = getattr(self.cfg.rewards, "stair_gate_upright_min", 0.70)
        upright_gate = torch.clamp((upright_score - upright_min) / (1.0 - upright_min), 0.0, 1.0)

        base_height = self._get_base_heights()
        min_height = getattr(self.cfg.rewards, "stair_gate_base_height_min", 0.28)
        full_height = getattr(self.cfg.rewards, "stair_gate_base_height_full", 0.40)
        height_gate = torch.clamp((base_height - min_height) / max(full_height - min_height, 1e-6), 0.0, 1.0)

        return upright_gate * height_gate

    def _get_stair_reward_gate(self):
        """Gate positive stair rewards to upright, non-colliding attempts."""
        posture_gate = self._get_stair_posture_gate()

        contact_gate = torch.ones(self.num_envs, device=self.device)
        if hasattr(self, "contact_forces") and hasattr(self, "penalised_contact_indices"):
            if len(self.penalised_contact_indices) > 0:
                force_threshold = getattr(self.cfg.rewards, "stair_gate_bad_contact_force", 5.0)
                bad_contact = torch.any(
                    torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > force_threshold,
                    dim=1,
                )
                contact_gate = (~bad_contact).float()

        return posture_gate * contact_gate

    def _get_foot_contact_norm(self):
        if hasattr(self, "force_sensor_tensor") and torch.is_tensor(self.force_sensor_tensor):
            return torch.norm(self.force_sensor_tensor[:, :, :3], dim=-1)
        if not hasattr(self, "contact_forces") or not hasattr(self, "feet_indices"):
            return None
        return torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

    def _get_step_forward_score(self):
        min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
        cmd_x = torch.clamp(self.commands[:, 0], min=min_cmd_x)
        return torch.clamp(self.base_lin_vel[:, 0] / cmd_x, 0.0, 1.0)

    def _get_active_lift_mask(self, foot_like_tensor):
        if (
            not hasattr(self, "stair_lift_active")
            or not torch.is_tensor(self.stair_lift_active)
            or foot_like_tensor.shape[1] < 2
        ):
            return None

        lift_mask = torch.zeros_like(foot_like_tensor, dtype=torch.bool)
        lift_mask[:, :2] = self.stair_lift_active[:, :2]
        return lift_mask

    def _get_step_blocking_signal(self):
        context = self._get_step_lift_context()
        contact_norm = self._get_foot_contact_norm()
        if context is None or contact_norm is None:
            return None

        active, obstacle_height, foot_clearance, zeros = context
        margin = getattr(self.cfg.rewards, "step_block_clearance_margin", 0.04)
        target_clearance = torch.clamp(obstacle_height + margin, min=0.04)
        support_clearance = torch.clamp(foot_clearance.min(dim=1).values, min=0.0)
        low_clearance = torch.clamp(
            (target_clearance - support_clearance) / torch.clamp(target_clearance, min=0.04),
            0.0,
            1.0,
        )
        max_contact = contact_norm.max(dim=1).values
        return active, obstacle_height, foot_clearance, max_contact, low_clearance, zeros

    def _update_step_contact_state(self):
        signal = self._get_step_blocking_signal()
        if signal is None:
            self.step_contact_timer.zero_()
            self.step_jam_time.zero_()
            return

        active, _, _, max_contact, low_clearance, _ = signal
        contact_force = getattr(self.cfg.rewards, "step_contact_force_threshold", 80.0)
        jam_force = getattr(self.cfg.rewards, "step_jam_force_threshold", 280.0)
        jam_clearance = getattr(self.cfg.rewards, "step_jam_clearance_ratio", 0.45)
        jam_speed = getattr(self.cfg.rewards, "step_jam_min_speed", 0.08)
        contact_memory = getattr(self.cfg.rewards, "step_contact_memory_time", 0.25)

        step_contact = active & (max_contact > contact_force)
        jammed = active & (max_contact > jam_force) & (low_clearance > jam_clearance) & (self.base_lin_vel[:, 0] < jam_speed)

        self.step_contact_timer = torch.where(
            step_contact,
            torch.full_like(self.step_contact_timer, contact_memory),
            torch.clamp(self.step_contact_timer - self.dt, min=0.0),
        )
        self.step_jam_time = torch.where(jammed, self.step_jam_time + self.dt, torch.zeros_like(self.step_jam_time))

    def _get_step_leg_imbalance_signal(self):
        context = self._get_step_lift_context()
        if context is None:
            return None

        active, obstacle_height, foot_clearance, zeros = context
        margin = getattr(self.cfg.rewards, "step_lift_margin", 0.06)
        target_lift = torch.clamp(obstacle_height + margin, min=0.05).unsqueeze(1)
        per_foot_progress = torch.clamp(torch.clamp(foot_clearance, min=0.0) / target_lift, 0.0, 1.0)
        lead_progress = per_foot_progress.max(dim=1).values
        follow_progress = per_foot_progress.min(dim=1).values
        imbalance_start = getattr(self.cfg.rewards, "step_leg_imbalance_start", 0.35)
        imbalance = torch.clamp(
            (lead_progress - follow_progress - imbalance_start) / max(1.0 - imbalance_start, 1e-6),
            0.0,
            1.0,
        )
        return active, imbalance, lead_progress, zeros

    def _update_step_imbalance_state(self):
        signal = self._get_step_leg_imbalance_signal()
        if signal is None:
            self.step_imbalance_time.zero_()
            return

        active, imbalance, lead_progress, _ = signal
        trigger = getattr(self.cfg.rewards, "step_leg_imbalance_trigger", 0.25)
        high_leg = getattr(self.cfg.rewards, "step_leg_imbalance_min_lead", 0.65)
        bad_imbalance = active & (imbalance > trigger) & (lead_progress > high_leg)
        self.step_imbalance_time = torch.where(
            bad_imbalance,
            self.step_imbalance_time + self.dt,
            torch.zeros_like(self.step_imbalance_time),
        )

    def _reward_step_clearance(self):
        """Reward useful wheel/foot clearance when a front up-step is detected."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        clearance_margin = getattr(self.cfg.rewards, "step_clearance_margin", 0.04)
        sigma = getattr(self.cfg.rewards, "step_clearance_sigma", 0.04)

        target_clearance = torch.clamp(obstacle_height + clearance_margin, min=0.04).unsqueeze(1)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)

        per_foot_progress = torch.clamp(positive_clearance / target_clearance, 0.0, 1.0)
        lift_mask = self._get_active_lift_mask(per_foot_progress)
        if lift_mask is None:
            return zeros
        lifting = lift_mask.any(dim=1)
        if not torch.any(active & lifting):
            return zeros

        selected_progress = (per_foot_progress * lift_mask.float()).sum(dim=1) / torch.clamp(
            lift_mask.float().sum(dim=1),
            min=1.0,
        )
        forward_score = self._get_step_forward_score()
        reward = selected_progress * (0.4 + 0.6 * forward_score)

        return reward * active.float() * lifting.float() * self._get_stair_posture_gate()

    def _reward_stair_ff_tracking(self):
        """Strongly distill the full stair IK teacher into policy action and body state."""
        if (
            not hasattr(self, "last_stair_ff_teacher_offsets")
            or not hasattr(self, "stair_lift_active")
            or not hasattr(self, "actions")
            or not torch.any(self.stair_lift_active)
        ):
            return torch.zeros(self.num_envs, device=self.device)

        tracked_joint_names = [
            ("FL_thigh_joint", 0),
            ("FL_calf_joint", 0),
            ("FR_thigh_joint", 1),
            ("FR_calf_joint", 1),
        ]

        action_scale = torch.full_like(self.actions, float(self.cfg.control.action_scale))
        action_scale[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction
        teacher_action = torch.where(
            torch.abs(action_scale) > 1e-6,
            self.last_stair_ff_teacher_offsets / action_scale,
            torch.zeros_like(self.last_stair_ff_teacher_offsets),
        )
        clip_actions = float(self.cfg.normalization.clip_actions)
        teacher_action = torch.clamp(teacher_action, -clip_actions, clip_actions)
        if getattr(self.cfg.control, "stair_ff_anneal_enabled", False):
            missing_teacher_scale = 1.0 - self._get_stair_ff_anneal_scale().detach().item()
            min_target_scale = float(getattr(self.cfg.rewards, "stair_ff_tracking_min_action_target_scale", 0.15))
            missing_teacher_scale = max(missing_teacher_scale, min_target_scale)
        else:
            missing_teacher_scale = 1.0
        target_action_delta = missing_teacher_scale * teacher_action

        action_source = getattr(self, "last_residual_delta", self.actions)
        action_sq_error = torch.zeros(self.num_envs, device=self.device)
        joint_sq_error = torch.zeros(self.num_envs, device=self.device)
        active_count = torch.zeros(self.num_envs, device=self.device)

        for joint_name, side in tracked_joint_names:
            if joint_name not in self.dof_names:
                continue
            joint_idx = self.dof_names.index(joint_name)
            active = self.stair_lift_active[:, side].float()
            target_joint_pos = self.default_dof_pos[:, joint_idx] + self.last_stair_ff_teacher_offsets[:, joint_idx]
            joint_sq_error += torch.square(self.dof_pos[:, joint_idx] - target_joint_pos) * active
            action_sq_error += torch.square(action_source[:, joint_idx] - target_action_delta[:, joint_idx]) * active
            active_count += active

        active_env = active_count > 0.0
        active_count_safe = torch.clamp(active_count, min=1.0)
        mean_joint_error = torch.sqrt(joint_sq_error / active_count_safe)
        mean_action_error = torch.sqrt(action_sq_error / active_count_safe)

        joint_sigma = max(
            float(
                getattr(
                    self.cfg.rewards,
                    "stair_ff_tracking_joint_sigma",
                    getattr(self.cfg.rewards, "stair_ff_tracking_sigma", 0.35),
                )
            ),
            1e-6,
        )
        action_sigma = max(float(getattr(self.cfg.rewards, "stair_ff_tracking_action_sigma", 0.45)), 1e-6)
        action_weight = float(getattr(self.cfg.rewards, "stair_ff_tracking_action_weight", 0.75))
        joint_weight = float(getattr(self.cfg.rewards, "stair_ff_tracking_joint_weight", 0.25))

        action_score = 1.0 / (1.0 + torch.square(mean_action_error / action_sigma))
        joint_score = 1.0 / (1.0 + torch.square(mean_joint_error / joint_sigma))
        reward = action_weight * action_score + joint_weight * joint_score

        progress_target = float(getattr(self.cfg.rewards, "stair_ff_tracking_progress_target", 2.0))
        progress_floor = float(getattr(self.cfg.rewards, "stair_ff_tracking_progress_floor", 0.15))
        x_progress = self.root_states[:, 0] - self.env_origins[:, 0]
        progress_gate = progress_floor + (1.0 - progress_floor) * torch.clamp(
            x_progress / max(progress_target, 1e-6),
            0.0,
            1.0,
        )
        drive_gate = 0.25 + 0.75 * self._get_step_forward_score()

        min_gate = float(getattr(self.cfg.rewards, "stair_ff_tracking_min_gate", 0.25))
        posture_gate = torch.clamp(self._get_stair_posture_gate(), min=min_gate, max=1.0)
        return reward * progress_gate * drive_gate * active_env.float() * posture_gate

    def _reward_step_lift(self):
        """Reward reaching a useful lift height while the front step is active."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_lift_min_height", 0.05)
        margin = getattr(self.cfg.rewards, "step_lift_margin", 0.06)
        sigma = getattr(self.cfg.rewards, "step_lift_sigma", 0.05)
        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)

        lift_error = torch.clamp(target_lift.unsqueeze(1) - positive_clearance, min=0.0)
        per_foot_lift = torch.exp(-torch.square(lift_error / sigma))
        lift_mask = self._get_active_lift_mask(per_foot_lift)
        if lift_mask is None:
            return zeros
        lifting = lift_mask.any(dim=1)
        if not torch.any(active & lifting):
            return zeros

        selected_lift = (per_foot_lift * lift_mask.float()).sum(dim=1) / torch.clamp(
            lift_mask.float().sum(dim=1),
            min=1.0,
        )
        forward_score = self._get_step_forward_score()
        reward = selected_lift * (0.4 + 0.6 * forward_score)

        return reward * active.float() * lifting.float() * self._get_stair_posture_gate()

    def _reward_step_pre_lift(self):
        """Small bonus for using height scan to lift before contact."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, zeros = context
        if not torch.any(active):
            return zeros

        contact_norm = self._get_foot_contact_norm()
        if contact_norm is None:
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_pre_lift_min_height", 0.06)
        margin = getattr(self.cfg.rewards, "step_pre_lift_margin", 0.07)
        max_lift_contact = getattr(self.cfg.rewards, "step_pre_lift_max_contact_force", 35.0)

        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)
        lift_score = torch.clamp(positive_clearance / torch.clamp(target_lift.unsqueeze(1), min=0.04), 0.0, 1.0)

        # A real pre-lift should happen with low foot contact; if the stair edge
        # is pushing the wheel up, contact force is usually high.
        low_contact_score = torch.clamp(
            (max_lift_contact - contact_norm) / max(max_lift_contact, 1e-6),
            0.0,
            1.0,
        )
        lead_lift = lift_score.max(dim=1).values
        unloaded_lift = (lift_score * low_contact_score).max(dim=1).values
        reward = 0.45 * lead_lift + 0.55 * unloaded_lift

        return reward * active.float() * self._get_stair_reward_gate()

    def _reward_step_reactive_lift(self):
        """Reward lifting and unloading shortly after contacting the stair."""

        signal = self._get_step_blocking_signal()
        if signal is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, foot_clearance, max_contact, _, zeros = signal
        recent_contact = self.step_contact_timer > 0.0
        if not torch.any(active & recent_contact):
            return zeros

        min_lift = getattr(self.cfg.rewards, "step_reactive_lift_min_height", 0.08)
        margin = getattr(self.cfg.rewards, "step_reactive_lift_margin", 0.07)
        target_lift = torch.clamp(obstacle_height + margin, min=min_lift)
        positive_clearance = torch.clamp(foot_clearance, min=0.0)
        per_foot_progress = torch.clamp(positive_clearance / torch.clamp(target_lift.unsqueeze(1), min=0.04), 0.0, 1.0)
        lead_progress = per_foot_progress.max(dim=1).values
        follow_progress = per_foot_progress.min(dim=1).values
        lift_progress = 0.45 * lead_progress + 0.55 * follow_progress

        unload_low = getattr(self.cfg.rewards, "step_reactive_unload_force_low", 80.0)
        unload_high = getattr(self.cfg.rewards, "step_reactive_unload_force_high", 300.0)
        unload_score = 1.0 - torch.clamp((max_contact - unload_low) / max(unload_high - unload_low, 1e-6), 0.0, 1.0)

        forward_score = self._get_step_forward_score()

        reward = lift_progress * (0.25 + 0.75 * forward_score) * (0.5 + 0.5 * unload_score)
        return reward * active.float() * recent_contact.float() * self._get_stair_reward_gate()

    def _reward_step_leg_imbalance(self):
        """Penalize camping with one leg high while the other leg never follows."""

        signal = self._get_step_leg_imbalance_signal()
        if signal is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, imbalance, lead_progress, zeros = signal
        if not torch.any(active):
            return zeros

        grace_time = getattr(self.cfg.rewards, "step_leg_imbalance_grace_time", 0.25)
        time_scale = getattr(self.cfg.rewards, "step_leg_imbalance_time_scale", 0.35)
        time_gate = torch.clamp(
            (self.step_imbalance_time - grace_time) / max(time_scale, 1e-6),
            0.0,
            1.0,
        )
        return imbalance * lead_progress * time_gate * active.float() * self._get_stair_posture_gate()

    def _reward_step_bump(self):
        """Penalize sustained jamming, not the first probing contact."""

        grace_time = getattr(self.cfg.rewards, "step_jam_grace_time", 0.12)
        time_scale = getattr(self.cfg.rewards, "step_jam_time_scale", 0.20)
        jam_score = torch.clamp((self.step_jam_time - grace_time) / max(time_scale, 1e-6), 0.0, 1.0)
        return jam_score * self._get_stair_posture_gate()

    def _reward_step_drive(self):
        """Dense anti-camping reward for continuing to drive into the stair."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_speed = getattr(self.cfg.rewards, "step_drive_min_speed", 0.06)
        cmd_x = torch.clamp(self.commands[:, 0], min=min_speed + 1e-3)
        vel_score = torch.clamp(
            (self.base_lin_vel[:, 0] - min_speed) / torch.clamp(cmd_x - min_speed, min=1e-3),
            0.0,
            1.0,
        )

        x_progress = self.root_states[:, 0] - self.env_origins[:, 0]
        progress_target = getattr(self.cfg.rewards, "step_drive_progress_target", 1.6)
        progress_score = torch.clamp(x_progress / max(progress_target, 1e-6), 0.0, 1.0)

        reward = 0.7 * vel_score + 0.3 * progress_score
        return reward * active.float() * self._get_stair_posture_gate()

    def _reward_step_progress(self):
        """Reward forward progress when a front step is detected."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, _, _, zeros = context
        if not torch.any(active):
            return zeros

        progress = self._get_step_forward_score()
        return progress * active.float() * self._get_stair_reward_gate()

    def _reward_step_up(self):
        """Reward actual terrain-height migration, not just lifting a foot."""

        context = self._get_stair_height_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, climbed_height, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_height = getattr(self.cfg.rewards, "step_success_min_height", 0.025)
        height_scale = torch.clamp(torch.maximum(obstacle_height, climbed_height), min=min_height)
        up_progress = torch.clamp(climbed_height / height_scale, 0.0, 1.0)

        forward_score = self._get_step_forward_score()

        return up_progress * (0.5 + 0.5 * forward_score) * active.float() * self._get_stair_posture_gate()

    def _reward_step_success(self):
        """Reward a completed stair transition that is followed by stable travel."""

        context = self._get_stair_height_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, obstacle_height, climbed_height, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_height = getattr(self.cfg.rewards, "step_success_min_height", 0.025)
        start_ratio = getattr(self.cfg.rewards, "step_success_start_ratio", 0.35)
        complete_ratio = getattr(self.cfg.rewards, "step_success_complete_ratio", 0.70)
        height_scale = torch.clamp(torch.maximum(obstacle_height, climbed_height), min=min_height)
        height_ratio = torch.clamp(climbed_height / height_scale, 0.0, 1.0)
        height_complete = torch.clamp((height_ratio - start_ratio) / max(complete_ratio - start_ratio, 1e-6), 0.0, 1.0)

        base_height = self._get_base_heights()
        min_base_height = getattr(self.cfg.rewards, "step_success_min_base_height", 0.30)
        full_base_height = getattr(self.cfg.rewards, "step_success_full_base_height", 0.40)
        base_score = torch.clamp(
            (base_height - min_base_height) / max(full_base_height - min_base_height, 1e-6),
            0.0,
            1.0,
        )

        speed_score = self._get_step_forward_score()
        recovery_score = 0.5 + 0.5 * speed_score

        x_progress = self.root_states[:, 0] - self.env_origins[:, 0]
        min_x_progress = getattr(
            self.cfg.rewards,
            "step_success_min_x_progress",
            getattr(self.cfg.rewards, "step_success_min_distance", 1.0),
        )
        full_x_progress = getattr(
            self.cfg.rewards,
            "step_success_full_x_progress",
            getattr(self.cfg.rewards, "step_success_full_distance", 2.0),
        )
        x_progress_score = torch.clamp(
            (x_progress - min_x_progress) / max(full_x_progress - min_x_progress, 1e-6),
            0.0,
            1.0,
        )

        height_score = 0.25 * height_ratio + 0.75 * height_complete
        success_score = height_score * base_score * recovery_score * x_progress_score
        return success_score * active.float() * self._get_stair_posture_gate()

    def _reward_step_stall(self):
        """Penalize stopping at the step edge instead of attempting to climb."""

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, device=self.device)

        active, _, _, zeros = context
        if not torch.any(active):
            return zeros

        min_speed = getattr(self.cfg.rewards, "step_stall_min_speed", 0.08)
        speed_stall = torch.clamp((min_speed - self.base_lin_vel[:, 0]) / max(min_speed, 1e-6), 0.0, 1.0)
        x_progress = self.root_states[:, 0] - self.env_origins[:, 0]
        progress_target = float(getattr(self.cfg.rewards, "step_stall_progress_target", 2.4))
        progress_stall = 1.0 - torch.clamp(x_progress / max(progress_target, 1e-6), 0.0, 1.0)
        stall_score = torch.maximum(speed_stall, progress_stall)
        stalled = (self.base_lin_vel[:, 0] < min_speed) | (progress_stall > 0.35)
        posture_gate = torch.clamp(self._get_stair_posture_gate(), min=0.25, max=1.0)
        return stall_score * (active & stalled).float() * posture_gate

    def _reward_opposite_base_vel(self):
        cmd_x = self.commands[:, 0]
        backward = (cmd_x > 0.05) & (self.base_lin_vel[:, 0] < -0.03)
        penalty = torch.clamp(-self.base_lin_vel[:, 0] / torch.clamp(cmd_x, min=0.05), 0.0, 1.0)
        return backward.float() * penalty


class D1HMoEDiscCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        # Stage A: keep speed near the previously successful 0.4 m/s band.
        # Terrain difficulty is handled only by the forward-distance curriculum.
        curriculum = False
        max_curriculum_x = 0.55
        max_curriculum_x_back = 0.0
        max_curriculum_y = 0.0
        max_curriculum_yaw = 0.0
        resampling_time = 10.0
        heading_command = True
        zero_command_ratio = 0.0
        startup_freeze_time = 0.0

        class ranges:
            # Values below 0.2 are zeroed by the base command sampler.
            lin_vel_x = [0.32, 0.44]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class terrain(D1HMoEBaseCfg.terrain):
        # Stage 1: only clean up-stairs, starting at the easiest row.
        curriculum = True
        max_init_terrain_level = 2
        # Terrain order: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles].
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

        # Fifteen rows provide a smooth 2.8-17.8 cm stair progression.
        # In utils/terrain.py, up-stair height uses:
        #   height(row i) = min + (max - min) * i / num_rows
        # With num_rows=15 and step_height=[0.035, 0.185], rows 0..14 are
        # approximately 3.5, 4.5, ..., 17.5 cm. This extends the old distribution
        # instead of suddenly jumping to a fixed 17 cm stair.
        num_rows = 10
        step_height = [0.05, 0.17]
        step_width_range = [0.40, 0.55]
        slope = [0.0, 0.02]
        slope_treshold = 0.20
        
        # Stair-specific curriculum distances.
        # Move up only when the robot travels far enough.
        # Move down when it almost fails to make forward progress.
        curriculum_move_up_distance = 3.0
        curriculum_move_down_distance = 1.4
        curriculum_move_up_max_ff_scale = 0.75
        curriculum_move_up_strict_min_level = 3
        curriculum_move_up_min_height_ratio = 0.55
        curriculum_move_up_min_base_height = 0.38



    class domain_rand(D1HMoEBaseCfg.domain_rand):
        # Stair-up is a fine contact skill. Remove early noise sources that make
        # residual credit assignment look random.
        randomize_friction = True
        friction_range = [0.8, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 0.0]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        randomize_base_com = False
        added_com_range = [0.0, 0.0]
        push_robots = False
        disturbance = False
        randomize_motor = False
        motor_strength_range = [1.0, 1.0]
        randomize_lag_timesteps = False
        lag_timesteps = 2

    class control(D1HMoEBaseCfg.control):
        stair_ff_enabled = True

        # IK rescue feedforward timing.  The follow-up is phase based: 0.64 means
        # the second leg starts after the first leg has finished roughly two thirds
        # of its rescue trajectory.  This is less aggressive than 0.50, but much
        # less likely to miss the second leg than the old 0.42 s impact-gated rule.
        stair_ff_duration = 0.6
        stair_ff_followup_phase = 0.8
        stair_ff_followup_phase_init = 0.00
        stair_ff_cooldown = 0.20
        stair_ff_extend_enabled = True
        stair_ff_extend_ratio = 0.30

        # Follow-up rescue uses looser gates than the first trigger.  This helps
        # the trailing leg lift when the leading leg has already climbed and the
        # body posture has degraded slightly.
        stair_ff_followup_min_upright = 0.45
        stair_ff_followup_min_base_height = 0.26
        stair_ff_opposite_jam_force = 40.0
        stair_ff_opposite_jam_abs_speed = 0.1
        stair_ff_opposite_jam_speed_ratio = 0.3
        stair_ff_trailing_jam_min_first_phase = 0.80

        # B-scheme IK feedforward strength and shaping.
        stair_ff_k = 1.0
        stair_ff_phase_start = 0.20
        stair_ff_ramp_ratio = 0.06
        stair_ff_max_offset = 1.20
        stair_ff_final_max_offset = 0.65

        # IK geometry.
        stair_ff_l1 = 0.25
        stair_ff_l2 = 0.25
        stair_ff_wheel_radius = 0.085
        stair_ff_step_height = 0.16
        stair_ff_x0 = 0.0
        stair_ff_x1 = 0.13
        stair_ff_z_hip = 0.45

        # Rounded vertical-first Bezier stair trajectory.
        # z_peak = step_height + wheel_radius + clear_margin.
        # This replaces the old cycloid + h_margin trajectory.
        stair_ff_traj_type = "bezier_vertical_first"
        stair_ff_clear_margin = 0.045

        # Bezier control ratios:
        # P0 = start
        # P1 = same x, high z: makes initial motion mostly upward
        # P2 = near start x, peak z: early clearance
        # P3 = forward x, peak z: high forward transfer
        # P4 = end x, above landing: smooth downward landing
        # P5 = end
        stair_ff_bezier_p1_z_ratio = 0.70
        stair_ff_bezier_p2_x_ratio = 0.05
        stair_ff_bezier_p3_x_ratio = 0.65
        stair_ff_bezier_p4_z_ratio = 0.25

        # Opposite support-leg compensation.
        # When one leg swings, the other leg is extended slightly to support the base.
        stair_ff_support_enabled = True
        stair_ff_support_hip_lift = 0.050
        stair_ff_support_k = 0.90
        stair_ff_support_ramp_ratio = 0.35
        stair_ff_support_max_offset = 0.35

        # Default stance angles used to compute IK-based support compensation.
        stair_ff_support_default_thigh = 0.8
        stair_ff_support_default_calf = -1.5

        # Blind trigger: sustained contact plus mild blocked-motion evidence.
        # These are intentionally not too strict, because the robot has no exteroceptive
        # perception and must react from proprioceptive/contact information.
        stair_ff_contact_threshold = 42.0
        stair_ff_contact_force_axis = "horizontal"
        stair_ff_min_forward_travel = 0.12
        stair_ff_contact_stable_frames = 3
        stair_ff_min_cmd_x = 0.12
        stair_ff_jam_speed_ratio = 0.75
        stair_ff_jam_abs_speed = 0.12
        stair_ff_min_upright = 0.62
        stair_ff_min_base_height = 0.30

        # Feedforward execution anneals away while the full IK target remains as a teacher.
        stair_ff_anneal_enabled = True
        stair_ff_anneal_mode = "cosine"
        stair_ff_anneal_start_iter = 300
        stair_ff_anneal_iterations = 4500
        stair_ff_anneal_override_scale = None
        stair_ff_anneal_iter_offset = 0.0
        stair_ff_anneal_final_scale = 0.0
        stair_ff_anneal_steps_per_iter = 32
        stair_ff_anneal_adaptive_enabled = True
        stair_ff_anneal_good_progress = 2.5
        stair_ff_anneal_bad_progress = 1.8
        stair_ff_anneal_good_promote_rate = 0.25
        stair_ff_anneal_bad_move_down_rate = 0.12
        stair_ff_anneal_min_contact_hit_ratio = 0.05
        stair_ff_anneal_good_contact_hit_ratio = 0.12
        stair_ff_anneal_fast_speed = 1.5
        stair_ff_anneal_normal_speed = 1.0
        stair_ff_anneal_slow_speed = 0.35
        stair_ff_anneal_speed_smoothing = 0.3

    class rewards(D1HMoEBaseCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.06
        distance_sigma = 0.08
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.98
        soft_torque_limit = 0.98
        base_height_target = 0.45
        base_height_scale = 0.04
        base_height_deadband = 0.03
        stair_ff_tracking_sigma = 0.35
        stair_ff_tracking_joint_sigma = 0.35
        stair_ff_tracking_action_sigma = 0.45
        stair_ff_tracking_action_weight = 0.75
        stair_ff_tracking_joint_weight = 0.25
        stair_ff_tracking_min_gate = 0.25
        stair_ff_tracking_min_action_target_scale = 0.15
        stair_ff_tracking_progress_target = 2.0
        stair_ff_tracking_progress_floor = 0.15

        # Front height scan window used only by the stair rewards.
        step_clearance_front_x_min = 0.20
        step_clearance_front_x_max = 0.80
        step_clearance_center_x_abs = 0.15
        step_clearance_y_abs = 0.45

        # Clearance target = detected step height + margin.
        step_clearance_trigger_height = 0.025
        step_clearance_margin = 0.08
        step_clearance_max_obstacle_height = 0.24
        step_clearance_sigma = 0.08
        step_clearance_min_cmd_x = 0.08
        step_lift_min_height = 0.08
        step_lift_margin = 0.09
        step_lift_sigma = 0.07
        step_stall_min_speed = 0.14
        step_drive_min_speed = 0.06
        step_drive_progress_target = 1.6
        step_stall_progress_target = 2.4
        step_pre_lift_min_height = 0.08
        step_pre_lift_margin = 0.09
        step_pre_lift_sigma = 0.05
        step_pre_lift_max_contact_force = 160.0
        step_contact_force_threshold = 50.0
        step_contact_memory_time = 0.45
        step_block_clearance_margin = 0.04
        step_reactive_lift_min_height = 0.08
        step_reactive_lift_margin = 0.07
        step_reactive_unload_force_low = 120.0
        step_reactive_unload_force_high = 450.0
        step_jam_force_threshold = 300.0
        step_jam_clearance_ratio = 0.30
        step_jam_min_speed = 0.08
        step_jam_grace_time = 0.15
        step_jam_time_scale = 0.25
        stair_gate_upright_min = 0.65
        stair_gate_base_height_min = 0.28
        stair_gate_base_height_full = 0.40
        stair_gate_bad_contact_force = 5.0
        step_success_rear_x_min = -0.75
        step_success_rear_x_max = -0.20
        step_success_min_height = 0.025
        step_success_start_ratio = 0.35
        step_success_complete_ratio = 0.70
        step_success_min_base_height = 0.30
        step_success_full_base_height = 0.42
        step_success_min_speed = 0.10
        step_success_min_distance = 1.0
        step_success_full_distance = 2.2
        step_success_min_x_progress = 1.0
        step_success_full_x_progress = 2.2
        step_success_min_time = 3.0
        step_success_full_time = 8.0
        step_leg_imbalance_start = 0.35
        step_leg_imbalance_trigger = 0.25
        step_leg_imbalance_min_lead = 0.65
        step_leg_imbalance_grace_time = 0.25
        step_leg_imbalance_time_scale = 0.35

        class scales(D1HMoEBaseCfg.rewards.scales):
            # Disabled legacy aggregate tracker; this expert uses axis-specific tracking below.
            tracking_lin_vel = 0.0
            # Keep only a weak forward guardrail. Y/yaw rewards were mostly free
            # bonuses with zero commands, so they hide the real stair signal.
            tracking_lin_vel_x = 12.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 15.0
            heading = -25.0

            # Stability guardrails. They should prevent garbage motion, not dominate climbing.
            orientation = -25.0
            upward = 3.5
            ang_vel_xy = -0.4
            base_height = -13.0
            lin_vel_z = -0.5

            # Failure/contact penalties.
            termination = -400.0
            collision = -12.0
            collision_hard = -100.0
            collision_head = 0.0

            # Remove tiny regularizers that only add noise to the scalar reward.
            torques = 0.0
            powers = 0.0
            dof_acc = 0.0
            action_rate = -0.15
            action_smoothness = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0

            # Zero-command rewards are disabled because this expert never samples zero commands.
            stand_still = 0.0
            zero_base_vel = 0.0
            zero_yaw_rate = 0.0
            zero_wheel_vel = 0.0

            # Air-time is not a stair-success signal for this wheel-legged robot.
            feet_air_time = 0.0
            feet_contact_forces = 0.0
            feet_stumble = 0.0
            stumble = 0.0
            no_jump = 0.0

            # Disable weak geometry priors until the stair skill exists.
            body_pos_to_feet_x = 0.0
            body_feet_distance_x = -8.0
            body_feet_distance_y = 0.0
            body_symmetry_y = 0.0
            body_symmetry_z = 0.0

            # Main stair-up objective. Clearance/lift remain auxiliary; the
            # curriculum follows step_success.
            step_clearance = 1.5
            stair_ff_tracking = 40.0
            step_lift = 4.0
            step_pre_lift = 0.0
            step_reactive_lift = 0.0
            step_leg_imbalance = -10.0
            step_drive = 10.0
            step_progress = 24.0
            step_up = 60.0
            step_success = 100.0
            step_stall = -40.0
            step_bump = -20.0
            opposite_base_vel = -48.0

    class normalization(D1HMoEBaseCfg.normalization):
        # Keep exploration broad enough for stair actions, but prevent unbounded
        # residual samples from destroying the frozen base policy.
        clip_actions = 1.6

    class costs(D1HMoEBaseCfg.costs):
        class scales(D1HMoEBaseCfg.costs.scales):
            # Stage 1 uses reward/termination safety only. These costs were
            # near-zero but still drove the constrained loss.
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

        class d_values(D1HMoEBaseCfg.costs.d_values):
            # Keep the original zero-budget interpretation explicit.
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0


class D1HMoEDiscCfgPPO(D1HMoEBaseCfgPPO):
    class algorithm(D1HMoEBaseCfgPPO.algorithm):
        entropy_coef = 0.001
        residual_l2_coef = 0.05
        learning_rate = 3.0e-4
        learning_rate_min = 9.0e-5
        learning_rate_max = 3.0e-3
        schedule = "adaptive"
        desired_kl = 0.01
        gamma = 0.995
        lam = 0.95
        clip_param = 0.2
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        value_loss_coef = 1.0
        cost_value_loss_coef = 0.05
        cost_viol_loss_coef = 0.05

    class policy(D1HMoEBaseCfgPPO.policy):
        # Smaller residual expert: only actor is deployed, so keep it compact.
        # Critic is train-only and can be slightly larger for value stability.
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [256, 128]
        barlow_actor_hidden_dims = [128, 64]
        barlow_mlp_encoder_dims = [96, 48]
        barlow_obs_encoder_dims = [96, 48]
        barlow_latent_dim = 12
        init_noise_std = 0.45

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = "d1h_moe_disc"
        max_iterations = 20000
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

