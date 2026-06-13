import math

import torch

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
        self.stair_contact_hist = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.last_stair_ff_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_stair_trigger = torch.zeros(self.num_envs, 2, device=self.device)
        self.stair_followup_used = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.stair_ff_trigger_arm_sum = torch.zeros(self.num_envs, device=self.device)
        self.stair_ff_contact_hit_sum = torch.zeros(self.num_envs, device=self.device)
        self.stair_ff_active_sum = torch.zeros(self.num_envs, device=self.device)

        # Moving-window stair curriculum state.
        # Bucket 0 is intentionally close to the Jun12_11-25-56_ / 9000-iter state:
        #   terrain window: Level 1 / 2 / 3
        #   ff scale: 1.0
        #   command vx: fixed by cfg.commands.ranges.lin_vel_x
        self.stair_bucket_id = int(getattr(self.cfg.terrain, "stair_bucket_initial_id", 0))
        self.stair_bucket_last_update_iter = 0
        self.stair_bucket_good_windows = 0
        self.stair_bucket_bad_windows = 0
        self.stair_bucket_low_pass_rate = 0.0
        self.stair_bucket_main_pass_rate = 0.0
        self.stair_bucket_challenge_pass_rate = 0.0
        self.stair_bucket_collapse_rate = 0.0
        self.stair_bucket_low_count = 0
        self.stair_bucket_main_count = 0
        self.stair_bucket_challenge_count = 0

    def _get_terrain_max_level(self):
        if hasattr(self, "terrain_origins"):
            return int(self.terrain_origins.shape[0] - 1)
        if hasattr(self, "max_terrain_level"):
            return int(self.max_terrain_level - 1)
        return int(getattr(self.cfg.terrain, "num_rows", 10) - 1)

    def _get_stair_bucket_max_id(self):
        first_low = int(getattr(self.cfg.terrain, "stair_bucket_first_low_level", 1))
        max_level = self._get_terrain_max_level()
        # A bucket uses [low, main, challenge] = [first_low+b, first_low+b+1, first_low+b+2].
        # Therefore the largest valid bucket satisfies challenge <= max_level.
        geometry_max_bucket = max(0, max_level - first_low - 2)
        cfg_max_bucket = getattr(self.cfg.terrain, "stair_bucket_max_id", geometry_max_bucket)
        return int(max(0, min(cfg_max_bucket, geometry_max_bucket)))

    def _get_stair_bucket_levels(self, bucket_id=None):
        if bucket_id is None:
            bucket_id = int(getattr(self, "stair_bucket_id", getattr(self.cfg.terrain, "stair_bucket_initial_id", 0)))

        bucket_id = int(max(0, min(bucket_id, self._get_stair_bucket_max_id())))
        first_low = int(getattr(self.cfg.terrain, "stair_bucket_first_low_level", 1))
        max_level = self._get_terrain_max_level()

        low_level = max(0, min(first_low + bucket_id, max_level))
        main_level = max(0, min(low_level + 1, max_level))
        challenge_level = max(0, min(low_level + 2, max_level))
        return low_level, main_level, challenge_level

    def _get_stair_bucket_ff_scale(self, bucket_id=None):
        if bucket_id is None:
            bucket_id = int(getattr(self, "stair_bucket_id", getattr(self.cfg.terrain, "stair_bucket_initial_id", 0)))

        scales = getattr(self.cfg.terrain, "stair_bucket_ff_scales", [1.0])
        if len(scales) == 0:
            return 1.0
        bucket_id = int(max(0, min(bucket_id, len(scales) - 1)))
        return float(scales[bucket_id])

    def _sample_stair_bucket_levels(self, num_samples):
        low_level, main_level, challenge_level = self._get_stair_bucket_levels()
        probs = getattr(self.cfg.terrain, "stair_bucket_sample_probs", [0.25, 0.60, 0.15])
        p_low = float(probs[0])
        p_main = float(probs[1])
        # p_high is implicit; this makes the code robust if the three values do not sum exactly to 1.
        r = torch.rand(num_samples, device=self.device)

        levels = torch.full((num_samples,), challenge_level, dtype=torch.long, device=self.device)
        levels = torch.where(r < p_low + p_main, torch.full_like(levels, main_level), levels)
        levels = torch.where(r < p_low, torch.full_like(levels, low_level), levels)
        return levels

    def _resample_stair_bucket_env_origins(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        if not hasattr(self, "terrain_origins") or not hasattr(self, "terrain_types"):
            return

        new_levels = self._sample_stair_bucket_levels(len(env_ids))
        self.terrain_levels[env_ids] = new_levels
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_env_origins(self):
        # This override makes the very first episodes follow the same moving-window
        # bucket distribution used later in reset_idx(), instead of the default
        # uniform [0, max_init_terrain_level] initialization.
        super()._get_env_origins()

        if not getattr(self.cfg.terrain, "stair_bucket_curriculum", False):
            return
        if not getattr(self, "custom_origins", False):
            return
        if not hasattr(self, "terrain_levels") or not hasattr(self, "env_origins"):
            return

        self.stair_bucket_id = int(getattr(self.cfg.terrain, "stair_bucket_initial_id", 0))
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._resample_stair_bucket_env_origins(all_env_ids)

    def _get_stair_bucket_debug_episode(self):
        if not getattr(self.cfg.terrain, "stair_bucket_curriculum", False):
            return {}

        low_level, main_level, challenge_level = self._get_stair_bucket_levels()
        ff_scale = self._get_stair_bucket_ff_scale()
        return {
            "stair_bucket_id": torch.as_tensor(float(getattr(self, "stair_bucket_id", 0)), device=self.device),
            "stair_bucket_low_level": torch.as_tensor(float(low_level), device=self.device),
            "stair_bucket_main_level": torch.as_tensor(float(main_level), device=self.device),
            "stair_bucket_challenge_level": torch.as_tensor(float(challenge_level), device=self.device),
            "stair_bucket_ff_scale": torch.as_tensor(float(ff_scale), device=self.device),
            "stair_bucket_low_pass_rate": torch.as_tensor(float(getattr(self, "stair_bucket_low_pass_rate", 0.0)), device=self.device),
            "stair_bucket_main_pass_rate": torch.as_tensor(float(getattr(self, "stair_bucket_main_pass_rate", 0.0)), device=self.device),
            "stair_bucket_challenge_pass_rate": torch.as_tensor(float(getattr(self, "stair_bucket_challenge_pass_rate", 0.0)), device=self.device),
            "stair_bucket_collapse_rate": torch.as_tensor(float(getattr(self, "stair_bucket_collapse_rate", 0.0)), device=self.device),
            "stair_bucket_low_count": torch.as_tensor(float(getattr(self, "stair_bucket_low_count", 0)), device=self.device),
            "stair_bucket_main_count": torch.as_tensor(float(getattr(self, "stair_bucket_main_count", 0)), device=self.device),
            "stair_bucket_challenge_count": torch.as_tensor(float(getattr(self, "stair_bucket_challenge_count", 0)), device=self.device),
        }

    def _rate_or_minus_one(self, values, mask):
        count = int(mask.sum().item())
        if count == 0:
            return -1.0, 0
        return float(values[mask].float().mean().item()), count

    def _update_stair_bucket_curriculum(self, env_ids, step_success, x_progress, episode_time, terminated_early):
        if not getattr(self.cfg.terrain, "stair_bucket_curriculum", False):
            return

        if env_ids is None or len(env_ids) == 0:
            return

        low_level, main_level, challenge_level = self._get_stair_bucket_levels()
        levels = self.terrain_levels[env_ids]

        success_reward_threshold = float(getattr(self.cfg.terrain, "curriculum_success_reward_threshold", 3.2))
        success_down_threshold = float(getattr(self.cfg.terrain, "curriculum_success_down_threshold", 0.8))
        success_min_distance = float(getattr(self.cfg.terrain, "curriculum_success_min_distance", 1.8))
        success_min_episode_time = float(getattr(self.cfg.terrain, "curriculum_success_min_episode_time", 8.0))
        collapse_min_distance = float(getattr(self.cfg.terrain, "curriculum_move_down_min_distance", 0.6))

        passed = (
            (step_success > success_reward_threshold)
            & (x_progress > success_min_distance)
            & (episode_time > success_min_episode_time)
            & (~terminated_early)
        )
        collapsed = (
            ((step_success < success_down_threshold) & (x_progress < collapse_min_distance))
            | terminated_early
        )

        low_mask = levels == low_level
        main_mask = levels == main_level
        challenge_mask = levels == challenge_level

        self.stair_bucket_low_pass_rate, self.stair_bucket_low_count = self._rate_or_minus_one(passed, low_mask)
        self.stair_bucket_main_pass_rate, self.stair_bucket_main_count = self._rate_or_minus_one(passed, main_mask)
        self.stair_bucket_challenge_pass_rate, self.stair_bucket_challenge_count = self._rate_or_minus_one(passed, challenge_mask)
        self.stair_bucket_collapse_rate = float(collapsed.float().mean().item())

        steps_per_iter = max(int(getattr(self.cfg.control, "stair_ff_anneal_steps_per_iter", 32)), 1)
        current_iter = int(getattr(self, "common_step_counter", 0) // steps_per_iter)

        interval = int(getattr(self.cfg.terrain, "stair_bucket_update_interval", 300))
        cooldown = int(getattr(self.cfg.terrain, "stair_bucket_cooldown_iters", 300))
        if current_iter - int(getattr(self, "stair_bucket_last_update_iter", 0)) < max(interval, cooldown):
            return

        min_main_samples = int(getattr(self.cfg.terrain, "stair_bucket_min_main_samples", 96))
        min_challenge_samples = int(getattr(self.cfg.terrain, "stair_bucket_min_challenge_samples", 32))

        promote_main_rate = float(getattr(self.cfg.terrain, "stair_bucket_promote_main_rate", 0.65))
        promote_challenge_rate = float(getattr(self.cfg.terrain, "stair_bucket_promote_challenge_rate", 0.35))
        promote_max_collapse_rate = float(getattr(self.cfg.terrain, "stair_bucket_promote_max_collapse_rate", 0.25))
        promote_windows = int(getattr(self.cfg.terrain, "stair_bucket_promote_windows", 2))

        enough_promote_samples = (
            self.stair_bucket_main_count >= min_main_samples
            and self.stair_bucket_challenge_count >= min_challenge_samples
        )
        promote_ok = (
            enough_promote_samples
            and self.stair_bucket_main_pass_rate >= promote_main_rate
            and self.stair_bucket_challenge_pass_rate >= promote_challenge_rate
            and self.stair_bucket_collapse_rate <= promote_max_collapse_rate
        )

        if promote_ok:
            self.stair_bucket_good_windows += 1
        else:
            self.stair_bucket_good_windows = 0

        max_bucket = self._get_stair_bucket_max_id()
        if self.stair_bucket_good_windows >= promote_windows and self.stair_bucket_id < max_bucket:
            self.stair_bucket_id += 1
            self.stair_bucket_last_update_iter = current_iter
            self.stair_bucket_good_windows = 0
            self.stair_bucket_bad_windows = 0
            print(
                f"[stair bucket] promote to {self.stair_bucket_id}: "
                f"levels={self._get_stair_bucket_levels()}, "
                f"ff_scale={self._get_stair_bucket_ff_scale():.3f}, "
                f"main_pass={self.stair_bucket_main_pass_rate:.3f}, "
                f"challenge_pass={self.stair_bucket_challenge_pass_rate:.3f}, "
                f"collapse={self.stair_bucket_collapse_rate:.3f}"
            )
            return

        # Bucket demotion is intentionally conservative. Per-env difficulty is already
        # controlled by the moving-window sampler; global demotion should only happen
        # when the current bucket breaks the old/easy slice or causes broad collapse.
        demote_main_rate = float(getattr(self.cfg.terrain, "stair_bucket_demote_main_rate", 0.35))
        demote_low_rate = float(getattr(self.cfg.terrain, "stair_bucket_demote_low_rate", 0.50))
        demote_collapse_rate = float(getattr(self.cfg.terrain, "stair_bucket_demote_collapse_rate", 0.45))
        demote_windows = int(getattr(self.cfg.terrain, "stair_bucket_demote_windows", 3))

        low_is_broken = (self.stair_bucket_low_count >= min_challenge_samples) and (
            0.0 <= self.stair_bucket_low_pass_rate < demote_low_rate
        )
        main_is_broken = (self.stair_bucket_main_count >= min_main_samples) and (
            0.0 <= self.stair_bucket_main_pass_rate < demote_main_rate
        )
        demote_ok = low_is_broken or (main_is_broken and self.stair_bucket_collapse_rate > demote_collapse_rate)

        if demote_ok and self.stair_bucket_id > 0:
            self.stair_bucket_bad_windows += 1
        else:
            self.stair_bucket_bad_windows = 0

        if self.stair_bucket_bad_windows >= demote_windows and self.stair_bucket_id > 0:
            self.stair_bucket_id -= 1
            self.stair_bucket_last_update_iter = current_iter
            self.stair_bucket_good_windows = 0
            self.stair_bucket_bad_windows = 0
            print(
                f"[stair bucket] demote to {self.stair_bucket_id}: "
                f"levels={self._get_stair_bucket_levels()}, "
                f"ff_scale={self._get_stair_bucket_ff_scale():.3f}, "
                f"low_pass={self.stair_bucket_low_pass_rate:.3f}, "
                f"main_pass={self.stair_bucket_main_pass_rate:.3f}, "
                f"collapse={self.stair_bucket_collapse_rate:.3f}"
            )


    def reset_idx(self, env_ids):
        debug_episode = {}
        if len(env_ids) > 0 and hasattr(self, "stair_ff_trigger_arm_sum"):
            denom = torch.clamp(self.episode_length_buf[env_ids].float(), min=1.0)
            debug_episode["stair_ff_trigger_arm_ratio"] = torch.mean(self.stair_ff_trigger_arm_sum[env_ids] / denom)
            debug_episode["stair_ff_contact_hit_ratio"] = torch.mean(self.stair_ff_contact_hit_sum[env_ids] / denom)
            debug_episode["stair_ff_active_ratio"] = torch.mean(self.stair_ff_active_sum[env_ids] / denom)
            debug_episode["stair_ff_anneal_scale"] = self._get_stair_ff_anneal_scale()

        super().reset_idx(env_ids)

        if getattr(self.cfg.terrain, "stair_bucket_curriculum", False):
            debug_episode.update(self._get_stair_bucket_debug_episode())

        if len(debug_episode) > 0 and "episode" in self.extras:
            self.extras["episode"].update(debug_episode)

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
        self.stair_ff_trigger_arm_sum[env_ids] = 0.0
        self.stair_ff_contact_hit_sum[env_ids] = 0.0
        self.stair_ff_active_sum[env_ids] = 0.0

    def step(self, actions):
        actions = self._apply_stair_feedforward(actions)
        return super().step(actions)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_step_contact_state()
        self._update_step_imbalance_state()

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
        if not getattr(self.cfg.control, "stair_ff_require_step_context", True):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        context = self._get_step_lift_context()
        if context is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        arm = context[0]
        min_travel = getattr(self.cfg.control, "stair_ff_min_forward_travel", 0.15)
        if min_travel > 0.0:
            forward_travel = self.root_states[:, 0] - self.env_origins[:, 0]
            arm = arm & (forward_travel >= min_travel)

        if getattr(self.cfg.control, "stair_ff_require_blocking", True):
            min_cmd_x = getattr(self.cfg.rewards, "step_clearance_min_cmd_x", 0.03)
            cmd_x = torch.clamp(self.commands[:, 0], min=min_cmd_x)
            speed_fraction = getattr(self.cfg.control, "stair_ff_blocking_speed_fraction", 0.65)
            speed_max = getattr(self.cfg.control, "stair_ff_blocking_speed_max", 0.30)
            blocking_speed = torch.minimum(cmd_x * speed_fraction, torch.full_like(cmd_x, speed_max))
            arm = arm & (self.base_lin_vel[:, 0] < blocking_speed)

        return arm

    def _get_stair_ff_gate(self):
        gate = self.stair_lift_active.any(dim=1)
        context = self._get_step_lift_context()
        if context is not None:
            gate = gate | context[0]
        return gate.float()

    def _get_stair_ff_anneal_scale(self):
        override_scale = getattr(self.cfg.control, "stair_ff_anneal_override_scale", None)
        if override_scale is not None:
            return torch.as_tensor(float(override_scale), device=self.device)

        if not getattr(self.cfg.control, "stair_ff_anneal_enabled", False):
            return torch.ones((), device=self.device)

        steps_per_iter = max(int(getattr(self.cfg.control, "stair_ff_anneal_steps_per_iter", 32)), 1)
        iter_offset = float(getattr(self.cfg.control, "stair_ff_anneal_iter_offset", 0.0))
        train_iter = torch.as_tensor(
            float(getattr(self, "common_step_counter", 0)) / steps_per_iter + iter_offset,
            device=self.device,
        )
        start_iter = float(getattr(self.cfg.control, "stair_ff_anneal_start_iter", 0.0))
        end_iter = float(getattr(self.cfg.control, "stair_ff_anneal_end_iter", start_iter + 1.0))
        final_scale = float(getattr(self.cfg.control, "stair_ff_anneal_final_scale", 0.0))

        progress = torch.clamp((train_iter - start_iter) / max(end_iter - start_iter, 1e-6), 0.0, 1.0)
        cosine_scale = 0.5 * (1.0 + torch.cos(math.pi * progress))
        return final_scale + (1.0 - final_scale) * cosine_scale

    def _trigger_stair_lift(self, env_mask, side, episode_time, is_followup=False):
        trigger_mask = env_mask & ~self.stair_lift_active[:, side]
        if not torch.any(trigger_mask):
            return
        if not is_followup:
            self.stair_followup_used[trigger_mask] = False
        self.stair_lift_active[trigger_mask, side] = True
        self.stair_lift_phase[trigger_mask, side] = 0.0
        self.stair_lift_side[trigger_mask] = side
        self.last_stair_trigger[trigger_mask, side] = episode_time[trigger_mask] + self.dt
        if is_followup:
            self.stair_followup_used[trigger_mask, side] = True

    def _update_stair_feedforward_state(self):
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            self.last_stair_ff_signal.zero_()
            return

        contact_forces = self._get_stair_ff_contact_forces()
        if contact_forces is None:
            self.last_stair_ff_signal.zero_()
            return

        self.stair_contact_hist = torch.cat(
            [self.stair_contact_hist[:, :, 1:].clone(), contact_forces.unsqueeze(-1)],
            dim=2,
        )
        smooth_frames = int(getattr(self.cfg.control, "stair_ff_contact_smooth_frames", 3))
        smooth_frames = max(1, min(smooth_frames, self.stair_contact_hist.shape[2]))
        smooth_contact = self.stair_contact_hist[:, :, -smooth_frames:].mean(dim=2)

        period = max(getattr(self.cfg.control, "stair_ff_period", 0.65), 1e-6)
        threshold = getattr(self.cfg.control, "stair_ff_contact_threshold", 50.0)
        delay = getattr(self.cfg.control, "stair_ff_followup_delay_factor", 0.5) * period
        episode_time = self.episode_length_buf.float() * self.dt

        trigger_arm = self._get_stair_ff_trigger_arm()
        contact_hit = (smooth_contact > threshold) & trigger_arm.unsqueeze(1)
        self.stair_ff_trigger_arm_sum += trigger_arm.float()
        self.stair_ff_contact_hit_sum += contact_hit.any(dim=1).float()
        no_lift = ~self.stair_lift_active.any(dim=1)
        no_contact = ~contact_hit.any(dim=1)
        clear_mask = no_lift & (no_contact | ~trigger_arm)
        self.last_stair_trigger[clear_mask] = 0.0
        self.stair_followup_used[clear_mask] = False

        both_hit = contact_hit[:, 0] & contact_hit[:, 1]
        left_stronger = smooth_contact[:, 0] >= smooth_contact[:, 1]
        no_active = ~self.stair_lift_active.any(dim=1)

        left_first = no_active & ((contact_hit[:, 0] & ~contact_hit[:, 1]) | (both_hit & left_stronger))
        right_first = no_active & ((contact_hit[:, 1] & ~contact_hit[:, 0]) | (both_hit & ~left_stronger))
        self._trigger_stair_lift(left_first, 0, episode_time)
        self._trigger_stair_lift(right_first, 1, episode_time)

        left_since_right = episode_time - self.last_stair_trigger[:, 1]
        right_since_left = episode_time - self.last_stair_trigger[:, 0]
        followup_available = ~self.stair_followup_used.any(dim=1)
        left_followup = (
            ~self.stair_lift_active[:, 0]
            & (self.last_stair_trigger[:, 1] > 0.0)
            & followup_available
            & (((left_since_right >= 0.0) & contact_hit[:, 0]) | (left_since_right >= delay))
        )
        right_followup = (
            ~self.stair_lift_active[:, 1]
            & (self.last_stair_trigger[:, 0] > 0.0)
            & followup_available
            & (((right_since_left >= 0.0) & contact_hit[:, 1]) | (right_since_left >= delay))
        )
        self._trigger_stair_lift(left_followup, 0, episode_time, is_followup=True)
        self._trigger_stair_lift(right_followup, 1, episode_time, is_followup=True)

        self.stair_lift_phase = torch.where(
            self.stair_lift_active,
            self.stair_lift_phase + self.dt / period,
            self.stair_lift_phase,
        )
        done = self.stair_lift_phase >= 1.0
        self.stair_lift_active = self.stair_lift_active & ~done
        self.stair_lift_phase = torch.where(done, torch.zeros_like(self.stair_lift_phase), self.stair_lift_phase)

        phase = torch.clamp(self.stair_lift_phase, 0.0, 1.0)
        signal = 0.5 * (1.0 - torch.cos(2.0 * math.pi * phase))
        self.last_stair_ff_signal = signal * self.stair_lift_active.float()
        self.stair_ff_active_sum += self.stair_lift_active.any(dim=1).float()

    def _apply_stair_feedforward(self, actions):
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            return actions

        self._update_stair_feedforward_state()
        return actions

    def _get_stair_ff_joint_target_offsets(self):
        ff_offset = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        if not getattr(self.cfg.control, "stair_ff_enabled", True):
            return ff_offset
        if not torch.any(self.stair_lift_active):
            return ff_offset

        joint_amplitudes = getattr(self.cfg.control, "stair_ff_joint_amplitudes", {})
        for joint_name, amplitude in joint_amplitudes.items():
            if not hasattr(self, "dof_names") or joint_name not in self.dof_names:
                continue
            joint_idx = self.dof_names.index(joint_name)
            if joint_idx in self.foot_joint_indices:
                continue
            side = 0 if joint_name.startswith("FL_") else 1 if joint_name.startswith("FR_") else None
            if side is None:
                continue
            ff_offset[:, joint_idx] += self.last_stair_ff_signal[:, side] * amplitude

        k_ff = getattr(self.cfg.control, "stair_ff_k", 0.35)
        anneal_scale = self._get_stair_ff_anneal_scale()
        ff_gate = self._get_stair_ff_gate().unsqueeze(1)
        bucket_scale = self._get_stair_bucket_ff_scale() if getattr(self.cfg.terrain, "stair_bucket_curriculum", False) else 1.0
        return ff_gate * k_ff * anneal_scale * bucket_scale * ff_offset

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
        """Moving-window stair terrain curriculum.

        Bucket interpretation:
            current bucket b -> sample three adjacent levels:
                low        = first_low + b
                main       = first_low + b + 1
                challenge  = first_low + b + 2

        Sampling ratio is cfg.terrain.stair_bucket_sample_probs, default [0.25, 0.60, 0.15].
        The bucket is promoted only when the main level is already stable and the
        challenge level has started to succeed. This avoids upgrading because of
        easy replay levels only.
        """
        if not self.init_done:
            return

        if getattr(self.cfg.terrain, "stair_bucket_curriculum", False):
            x_progress = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            episode_time = self.episode_length_buf[env_ids].float() * self.dt

            if hasattr(self, "episode_sums") and "step_success" in self.episode_sums:
                step_success = self.episode_sums["step_success"][env_ids] / self.max_episode_length_s
            else:
                step_success = torch.zeros_like(x_progress)

            if hasattr(self, "time_out_buf"):
                terminated_early = self.reset_buf[env_ids] & (~self.time_out_buf[env_ids])
            else:
                terminated_early = self.reset_buf[env_ids].bool()

            self._update_stair_bucket_curriculum(
                env_ids=env_ids,
                step_success=step_success,
                x_progress=x_progress,
                episode_time=episode_time,
                terminated_early=terminated_early,
            )

            self._resample_stair_bucket_env_origins(env_ids)
            return

        # Fallback: original per-env stair curriculum.
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up_distance = getattr(self.cfg.terrain, "curriculum_move_up_distance", 3.0)
        move_down_expected_factor = getattr(self.cfg.terrain, "curriculum_move_down_expected_factor", 0.30)
        move_down_min_distance = getattr(self.cfg.terrain, "curriculum_move_down_min_distance", 1.0)
        success_reward_threshold = getattr(self.cfg.terrain, "curriculum_success_reward_threshold", 0.85)
        success_down_threshold = getattr(self.cfg.terrain, "curriculum_success_down_threshold", 0.15)
        success_min_distance = getattr(self.cfg.terrain, "curriculum_success_min_distance", 1.8)
        success_min_episode_time = getattr(self.cfg.terrain, "curriculum_success_min_episode_time", 8.0)
        allow_distance_promotion = getattr(self.cfg.terrain, "curriculum_allow_distance_promotion", False)
        max_allowed_level = min(
            getattr(self.cfg.terrain, "curriculum_max_terrain_level", self.max_terrain_level - 1),
            self.max_terrain_level - 1,
        )

        expected_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
        episode_time = self.episode_length_buf[env_ids].float() * self.dt
        if hasattr(self, "episode_sums") and "step_success" in self.episode_sums:
            step_success = self.episode_sums["step_success"][env_ids] / self.max_episode_length_s
        else:
            step_success = torch.zeros_like(distance)

        move_up_by_success = (
            (step_success > success_reward_threshold)
            & (distance > success_min_distance)
            & (episode_time > success_min_episode_time)
        )
        move_up_by_distance = (distance > move_up_distance) & allow_distance_promotion
        move_up = move_up_by_success | move_up_by_distance
        move_down_distance = torch.clamp(expected_distance * move_down_expected_factor, min=move_down_min_distance)
        move_down = (step_success < success_down_threshold) & (distance < move_down_distance) & ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, max_allowed_level)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

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
        stall_score = torch.clamp((min_speed - self.base_lin_vel[:, 0]) / max(min_speed, 1e-6), 0.0, 1.0)
        stalled = self.base_lin_vel[:, 0] < min_speed
        return (active & stalled).float() * self._get_stair_posture_gate()

    def _reward_opposite_base_vel(self):
        cmd_x = self.commands[:, 0]
        backward = (cmd_x > 0.05) & (self.base_lin_vel[:, 0] < -0.03)
        penalty = torch.clamp(-self.base_lin_vel[:, 0] / torch.clamp(cmd_x, min=0.05), 0.0, 1.0)
        return backward.float() * penalty


class D1HMoEDiscCfg(D1HMoEBaseCfg):
    class commands(D1HMoEBaseCfg.commands):
        # Stage-A objective: climb higher stairs at an almost fixed forward speed.
        # Speed curriculum and feedforward annealing are intentionally postponed
        # until the moving-window terrain bucket reaches the target height.
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
            # Verified stable command band around 0.4 m/s.
            lin_vel_x = [0.36, 0.46]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class terrain(D1HMoEBaseCfg.terrain):
        curriculum = True
        max_init_terrain_level = 3

        # Terrain order: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles].
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

        # Use more rows so the low-level heights stay close to the previous
        # [0.035, 0.15] / 10-row run, while the final row reaches about 17 cm.
        # In utils/terrain.py: difficulty = row / num_rows, so row 12 height is:
        # 0.035 + (0.18125 - 0.035) * 12 / 13 = 0.170 m.
        num_rows = 13
        step_height = [0.035, 0.18125]

        step_width_range = [0.40, 0.55]
        slope = [0.0, 0.02]
        slope_treshold = 0.3

        # Original per-env stair curriculum thresholds are reused as pass/fail
        # definitions for bucket promotion/demotion.
        curriculum_move_up_distance = 6.0
        curriculum_move_down_expected_factor = 0.18
        curriculum_move_down_min_distance = 0.6
        curriculum_success_reward_threshold = 3.2
        curriculum_success_down_threshold = 0.8
        curriculum_success_min_distance = 1.8
        curriculum_success_min_episode_time = 8.0
        curriculum_allow_distance_promotion = False
        curriculum_max_terrain_level = 12

        # Moving-window bucket curriculum.
        # Bucket b samples: [first_low+b, first_low+b+1, first_low+b+2].
        # Default bucket 0 samples Level 1 / 2 / 3, close to the 9000-iter
        # terrain_level state, without the 17cm OOD jump.
        stair_bucket_curriculum = True
        stair_bucket_initial_id = 0
        stair_bucket_first_low_level = 1
        stair_bucket_max_id = 9
        stair_bucket_sample_probs = [0.25, 0.60, 0.15]

        # Bucket ff scale. Base amplitude remains 0.30 / -0.60; these values
        # only multiply that base amplitude smoothly as the terrain window moves.
        stair_bucket_ff_scales = [
            1.00,  # bucket 0: L1/L2/L3
            1.04,  # bucket 1: L2/L3/L4
            1.08,  # bucket 2: L3/L4/L5
            1.12,  # bucket 3: L4/L5/L6
            1.17,  # bucket 4: L5/L6/L7
            1.22,  # bucket 5: L6/L7/L8
            1.28,  # bucket 6: L7/L8/L9
            1.34,  # bucket 7: L8/L9/L10
            1.38,  # bucket 8: L9/L10/L11
            1.40,  # bucket 9: L10/L11/L12, challenge reaches ~17cm
        ]

        # Promotion: main level should already be reliable, and challenge level
        # should have begun to succeed. This avoids upgrading because the replay
        # slice is easy.
        stair_bucket_update_interval = 300
        stair_bucket_cooldown_iters = 300
        stair_bucket_min_main_samples = 96
        stair_bucket_min_challenge_samples = 32
        stair_bucket_promote_main_rate = 0.65
        stair_bucket_promote_challenge_rate = 0.35
        stair_bucket_promote_max_collapse_rate = 0.25
        stair_bucket_promote_windows = 2

        # Demotion is conservative. A bucket should fall back only when the new
        # window breaks the main/easy levels, not merely because challenge samples fail.
        stair_bucket_demote_main_rate = 0.35
        stair_bucket_demote_low_rate = 0.50
        stair_bucket_demote_collapse_rate = 0.45
        stair_bucket_demote_windows = 3

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
        stair_ff_period = 0.65
        stair_ff_k = 1.0
        stair_ff_contact_threshold = 100.0
        stair_ff_contact_force_axis = "horizontal"
        stair_ff_require_step_context = True
        stair_ff_min_forward_travel = 0.15
        stair_ff_require_blocking = True
        stair_ff_blocking_speed_fraction = 0.65
        stair_ff_blocking_speed_max = 0.30
        stair_ff_followup_delay_factor = 0.35
        stair_ff_contact_smooth_frames = 3

        # Stage A: do NOT anneal the feedforward yet.
        # It will be annealed only after the terrain bucket reaches the target height.
        stair_ff_anneal_enabled = False
        stair_ff_anneal_start_iter = 1000
        stair_ff_anneal_end_iter = 14000
        stair_ff_anneal_final_scale = 1.0
        stair_ff_anneal_steps_per_iter = 32
        stair_ff_anneal_iter_offset = 0.0
        stair_ff_anneal_override_scale = None

        # Keep the 9000-iter base amplitude. The bucket curriculum multiplies this
        # by stair_bucket_ff_scales, avoiding a sudden 0.43/-0.86 OOD action jump.
        stair_ff_joint_amplitudes = {
            "FL_thigh_joint": 0.30,
            "FL_calf_joint": -0.60,
            "FR_thigh_joint": 0.30,
            "FR_calf_joint": -0.60,
        }

    class rewards(D1HMoEBaseCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.07
        distance_sigma = 0.08
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.98
        soft_torque_limit = 0.98
        base_height_target = 0.45
        base_height_scale = 0.05
        base_height_deadband = 0.01

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
        step_stall_min_speed = 0.12
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
            tracking_lin_vel_x = 10.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 0.0
            heading = -2.0

            # Stability guardrails. They should prevent garbage motion, not dominate climbing.
            orientation = -14.0
            upward = 0.0
            ang_vel_xy = -0.30
            base_height = -7.5
            lin_vel_z = -0.3

            # Failure/contact penalties.
            termination = -400.0
            collision = -16.0
            collision_hard = -140.0
            collision_head = 0.0

            # Remove tiny regularizers that only add noise to the scalar reward.
            torques = 0.0
            powers = 0.0
            dof_acc = 0.0
            action_rate = -0.03
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
            step_clearance = 1.0
            step_lift = 2.0
            step_pre_lift = 0.0
            step_reactive_lift = 0.0
            step_leg_imbalance = -8.0
            step_progress = 15.0
            step_up = 60.0
            step_success = 80.0
            step_stall = -12.0
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
        learning_rate = 5.0e-4
        learning_rate_min = 1.0e-4
        learning_rate_max = 3.0e-3
        schedule = "adaptive"
        desired_kl = 0.015
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
        actor_hidden_dims = [512, 256, 128]
        barlow_actor_hidden_dims = [512, 256, 128]
        barlow_mlp_encoder_dims = [128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 0.45

    class runner(D1HMoEBaseCfgPPO.runner):
        experiment_name = "d1h_moe_disc"
        max_iterations = 20000
        num_steps_per_env = 32
        save_interval = 200
