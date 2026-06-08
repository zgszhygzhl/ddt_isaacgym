import time
import os
from collections import deque
import statistics
import warnings

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
from isaacgym import gymapi
from global_config import ROOT_DIR

# from modules import ActorCriticRMA,ActorCriticRmaTrans,ActorCriticSF,ActorCriticBarlowTwins,ActorCriticStateTransformer,ActorCriticTransBarlowTwins,ActorCriticMixedBarlowTwins,ActorCriticRnnBarlowTwins,ActorCriticVqvae
from modules import ActorCriticBarlowTwins 
from algorithm import NP3O
from envs.vec_env import VecEnv
from utils.helpers import hard_phase_schedualer, partial_checkpoint_load
from copy import copy, deepcopy
from utils import get_load_path
from utils.video_recorder import FfmpegVideoWriter

class OnConstraintPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.current_learning_iteration = 0

        # self.phase1_end = self.cfg["phase1_end"] 
 
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCriticBarlowTwins = actor_critic_class(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg)
        print("Policy architecture: ",actor_critic)

        checkpoint_dict = None
        resume_path = None
        if self.cfg['resume']:
            log_root = os.path.join(ROOT_DIR, 'logs', self.cfg['experiment_name'], self.cfg['resume_path'])
            resume_path = get_load_path(log_root, load_run=self.cfg['load_run'], checkpoint=self.cfg['checkpoint'])
            print("Resume model from: ",resume_path)
            checkpoint_dict = torch.load(resume_path, map_location=self.device)
            actor_critic.load_state_dict(checkpoint_dict['model_state_dict'])
        
        actor_critic.to(self.device)

        # Create algorithm
        self.alg_cfg['k_value'] = self.env.cost_k_values
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        if checkpoint_dict is not None and 'optimizer_state_dict' in checkpoint_dict:
            self.alg.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

        if checkpoint_dict is not None:
            checkpoint_iter = checkpoint_dict.get('iter')
            path_iter = self._extract_iteration_from_path(resume_path)
            if checkpoint_iter is None or checkpoint_iter < 0:
                checkpoint_iter = path_iter
            elif path_iter > int(checkpoint_iter):
                checkpoint_iter = path_iter
            self.current_learning_iteration = int(checkpoint_iter)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
            [self.env.cfg.costs.num_costs],
            self.env.cost_d_values_tensor
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0

        self.record_video = self.cfg.get("record_video", False) and self.log_dir is not None
        self.video_interval = int(self.cfg.get("video_interval", 500))
        self.video_duration = float(self.cfg.get("video_duration", 8.0))
        self.video_fps = int(self.cfg.get("video_fps", 30))
        self.video_num_envs = int(self.cfg.get("video_num_envs", 16))
        self.video_tile_rows = int(self.cfg.get("video_tile_rows", 4))
        self.video_tile_cols = int(self.cfg.get("video_tile_cols", 4))
        self.video_tile_width = int(self.cfg.get("video_tile_width", 320))
        self.video_tile_height = int(self.cfg.get("video_tile_height", 180))
        self.video_width = self.video_tile_cols * self.video_tile_width
        self.video_height = self.video_tile_rows * self.video_tile_height
        self.video_dir = None
        self.video_env_ids = []
        self.video_cam_handles = []
        self.video_writer = None
        self.video_steps_left = 0
        self.video_step_count = 0
        self.video_record_every = max(1, int(1.0 / (self.video_fps * self.env.dt)))
        self.video_black_tile = np.zeros(
            (self.video_tile_height, self.video_tile_width, 3),
            dtype=np.uint8,
        )

        self.env.reset()
        if self.record_video:
            self._setup_train_video_camera()

    def _extract_iteration_from_path(self, checkpoint_path):
        if checkpoint_path is None:
            return 0

        filename = os.path.basename(checkpoint_path)
        stem, _ = os.path.splitext(filename)
        if '_' not in stem:
            return 0

        try:
            return int(stem.rsplit('_', 1)[-1])
        except ValueError:
            return 0

    def _setup_train_video_camera(self):
        self.video_env_ids = []
        self.video_cam_handles = []
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.video_tile_width
        camera_props.height = self.video_tile_height

        max_cameras = min(
            self.env.num_envs,
            self.video_num_envs,
            self.video_tile_rows * self.video_tile_cols,
        )
        selected_env_ids = self._select_train_video_env_ids(max_cameras)
        for env_index in selected_env_ids:
            env_handle = self.env.envs[env_index]
            cam_handle = self.env.gym.create_camera_sensor(env_handle, camera_props)
            self.video_env_ids.append(env_index)
            self.video_cam_handles.append(cam_handle)

        self._update_train_video_camera_locations()

    def _select_train_video_env_ids(self, max_cameras):
        if max_cameras <= 0:
            return []

        terrain_types = getattr(self.env, "terrain_types", None)
        terrain_cfg = getattr(getattr(self.env, "cfg", None), "terrain", None)
        if terrain_types is None or terrain_cfg is None:
            return list(range(max_cameras))

        num_cols = int(getattr(terrain_cfg, "num_cols", 0))
        terrain_proportions = list(getattr(terrain_cfg, "terrain_proportions", []))
        if num_cols <= 0 or not terrain_proportions:
            return list(range(max_cameras))

        terrain_type_values = terrain_types.detach().cpu().tolist()
        terrain_type_to_env_ids = {}
        for env_index, terrain_type in enumerate(terrain_type_values):
            terrain_type_to_env_ids.setdefault(int(terrain_type), []).append(env_index)

        category_to_cols = {
            "smooth_slope": [],
            "rough_slope": [],
            "stairs_up": [],
            "stairs_down": [],
            "discrete": [],
            "stepping_stones": [],
            "gap": [],
            "pit": [],
        }
        cumulative = np.cumsum(np.asarray(terrain_proportions, dtype=np.float64))
        for col_index in range(num_cols):
            choice = col_index / num_cols + 0.001
            if choice < cumulative[0]:
                category_to_cols["smooth_slope"].append(col_index)
            elif len(cumulative) > 1 and choice < cumulative[1]:
                category_to_cols["rough_slope"].append(col_index)
            elif len(cumulative) > 3 and choice < cumulative[3]:
                if choice < cumulative[2]:
                    category_to_cols["stairs_up"].append(col_index)
                else:
                    category_to_cols["stairs_down"].append(col_index)
            elif len(cumulative) > 4 and choice < cumulative[4]:
                category_to_cols["discrete"].append(col_index)
            elif len(cumulative) > 5 and choice < cumulative[5]:
                category_to_cols["stepping_stones"].append(col_index)
            elif len(cumulative) > 6 and choice < cumulative[6]:
                category_to_cols["gap"].append(col_index)
            else:
                category_to_cols["pit"].append(col_index)

        selected_env_ids = []
        used_cols = set()
        primary_categories = [
            "stairs_up",
            "stairs_down",
            "smooth_slope",
            "rough_slope",
            "discrete",
            "stepping_stones",
            "gap",
            "pit",
        ]

        for category in primary_categories:
            for col_index in category_to_cols[category]:
                env_ids = terrain_type_to_env_ids.get(col_index)
                if env_ids:
                    selected_env_ids.append(env_ids[0])
                    used_cols.add(col_index)
                    break
            if len(selected_env_ids) >= max_cameras:
                return selected_env_ids[:max_cameras]

        refill_categories = [
            "stairs_up",
            "stairs_down",
            "rough_slope",
            "smooth_slope",
            "discrete",
            "stepping_stones",
            "gap",
            "pit",
        ]
        made_progress = True
        while len(selected_env_ids) < max_cameras and made_progress:
            made_progress = False
            for category in refill_categories:
                for col_index in category_to_cols[category]:
                    if col_index in used_cols:
                        continue
                    env_ids = terrain_type_to_env_ids.get(col_index)
                    if not env_ids:
                        continue
                    selected_env_ids.append(env_ids[0])
                    used_cols.add(col_index)
                    made_progress = True
                    break
                if len(selected_env_ids) >= max_cameras:
                    break

        if len(selected_env_ids) < max_cameras:
            selected_set = set(selected_env_ids)
            for env_index in range(self.env.num_envs):
                if env_index in selected_set:
                    continue
                selected_env_ids.append(env_index)
                selected_set.add(env_index)
                if len(selected_env_ids) >= max_cameras:
                    break

        return selected_env_ids[:max_cameras]

    def _update_train_video_camera_locations(self):
        for env_id, cam_handle in zip(self.video_env_ids, self.video_cam_handles):
            origin = self.env.env_origins[env_id].detach().cpu().numpy()
            env_handle = self.env.envs[env_id]

            cam_pos = gymapi.Vec3(
                float(origin[0] + 2.4),
                float(origin[1] - 3.3),
                float(origin[2] + 1.55),
            )
            cam_target = gymapi.Vec3(
                float(origin[0] + 0.25),
                float(origin[1] + 0.0),
                float(origin[2] + 0.72),
            )

            self.env.gym.set_camera_location(
                cam_handle,
                env_handle,
                cam_pos,
                cam_target,
            )

    def _start_train_video(self, iteration):
        if not self.record_video or not self.video_cam_handles:
            return

        self._close_train_video()
        self.video_dir = os.path.join(self.log_dir, 'videos')
        video_path = os.path.join(self.video_dir, f'train_iter_{iteration:06d}.mp4')
        self.video_writer = FfmpegVideoWriter(
            video_path,
            self.video_width,
            self.video_height,
            self.video_fps,
        )
        self.video_steps_left = max(1, int(np.ceil(self.video_duration / self.env.dt)))
        self.video_step_count = 0

    def _capture_train_video_frame(self):
        if self.video_writer is None or not self.video_cam_handles:
            return

        self._update_train_video_camera_locations()
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)

        if self.video_step_count % self.video_record_every == 0:
            tiles = []
            total_tiles = self.video_tile_rows * self.video_tile_cols
            for env_id, cam_handle in zip(self.video_env_ids, self.video_cam_handles):
                image = self.env.gym.get_camera_image(
                    self.env.sim,
                    self.env.envs[env_id],
                    cam_handle,
                    gymapi.IMAGE_COLOR,
                )
                frame = np.asarray(image, dtype=np.uint8).reshape(
                    (self.video_tile_height, self.video_tile_width, 4)
                )[:, :, :3]
                tiles.append(frame)

            while len(tiles) < total_tiles:
                tiles.append(self.video_black_tile)

            rows = []
            for row_index in range(self.video_tile_rows):
                row_start = row_index * self.video_tile_cols
                row_tiles = tiles[row_start:row_start + self.video_tile_cols]
                rows.append(np.concatenate(row_tiles, axis=1))

            mosaic_frame = np.concatenate(rows, axis=0)
            self.video_writer.write(mosaic_frame)

        self.video_step_count += 1
        self.video_steps_left -= 1
        if self.video_steps_left <= 0:
            self._close_train_video()

    def _close_train_video(self):
        if self.video_writer is None:
            return

        try:
            self.video_writer.close()
        except Exception as error:
            warnings.warn(f'Failed to finalize training video: {error}')
        finally:
            self.video_writer = None
            self.video_steps_left = 0
            self.video_step_count = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.act_shed,self.imi_shed,self.lag_shed = hard_phase_schedualer(max_iters=tot_iter,
        #             phase1_end=self.phase1_end)

        #imitation_mode
        if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
            self.alg.actor_critic.imitation_mode()
            
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.alg.actor_critic, "set_learning_iteration"):
                self.alg.actor_critic.set_learning_iteration(it)

            if self.record_video and it % self.video_interval == 0:
                self._start_train_video(it)

            # act_teacher_flag = self.act_shed[it]
            # imi_flag = self.imi_shed[it]
            # lag_flag = self.lag_shed[it]

            # self.alg.set_imi_flag(imi_flag)
            # self.alg.actor_critic.set_teacher_act(act_teacher_flag)
            # # self.env.randomize_lag_timesteps = lag_flag
            # # if self.env.randomize_lag_timesteps:
            # #     print("lag is on")
            # # else:
            # #     print("lag is off")
            # if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
            #     step_size = 1/int(tot_iter/2)
            #     imi_weight = max(0,1 - it * step_size)
            #     print("imi_weight:",imi_weight)
            #     self.alg.set_imi_weight(imi_weight)
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                   
                    actions = self.alg.act(obs, critic_obs, infos)
                    obs, privileged_obs, rewards,costs,dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    self._capture_train_video_frame()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs,rewards,costs,dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device),costs.to(self.device),dones.to(self.device)
                    self.alg.process_env_step(rewards,costs,dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                self.alg.compute_cost_returns(critic_obs)

            #update k value for better expolration
            k_value = self.alg.update_k_value(it)
            
            mean_value_loss,mean_cost_value_loss,mean_viol_loss,mean_surrogate_loss, mean_imitation_loss,obs_batch_min,obs_batch_max = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), iteration=it)
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        self.save(
            os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
            iteration=self.current_learning_iteration,
        )
        self._close_train_video()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        #mean_std = self.alg.actor_critic.std.mean()
        mean_std = self.alg.actor_critic.get_std().mean()
        residual_mean_std = None
        if hasattr(self.alg.actor_critic, "get_residual_std"):
            if hasattr(self.alg.actor_critic, "get_effective_std"):
                residual_mean_std = self.alg.actor_critic.get_effective_std().mean()
            else:
                residual_mean_std = self.alg.actor_critic.get_residual_std().mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        #mean_kl_loss,mean_recons_loss,mean_vel_recons_loss
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/cost_value_function', locs['mean_cost_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_viol_loss', locs['mean_viol_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_imitation_loss', locs['mean_imitation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if residual_mean_std is not None:
            self.writer.add_scalar('Policy/residual_mean_noise_std', residual_mean_std.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_current_alpha"):
            self.writer.add_scalar('Policy/residual_alpha', self.alg.actor_critic.last_current_alpha.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_delta_norm"):
            self.writer.add_scalar('Policy/residual_delta_norm', self.alg.actor_critic.last_delta_norm.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_saturation_ratio"):
            self.writer.add_scalar('Policy/action_saturation_ratio', self.alg.actor_critic.last_saturation_ratio.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        self.writer.add_scalar('Data/obs_max', locs['obs_batch_max'], locs['it'])
        self.writer.add_scalar('Data/obs_min', locs['obs_batch_min'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        if iteration is None:
            iteration = self.current_learning_iteration

        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_teacher
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
