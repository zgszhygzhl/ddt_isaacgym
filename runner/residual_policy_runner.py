import os

import torch

from algorithm import NP3O
from global_config import ROOT_DIR
from runner.on_constraint_policy_runner import OnConstraintPolicyRunner
from utils import get_load_path


class ResidualPolicyRunner(OnConstraintPolicyRunner):
    def __init__(self, env, train_cfg, actor_critic, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.current_learning_iteration = 0

        checkpoint_dict = None
        resume_path = None
        if self.cfg["resume"]:
            log_root = os.path.join(ROOT_DIR, "logs", self.cfg["experiment_name"], self.cfg["resume_path"])
            resume_path = get_load_path(log_root, load_run=self.cfg["load_run"], checkpoint=self.cfg["checkpoint"])
            print("Resume model from: ", resume_path)
            checkpoint_dict = torch.load(resume_path, map_location=self.device)
            actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict=False)

        actor_critic.to(self.device)
        self.alg_cfg["k_value"] = self.env.cost_k_values
        self.alg = NP3O(actor_critic, device=self.device, **self.alg_cfg)
        if checkpoint_dict is not None and "optimizer_state_dict" in checkpoint_dict:
            self.alg.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        if checkpoint_dict is not None:
            checkpoint_iter = checkpoint_dict.get("iter")
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
            self.env.cost_d_values_tensor,
        )

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
        self.video_black_tile = torch.zeros(1).new_zeros((self.video_tile_height, self.video_tile_width, 3), dtype=torch.uint8).cpu().numpy()

        self.env.reset()
        if self.record_video:
            self._setup_train_video_camera()

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic