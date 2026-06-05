import importlib
import os
import sys
from copy import deepcopy
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymutil
import isaacgym

from configs import *  # noqa: F401,F403
from global_config import ROOT_DIR
from modules.model_loader import load_actor_critic_checkpoint
from modules.residual_expert_actor_critic import ResidualExpertActorCritic
from runner.residual_policy_runner import ResidualPolicyRunner
from utils.helpers import class_to_dict, update_cfg_from_args
from utils.task_registry import task_registry


def get_residual_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "d1h_disc", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Override experiment name."},
        {"name": "--run_name", "type": str, "help": "Override run name."},
        {"name": "--load_run", "type": str, "help": "Run name to load when resume=True."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint id to load when resume=True."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL algorithm."},
        {"name": "--num_envs", "type": int, "help": "Override number of environments."},
        {"name": "--seed", "type": int, "help": "Override random seed."},
        {"name": "--max_iterations", "type": int, "help": "Override max learning iterations."},
        {"name": "--base_ckpt", "type": str, "default": None, "help": "Checkpoint path for the frozen base policy."},
        {"name": "--residual_alpha", "type": float, "default": 0.3, "help": "Scale factor for the residual expert mean."},
    ]

    args = gymutil.parse_arguments(description="Train residual expert policy.", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def build_actor_critic(module_name, class_name, env, policy_cfg):
    actor_critic_module = importlib.import_module(module_name)
    actor_critic_class = getattr(actor_critic_module, class_name)
    return actor_critic_class(
        env.cfg.env.n_proprio,
        env.cfg.env.n_scan,
        env.num_obs,
        env.cfg.env.n_priv_latent,
        env.cfg.env.history_len,
        env.num_actions,
        **deepcopy(policy_cfg),
    )


def build_log_dir(train_cfg):
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)


def train(args):
    if not args.resume and not args.base_ckpt:
        raise ValueError("Fresh residual training requires --base_ckpt.")

    env, _ = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)

    policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    policy_cfg = train_cfg_dict["policy"]

    base_actor_critic = build_actor_critic("modules", policy_class_name, env, policy_cfg)
    residual_actor_critic = build_actor_critic("modules", policy_class_name, env, policy_cfg)

    if args.base_ckpt:
        load_actor_critic_checkpoint(base_actor_critic, args.base_ckpt, args.rl_device)

    actor_critic = ResidualExpertActorCritic(
        base_actor_critic=base_actor_critic,
        residual_actor_critic=residual_actor_critic,
        alpha=args.residual_alpha,
        freeze_base=True,
    )

    log_dir = build_log_dir(train_cfg)
    print("[train_residual] task           =", args.task)
    print("[train_residual] base_ckpt      =", args.base_ckpt)
    print("[train_residual] residual_alpha =", args.residual_alpha)
    print("[train_residual] log_dir        =", log_dir)

    runner = ResidualPolicyRunner(env, train_cfg_dict, actor_critic, log_dir=log_dir, device=args.rl_device)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    args = get_residual_args()
    train(args)