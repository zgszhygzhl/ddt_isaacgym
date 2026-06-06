import copy
import importlib
import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import get_euler_xyz
import isaacgym
import torch
import torch.nn as nn

from configs import *  # noqa: F401,F403
from global_config import ROOT_DIR
from modules.model_loader import load_actor_critic_checkpoint
from modules.residual_expert_actor_critic import ResidualExpertActorCritic
from utils.helpers import class_to_dict, get_load_path
from utils.logger import Logger
from utils.math import wrap_to_pi
from utils.task_registry import task_registry
from utils.video_recorder import FfmpegVideoWriter


class ResidualPolicyExporter(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs):
        with torch.no_grad():
            base_mean = self.actor_critic.base_actor_critic.act_inference(obs)
        residual_mean = self.actor_critic.residual_actor_critic.act_inference(obs)
        return torch.clamp(base_mean + self.actor_critic.alpha * residual_mean, -1.0, 1.0)


def get_residual_play_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "d1h_moe_disc", "help": "Residual task name."},
        {"name": "--base_task", "type": str, "default": "d1h_moe_base", "help": "Task name used to build the frozen base policy."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Unused for play; kept for compatibility."},
        {"name": "--experiment_name", "type": str, "help": "Override experiment name."},
        {"name": "--run_name", "type": str, "help": "Unused for play; kept for compatibility."},
        {"name": "--load_run", "type": str, "default": "-1", "help": "Residual run name to load. Use -1 for the latest run."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Residual checkpoint id to load. Use -1 for the latest checkpoint."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times."},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL policy."},
        {"name": "--num_envs", "type": int, "default": 6, "help": "Override number of environments."},
        {"name": "--seed", "type": int, "help": "Override random seed."},
        {"name": "--max_iterations", "type": int, "help": "Unused for play; kept for compatibility."},
        {"name": "--base_ckpt", "type": str, "default": None, "help": "Optional checkpoint path used to pre-load the frozen base policy before loading the wrapper checkpoint."},
        {"name": "--residual_alpha", "type": float, "default": 0.3, "help": "Scale factor for the residual expert mean."},
        {"name": "--max_steps", "type": int, "default": 2000, "help": "Maximum rollout steps for inference."},
        {"name": "--enable_noise", "action": "store_true", "default": False, "help": "Enable observation noise during inference."},
        {"name": "--enable_domain_rand", "action": "store_true", "default": False, "help": "Enable pushes, disturbances, and domain randomization during inference."},
        {"name": "--disable_record_frames", "action": "store_true", "default": False, "help": "Skip camera video recording."},
        {"name": "--disable_plot_states", "action": "store_true", "default": False, "help": "Skip state logging and plots."},
        {"name": "--disable_export", "action": "store_true", "default": False, "help": "Skip JIT and ONNX export."},
        {"name": "--record_fps", "type": int, "default": 30, "help": "Recorded video FPS."},
    ]

    args = gymutil.parse_arguments(description="Play residual expert policy.", custom_parameters=custom_parameters)
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


def normalize_load_run(load_run):
    if load_run in (None, "", -1, "-1"):
        return -1
    return load_run


def prepare_env_cfg(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs

    if args.seed is not None:
        env_cfg.seed = args.seed

    if not args.enable_noise and hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False

    if not args.enable_domain_rand and hasattr(env_cfg, "domain_rand"):
        for attr_name in [
            "push_robots",
            "disturbance",
            "randomize_friction",
            "randomize_restitution",
            "randomize_base_mass",
            "randomize_base_com",
            "randomize_motor",
            "randomize_kpkd",
            "randomize_lag_timesteps",
        ]:
            if hasattr(env_cfg.domain_rand, attr_name):
                setattr(env_cfg.domain_rand, attr_name, False)

    return env_cfg


def resolve_resume_path(train_cfg, args):
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name

    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return get_load_path(
        log_root,
        load_run=normalize_load_run(args.load_run),
        checkpoint=args.checkpoint,
    )


def export_policy(actor_critic, num_obs, export_dir, stem):
    os.makedirs(export_dir, exist_ok=True)
    exporter = ResidualPolicyExporter(copy.deepcopy(actor_critic).to("cpu").eval())
    dummy_obs = torch.randn(1, num_obs)

    jit_path = os.path.join(export_dir, f"{stem}_residual_wrapper.pt")
    onnx_path = os.path.join(export_dir, f"{stem}_residual_wrapper.onnx")

    traced = torch.jit.trace(exporter, dummy_obs)
    traced.save(jit_path)

    torch.onnx.export(
        exporter,
        dummy_obs,
        onnx_path,
        input_names=["observations"],
        output_names=["actions"],
        verbose=False,
        opset_version=13,
        export_params=True,
    )

    return jit_path, onnx_path


def setup_camera(env):
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    origin = env.env_origins[0].cpu().numpy()

    cam_pos = gymapi.Vec3(float(origin[0] + 3.0), float(origin[1] - 5.0), float(origin[2] + 2.0))
    cam_target = gymapi.Vec3(float(origin[0] + 0.5), float(origin[1]), float(origin[2] + 0.5))
    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)
    return cam_handle, camera_props


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = prepare_env_cfg(env_cfg, args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    _, base_train_cfg = task_registry.get_cfgs(args.base_task)
    train_cfg_dict = class_to_dict(train_cfg)
    base_train_cfg_dict = class_to_dict(base_train_cfg)

    policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    base_policy_class_name = base_train_cfg_dict["runner"]["policy_class_name"]
    policy_cfg = train_cfg_dict["policy"]
    base_policy_cfg = base_train_cfg_dict["policy"]

    base_actor_critic = build_actor_critic("modules", base_policy_class_name, env, base_policy_cfg)
    residual_actor_critic = build_actor_critic("modules", policy_class_name, env, policy_cfg)

    if args.base_ckpt:
        load_actor_critic_checkpoint(base_actor_critic, args.base_ckpt, args.rl_device)

    actor_critic = ResidualExpertActorCritic(
        base_actor_critic=base_actor_critic,
        residual_actor_critic=residual_actor_critic,
        alpha=args.residual_alpha,
        freeze_base=True,
    )

    resume_path = resolve_resume_path(train_cfg, args)
    checkpoint = torch.load(resume_path, map_location=args.rl_device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.to(env.device)
    actor_critic.eval()

    run_dir = os.path.dirname(resume_path)
    checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
    export_dir = os.path.join(run_dir, "play_exports")
    os.makedirs(export_dir, exist_ok=True)

    print("[play_residual] task                =", args.task)
    print("[play_residual] base_task           =", args.base_task)
    print("[play_residual] checkpoint          =", resume_path)
    print("[play_residual] base_ckpt           =", args.base_ckpt)
    print("[play_residual] residual_alpha      =", args.residual_alpha)
    print("[play_residual] num_envs            =", env_cfg.env.num_envs)
    print("[play_residual] enable_noise        =", args.enable_noise)
    print("[play_residual] enable_domain_rand  =", args.enable_domain_rand)

    if not args.disable_export:
        jit_path, onnx_path = export_policy(actor_critic, env.num_obs, export_dir, checkpoint_stem)
        print("[play_residual] jit_export          =", jit_path)
        print("[play_residual] onnx_export         =", onnx_path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    status_every = max(1, int(1.0 / env.dt))
    velocity_plot_path = os.path.join(export_dir, f"{checkpoint_stem}_velocity_tracking.png")

    record_frames = not args.disable_record_frames
    plot_states = not args.disable_plot_states

    video = None
    cam_handle = None
    camera_props = None
    record_every = max(1, int(1.0 / (args.record_fps * env.dt)))

    if record_frames:
        cam_handle, camera_props = setup_camera(env)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        first_img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
        first_img = first_img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        video = FfmpegVideoWriter(
            os.path.join(export_dir, f"{checkpoint_stem}_play.mp4"),
            camera_props.width,
            camera_props.height,
            args.record_fps,
        )
        video.write(first_img)

    with torch.no_grad():
        for step in range(args.max_steps):
            actions = actor_critic.act_inference(obs)
            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)

            _, _, base_heading = get_euler_xyz(env.base_quat[robot_index : robot_index + 1])
            actual_heading = wrap_to_pi(base_heading)[0].item()
            command_heading = env.commands[robot_index, 3].item() if env.commands.shape[1] > 3 else 0.0

            if step % status_every == 0:
                print(
                    f"step={step:05d} "
                    f"cmd[x,y,yaw]=({env.commands[robot_index, 0].item():+.3f}, "
                    f"{env.commands[robot_index, 1].item():+.3f}, "
                    f"{env.commands[robot_index, 2].item():+.3f}) "
                    f"actual[x,y,yaw]=({env.base_lin_vel[robot_index, 0].item():+.3f}, "
                    f"{env.base_lin_vel[robot_index, 1].item():+.3f}, "
                    f"{env.base_ang_vel[robot_index, 2].item():+.3f}) "
                    f"heading(cmd,actual)=({command_heading:+.3f}, {actual_heading:+.3f}) "
                    f"rew={rewards.mean().item():+.4f} cost={costs.mean().item():+.4f} resets={int(dones.sum().item())}"
                )

            if record_frames:
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                if step % record_every == 0:
                    img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
                    img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                    video.write(img)

            if plot_states:
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "command_heading": command_heading,
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "base_heading": actual_heading,
                        "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        "base_height": env._get_base_heights()[robot_index].item(),
                        "command_height": env.cfg.rewards.base_height_target,
                        "torques": env.torques[robot_index, :].tolist(),
                        "velocities": env.dof_vel[robot_index, :].tolist(),
                    }
                )

    if video is not None:
        video.close()

    if plot_states:
        plot_metadata = {
            "joint_names": list(getattr(env, "dof_names", [])),
            "logged_joint_name": getattr(env, "dof_names", [f"joint_{joint_index}"])[joint_index],
        }
        logger.plot_states(save_path=velocity_plot_path, show=False, plot_metadata=plot_metadata)
        print("[play_residual] velocity_plot       =", velocity_plot_path)


if __name__ == "__main__":
    play(get_residual_play_args())