import copy
import importlib
import math
import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import get_euler_xyz
import isaacgym
import numpy as np
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


class ResidualFullObsExporter(nn.Module):
    """
    Debug exporter:
        full_obs -> base.act_inference(full_obs) + alpha * residual.act_inference(full_obs)

    这个导出的 ONNX 会是 [1, env.num_obs]，例如 616 维。
    它适合调试，不适合直接替换 DDT 部署端 ONNX。
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs):
        base_mean = self.actor_critic.base_actor_critic.act_inference(obs)
        residual_mean = self.actor_critic.residual_actor_critic.act_inference(obs)
        return torch.clamp(base_mean + self.actor_critic.alpha * residual_mean, -1.0, 1.0)


class ResidualDeployExporter(nn.Module):
    """
    Deploy-style exporter:
        obs_actor, obs_hist_actor -> base_actor + alpha * residual_actor

    这个接口和未来 MoE 更接近。
    未来门控网络应该在这个层级融合多个 residual expert：
        final = base + sum_i w_i * alpha_i * expert_i
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.base_actor = actor_critic.base_actor_critic.actor_teacher_backbone
        self.residual_actor = actor_critic.residual_actor_critic.actor_teacher_backbone
        self.alpha = actor_critic.alpha

    def forward(self, obs_actor, obs_hist_actor):
        base_mean = self.base_actor(obs_actor, obs_hist_actor)
        residual_mean = self.residual_actor(obs_actor, obs_hist_actor)
        return torch.clamp(base_mean + self.alpha * residual_mean, -1.0, 1.0)


def get_residual_play_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "d1h_moe_disc", "help": "Residual task name."},
        {"name": "--base_task", "type": str, "default": "d1h_moe_base", "help": "Task name used to build the frozen base policy."},

        {"name": "--resume", "action": "store_true", "default": False, "help": "Unused for play; kept for compatibility."},
        {"name": "--experiment_name", "type": str, "help": "Override experiment name."},
        {"name": "--run_name", "type": str, "help": "Unused for play; kept for compatibility."},
        {"name": "--load_run", "type": str, "default": "-1", "help": "Residual run name to load. Use -1 for the latest run."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Residual checkpoint id to load. Use -1 for latest checkpoint."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times."},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL policy."},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Override number of environments."},
        {"name": "--seed", "type": int, "help": "Override random seed."},
        {"name": "--max_iterations", "type": int, "help": "Unused for play; kept for compatibility."},

        {"name": "--base_ckpt", "type": str, "default": None, "help": "Optional base checkpoint. Usually unnecessary if loading a full residual wrapper checkpoint."},
        {"name": "--residual_alpha", "type": float, "default": 0.3, "help": "Scale factor for the residual expert mean."},
        {"name": "--max_steps", "type": int, "default": 2000, "help": "Maximum rollout steps for inference."},

        # Fixed command override for play.
        # 默认 None 表示不覆盖原配置。
        {"name": "--cmd_x", "type": float, "default": None, "help": "Fixed target linear velocity x during inference."},
        {"name": "--cmd_y", "type": float, "default": None, "help": "Fixed target linear velocity y during inference. If any command override is used and this is not set, it defaults to 0.0."},
        {"name": "--cmd_yaw", "type": float, "default": None, "help": "Fixed target yaw rate during inference. Used when --command_mode yaw. If any command override is used and this is not set, it defaults to 0.0."},
        {"name": "--cmd_heading", "type": float, "default": None, "help": "Fixed target heading during inference. Used when --command_mode heading. If not set, defaults to 0.0."},
        {"name": "--command_mode", "type": str, "default": "yaw", "help": "Command mode for play override: yaw or heading."},
        {"name": "--cmd_resampling_time", "type": float, "default": 1e9, "help": "Command resampling time during inference when command override is used."},

        {"name": "--enable_noise", "action": "store_true", "default": False, "help": "Enable observation noise during inference."},
        {"name": "--enable_domain_rand", "action": "store_true", "default": False, "help": "Enable pushes, disturbances, and domain randomization during inference."},

        {"name": "--disable_record_frames", "action": "store_true", "default": False, "help": "Skip camera video recording."},
        {"name": "--disable_plot_states", "action": "store_true", "default": False, "help": "Skip state logging and plots."},

        # Export: default none，避免每次都导出 616 维 debug ONNX 造成误解。
        # 可选: none / full_obs / deploy / both
        {"name": "--export_mode", "type": str, "default": "none", "help": "Export mode: none, full_obs, deploy, both."},
        {"name": "--disable_export", "action": "store_true", "default": False, "help": "Compatibility option. If set, skip export."},

        # Mosaic recording
        {"name": "--record_fps", "type": int, "default": 30, "help": "Recorded video FPS."},
        {"name": "--record_num_envs", "type": int, "default": 4, "help": "Number of envs to record into a mosaic video."},
        {"name": "--record_env_ids", "type": str, "default": "auto", "help": "Comma-separated env ids, e.g. 0,1,2,3. Use auto to select automatically."},
        {"name": "--mosaic_cols", "type": int, "default": 2, "help": "Number of columns in the mosaic video."},
        {"name": "--camera_width", "type": int, "default": 640, "help": "Single-camera width."},
        {"name": "--camera_height", "type": int, "default": 360, "help": "Single-camera height."},
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


def parse_record_env_ids(record_env_ids):
    if record_env_ids is None:
        return None

    text = str(record_env_ids).strip().lower()
    if text in ("", "auto", "-1"):
        return None

    env_ids = []
    for part in text.split(","):
        part = part.strip()
        if part:
            env_ids.append(int(part))
    return env_ids


def has_command_override(args):
    return (
        args.cmd_x is not None
        or args.cmd_y is not None
        or args.cmd_yaw is not None
        or args.cmd_heading is not None
    )


def get_command_mode(args):
    command_mode = str(args.command_mode).strip().lower()
    if command_mode not in ("yaw", "heading"):
        raise ValueError(f"--command_mode must be 'yaw' or 'heading', got: {args.command_mode}")
    return command_mode


def apply_fixed_command_cfg(env_cfg, args):
    """
    在创建环境前固定 command range。
    这样 env.reset() / resample_commands() 采样到的就是固定目标速度，
    策略观测里的 command 和真实 env.commands 保持一致。
    """
    if not has_command_override(args):
        return env_cfg

    command_mode = get_command_mode(args)

    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "ranges"):
        raise AttributeError("env_cfg.commands.ranges is required for command override.")

    # 推理时关闭命令 curriculum、zero command 和启动冻结，避免你设置了速度但环境又改回 0。
    if hasattr(env_cfg.commands, "curriculum"):
        env_cfg.commands.curriculum = False

    if hasattr(env_cfg.commands, "zero_command_ratio"):
        env_cfg.commands.zero_command_ratio = 0.0

    if hasattr(env_cfg.commands, "startup_freeze_time"):
        env_cfg.commands.startup_freeze_time = 0.0

    if hasattr(env_cfg.commands, "resampling_time"):
        env_cfg.commands.resampling_time = args.cmd_resampling_time

    # 如果用户只设置了 cmd_x，则默认 y/yaw 为 0，避免还随机横移或随机转向。
    if args.cmd_x is not None:
        env_cfg.commands.ranges.lin_vel_x = [args.cmd_x, args.cmd_x]

    cmd_y = args.cmd_y
    if cmd_y is None:
        cmd_y = 0.0
    env_cfg.commands.ranges.lin_vel_y = [cmd_y, cmd_y]

    if command_mode == "yaw":
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = False

        cmd_yaw = args.cmd_yaw
        if cmd_yaw is None:
            cmd_yaw = 0.0
        env_cfg.commands.ranges.ang_vel_yaw = [cmd_yaw, cmd_yaw]

    else:
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = True

        cmd_heading = args.cmd_heading
        if cmd_heading is None:
            cmd_heading = 0.0
        env_cfg.commands.ranges.heading = [cmd_heading, cmd_heading]

    return env_cfg


def force_env_commands(env, args):
    """
    在 env.reset() 后立即把 env.commands 改成固定值。
    这一步主要保证初始 obs 之前 command 已经正确。
    后续 rollout 中因为 command ranges 已经固定，通常不需要每步强行覆盖。
    """
    if not has_command_override(args):
        return

    if not hasattr(env, "commands"):
        return

    command_mode = get_command_mode(args)
    device = env.commands.device

    if args.cmd_x is not None:
        env.commands[:, 0] = float(args.cmd_x)

    cmd_y = args.cmd_y
    if cmd_y is None:
        cmd_y = 0.0
    if env.commands.shape[1] > 1:
        env.commands[:, 1] = float(cmd_y)

    if command_mode == "yaw":
        cmd_yaw = args.cmd_yaw
        if cmd_yaw is None:
            cmd_yaw = 0.0
        if env.commands.shape[1] > 2:
            env.commands[:, 2] = float(cmd_yaw)

        if env.commands.shape[1] > 3:
            # yaw-rate 模式下 heading 不作为控制目标；这里置 0 只是为了日志清楚。
            env.commands[:, 3] = 0.0

    else:
        cmd_heading = args.cmd_heading
        if cmd_heading is None:
            cmd_heading = 0.0

        if env.commands.shape[1] > 3:
            env.commands[:, 3] = float(cmd_heading)

        # heading 模式下，commands[:, 2] 通常由环境根据 heading error 生成。
        # 这里手动初始化一次，保证第一帧 obs 不会用到旧 yaw command。
        if env.commands.shape[1] > 2 and hasattr(env, "base_quat"):
            _, _, heading = get_euler_xyz(env.base_quat)
            heading = wrap_to_pi(heading)

            target_heading = torch.ones_like(heading, device=device) * float(cmd_heading)

            heading_kp = 1.0
            max_yaw_rate = 1.0
            if hasattr(env.cfg, "commands"):
                heading_kp = getattr(env.cfg.commands, "heading_kp", heading_kp)
                max_yaw_rate = getattr(env.cfg.commands, "max_yaw_rate", max_yaw_rate)

            env.commands[:, 2] = torch.clamp(
                heading_kp * wrap_to_pi(target_heading - heading),
                -max_yaw_rate,
                max_yaw_rate,
            )


def prepare_env_cfg(env_cfg, args):
    explicit_env_ids = parse_record_env_ids(args.record_env_ids)

    num_envs = args.num_envs
    if not args.disable_record_frames:
        num_envs = max(num_envs, args.record_num_envs)
        if explicit_env_ids:
            num_envs = max(num_envs, max(explicit_env_ids) + 1)

    env_cfg.env.num_envs = num_envs

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

    env_cfg = apply_fixed_command_cfg(env_cfg, args)

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


def export_policy(actor_critic, num_obs, export_dir, stem, export_mode):
    export_mode = "none" if export_mode is None else str(export_mode).lower()
    if export_mode not in ("none", "full_obs", "deploy", "both"):
        raise ValueError(f"Unknown export_mode: {export_mode}. Use none, full_obs, deploy, or both.")

    paths = {}
    if export_mode == "none":
        return paths

    os.makedirs(export_dir, exist_ok=True)

    if export_mode in ("full_obs", "both"):
        full_exporter = ResidualFullObsExporter(copy.deepcopy(actor_critic).to("cpu").eval())
        dummy_obs = torch.randn(1, num_obs)

        full_jit_path = os.path.join(export_dir, f"{stem}_residual_full_obs_wrapper.pt")
        full_onnx_path = os.path.join(export_dir, f"{stem}_residual_full_obs_wrapper.onnx")

        traced = torch.jit.trace(full_exporter, dummy_obs)
        traced.save(full_jit_path)

        torch.onnx.export(
            full_exporter,
            dummy_obs,
            full_onnx_path,
            input_names=["observations"],
            output_names=["actions"],
            verbose=False,
            opset_version=13,
            export_params=True,
        )

        paths["full_obs_jit"] = full_jit_path
        paths["full_obs_onnx"] = full_onnx_path

    if export_mode in ("deploy", "both"):
        deploy_exporter = ResidualDeployExporter(copy.deepcopy(actor_critic).to("cpu").eval())

        obs_dim = actor_critic.base_actor_critic.num_prop - 3
        hist_len = actor_critic.base_actor_critic.num_hist

        dummy_obs_actor = torch.randn(1, obs_dim)
        dummy_obs_hist = torch.randn(1, hist_len, obs_dim)

        deploy_jit_path = os.path.join(export_dir, f"{stem}_residual_deploy.pt")
        deploy_onnx_path = os.path.join(export_dir, f"{stem}_residual_deploy.onnx")

        traced = torch.jit.trace(deploy_exporter, (dummy_obs_actor, dummy_obs_hist))
        traced.save(deploy_jit_path)

        torch.onnx.export(
            deploy_exporter,
            (dummy_obs_actor, dummy_obs_hist),
            deploy_onnx_path,
            input_names=["nn_input0", "nn_input1"],
            output_names=["nn_output"],
            verbose=False,
            opset_version=13,
            export_params=True,
        )

        paths["deploy_jit"] = deploy_jit_path
        paths["deploy_onnx"] = deploy_onnx_path

    return paths


def select_record_env_ids(env, args):
    explicit_ids = parse_record_env_ids(args.record_env_ids)
    if explicit_ids:
        env_ids = [i for i in explicit_ids if 0 <= i < env.num_envs]
        if len(env_ids) == 0:
            raise ValueError(f"No valid env ids in --record_env_ids={args.record_env_ids}")
        return env_ids

    target_n = min(args.record_num_envs, env.num_envs)

    # 尽量选择不同 terrain type 的环境；如果没有 terrain_types 属性，则退化成 0,1,2,3。
    selected = []
    used_types = set()

    terrain_types = getattr(env, "terrain_types", None)
    terrain_levels = getattr(env, "terrain_levels", None)

    if terrain_types is not None:
        for i in range(env.num_envs):
            try:
                t = int(terrain_types[i].item())
            except Exception:
                t = int(terrain_types[i])

            if t not in used_types:
                selected.append(i)
                used_types.add(t)

            if len(selected) >= target_n:
                break

    if len(selected) < target_n:
        for i in range(env.num_envs):
            if i not in selected:
                selected.append(i)
            if len(selected) >= target_n:
                break

    # 打印记录环境的地形信息，方便你知道四格视频分别是什么。
    labels = []
    for i in selected:
        label = f"env{i}"
        if terrain_types is not None:
            try:
                label += f"/type{int(terrain_types[i].item())}"
            except Exception:
                label += f"/type{terrain_types[i]}"
        if terrain_levels is not None:
            try:
                label += f"/level{int(terrain_levels[i].item())}"
            except Exception:
                label += f"/level{terrain_levels[i]}"
        labels.append(label)

    print("[play_residual] record_envs         =", ", ".join(labels))
    return selected


def setup_cameras(env, env_ids, args):
    camera_props = gymapi.CameraProperties()
    camera_props.width = args.camera_width
    camera_props.height = args.camera_height

    cameras = []
    for env_id in env_ids:
        cam_handle = env.gym.create_camera_sensor(env.envs[env_id], camera_props)
        origin = env.env_origins[env_id].cpu().numpy()

        cam_pos = gymapi.Vec3(
            float(origin[0] + 3.0),
            float(origin[1] - 5.0),
            float(origin[2] + 2.0),
        )
        cam_target = gymapi.Vec3(
            float(origin[0] + 0.5),
            float(origin[1]),
            float(origin[2] + 0.5),
        )
        env.gym.set_camera_location(cam_handle, env.envs[env_id], cam_pos, cam_target)
        cameras.append((env_id, cam_handle))

    return cameras, camera_props


def make_mosaic(frames, mosaic_cols):
    if len(frames) == 0:
        raise ValueError("No frames to mosaic.")

    h, w, c = frames[0].shape
    cols = max(1, int(mosaic_cols))
    rows = int(math.ceil(len(frames) / cols))

    mosaic = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)

    for idx, frame in enumerate(frames):
        r = idx // cols
        col = idx % cols
        mosaic[r * h:(r + 1) * h, col * w:(col + 1) * w, :] = frame

    return mosaic


def capture_mosaic_frame(env, cameras, camera_props, mosaic_cols):
    frames = []
    for env_id, cam_handle in cameras:
        img = env.gym.get_camera_image(
            env.sim,
            env.envs[env_id],
            cam_handle,
            gymapi.IMAGE_COLOR,
        )
        img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        frames.append(img.copy())

    return make_mosaic(frames, mosaic_cols)


def log_action_decomposition(logger_data, actor_critic, robot_index):
    if hasattr(actor_critic, "last_base_mean"):
        logger_data["base_action_0"] = actor_critic.last_base_mean[robot_index, 0].item()
    if hasattr(actor_critic, "last_residual_mean"):
        logger_data["residual_action_0"] = actor_critic.last_residual_mean[robot_index, 0].item()
    if hasattr(actor_critic, "last_final_mean"):
        logger_data["final_action_0"] = actor_critic.last_final_mean[robot_index, 0].item()


def print_command_config(env_cfg, args):
    print("[play_residual] command_override    =", has_command_override(args))
    print("[play_residual] command_mode        =", args.command_mode)

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ranges"):
        print("[play_residual] cmd range x         =", env_cfg.commands.ranges.lin_vel_x)
        print("[play_residual] cmd range y         =", env_cfg.commands.ranges.lin_vel_y)
        print("[play_residual] cmd range yaw       =", env_cfg.commands.ranges.ang_vel_yaw)
        if hasattr(env_cfg.commands.ranges, "heading"):
            print("[play_residual] cmd range heading   =", env_cfg.commands.ranges.heading)

    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "heading_command"):
            print("[play_residual] heading_command     =", env_cfg.commands.heading_command)
        if hasattr(env_cfg.commands, "resampling_time"):
            print("[play_residual] resampling_time     =", env_cfg.commands.resampling_time)
        if hasattr(env_cfg.commands, "zero_command_ratio"):
            print("[play_residual] zero_command_ratio  =", env_cfg.commands.zero_command_ratio)
        if hasattr(env_cfg.commands, "startup_freeze_time"):
            print("[play_residual] startup_freeze_time =", env_cfg.commands.startup_freeze_time)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = prepare_env_cfg(env_cfg, args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.reset()
    force_env_commands(env, args)
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

    missing_keys, unexpected_keys = actor_critic.load_state_dict(state_dict, strict=False)
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
    print("[play_residual] env.num_obs         =", env.num_obs)
    print("[play_residual] n_proprio           =", env.cfg.env.n_proprio)
    print("[play_residual] actor_obs_dim       =", actor_critic.base_actor_critic.num_prop - 3)
    print("[play_residual] history_len         =", actor_critic.base_actor_critic.num_hist)
    print_command_config(env_cfg, args)

    if len(missing_keys) > 0:
        print("[play_residual] missing_keys        =", missing_keys)
    if len(unexpected_keys) > 0:
        print("[play_residual] unexpected_keys     =", unexpected_keys)

    if (not args.disable_export) and args.export_mode != "none":
        export_paths = export_policy(actor_critic, env.num_obs, export_dir, checkpoint_stem, args.export_mode)
        for name, path in export_paths.items():
            print(f"[play_residual] {name:<20} = {path}")

    record_frames = not args.disable_record_frames
    plot_states = not args.disable_plot_states

    record_env_ids = select_record_env_ids(env, args) if record_frames else [0]
    robot_index = record_env_ids[0] if len(record_env_ids) > 0 else 0

    logger = Logger(env.dt)
    joint_index = 1
    status_every = max(1, int(1.0 / env.dt))
    velocity_plot_path = os.path.join(export_dir, f"{checkpoint_stem}_velocity_tracking.png")

    video = None
    cameras = None
    camera_props = None
    record_every = max(1, int(1.0 / (args.record_fps * env.dt)))

    if record_frames:
        cameras, camera_props = setup_cameras(env, record_env_ids, args)

        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        first_mosaic = capture_mosaic_frame(env, cameras, camera_props, args.mosaic_cols)
        video_path = os.path.join(export_dir, f"{checkpoint_stem}_play_mosaic.mp4")

        video = FfmpegVideoWriter(
            video_path,
            first_mosaic.shape[1],
            first_mosaic.shape[0],
            args.record_fps,
        )
        video.write(first_mosaic)
        print("[play_residual] video               =", video_path)

    with torch.no_grad():
        for step in range(args.max_steps):
            actions = actor_critic.act_inference(obs)
            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)

            _, _, base_heading = get_euler_xyz(env.base_quat[robot_index: robot_index + 1])
            actual_heading = wrap_to_pi(base_heading)[0].item()
            command_heading = env.commands[robot_index, 3].item() if env.commands.shape[1] > 3 else 0.0

            if step % status_every == 0:
                print(
                    f"step={step:05d} "
                    f"env={robot_index} "
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
                    mosaic = capture_mosaic_frame(env, cameras, camera_props, args.mosaic_cols)
                    video.write(mosaic)

            if plot_states:
                logger_data = {
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
                log_action_decomposition(logger_data, actor_critic, robot_index)
                logger.log_states(logger_data)

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