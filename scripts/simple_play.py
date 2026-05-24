import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import *
from isaacgym import gymapi
from modules import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR
from utils.video_recorder import FfmpegVideoWriter

from PIL import Image as im

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 6
    # env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0]
    # env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    # #env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.domain_rand.push_robots = False
    # #env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_base_com = False
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_motor = False
    # env_cfg.domain_rand.randomize_lag_timesteps = False
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.control.use_filter = True
    # env_cfg.domain_rand.disturbance = False
    # env_cfg.domain_rand.randomize_kpkd = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 强制 reset 一次，让 root state 和 dof state 都按 cfg 初始化
    env.reset()
    obs = env.get_observations()
    # load policy partial_checkpoint_load
    # train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    # print("policy", policy)
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticBarlowTwins = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    # print("asdasdsa", policy)

    if args.load_run is not None:
      train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
      train_cfg.runner.checkpoint = args.checkpoint
    log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)

    model_dict = torch.load(resume_path)
    print("resume_path", resume_path)
    # export the policy
    policy.load_state_dict(model_dict['model_state_dict'])
    # policy.half()
    policy.eval()
    policy = policy.to(env.device)
    policy.save_torch_jit_policy('model.pt',env.device)

    # clear images under frames folder
    # frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    # delete_files_in_directory(frames_path)
    # logger for plot
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    start_state_log = 1000 # number of steps before plotting states

    stop_state_log = 2000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards


    # set fixed world/environment camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)

    origin = env.env_origins[0].cpu().numpy()

    cam_pos = gymapi.Vec3(
      float(origin[0] + 3.0),
      float(origin[1] - 5.0),
      float(origin[2] + 2.0),
    )

    cam_target = gymapi.Vec3(
      float(origin[0] + 0.5),
      float(origin[1] + 0.0),
      float(origin[2] + 0.5),
    )

    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)

    img_idx = 0
    video_duration = 10
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')

    video = None
    record_fps = 30
    record_every = max(1, int(1.0 / (record_fps * env.dt)))

    if RECORD_FRAMES:
      env.gym.step_graphics(env.sim)
      env.gym.render_all_camera_sensors(env.sim)
      img = env.gym.get_camera_image(
        env.sim,
        env.envs[0],
        cam_handle,
        gymapi.IMAGE_COLOR,
      ).reshape((camera_props.height, camera_props.width, 4))[:, :, :3]

      video = FfmpegVideoWriter(
        os.path.join(ROOT_DIR, 'record_h264.mp4'),
        camera_props.width,
        camera_props.height,
        record_fps,
      )
      video.write(img)
      img_idx += 1

    for i in range(num_frames):
        # env.commands[:,0] = 1.0
        # env.commands[:,1] = 0
        # env.commands[:,2] = 0
        # env.commands[:,3] = 0
        actions = policy.act_teacher(obs.half())
        # actions = torch.clamp(actions,-1.2,1.2)
        # actions = policy(obs.detach())
        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)
        if RECORD_FRAMES and i % record_every == 0:
          img = env.gym.get_camera_image(
            env.sim,
            env.envs[0],
            cam_handle,
            gymapi.IMAGE_COLOR,
          ).reshape((camera_props.height, camera_props.width, 4))[:, :, :3]

          video.write(img)
          img_idx += 1
        if PLOT_STATES:
            if i < stop_state_log and i > start_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        'base_height': env._get_base_heights()[robot_index].item(),
                        'command_height': env.cfg.rewards.base_height_target,
                        'torques': env.torques[robot_index, :].tolist(),
                        'velocities': env.dof_vel[robot_index, :].tolist(),
                    }
                )
            elif i==stop_state_log:
                logger.plot_states()
            # if  0 < i < stop_rew_log:
            #     if infos["episode"]:
            #         num_episodes = torch.sum(env.reset_buf).item()
            #         if num_episodes>0:
            #             logger.log_rewards(infos["episode"], num_episodes)
            # elif i==stop_rew_log:
            #     logger.print_rewards()

    if video is not None:
      video.close()

    #test model profile
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    #      for i in range(1000):
    #         with torch.no_grad():
    #           actions = policy.act_teacher(obs.half())
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

if __name__ == '__main__':
    RECORD_FRAMES = True
    PLOT_STATES = True
    args = get_args()
    play(args)
