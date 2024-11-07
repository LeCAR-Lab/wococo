import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, diff_quat

import numpy as np
import torch
from termcolor import colored

command_state = {
    'vel_forward': 0.0,
    'vel_side': 0.0,
    'orientation': 0.0,
}

override = False
groupdance = False

def play(args):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.env.num_envs = 1
    if not groupdance: env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    if env_cfg.terrain.mesh_type == 'trimesh':
        env_cfg.terrain.terrain_types = ['flat', 'rough', 'low_obst']  # do not duplicate!
        env_cfg.terrain.terrain_proportions = [0.0, 0.5, 0.5]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 70
    env_cfg.task.num_seq = 100
    # env_cfg.domain_rand.randomize_torque_rfi = True
    # env_cfg.domain_rand.randomize_rfi_lim = False
    # env_cfg.domain_rand.randomize_pd_gain = True
    # env_cfg.domain_rand.randomize_link_mass = True
    # env_cfg.domain_rand.randomize_base_com = True
    # env_cfg.domain_rand.randomize_ctrl_delay = False
    # env_cfg.domain_rand.ctrl_delay_step_range = [1,1]
    env_cfg.rewards.scales.curiosity = 0

    env_cfg.env.test = True
    if args.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard
        from legged_gym.utils import key_response_fn

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if groupdance:
        env.env_origins *= 0
        env.env_origins[0,1] = 10.0
        env.env_origins[0,0] = 10.0
        for i in range(1,4):
            env.env_origins[i,1] = 10.0 + (i-2)*1.5
            env.env_origins[i,0] = 7.0
        for i in range(4,9):
            env.env_origins[i,1] = 10.0 + (i-6)*1.5
            env.env_origins[i,0] = 5.0
        # compose dance
        env.contact_sequence[0] = torch.zeros_like(env.contact_sequence[0])
        env.resample_sequence = lambda x: None
        dance_steps = \
    [[0, 1,0,0],
    [0, 0,0,1],
    [0, 1,0,0],
    [0, 0,0,1]]
        env.contact_sequence[0,:,:] = torch.tensor(dance_steps).to(env.device).repeat(1,25)
        env.period_contact[0] = 0.4

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards


    obs, _ = env.get_observations()

    
    
    # diff_rot_left = torch.abs(diff_quat(self._rigid_body_rot[:, left_elbow_index], self._rigid_body_rot[:, torso_index]))
    # diff_rot_right = torch.abs(diff_quat(self._rigid_body_rot[:, right_elbow_index], self._rigid_body_rot[:, torso_index]))

    
    
    #import ipdb; ipdb.set_trace()
    # obs[:, 9:12] = torch.Tensor([0.5, 0, 0])
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('Loaded policy from: ', task_registry.loaded_policy_path)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, exported_policy_name)
        print('Exported policy as jit script to: ', os.path.join(path, exported_policy_name))

    if args.joystick:
        print(colored("joystick on", "green"))
        key_response = key_response_fn(mode='vel')
        def on_press(key):
            global command_state
            try:
                # print(key.char)
                key_response(key, command_state, env)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    for i in range(10*int(env.max_episode_length)):
        # obs[:, -19:] = 0 # will destroy the performance
        actions = policy(obs.detach()) 
        # actions[:, 11:12] += 10.
        # actions[:, 15:16] += 10.
        # print(actions)
        obs, rews, dones, infos = env.step(actions.detach())
        if groupdance:
            obs[1:,47:51]= obs[0,47:51].clone().unsqueeze(0)
            left = obs[1:4,[47,49]]
            right = obs[1:4,[48,50]]
            obs[1:4,[47,49]] = right.clone()
            obs[1:4,[48,50]] = left.clone()
        
        left_elbow_index = env._body_list.index("left_elbow_link") - 1
        right_elbow_index = env._body_list.index("right_elbow_link") - 1
        assert left_elbow_index >= 0
        assert right_elbow_index >= 0
        # print(env._rigid_body_pos[:, left_elbow_index] - env.base_pos, env._rigid_body_pos[:, right_elbow_index] - env.base_pos)
        # print(env._error_upperbody_rot())
        ## DEBUG
        # for idx in range(19):
        #     print(f"{env.dof_names[idx]}: {obs[:, idx]}")
        # ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        # print(env.dof_names)
        # torso_index = env._body_list.index("torso_link") - 1
        # left_elbow_index = env._body_list.index("left_elbow_link") - 1
        # right_elbow_index = env._body_list.index("right_elbow_link") - 1
        # assert torso_index >= 0
        # assert left_elbow_index >= 0
        # assert right_elbow_index >= 0
        # print("left diff", diff_quat(env._rigid_body_rot[:, torso_index], env._rigid_body_rot[:, left_elbow_index]))
        # print("right diff", diff_quat(env._rigid_body_rot[:, torso_index], env._rigid_body_rot[:, right_elbow_index]))



        # print("obs = ", obs)
        # print("actions = ", actions)
        # print()
        # exit()
        if override: 
            obs[:,9] = 0.5
            obs[:,10] = 0.0
            obs[:,11] = 0.0

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    # 'dof_pos_target': env.actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
