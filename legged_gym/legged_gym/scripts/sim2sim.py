# import time

# import mujoco.viewer
# import mujoco
# import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import torch
# import yaml

import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, diff_quat

import numpy as np
import torch
# from termcolor import colored

import time

import mujoco.viewer
import mujoco
import yaml
from legged_gym.envs.base.lpf import ActionFilterButterTorch

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def global_obs(base_pos,base_quat):
    # x y yaw
    # xy = base_pos[:,:2] - env_origins[:,:2]
    xy = base_pos[:2]
    qw, qx, qy, qz,  = base_quat[ 0], base_quat[1], base_quat[2], base_quat[3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw = np.array([yaw])  
    return np.concatenate([xy, yaw], axis=-1)
if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/scripts/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config:',config)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]

        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        clip_observations = config["clip_observations"]
        clip_actions = config["clip_actions"]

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        joint_hist_len = config["joint_hist_len"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)

    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    # projected_gravity = torch.tensor([0,0,-1],dtype=torch.float)    

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)
    
    dof_bias = np.zeros(num_actions, dtype=np.float32)
    # q, dq, qtgt hist reset
    qj_start = d.qpos[7:]

    joint_hist = np.zeros((joint_hist_len, num_actions * 3), dtype=np.float32)

    joint_hist[:, 0:num_actions] = (qj_start-dof_bias)
    joint_hist[:, num_actions:num_actions] = 0.
    joint_hist[:, 2*num_actions:3*num_actions] = \
        (qj_start - default_angles -dof_bias) / action_scale
    
    print('joint_hist:',joint_hist)
    
    # init low pass filter
    action_filt = True
    action_cutfreq = 4.0
    torque_effort_scale = 1.0
    ctrl_dt = simulation_dt * control_decimation
    print('ctrl_dt:',ctrl_dt)

    if action_filt:
        action_filter = ActionFilterButterTorch(lowcut=np.zeros(num_actions),
                                                    highcut=np.ones(num_actions) * action_cutfreq, 
                                                    sampling_rate=1./ctrl_dt, num_joints= num_actions, 
                                                    device="cpu")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        viewer.cam.lookat[0] = 0  
        viewer.cam.lookat[1] = 0
        viewer.cam.lookat[2] = 1 
        viewer.cam.distance = 8  
        viewer.cam.elevation = -20  
        viewer.cam.azimuth = 90   
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                lin_vel = d.qvel[0:3]
                base_pos = d.qpos[0:3]


                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale
                lin_vel = lin_vel * lin_vel_scale

                count = counter * simulation_dt
                phase = np.array([0., 1., 0., 1.])
                # phase = np.array([0., 0., 0., 0.])

                # q, dq, joint action hist update
                joint_hist[1:, :] = joint_hist[:-1, :].copy()
                joint_hist[0, 0:num_actions] = qj.copy() - dof_bias
                joint_hist[0, num_actions:2*num_actions] = dqj.copy()
                joint_hist[0, 2*num_actions:3*num_actions] = action.copy()

                print('joint_hist:',joint_hist)
                print('global_obs(base_pos,quat):',global_obs(base_pos,quat))
                
                obs[:19] = (qj - dof_bias) * dof_pos_scale
                obs[19:38] = dqj * dof_vel_scale
                obs[38:41] = omega
                obs[41:44] = lin_vel
                obs[44:47] = gravity_orientation
                obs[47:51] = phase
                obs[51:54] = global_obs(base_pos,quat)
                obs[54:73] = action
                obs[73:130] = joint_hist[1,:]
                obs[130:187] = joint_hist[2,:]
                obs = np.clip(obs, -clip_observations, clip_observations)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # print('obs_tensor:',obs_tensor)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -clip_actions, clip_actions)
                actions_filter = action.copy()
                actions_filter = torch.tensor(actions_filter)
                if action_filt:
                    actions_filter = action_filter.filter(actions_filter.reshape(num_actions)).reshape(1, num_actions)

                # transform action to target_dof_pos
                target_dof_pos = actions_filter * action_scale + default_angles
                print('target_dof_pos:',target_dof_pos)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

