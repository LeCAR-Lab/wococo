from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
from typing import Optional, Any
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from termcolor import colored
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi, yaw_quat
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch
from legged_gym.utils.visualization import SingleLine

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.viewer.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        # init low pass filter
        if self.cfg.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_envs*self.num_actions),
                                                        highcut=np.ones(self.num_envs*self.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_envs * self.num_actions, 
                                                        device=self.device)
            
        self.num_obs_noise = self.cfg.env.num_noise # obs_dim for all obs
        assert self.cfg.env.num_observations == self.num_obs_noise

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        actions = self.actions.clone()
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()

        if self.cfg.control.action_filt:
            # # older version
            # actions_np = self.actions.cpu().numpy().reshape(self.num_envs * self.num_actions)
            # actions_np = self.action_filter.filter_old(actions_np)
            # actions_torch_old = torch.tensor(actions_np.reshape(self.num_envs, self.num_actions),
            #                             dtype=self.actions.dtype, device=self.device, requires_grad=self.actions.requires_grad)
            
            # torch version
            actions = self.action_filter.filter(actions.reshape(self.num_envs * self.num_actions)).reshape(self.num_envs, self.num_actions)

        else:
            pass
            # actions = actions.clone()

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            self.extras["observations"]["critic"] = self.privileged_obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        # q, dq, joint action hist update
        self.joint_hist[:, 1:, :] = self.joint_hist[:, :-1, :].clone()
        self.joint_hist[:, 0, 0:self.num_dof] = self.dof_pos.clone() - self.dof_bias
        self.joint_hist[:, 0, self.num_dof:2*self.num_dof] = self.dof_vel.clone()
        self.joint_hist[:, 0, 2*self.num_dof:3*self.num_dof] = self.actions.clone()
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            
    def check_termination(self):
        """ Check if environments need to be reset
        """ 
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        #print("Terminated by contact: ", torch.sum(self.reset_buf).item())
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)

        
        # Termination for knee distance too close
        if self.cfg.asset.terminate_by_knee_distance and self.knee_distance.shape:
            # print("terminate_by knee_distance")
            self.reset_buf |= torch.any(self.knee_distance < self.cfg.asset.termination_scales.min_knee_distance, dim=1)
            #print("Terminated by knee distance: ", torch.sum(self.reset_buf).item())
        
        if self.cfg.asset.terminate_by_feet_distance and self.feet_distance.shape:
            self.reset_buf |= torch.any(self.feet_distance < self.cfg.asset.termination_scales.min_feet_distance, dim=1)
                    
        # Termination for velocities
        if self.cfg.asset.terminate_by_lin_vel:
            # print("terminate_by lin_vel")
            self.reset_buf |= torch.any(torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > self.cfg.asset.termination_scales.base_vel, dim=1)
            #print("Terminated by lin vel: ", torch.sum(self.reset_buf).item())
        # print(self.reset_buf)

        # Termination for angular velocities
        if self.cfg.asset.terminate_by_ang_vel:
            # print("terminate_by ang_vel")
            #print(self.base_ang_vel)
            #print(torch.norm(self.base_ang_vel, dim=-1, keepdim=True))
            self.reset_buf |= torch.any(torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > self.cfg.asset.termination_scales.base_ang_vel, dim=1)
            #print("Terminated by ang vel: ", torch.sum(self.reset_buf).item())
        # print(self.reset_buf)
        
        # Termination for gravity in x-direction
        if self.cfg.asset.terminate_by_gravity:
            # print("terminate_by gravity")
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.cfg.asset.termination_scales.gravity_x, dim=1)
            
            # Termination for gravity in y-direction
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.cfg.asset.termination_scales.gravity_y, dim=1)
            
            # print(self.reset_buf)
            #print("Terminated by gravity: ", torch.sum(self.reset_buf).item())
        
        # Termination for low height
        if self.cfg.asset.terminate_by_low_height:
            # print("terminate_by low_height")
            self.reset_buf |= torch.any(self.root_states[:, 2:3] < self.cfg.asset.termination_scales.base_height, dim=1)
            #print("Terminated by low height: ", torch.sum(self.reset_buf).item())

        # Termination by global y coordination
        if self.cfg.asset.terminate_by_xy:
            self.reset_buf |= torch.abs(self.root_states[:,0]-self.env_origins[:,0]) > self.cfg.asset.termination_scales.global_xy
            self.reset_buf |= torch.abs(self.root_states[:,1]-self.env_origins[:,1]) > self.cfg.asset.termination_scales.global_xy
            
        # Termination for close hip
        if self.cfg.asset.terminate_by_hip_yaw:
            self.reset_buf |= torch.abs(self.dof_pos[:, 0]- self.dof_pos[:, 5]) > self.cfg.asset.termination_scales.hip_yaw_sum
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        #print("Terminated by time out: ", torch.sum(self.time_out_buf).item())
        

        #print()
        self.reset_buf |= self.time_out_buf
        

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._episodic_domain_randomization(env_ids)
        if self.cfg.control.action_filt:
            # # older version
            # filter_action_ids = np.concatenate([np.arange(self.num_actions) + env_id.cpu().numpy() * self.num_actions for env_id in env_ids])
            # self.action_filter.reset_by_ids(filter_action_ids)
            # torch version
            filter_action_ids_torch = torch.concat([torch.arange(self.num_actions,dtype=torch.int64, device=self.device) + env_id * self.num_actions for env_id in env_ids])
            self.action_filter.reset_hist(filter_action_ids_torch)

        self._resample_commands(env_ids)
        
        # reset buffers
        if self.cfg.domain_rand.randomize_dof_bias:
            self.dof_bias[env_ids] = self.dof_bias[env_ids].uniform_(-self.cfg.domain_rand.max_dof_bias, self.cfg.domain_rand.max_dof_bias)
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # q, dq, qtgt hist reset
        self.joint_hist[env_ids, :, 0:self.num_dof] = (self.dof_pos[env_ids]-self.dof_bias[env_ids]).unsqueeze(1)
        self.joint_hist[env_ids, :, self.num_dof:2*self.num_dof] = 0.
        self.joint_hist[env_ids, :, 2*self.num_dof:3*self.num_dof] = \
            (self.dof_pos[env_ids] - self.default_dof_pos - self.dof_bias[env_ids]).unsqueeze(1) / self.cfg.control.action_scale
        
        self.extras['cost'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0], 
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        # if self.cfg.terrain.mesh_type == 'box':
        #     self.feet_box_indices[env_ids] = torch.randint(0, self.cfg.terrain.box.num_box, (len(env_ids), 2), device=self.device, requires_grad=False)

        # print(self.feet_box_indices)
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  
                                    # self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos - self.dof_bias) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
            
        # obs_buf_denoise = self.obs_buf.clone()
        
        # import pdb; pdb.set_trace()
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.env.num_privileged_obs is not None:
            # privileged obs
            self.privileged_info = torch.cat([
                self._base_com_bias,
                self._ground_friction_values[:, self.feet_indices],
                self._link_mass_scale,
                self._kp_scale,
                self._kd_scale,
                self._rfi_lim_scale,
                self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, 6),
            ], dim=1)
            self.privileged_obs_buf = torch.cat([obs_buf_denoise, self.privileged_info], dim=1)
    
    def symmetric_fn(self, batch_obs, batch_critic_obs, batch_actions):
        raise NotImplementedError
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        self.terrain = None
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
            # self._create_ground_plane()
        elif mesh_type is not None:
            raise ValueError(mesh_type,"Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        import pdb; pdb.set_trace()
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """   
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
            
    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(self._ground_friction_values[0])):
                props[s].friction = self.friction_coeffs[env_id]
                # import pdb; pdb.set_trace()
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.control.torque_effort_scale
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (before randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (before randomization)")
        # print()
        
        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            torso_index = self._body_list.index("torso_link")
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.x[0], self.cfg.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.y[0], self.cfg.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.z[0], self.cfg.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i, body_name in enumerate(self.cfg.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.cfg.domain_rand.link_mass_range[0], self.cfg.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            raise Exception("index 0 is for world, 13 is for torso!")
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (after randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (afters randomization)")
        # print()

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale + self.dof_bias
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self._kd_scale * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.dof_vel) - self._kd_scale * self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.cfg.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.cfg.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.4, 0.4, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids,12] += 1.2
        self.dof_pos[env_ids,16] -= 1.2
        # if self.cfg.init_state.randomize_upperbody:
        #     self.dof_pos[env_ids, 11:] += torch_rand_float(-2., 2., (len(env_ids), self.num_dof-11), device=self.device)
            
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # print("after reset dof"); import pdb; pdb.set_trace()
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # trimesh
            # import pdb; pdb.set_trace()
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-.3, .3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            # import pdb; pdb.set_trace()
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            
        if self.cfg.domain_rand.randomize_yaw:  # yaw
            _yaw = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_yaw_range[0], self.cfg.domain_rand.init_yaw_range[1])
        else:
            _yaw = torch.zeros_like(self.root_states[env_ids, 3])
        roll = torch.zeros_like(self.root_states[env_ids, 3])
        pitch = torch.zeros_like(self.root_states[env_ids, 3])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, _yaw)
            
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states_all),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel xy delta
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up

        #import ipdb; ipdb.set_trace()
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[:,:self.cfg.env.num_noise])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.                                             # commands
        noise_vec[9                       :   9+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+  self.num_actions    :   9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions    :                       ] = 0. # previous actions
        return noise_vec

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        
        if self.cfg.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.rfi_lim_range[0], self.cfg.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_actions), device=self.device)
        # print(self._kp_scale[env_ids[0]])
    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        

        # create some wrapper tensors for different slices
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states_all = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.root_states_all
            
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # init rigid body state
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 10:13]
        
        # initialize some data used later on
        self.common_step_counter = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # normalization
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        self.dof_bias = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        if self.cfg.domain_rand.randomize_dof_bias:
            self.dof_bias.uniform_(-self.cfg.domain_rand.max_dof_bias, self.cfg.domain_rand.max_dof_bias)
        
        self.joint_hist = torch.zeros(self.num_envs, self.cfg.env.joint_hist_len, self.num_dof * 3, device=self.device) # q, dq, joint action

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0], 
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)  

        # init buffer that used for assigning box for each feet
        # self.feet_box_indices = torch.randint(0, self.cfg.terrain.box.num_box, (self.num_envs, 2), device=self.device, requires_grad=False)        
            
    def _init_domain_params(self):
        # init params for domain randomization
        # init 0 for values
        # init 1 for scales
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_friction_values = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)        
        self._link_mass_scale = torch.ones(self.num_envs, len(self.cfg.domain_rand.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    
    
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # init env domain params
        self._init_domain_params()

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # import pdb; pdb.set_trace()
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.marker_handles = []
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self._body_list = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.hand_indices = torch.zeros(2, dtype=torch.long, device=self.device, requires_grad=False)
        self.hand_indices[0] = self._body_list.index("left_hand")
        self.hand_indices[1] = self._body_list.index("right_hand")

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        print("terminate by", termination_contact_names)
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
        elif self.cfg.terrain.mesh_type in ["box"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # box info
            # self.box_positions = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            # self.box_dims = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # self.terrain_box_positions = torch.from_numpy(self.terrain.box_positions).to(self.device).to(torch.float)
            # self.terrain_box_dims = torch.from_numpy(self.terrain.box_dims).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
            # self.box_positions[:] = self.terrain_box_positions[self.terrain_levels, self.terrain_types].clone()
            # self.box_dims[:] = self.terrain_box_dims[self.terrain_levels, self.terrain_types].clone()
            self.terrain_box_upper_vetices = torch.from_numpy(self.terrain.box_upper_vetices).to(self.device).to(torch.float)
            self.box_upper_vertices = self.terrain_box_upper_vetices[self.terrain_levels, self.terrain_types].clone()
            
            # self.terrain_motion_pool = torch.from_numpy(self.terrain.motion_pool).to(self.device).to(torch.long)
            self.terrain_sequence_pool_size = torch.from_numpy(self.terrain.sequence_pool_size).to(self.device).to(torch.long)
            # self.motion_pool = self.terrain_motion_pool[self.terrain_levels, self.terrain_types].clone()
            self.sequence_pool_size = self.terrain_sequence_pool_size[self.terrain_levels, self.terrain_types].clone()
            # import pdb; pdb.set_trace()
            # self.sequence_pool_size = self.motion_pool.shape[1]
            # self.sequence_length = self.motion_pool.shape[2]
            # import pdb; pdb.set_trace()
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s
        # import pdb; pdb.set_trace()
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    #-------------- Reference Motion ---------------
    def _load_motion(self):
        motion_path = self.cfg.motion.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        skeleton_path = self.cfg.motion.skeleton_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._motion_lib = MotionLibH1(motion_file=motion_path, device=self.device, masterfoot_conifg=None, fix_height=False,multi_thread=False,mjcf_file=skeleton_path) #multi_thread=True doesn't work
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        
        self.skeleton_trees = [sk_tree] * self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        self.motion_dt = self._motion_lib._motion_dt
        
    def _resample_motions(self, env_ids):
        if len(env_ids) == 0:
            return
        # self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        # self.motion_ids[env_ids] = torch.randint(0, self._motion_lib._num_unique_motions, (len(env_ids),), device=self.device)
        # print(self.motion_ids[:10])
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        self.env_origins_init_3Doffset[env_ids, :2] = torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        self.motion_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        motion_res = self._motion_lib.get_motion_state(self.motion_ids[env_ids], self.motion_times[env_ids])
        self.ref_base_pos_init[env_ids] = motion_res["root_pos"] + self.env_origins[env_ids] + self.env_origins_init_3Doffset[env_ids] # ZL: fix later. Ugly code. 
        self.ref_base_rot_init[env_ids] = motion_res["root_rot"]
        self.ref_base_vel_init[env_ids] = motion_res["root_vel"]
        self.ref_base_ang_vel_init[env_ids] = motion_res["root_ang_vel"]

        
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache
        
    def _update_motion_reference(self,):
        # if self.cfg.motion.recycle_motion:
        #     self.base_pos_init[motion_times <= self.dt, :3] = self.root_states[motion_times <= self.dt, :3]
        
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times)
        
        # self.ref_body_pos = motion_res["rg_pos"] + self.env_origins[:, None] # [num_envs, num_markers, 3]  # ZL: fix later. Ugly code.  
        self.ref_body_pos = motion_res["rg_pos"] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
        self.ref_body_vel = motion_res["body_vel"] # [num_envs, num_markers, 3]
        self.ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
        self.ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
        self.ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        # self.marker_coords[:] = motion_res["rg_pos"][:, 1:,] + self.env_origins[:, None]
        self.marker_coords[:] = motion_res["rg_pos"][:, 1:,] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
        # import ipdb; ipdb.set_trace()
    def _load_marker_asset(self):
        asset_path = self.cfg.motion.marker_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        marker_asset_options = gymapi.AssetOptions()
        marker_asset_options.angular_damping = 0.0
        marker_asset_options.linear_damping = 0.0
        marker_asset_options.max_angular_velocity = 0.0
        marker_asset_options.density = 0
        marker_asset_options.fix_base_link = True
        marker_asset_options.thickness = 0.0
        marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # set no collision
        marker_asset_options.disable_gravity = True
        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, marker_asset_options)
        return
    
    #------------ helper functions ---------------
    def _get_rigid_body_pos(self, body_name):
        return self._rigid_body_pos[:, self._body_list.index(body_name)-1, :]
    
    @property
    def knee_distance(self):
        left_knee_pos = self._get_rigid_body_pos("left_knee_link")
        right_knee_pos = self._get_rigid_body_pos("right_knee_link")
        # print(f"left knee pos: {left_knee_pos}")
        dist_knee = torch.norm(left_knee_pos - right_knee_pos, dim=-1, keepdim=True)
        # print("dist knee shape", dist_knee.shape)
        return dist_knee
  
    @property
    def feet_distance(self):
        left_foot_pos = self._get_rigid_body_pos("left_ankle_link")
        right_foot_pos = self._get_rigid_body_pos("right_ankle_link")
        dist_feet = torch.norm(left_foot_pos - right_foot_pos, dim=-1, keepdim=True)
        return dist_feet
    
    def get_ee_pos(self):
        left_foot_pos = self._rigid_body_pos[:, self.feet_indices[0] - 1]
        right_foot_pos = self._rigid_body_pos[:, self.feet_indices[1] - 1]
        left_hand_pos = self._rigid_body_pos[:, self.hand_indices[0] - 1]
        right_hand_pos = self._rigid_body_pos[:, self.hand_indices[1] - 1]
        return left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos
    
    def get_ee_pos_b(self):
        left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos = self.get_ee_pos()
        left_foot_pos_b = quat_rotate_inverse(self.base_quat, left_foot_pos - self.base_pos)
        right_foot_pos_b = quat_rotate_inverse(self.base_quat, right_foot_pos - self.base_pos)
        left_hand_pos_b = quat_rotate_inverse(self.base_quat, left_hand_pos - self.base_pos)
        right_hand_pos_b = quat_rotate_inverse(self.base_quat, right_hand_pos - self.base_pos) 
        return left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b
    
    def get_ee_pos_yawb(self):
        left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos = self.get_ee_pos()
        yawquat = yaw_quat(self.base_quat)
        left_foot_pos_b = quat_rotate_inverse(yawquat, left_foot_pos - self.base_pos)
        right_foot_pos_b = quat_rotate_inverse(yawquat, right_foot_pos - self.base_pos)
        left_hand_pos_b = quat_rotate_inverse(yawquat, left_hand_pos - self.base_pos)
        right_hand_pos_b = quat_rotate_inverse(yawquat, right_hand_pos - self.base_pos) 
        return left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b
    
    @property
    def hand_state(self):
        left_hand_index = self.hand_indices[0] - 1
        right_hand_index = self.hand_indices[1] - 1    
        assert (left_hand_index >= 0) and (right_hand_index >= 0)
        left_hand_pos = self._rigid_body_pos[:, left_hand_index]
        right_hand_pos = self._rigid_body_pos[:, right_hand_index]
        left_hand_ori = self._rigid_body_rot[:, left_hand_index].clone()
        right_hand_ori = self._rigid_body_rot[:, right_hand_index].clone()
        left_state = torch.cat([left_hand_pos, left_hand_ori], dim=-1).unsqueeze(1)
        right_state = torch.cat([right_hand_pos, right_hand_ori], dim=-1).unsqueeze(1)
        return torch.cat([left_state, right_state], dim=1)
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques (rel)
        return torch.sum(torch.square(self.torques / self.torque_limits), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques/self.torque_limits) - self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_close_feet(self):
        # returns 1 if two feet are too close
        return (self.feet_distance < 0.24).squeeze(-1) * 1.0
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
            # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_freeze_upper_body(self):
        return torch.mean(torch.square(self.dof_pos[:, 10:] - self.default_dof_pos[:, 10:]), dim=1)
        
    def _reward_tracking_dof_vel(self):
        # Tracking of dof velocity commands
        dof_vel_error = torch.sum(torch.square(self.joint_vel_reference - self.dof_vel), dim=1)
        return torch.exp(-dof_vel_error/self.cfg.rewards.tracking_sigma)    
    
        
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_slippage(self):
        assert self._rigid_body_vel.shape[1] == 20
        foot_vel = self._rigid_body_vel[:, self.feet_indices-1]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             2 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_vel_x(self):
        return 1. * torch.abs(self.base_lin_vel[:,0])

    def _reward_hand_vel_w(self):
        return self._rigid_body_vel[:,self.hand_indices[0]-1,:].norm(dim=-1) + self._rigid_body_vel[:,self.hand_indices[1]-1,:].norm(dim=-1)