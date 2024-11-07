import torch
import numpy as np
from typing import List
from legged_gym.utils.math import wrap_to_pi, yaw_quat, in_poly_2d, batch_rand_int, diff_quat
from termcolor import colored


class BaseDino:
    def __init__(self, left_pos: List, left_quat: List, right_pos: List, right_quat: List):
        """ Define the basic upperbody motion
        Args:
            pos: [x, y, z]
            quat: [x, y, z, w]
        """
        self.left_pos = left_pos
        self.left_quat = left_quat
        self.right_pos = right_pos
        self.right_quat = right_quat
        
class JurassicPreset:
    class T_rex:
        pool = [ 
            BaseDino(
                left_pos = [0.0185, 0.2135, 0.10661],
                left_quat = [0., 0., 0., 1.],
                right_pos = [0.0185, -0.2135, 0.10661],
                right_quat = [0., 0., 0., 1.]
            ),
            BaseDino( ##Zombie Jump
                left_pos = [0.34727, 0.21145, 0.52434],
                left_quat = [-0.00628842, -0.0498768, 0.00441129, 0.998726],
                right_pos = [0.34727, -0.21145, 0.52434],
                right_quat = [0.00628842, -0.0498768, -0.00441129, 0.998726],
            ),
            BaseDino(
                left_pos = [0.10774, 0.29574, 0.13246],
                left_quat = [0.17598, -0.086496, -0.2936, 0.9356],
                right_pos = [0.10774, -0.29574, 0.13246],
                right_quat = [-0.17598, -0.086496, 0.2936, 0.9356]
            ),
            # BaseDino(
            #     left_pos=[0.0940048, 0.488889, 0.614446],
            #     left_quat=[-0.488116, 0.665247, -0.355721, -0.438921],
            #     right_pos=[0.0940048, -0.488889, 0.614446],
            #     right_quat=[0.488116, 0.665247, 0.355721, -0.438921],
            # ),
            BaseDino(
                left_pos=[0.016449, 0.53872, 0.50759],
                left_quat=[0.776049, 0.0326573, 0.0330757, 0.628957],
                right_pos=[0.016449, -0.53872, 0.50759],
                right_quat=[-0.776049, 0.0326573, -0.0330757, 0.628957],
            ),
        ]
    
class JurassicPark:
    """ This class is used to generate upperbody motion for regularizing motion, and hopefully for more complex tasks like box lifting...
    """
    def __init__(self, num_envs: int, num_steps_: int, motion_preset: str, device: str):
        """
        Args:
            num_envs: number of environments
            num_steps: number of steps per environment
            motion_preset: the preset of the motion
            device: torch.device
        """
        num_steps = 1
        print(colored("[Warning]: We overwrite num_steps to 1, since we only support 1 step for now", "red"))
        
        self.num_envs = num_envs
        self.num_steps = num_steps
        
        self.motion_preset = motion_preset
        self.device = device
        
        
        ## Load motion pool from presets
        presets = JurassicPreset()
        self.motion_pool = eval(f"presets.{motion_preset}.pool")
        
        left_pos_pool = []
        right_pos_pool = []
        left_quat_pool = []
        right_quat_pool = []
        
        for dino in self.motion_pool:
            left_pos_pool.append(dino.left_pos)
            right_pos_pool.append(dino.right_pos)
            left_quat_pool.append(dino.left_quat)
            right_quat_pool.append(dino.right_quat)
            
        # move to torch
        self.left_pos_pool = torch.tensor(left_pos_pool, device=device, dtype=torch.float)
        self.right_pos_pool = torch.tensor(right_pos_pool, device=device, dtype=torch.float)
        self.left_quat_pool = torch.tensor(left_quat_pool, device=device, dtype=torch.float)
        self.right_quat_pool = torch.tensor(right_quat_pool, device=device, dtype=torch.float)
        
        self.pool_size = torch.ones((num_envs, num_steps), device=device, dtype=torch.long) * len(self.motion_pool)
        
        self.left_pos = torch.zeros((num_envs, 3, num_steps), device=device, dtype=torch.float)
        self.right_pos = torch.zeros((num_envs, 3, num_steps), device=device, dtype=torch.float)
        self.left_quat = torch.zeros((num_envs, 4, num_steps), device=device, dtype=torch.float)
        self.right_quat = torch.zeros((num_envs, 4, num_steps), device=device, dtype=torch.float)
        
    def resample_motion(self, env_ids):
        if len(env_ids) <= 0:
            return
        # print("here")
        sequence_id = batch_rand_int(self.pool_size[env_ids]).to(device=self.device)
        # import pdb; pdb.set_trace()
        self.left_pos[env_ids, :, :] = self.left_pos_pool[sequence_id].transpose(1, 2)
        self.right_pos[env_ids, :, :] = self.right_pos_pool[sequence_id].transpose(1, 2)
        self.left_quat[env_ids, :, :] = self.left_quat_pool[sequence_id].transpose(1, 2)
        self.right_quat[env_ids, :, :] = self.right_quat_pool[sequence_id].transpose(1, 2)
        
        
    def get_ref_motion(self, step_id):
        step_id = 0
        # print(colored("Warning: We only support step_id = 0 for now", "red"))
        return self.left_pos[:, :, step_id].clone(), \
                self.left_quat[:, :, step_id].clone(), \
                self.right_pos[:, :, step_id].clone(), \
                self.right_quat[:, :, step_id].clone()
                
    def get_ref_motion_obs(self, step_id):
        step_id = 0
        # print(colored("Warning: We only support step_id = 0 for now", "red"))
        return torch.concat((self.left_pos[:, :, step_id], self.left_quat[:, :, step_id], self.right_pos[:, :, step_id], self.right_quat[:, :, step_id]), dim=1)