import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from termcolor import colored
import math

class myNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(myNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for layer_index in range(len(hidden_sizes)):
            if layer_index == len(hidden_sizes) - 1:
                layers.append(nn.Linear(hidden_sizes[layer_index], output_size))
            else:
                layers.append(nn.Linear(hidden_sizes[layer_index], hidden_sizes[layer_index + 1]))
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        x = self.mlp(x)
        return (x>0.)

class NHashCuriosity:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.hashnn = myNN(cfg.curiosity.obs_dim*2, cfg.curiosity.hidden_sizes_hash, cfg.curiosity.pred_dim).to(device)
        self.bin_cnt = torch.zeros(2**cfg.curiosity.pred_dim, dtype=torch.long, device=self.device, requires_grad=False)
        # normalize observation
        self.obs_lb = torch.Tensor(cfg.curiosity.obs_lb).unsqueeze(0).to(device)
        self.obs_ub = torch.Tensor(cfg.curiosity.obs_ub).unsqueeze(0).to(device)
    def bin2int(self, bins):
        # bins: (n, pred_dim) of boolean
        base_ = 2 ** torch.arange(end=self.cfg.curiosity.pred_dim, device=self.device).unsqueeze(0)
        ints = torch.sum(bins * base_, dim=-1)
        ints = torch.clip(ints, min=0, max = 2**self.cfg.curiosity.pred_dim-1)
        return ints
    def normalize(self, obs):
        obs_ = obs.clip(min=self.obs_lb, max=self.obs_ub)
        obs_ = (obs_-self.obs_lb) / (self.obs_ub - self.obs_lb) * math.pi # only half arc
        obs_ = torch.cat([torch.cos(obs_), torch.sin(obs_)], dim=-1)
        assert obs_.shape[-1] == self.cfg.curiosity.obs_dim*2
        return obs_
    def sym_map(self, obs):
        assert obs.shape[-1] == 29  # hardcode
        obs2 = torch.zeros_like(obs)
        obs2[:,0] = obs[:,0] * 1.
        obs2[:,1] = obs[:,1] * (-1.)
        obs2[:,2] = obs[:,2] * 1.  # xyz
        obs2[:,3] = obs[:,3] * (-1.)
        obs2[:,4] = obs[:,4] * 1.
        obs2[:,5] = obs[:,5] * (-1.)
        obs2[:,6] = obs[:,6] * 1.  # quat
        obs2[:,7:13] = obs[:,7:13] * torch.Tensor([[1,-1,1,-1,1,-1]]).to(obs.device)  # twist
        obs2[:,13:16] = obs[:,16:19] * torch.Tensor([[1,-1,1]]).to(obs.device)
        obs2[:,16:19] = obs[:,13:16] * torch.Tensor([[1,-1,1]]).to(obs.device)
        obs2[:,19:22] = obs[:,22:25] * torch.Tensor([[1,-1,1]]).to(obs.device)
        obs2[:,22:25] = obs[:,19:22] * torch.Tensor([[1,-1,1]]).to(obs.device) # ee
        obs2[:,25] = obs[:,26] * 1.
        obs2[:,26] = obs[:,25] * 1.
        obs2[:,27] = obs[:,28] * 1.
        obs2[:,28] = obs[:,27] * 1.  # ee contact
        return obs2
    def update_curiosity(self, obs:torch.Tensor, shift=0.)-> torch.Tensor:
        # obs: (n_env, n_obs), shift: (n_env, 1)
        n_env = len(obs)
        obs_ = torch.cat([obs, self.sym_map(obs)], dim=0)
        obs_ = self.normalize(obs_)
        obs_ *= (1 + shift.repeat(2,1))  # multiply radius
        hash_ = self.hashnn(obs_)
        bin_idx = self.bin2int(hash_) # (2*n_env,)
        newbin = torch.zeros_like(self.bin_cnt).repeat(2*n_env,1)
        newbin[torch.arange(2*n_env, device=self.device), bin_idx] += 1
        self.bin_cnt += torch.sum(newbin, dim=0)
        rew = 1.0 / torch.sqrt(1.0 + self.bin_cnt[bin_idx])
        return (0.5 * rew[:n_env] + 0.5 * rew[n_env:]).detach()


class RNDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(RNDNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0])) 
        layers.append(nn.ELU())
        for layer_index in range(len(hidden_sizes)):
            if layer_index == len(hidden_sizes) - 1:
                layers.append(nn.Linear(hidden_sizes[layer_index], output_size))
            else:
                layers.append(nn.Linear(hidden_sizes[layer_index], hidden_sizes[layer_index + 1]))
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class RNDCuriosity:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        self.rnd_pred = RNDNN(cfg.curiosity.obs_dim, cfg.curiosity.hidden_sizes_pred, cfg.curiosity.pred_dim).to(device)
        self.rnd_target = RNDNN(cfg.curiosity.obs_dim, cfg.curiosity.hidden_sizes_target, cfg.curiosity.pred_dim).to(device)
        for param in self.rnd_target.parameters(): param.requires_grad = False
        self.optimizer = torch.optim.SGD(self.rnd_pred.parameters(), lr=cfg.curiosity.lr)
        
    def update_curiosity(self, obs:torch.Tensor)-> torch.Tensor:
        # obs: (n_env, n_obs)
        pred = self.rnd_pred(obs)
        target = self.rnd_target(obs)
        rew = torch.norm(pred - target, dim=-1)
        loss = torch.mean((pred - target)**2)
        loss.requires_grad= True
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return rew.detach()



class DualDiscreteCuriosity:
    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        
        self.init_buffer()
        
    
    def init_buffer(self, ):
        self.curiosity_items = [attr for attr in dir(self.cfg.curiosity) if not attr.startswith('__')]
        assert len(self.curiosity_items) % 2 == 0
        print(colored(f"curiosity_items - {self.curiosity_items}", "yellow"))
        
        for i in range(0, len(self.curiosity_items), 2):
            xbinset = getattr(self.cfg.curiosity, self.curiosity_items[i])
            ybinset = getattr(self.cfg.curiosity, self.curiosity_items[i+1])
            
            n_xbin = int( ( xbinset[1] - xbinset[0] + 1e-8 ) // xbinset[2] )
            n_ybin = int( ( ybinset[1] - ybinset[0] + 1e-8 ) // ybinset[2] )
            
            curiosity_bincnt_xy = torch.zeros(n_xbin, n_ybin, device=self.device, dtype=torch.float, requires_grad=False)
            setattr(self, f"{self.curiosity_items[i]}_{self.curiosity_items[i+1]}", curiosity_bincnt_xy)
        
        
    def update_curiosity(self, curiosity_dict):
        
        for i in range(0, len(self.curiosity_items), 2):
            obs_x = curiosity_dict[self.curiosity_items[i]]
            obs_y = curiosity_dict[self.curiosity_items[i+1]]
            
            xbinset = getattr(self.cfg.curiosity, self.curiosity_items[i])
            ybinset = getattr(self.cfg.curiosity, self.curiosity_items[i+1])
            
            n_xbin = int( ( xbinset[1] - xbinset[0] + 1e-8 ) // xbinset[2] )
            n_ybin = int( ( ybinset[1] - ybinset[0] + 1e-8 ) // ybinset[2] )
        
            xls_ = torch.div(torch.clip(obs_x, min=xbinset[0], max=xbinset[1]) - xbinset[0], xbinset[2], rounding_mode="floor").long()
            yls_ = torch.div(torch.clip(obs_y, min=ybinset[0], max=ybinset[1]) - ybinset[0], ybinset[2], rounding_mode="floor").long()
            
            
            xls_ = torch.clip(xls_, min=0, max=n_xbin - 1).long()
            yls_ = torch.clip(yls_, min=0, max=n_ybin - 1).long()
        
         
            curiosity_bincnt_xy = getattr(self, f"{self.curiosity_items[i]}_{self.curiosity_items[i+1]}")
            newbin_xyl = torch.zeros_like(curiosity_bincnt_xy).repeat(self.num_envs,1,1)
            newbin_xyl[torch.arange(self.num_envs, device=self.device), xls_, yls_] += 1
            curiosity_bincnt_xy += torch.sum(newbin_xyl, dim=0)
            setattr(self, f"{self.curiosity_items[i]}_{self.curiosity_items[i+1]}", curiosity_bincnt_xy)
        
    def get_curiosity_list(self, curiosity_dict: Dict) -> List:
        curiosity_list = []
        
        for i in range(0, len(self.curiosity_items), 2):
            # print("i", i, self.curiosity_items[i], self.curiosity_items[i+1])
            # import pdb; pdb.set_trace()
            
            obs_x = curiosity_dict[self.curiosity_items[i]]
            obs_y = curiosity_dict[self.curiosity_items[i+1]]
            # print(obs_x, obs_y)
            # import pdb; pdb.set_trace()
            
            xbinset = getattr(self.cfg.curiosity, self.curiosity_items[i])
            ybinset = getattr(self.cfg.curiosity, self.curiosity_items[i+1])
            
            n_xbin = int( ( xbinset[1] - xbinset[0] + 1e-8 ) // xbinset[2] )
            n_ybin = int( ( ybinset[1] - ybinset[0] + 1e-8 ) // ybinset[2] )
            # import pdb; pdb.set_trace()
        
            xls_ = torch.div(torch.clip(obs_x, min=xbinset[0], max=xbinset[1]) - xbinset[0], xbinset[2], rounding_mode="floor").long()
            yls_ = torch.div(torch.clip(obs_y, min=ybinset[0], max=ybinset[1]) - ybinset[0], ybinset[2], rounding_mode="floor").long()
            
            xls_ = torch.clip(xls_, min=0, max=n_xbin - 1).long()
            yls_ = torch.clip(yls_, min=0, max=n_ybin - 1).long()
         
            curiosity_bincnt_xy = getattr(self, f"{self.curiosity_items[i]}_{self.curiosity_items[i+1]}")
            # import pdb; pdb.set_trace()
            curiosity_list.append(curiosity_bincnt_xy[xls_, yls_])
        
        # import pdb; pdb.set_trace()
        return curiosity_list
        
        