from agent_arena_no_gathering import *
from environments.evil_wgw_env import EvilWindyGridWorld
from tabular_methods import QTable
import numpy as np
from hessian import hessian

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class OracleQDataset(Dataset):
    def __init__(self, env):
        super().__init__()
        
        qt = QTable(env)
        qt.run_policy_iteration(pol_eval_times=10, tol=1e-8)
        
        self.q = qt.q
        
        self.env = env
        
    def __getitem__(self, i):
        valid = False
        while not valid:
            i = np.random.randint(0, self.env.w, (1,))[0]
            j = np.random.randint(0, self.env.h, (1,))[0]

            self.env.reset()
            valid, state = self.env.set_pos((i, j))
            
        return torch.Tensor(state), torch.Tensor(self.q[i, j, :])
    
    
    def __len__(self):
        return 100000
    
    
    
class DQNOracleAgent(nn.Module):
    def __init__(self, state_shape=(7, 10), hidden=16, num_action=4):
        super().__init__()
        self.linear1 = nn.Linear(np.prod(state_shape), hidden)
        self.linear2 = nn.Linear(hidden, num_action)
        
    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)    
        return x
    
    
    
class OracleQRDataset(Dataset):
    def __init__(self, env, visual_env):
        super().__init__()
        self.env = visual_env
        qt = QTable(env)
        qt.qrq_learning(max_num_iter=200, lr=1e-1);
        
        self.z = qt.z

        
    def __getitem__(self, i):
        valid = False
        while not valid:
            i = np.random.randint(0, self.env.w, (1,))[0]
            j = np.random.randint(0, self.env.h, (1,))[0]

            self.env.reset()
            valid, state = self.env.set_pos((i, j))
            
        return torch.Tensor(state), torch.Tensor(self.z[i, j, :, :])
    
    
    def __len__(self):
        return 100000
    
    
    
class QROracleAgent(nn.Module):
    def __init__(self, state_shape=(7, 10), hidden=16, num_action=4, num_atoms=32):
        super().__init__()
        self.num_action = num_action
        self.num_atoms = num_atoms
        self.linear1 = nn.Linear(np.prod(state_shape), hidden)
        self.linear2 = nn.Linear(hidden, num_action * num_atoms)
        
    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x).view(-1, self.num_action, self.num_atoms)  
        return x