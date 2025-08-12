import os
import sys
sys.path.append("../")
import json
import pickle
import random
from sim.sim_use import make_random_paire_list_instr
from exploration.history import History
from exploration.imgep.features import Features
import numpy as np
from exploration.env.func import Env
from  exploration.imgep.OptimizationPolicy import OptimizationPolicykNN
from exploration.imgep.goal_generator import GoalGenerator
from exploration.imgep.intrinsic_reward import IR
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import distributions
def env_observation2_subset(output):
    out_list = []
    for o in pp:
            #print(j,o,output[o])
            out_list.append(np.array(output[o]).flatten())
    return np.concatenate(out_list)
class IMGEP_:
    """
    N: int. The experimental budget
    N_init: int. Number of experiments at random
    H: History. Buffer containing codes and signature pairs
    G: GoalGenerator.
    Pi: OptimizationPolicy.
    """
    def __init__(self,
                 N:int,
                 N_init:int,
                 E:Env,
                 H:History,
                 G:GoalGenerator,
                 Pi:OptimizationPolicykNN,
                 ir:IR,
                 modules:list[dict],
                 periode:int = 1,
                 max_len:int = 100):
        self.N = N
        self.env = E
        self.H = H
        self.G = G
        self.N_init = N_init
        self.Pi = Pi
        self.ir = ir
        self.periode = periode
        self.modules = modules
        self.max_len = max_len
        self.start = 0
        self.periode_expl = 10
        self.k = 0
    def take(self,sample:dict,N_init:int):
        """Takes the ``N_init`` first steps from the ``sample`` dictionnary to initialize the exploration.
        Then the iterator i is set to N_init directly
        """
        print("sampl", sample.keys())
        for key in sample["memory_perf"].keys():
            self.H.memory_perf[key]= list(sample["memory_perf"][key][:N_init])
        self.H.memory_program["core0"] = sample["memory_program"]["core0"][:N_init]
        self.H.memory_program["core1"] = sample["memory_program"]["core1"][:N_init]
        self.start = N_init

class Normalize:
    def __init__(self):
        self.min_=None
        self.max_= None
    def fit(self,x):
        self.min_ = x.min(axis=-1)
        self.max_ = x.max(axis=-1)
    def transform(self,x):
        if self.min_==None or self.max_==None:
            raise TypeError(f"User must call method Normalize.fit before calling method Normalize.transform")
        return (x - self.min_)/(self.max_-self.min_)

class IMGEP(IMGEP_):
    def __init__(self,
                 N:int,
                 N_init:int,
                 E:Env,
                 H:History,
                 G:GoalGenerator, 
                 Pi:OptimizationPolicykNN,
                 Norm:Normalize,
                 periode:int = 1,
                 max_len:int = 100):
        super(IMGEP,self).__init__(N,N_init, En,H,G,Pi,ir=ir, modules = modules, periode = periode, max_len = max_len)
        self.norm = Norm
    def __call__(self,intr_reward=True):
        
        """Performs the exploration.
        intr_reward:bool. If True, the exploration uses intrinsic reward based on diversity
        """
        time_explor = 0
        for i in range(self.start,self.N+1):
            if i%100==0:
                print(f"{i} iterations")
            if i<time_explor:
                continue
            if i<=self.N_init:
                parameter = make_random_paire_list_instr(self.max_len,num_addr=self.env.num_addr)
            else:
                if (i-self.N_init)%self.periode==0 and i>self.N_init:

                    in_ = np.array(np.array(self.H.memory_tab))
                    self.norm.fit(in_)
                    in_0 = self.norm.transform(in_)
                    in_0 = normalize(in_)
                    U,sigma,Vh =np.linalg.svd(in_0)
                    probs = sigma/np.sum(sigma)
                    idx_axis = np.random.choice(in_.shape[1],1,p=probs)
                if (i-self.N_init)>1:
                    previous_module = module
                goal = self.G(self.H, module = idx_axis,norm=norm)
                parameter = self.Pi(goal,self.H, module)
            observation = self.env(parameter)
            self.H.store({"program":parameter}|observation)
            self.H.memory_tab.append(env_observation2_subset(observation))

class GoalGenerator(Features):
    def __init__(self,
                 num_bank:int):
        super().__init__()
        self.num_bank = num_bank
        self.k_time = 0
        self.k_miss = 0
        self.modules = modules
    def __call__(self,H:History,module:int,norm:Normalize)->np.ndarray:
        min_ = sigma.min(axis=-1)
        max_ = sigma.max(axis=-1)
        out = np.random.uniform((1-np.sign(min_)*0.6)*min_,4.0*max_)
        return out

if __name__=="__main__":

    num_bank = 4
    num_addr = 20


    En = Env(repetition=1,num_banks = num_bank,num_addr = num_addr)
    pp = ['shared_cache_miss',
          'time_core0_together',
          'time_core1_together',
          'time_core0_alone',
          'time_core1_alone',
          'diff_ratios_core0',
          'diff_ratios_core1',
          'diff_time0',
          'diff_time1',
          'miss_ratios_detailled',
          'miss_ratios_core0_detailled',
          'miss_ratios_core1_detailled',
        ]
    periode  =5
    max_len = 50
    num_banks = 4
    num_addr = 20
    mutation_rate = 0.1
    N = 100
    N_init = 100
    k =2
    min_len=5


    H = History(N)
    Pi = OptimizationPolicykNN(k=k,mutation_rate=mutation_rate,max_len=max_len,num_addr=num_addr,num_bank=num_bank,min_instr=min_len,max_instr=max_len)
    G = GoalGenerator(num_banks)
    imgep = IMGEP(N,N_init, En,H,None,Pi,ir=None, modules = None, periode = periode, max_len = max_len)
    imgep(intr_reward=False)


    in_ = np.array(np.array(H.memory_tab))
    def normalize(in_:torch.Tensor):
        min_ = in_.min(axis=0)
        max_ = in_.max(axis=0)
        return (in_ - min_)/(max_-min_)
    print(in_.shape)
    in_0 = normalize(in_)
    U,sigma,Vh =np.linalg.svd(in_0)
    C = in_0@Vh.transpose()
    print(C.shape)
    probs = sigma/np.sum(sigma)
    idx_axis = np.random.choice(in_.shape[1],1,p=probs)
    print("idx_axis", idx_axis)
    print("C",C[:,idx_axis].shape)
    idx_codes,_ = Pi.feature2closest_code(C[:,idx_axis].reshape((1,-1)),np.array([5.0]))
    print("idx_codes", idx_codes)
