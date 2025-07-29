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
class GoalGenerator(Features):
    def __init__(self,
                 num_bank:int,
                 modules:list,
                 H:History,
                 pi:OptimizationPolicykNN
                 ):
        #super().__init__()
        Features.__init__(self)
        self.num_bank = num_bank
        self.k_time = 0
        self.k_miss = 0
        self.modules = modules
        self.history = H
        self.pi = pi
        
    def rand(self,stat)->np.ndarray:
        min_ = stat.min(axis=-1)
        max_ = stat.max(axis=-1)
        out = np.random.uniform((1-np.sign(min_)*0.6)*min_,4.0*max_)
        return out
    def diversity_representation(self,module:dict):
        feature = self.data2feature(self.history.memory_perf,module)
        hist,_ = np.histogram(feature,bins=module["bins"])
        return hist
    def __call__(self,H:History, module:str)->np.ndarray:
        assert module in self.modules, f"module {module} unknown"
        stat = self.diversity_representation(module)
        out = self.rand(stat)
        return out
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
class IMGEP(IMGEP_):
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
        super(IMGEP,self).__init__(N,N_init, En,H,G,Pi,ir=ir, modules = modules, periode = periode, max_len = max_len)
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
                if intr_reward:
                    #Sample target goal
                    if (i-self.N_init)%(self.periode_expl*self.periode)==0:
                        self.ir(self.N)
                        time_explor = i + self.ir.num_iteration*len(self.ir.modules)
                        print("time explor", time_explor)
                        module = self.ir.choice()
                        goal = self.G(self.H, module = module)
                        print("i",i,"module type",module["type"],"diversity",module["diversity"])
                        print("probs", self.ir.prob())
                        continue
                    elif (i-self.N_init)%self.periode==0:
                        module = self.ir.choice()
                        goal = self.G(self.H, module = module)
                else:
                    if (i-self.N_init)%self.periode==0 and i>self.N_init:
                        module = random.choice(self.modules)
                    if (i-self.N_init)>1:
                        previous_module = module
                    goal = self.G(self.H, module = module)
                parameter = self.Pi(goal,self.H, module)
            observation = self.env(parameter)
            self.H.store({"program":parameter}|observation)
            self.H.memory_tab.append(env_observation2_subset(observation))


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
    G = GoalGenerator(num_bank,None,H,Pi)
    imgep = IMGEP(N,N_init, En,H,None,Pi,ir=None, modules = None, periode = periode, max_len = max_len)
    imgep(intr_reward=False)


    class ae(nn.Module):
        def __init__(self,size):
            nn.Module.__init__(self)
            self.size = size
            self.linear1 = nn.Linear(size,64)
            self.linear2 = nn.Linear(64,32)
            self.linear3 = nn.Linear(32,16)
            self.linear4 = nn.Linear(16,8)

            self.linear5 = nn.Linear(8,16)
            self.linear6 = nn.Linear(16,32)
            self.linear7 = nn.Linear(32,64)
            self.linear8 = nn.Linear(64,size)
            self.actv = nn.ReLU()

        def encoder(self,x):
            out = self.linear1(x)
            out = self.actv(out)
            out = self.linear2(out)
            out = self.actv(out)
            out = self.linear3(out)
            out = self.actv(out)
            out = self.linear4(out)
            return out
        def decoder(self,x):
            out = self.linear5(x)
            out = self.actv(out)
            out = self.linear6(out)
            out = self.actv(out)
            out = self.linear7(out)
            out = self.actv(out)
            out = self.linear8(out)
            return out
        def forward(self,x):
            out = self.encoder(x)
            out = self.decoder(out)
            return out
    aa = ae(58)
    in_ = torch.Tensor(np.array(H.memory_tab))
    print(in_.shape)
    print(aa(in_).shape)
