import sys
sys.path.append("../")
from exploration.env.func import Env
from exploration.history import History
from  exploration.weak_imgep.OptimizationPolicy import OptimizationPolicyWeak
from sim.sim_use import make_random_paire_list_instr


class WeakIMGEP:
    def __init__(self,
                N:int,
                N_init:int,
                E:Env,
                H:History,
                Pi:OptimizationPolicyWeak,
                max_len = 100):
        """
        N: int. The experimental budget
        N_init: int. Number of experiments at random
        H: History. Buffer containing codes and signature pairs
        G: GoalGenerator.
        Pi: OptimizationPolicy.
        """
        self.N = N
        self.env = E
        self.H = H
        self.N_init = N_init
        self.Pi = Pi
        self.max_len = max_len
        self.start = 0
    def take(self,sample:dict,N_init:int): 
        print("sampl", sample.keys())
        for key in sample["memory_perf"].keys():
            self.H.memory_perf[key]= list(sample["memory_perf"][key][:N_init])
        self.H.memory_program["core0"] = sample["memory_program"]["core0"][:N_init]
        self.H.memory_program["core1"] = sample["memory_program"]["core1"][:N_init]
        self.start = N_init
    def __call__(self):
        for i in range(self.start,self.N):
            if i<self.N_init:
                parameter = make_random_paire_list_instr(self.max_len)
            else:
                parameter = self.Pi(self.H,)
            observation = self.env(parameter)
            self.H.store({"program":parameter}|observation)
