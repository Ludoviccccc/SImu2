import sys
sys.path.append("../")
from exploration.env.func import Env
from exploration.history import History
from  exploration.imgep.OptimizationPolicy import OptimizationPolicykNN
from exploration.imgep.goal_generator import GoalGenerator
from sim.sim_use import make_random_paire_list_instr
import random

from exploration.imgep.intrinsic_reward import IR

class IMGEP:
    """
    N: int. The experimental budget
    N_init: int. Number of experiments at random
    H: History. Buffer containing codes and signature pairs
    G: GoalGenerator.
    Pi: OptimizationPolicy.
    """
    def __init__(self,N:int,N_init:int,E:Env,H:History,G:GoalGenerator, Pi:OptimizationPolicykNN,ir:IR,modules:list[dict],periode:int = 1,max_len:int = 100):
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
            if i<self.N_init:
                parameter = make_random_paire_list_instr(self.max_len,num_addr=self.env.num_addr)
            else:
                if intr_reward:
                    #Sample target goal
                    if (i-self.N_init)%(self.periode_expl*self.periode)==0:
                        self.ir(self.N)
                        time_explor = i + self.ir.num_iteration*len(self.ir.modules)
                        print("time explor", time_explor)
                        module = self.ir.choice()
                        continue
                    elif (i-self.N_init)%self.periode==0:
                        module = self.ir.choice()
                else:
                    if (i-self.N_init)%self.periode==0:
                        module = random.choice(self.modules)
                if (i-self.N_init)>=self.periode:
                    goal = self.G(self.H, module = module)
                parameter = self.Pi(goal,self.H, module)
            observation = self.env(parameter)
            self.H.store({"program":parameter}|observation)
