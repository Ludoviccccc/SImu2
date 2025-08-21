import os
import sys
sys.path.append("../")
import json
import pickle
import random
from sim.sim_use import make_random_paire_list_instr
from exploration.history import History
import numpy as np
from exploration.random.func import RANDOM
from exploration.env.func import Env
from  exploration.imgep.OptimizationPolicy import OptimizationPolicykNN

class GoalGenerator():
    def __init__(self,
                 num_bank:int,
                 ):
        super().__init__()
        self.num_bank = num_bank
        self.k_time = 0
        self.k_miss = 0
    def __call__(self,coords:np.ndarray)->np.ndarray:
        min_ = coords.min(axis=-1)
        max_ = coords.max(axis=-1)
        out = np.random.uniform((1-np.sign(min_)*0.6)*min_,4.0*max_)
        return out
def env_observation2_subset(output):
    out_list = []
    for o in pp:
            out_list.append(np.array(output[o]).flatten())
    return np.concatenate(out_list)
class OptimizationPolicykNN_(OptimizationPolicykNN):
    def __init__(self,
                k=4,
                mutation_rate = 0.1,
                max_len=50,
                num_addr = 20,
                num_bank = 4,
                min_instr = 5,
                max_instr = 50):
                super(OptimizationPolicykNN_,self).__init__(
                k=4,
                mutation_rate = 0.1,
                max_len=50,
                num_addr = 20,
                num_bank = 4,
                min_instr = 5,
                max_instr = 50)
    def select_closest_codes(self,H,coords:np.ndarray,goal:np.ndarray):
        idx = self.feature2closest_code(coords.reshape((1,-1)),goal)
        output = {"program": {"core0":[],"core1":[]},}
        for id_ in idx:
            output["program"]["core0"].append(H.memory_program["core0"][id_])
            output["program"]["core1"].append(H.memory_program["core1"][id_])
        return output
    def __call__(self,goal:np.ndarray,H:History,coords:np.ndarray)->dict:
        closest_codes = self.select_closest_codes(H,coords,goal) #most promising sample from the history
        output = self.mix(closest_codes) #expansion strategie: small random mutation
        return output
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
                 periode:int = 1,
                 max_len:int = 100):
        self.N = N
        self.env = E
        self.H = H
        self.G = G
        self.N_init = N_init
        self.Pi = Pi
        self.periode = periode
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
        for j in range(len(H)):
            
            dd = {key:sample["memory_perf"][key][j] for key in sample["memory_perf"].keys() if key in pp}
            H.memory_tab.append(env_observation2_subset(dd))

class Normalize:
    """Affine normalization
    """
    def __init__(self):
        self.min_=None
        self.max_= None
        self.g = 1
    def fit(self,x):
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        self.g=0
    def transform(self,x):
        if self.g: 
            raise TypeError(f"User must call method Normalize.fit before calling method Normalize.transform")
        return (x - self.min_)/(self.max_-self.min_)

class IMGEP(IMGEP_):
    def __init__(self,
                 N:int,
                 N_init:int,
                 E:Env,
                 H:History,
                 G:GoalGenerator, 
                 Pi:OptimizationPolicykNN_,
                 Norm:Normalize,
                 periode:int = 1,
                 max_len:int = 100):
        super(IMGEP,self).__init__(N,N_init, En,H,G,Pi, periode = periode, max_len = max_len)
        self.norm = Norm
    def __call__(self):
        
        """Performs the exploration.
        """
        for i in range(self.start,self.N+1):
            if i%100==0:
                print(f"{i} iterations")
            if i<self.N_init:
                parameter = make_random_paire_list_instr(self.max_len,num_addr=self.env.num_addr)
            else:
                if (i-self.N_init)%(self.periode*10)==0:
                    in_ = np.array(np.array(self.H.memory_tab))
                    self.norm.fit(in_)
                    in_0 = self.norm.transform(in_)
                    U,sigma,Vh =np.linalg.svd(in_0)
                if (i-self.N_init)%self.periode==0:
                    idx_axis = np.random.randint(0,in_.shape[1])
                in_ = np.array(self.H.memory_tab)
                coords = self.norm.transform(in_)@Vh.transpose()
                goal = self.G(coords[:,idx_axis])
                parameter = self.Pi(goal,self.H, coords[:,idx_axis])
            observation = self.env(parameter)
            self.H.store({"program":parameter}|observation)
            self.H.memory_tab.append(env_observation2_subset(observation))




if __name__=="__main__":

    num_bank = 4
    num_addr = 20


    En = Env(repetition=1,num_banks = num_bank,num_addr = num_addr)
    pp = ['shared_cache_miss',
          'general_shared_cache_miss',
          'general_shared_cache_miss_core0',
          'general_shared_cache_miss_core1',
          'miss_ratios',
          'miss_ratios_global',
          'miss_ratios_global0',
          'miss_ratios_global1',
          'miss_ratios_core0',
          'miss_ratios_core1',
          'time_core0_together',
          'time_core1_together',
          'time_core0_alone',
          'time_core1_alone',
          'miss_count',
          'miss_count_core0',
          'miss_count_core1',
          'diff_ratios_core0',
          'diff_ratios_core1',
          'diff_time0',
          'diff_time1',
          'miss_ratios_detailled',
          'miss_ratios_core0_detailled',
          'miss_ratios_core1_detailled',
    ]
    periode  = 20
    max_len = 50
    num_banks = 4
    num_addr = 20
    mutation_rate = 0.1
    N = 10000
    N_init = 1000
    k = 20
    min_len=5


    H = History(N)
    Pi = OptimizationPolicykNN_(k=k,mutation_rate=mutation_rate,max_len=max_len,num_addr=num_addr,num_bank=num_bank,min_instr=min_len,max_instr=max_len)
    G = GoalGenerator(num_banks)
    Norm = Normalize()
    imgep = IMGEP(N,N_init, En,H,G,Pi,Norm, periode = periode, max_len = max_len)
    folder = "all_data/svd_results"
    while True:
        print("opening data")
        try:
            with open(f"{folder}/history_rand_N_{N}_0.pkl","rb") as f:
                sample_rand = pickle.load(f)
                content_random = sample_rand["memory_perf"]
            break
        except:
            print("start random exploration")
            H_rand = History(max_size=N)
            H2_rand = History(max_size=N)
            rand = RANDOM(N = N,E = En, H = H_rand, H2 = H2_rand,max_=max_len)
            rand()
            H2_rand.save_pickle(f"{folder}/history_rand_N_{N}")
    imgep.take(sample_rand,N_init)
    imgep()
    folder = "all_data/svd_results"
    H.save_pickle(f"{folder}/history_kNN_{k}_N_{N}_svd")
