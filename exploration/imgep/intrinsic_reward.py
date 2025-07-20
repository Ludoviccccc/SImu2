import numpy as np
import sys
sys.path.append("../../")
from exploration.history import History
from exploration.imgep.goal_generator import GoalGenerator
from exploration.imgep.OptimizationPolicy import OptimizationPolicykNN
from exploration.env.func import Env
class IR:
    """
    Intrinsic reward class
    env:Env. The environment 
    modules: dict.
    H: History. Buffer containing codes and signature pairs
    G: GoalGenerator.
    Pi: OptimizationPolicy.
    num_iteration: int. Number of iterationt to evaluate te learning progress
    epsilon:float in [0,1]. The module chooses a random module with probability epsilon
    """
    def __init__(self,
            env:Env,
            modules,
            history:History,
            goal_module:GoalGenerator,
            Pi:OptimizationPolicykNN,
            num_iteration:int,
            window:int=2,
            window_total_progress:int=5,
            epsilon:float=.3
            ):
        self.env = env
        self.epsilon = epsilon
        self.history = history
        self.modules = modules
        self.goal_module = goal_module
        self.Pi = Pi
        self.diversity = {}
        self.window = window
        self.window_total_progress = window_total_progress
        self.calls = 0
        self.num_iteration = num_iteration


    def progress(self):
        for module in self.modules:
            if np.abs(module["diversity"][0])==0:
                module["progress"] = np.abs(module["diversity"][-1] - module["diversity"][0])
            else:
                module["progress"] = np.abs(module["diversity"][-1] - module["diversity"][0])/np.abs(module["diversity"][0])
    def prob(self)->dict:
        #self.progress()
        sum_ = 0
        probs = []
        #constant to normalize: sum of all progress accross all modules
        for module in self.modules:
            sum_ += np.mean(module["total_progress"],axis=0)
        if sum_!=0:
            for module in self.modules:
                #print(module["type"], module["progress"])
                probs.append(np.mean(module["total_progress"],axis=0)/sum_)
        else:
            probs = [(1.0/len(self.modules))]*len(self.modules)
        return probs
    def choice(self):
        """
        choose the module to explore, the choice is random, based on the learing progress,
        itself based on the diversity
        """
        vec = np.zeros(len(self.modules))
        if self.calls==0:
            C = 1
            vec[np.random.randint(0,len(self.modules))] = 1.0
            probs = vec
        else:
            probs = self.prob()
            C = np.random.binomial(1,self.epsilon)
            if C:
                vec[np.random.randint(0,len(self.modules))] = 1.0
            probs = (1-C)*np.array(probs)+ C*vec
        return self.modules[int(np.random.choice(len(self.modules), 1, p=probs))]
    def __call__(self,N:int):
        """
        Evaluates progress made by exploring each module
        """
        for module in self.modules:
            for module_ in self.modules:
                self.eval_module_diversity(module_)
            goal = self.goal_module(self.history,module)
            for j in range(self.num_iteration):
                if len(self.history.memory_program["core0"])<N+1:
                    parameter = self.Pi(goal,self.history,module)
                    self.history.store({"program":parameter}|self.env(parameter))
                    print("len",len(self.history.memory_program["core0"]))
            for module_ in self.modules:
                self.eval_module_diversity(module_)
            self.progress()
            if "total_progress" not in module:
                module["total_progress"] = [np.sum([module["progress"] for module in self.modules])]
            else:
                module["total_progress"].append(np.sum([module["progress"] for module in self.modules]))
            module["total_progress"] = module["total_progress"][-self.window_total_progress:]
        self.calls+=1
    def eval_module_diversity(self,module:dict):
        feature = self.goal_module.data2feature(self.history.memory_perf, module)
        if module["type"] in ["miss_ratios","time_diff","time","miss_ratios_detailled","miss_count"]:
            bins = module["bins"]
            hist,_ = np.histogram(feature,bins =bins)
            div = sum(hist>0)
        elif module["type"]==f"miss_bank":
            bins = module["bins"]
            hist0,_,_ = np.histogram2d(feature[0,:],feature[2,:], bins=[bins, bins])
            hist1,_,_ = np.histogram2d(feature[1,:],feature[2,:], bins=[bins, bins])
            div = .5*(np.sum(hist0>0)+np.sum(hist1>0))
        elif module["type"]=="diff_ratios_bank":
            bins = module["bins"]

            hist,_,_ = np.histogram2d(feature[0,:],feature[1,:], bins=[bins, bins])
            div = np.sum(hist>0)
        elif module["type"] in ["time_vector"]:
            bins = module["bins"]
            hist1,_,_ = np.histogram2d(feature[0,:],feature[2,:], bins=[bins, bins])
            hist2,_,_ = np.histogram2d(feature[1,:],feature[3,:], bins=[bins, bins])
            div = np.sum(hist1>0) + np.sum(hist2>0)
        else:
            TypeError(f"module {module} not known")

        #Stores the result
        if "diversity" in module.keys():
            module["diversity"].append(div)
            module["diversity"] =module["diversity"][-self.window:]
        else:
            module["diversity"] = [div]
