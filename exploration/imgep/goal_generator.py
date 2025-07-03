import random
import numpy as np
import sys
sys.path.append("../../")
from exploration.history import History
from exploration.imgep.features import Features
class GoalGenerator(Features):
    def __init__(self,
                 num_bank:int,
                 modules:list):
        super().__init__()
        self.num_bank = num_bank
        self.k_time = 0
        self.k_miss = 0
        self.modules = modules
    def __call__(self,H:History, module:str)->np.ndarray:
        assert module in self.modules, f"module {module} unknown"
        stats_ = H.memory_perf
        stat = self.data2feature(stats_, module)
        min_ = stat.min(axis=-1)
        max_ = stat.max(axis=-1)
        out = np.random.uniform((1-np.sign(min_)*0.6)*min_,4.0*max_)
        return out
