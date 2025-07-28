import numpy as np
import pickle
import os.path
import copy
class History:
    def __init__(self, max_size = 100):
        self.max_size = max_size
        self.memory_program = {"core0":[],"core1":[]}
        self.memory_perf = {}
    def eviction(self):
        if len(self.memory_program)>self.max_size:
            self.memory_program["core0"] = self.memory_program["core0"][-self.max_size:]
            self.memory_program["core1"] = self.memory_program["core1"][-self.max_size:]
    def purge(self):
        self.memory_program["core0"] = [] 
        self.memory_program["core1"] = []
    def __len__(self):
        return len(self.memory_program["core0"])
    def store(self,sample:dict[list]):
        for j in range(len(sample["program"]["core0"])):
            self.memory_program["core0"].append(sample["program"]["core0"][j])
            self.memory_program["core1"].append(sample["program"]["core1"][j])
        for key in sample.keys():
            if key!="program":
                if key in self.memory_perf:
                    self.memory_perf[key].append(np.array(sample[key]))
                else:
                    self.memory_perf[key] = [np.array(sample[key])]
    def present_content(self):
        output  = {key:np.array(self.memory_perf[key]) for key in self.memory_perf.keys()}
        return output
    def save_pickle(self, name:str=None):
        k = 0
        name = f"{name}_{k}.pkl"
        while os.path.isfile(f"{name}_{k}.pkl"):
            k+=1
            name = f"data/{name}_{k}.pkl"
        output  = {"memory_perf":{key:np.array(self.memory_perf[key]) for key in self.memory_perf.keys()},
                "memory_program":{"core0":self.memory_program["core0"],"core1":self.memory_program["core1"]}}
        with open(name, "wb") as f:
            pickle.dump(output, f)
    def __getitem__(self,val):
        """
        returns slice of memory_perf and memory_program in this order
        """
        memory_program = {"core0":[],"core1":[]}
        memory_perf = {}
        for k in self.memory_perf.keys():
            memory_perf[k] = self.memory_perf[k][val]
        for k in self.memory_program.keys():
            memory_program[k] = self.memory_program[k][val]
        return memory_perf,memory_program
