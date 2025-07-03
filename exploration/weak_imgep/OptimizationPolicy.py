import random
import numpy as np

import sys
sys.path.append("../../")
from exploration.imgep.mutation import mutate_paire_instructions, mix_instruction_lists
from exploration.history import History

class OptimizationPolicyWeak():
    def __init__(self,
                k=4,
                mutation_rate = 0.1,
                max_len=50):
        super().__init__()
        self.k = k
        self.mutation_rate = mutation_rate
        self.max_len = max_len
        self.num_bank = 4
    def __call__(self,H:History)->dict:
        closest_codes = self.random_selection_codes(H) #most promising sample from the history
        output = self.mix(closest_codes) #expansion strategie: small random mutation
        return output
    def mix(self,programs):
        mix0, mix1 = mix_instruction_lists(programs["program"]["core0"],self.max_len), mix_instruction_lists(programs["program"]["core1"],self.max_len)
        output = self.light_code_mutation({"core0":mix0[:self.max_len],"core1":mix1[:self.max_len]}) #expansion strategie: small random mutation
        return output 
    def loss(self,goal:np.ndarray, elements:np.ndarray):
        if type(goal)!=float:
            a = goal.reshape(-1,1) 
        else:
            a = np.array([goal]).reshape(-1,1) 
        return  np.sum((a -elements)**2,axis=0)
    def random_selection_codes(self,H:History)->dict:
        assert len(H.memory_program)>0, "history empty"
        idx = random.sample(range(len(H.memory_program["core0"])),self.k)
        output = {"program": {"core0":[],"core1":[]},}
        for id_ in idx:
            output["program"]["core0"].append(H.memory_program["core0"][id_])
            output["program"]["core1"].append(H.memory_program["core1"][id_])
        return output
    def light_code_mutation(self,programs:dict[list[dict]]):
        mutated0, mutated1 = mutate_paire_instructions(programs["core0"], programs["core1"],mutation_rate = self.mutation_rate)
        return {"core0":[mutated0[:self.max_len]],"core1":[mutated1[:self.max_len]]}
