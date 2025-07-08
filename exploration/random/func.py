import sys
import random
sys.path.append("../")
sys.path.append("../sim")
sys.path.append("../exploration/")
from exploration.env.func import Env
from exploration.history import History
from sim.sim_use import runpgrms, make_random_list_instr, make_random_paire_list_instr
from sim.ddr import DDRMemory
from sim.class_mem_sim import *
def random_exploration(core0, core1,interconnect,ddr, budget:int=50, max_instr:int=50):
    program = runpgrms(core0, core1, max_instr, interconnect, ddr)
    list_intersection = []
    list_obj = []
    for i in range(budget):
        instructions0 = make_random_list_instr(length=random.randint(5,max_instr), core=0)
        instructions1 = make_random_list_instr(length=random.randint(5,max_instr), core=1)
        program(instructions0, instructions1)
        list_intersection.append(program.same_acces())
        list_obj.append(program.addr_obs())
        print("acces L3 core 0", program.list_acces_L30)
        print("acces L3 core 1", program.list_acces_L31)
        program.eviction()
    return list_intersection, list_obj

class RANDOM:
    def __init__(self,N:int,E:Env,H:History, H2:History=None,max_=50):
        """
        N: int. The experimental budget
        H: History. Buffer containing codes and signature pairs
        H2: History a second history to make comparaisons.
        max_l: int. Max length for of the instruction sequences
        E: Env. The environnement.
        """
        self.env = E
        self.H = H
        self.H2 = H2
        self.N = N
        self.max_ = max_
    def __call__(self):
        for i in range(self.N):
            parameter = make_random_paire_list_instr(self.max_)
            self.H.store({"program":parameter}|self.env(parameter))
            self.H2.store({"program":parameter}|self.env(parameter))
