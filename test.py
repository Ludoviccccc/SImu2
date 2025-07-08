
import pickle
import numpy as np
from exploration.random.func import RANDOM
from exploration.env.func import Env
from exploration.history import History
from exploration.imgep.goal_generator import GoalGenerator
from exploration.imgep.OptimizationPolicy import OptimizationPolicykNN
from exploration.imgep.imgep import IMGEP
import matplotlib.pyplot as plt
from visu import diversity_time_iteration2
from visu2 import comparaison3, comparaison_ratios_iterations, comparaison_ratios_global_iterations
from exploration.imgep.intrinsic_reward import IR
import json
import sys
if __name__=="__main__":
    #np.random.seed(0)
    folder1module = "data_1module"
    with open(sys.argv[1],"rb") as f:
        config = json.load(f)
    with open(sys.argv[2],"rb") as f:
        modules_dict = json.load(f)
    modules = modules_dict["modules"]
    #modules = config["modules"]
    periode = config["periode"]
    mutation_rate = config["mutation_rate"]
    N = config["N"]
    N_init = config["N_init"]
    num_bank = config["num_bank"]
    max_len = config["max_len"]
    folder = config["folder"]
    num_addr = config["num_addr"]
    num_iteration = config["num_iteration"]
    ks_ = config["ks"]
    print("nomb modules", len(modules))
    En = Env(repetition=1,num_banks = num_bank,num_addr = num_addr)

    H_rand = History(max_size= N)
    H2_rand = History(max_size=N)
    rand = RANDOM(N = N,E = En, H = H_rand, H2 = H2_rand,max_=max_len)

    try:
        with open(f"{folder}/history_rand_N_{N}_{0}.pkl", "rb") as f:
            sample_rand = pickle.load(f)
            content_random = sample_rand["memory_perf"]
    except:
        print("start random exploration")
        rand()
        print("done")
        #save results
        H_rand.save_pickle(f"{folder}/history_rand_N_{N}")
        with open(f"{folder}/history_rand_N_{N}_{0}.pkl", "rb") as f:
            sample_rand = pickle.load(f)
            content_random = sample_rand["memory_perf"]
    ks = ks_
    for intr_reward in [False]:
        for k in ks:
            print(f"start: k = {k}, N={N}")
            G = GoalGenerator(num_bank = num_bank,modules = modules)
            Pi = OptimizationPolicykNN(k=k,mutation_rate=mutation_rate,max_len=max_len,num_addr=num_addr,num_bank=num_bank)
            H_imgep = History(max_size=N)
            ir = IR(En,modules,H_imgep, G,Pi,num_iteration,window = 5)
            imgep = IMGEP(N,N_init, En,H_imgep,G,Pi,ir, modules = modules, periode = periode, max_len = max_len)
            imgep.take(sample_rand,N_init)
            imgep(intr_reward=intr_reward)
            if intr_reward:
                H_imgep.save_pickle(f"{folder}/history_kNN_{k}_N_{N}_lp")
            else:
                H_imgep.save_pickle(f"{folder}/history_kNN_{k}_N_{N}_no_lp")
            print(f"done")
