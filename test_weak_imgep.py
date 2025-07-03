
import pickle
from exploration.random.func import RANDOM
from exploration.env.func import Env
from exploration.history import History
from exploration.weak_imgep.OptimizationPolicy import OptimizationPolicyWeak
from exploration.weak_imgep.weakimgep import WeakIMGEP
import matplotlib.pyplot as plt
from visu import diversity_time_iteration2
from visu2 import comparaison3, comparaison_ratios_iterations, comparaison_ratios_global_iterations
import json
import sys
if __name__=="__main__":
    #np.random.seed(0)
    folder1module = "data_1module"
    with open(sys.argv[1],"rb") as f:
        config = json.load(f)
    mutation_rate = config["mutation_rate"]
    N = config["N"]
    N_init = config["N_init"]
    num_bank = config["num_bank"]
    max_len = config["max_len"]
    folder = config["folder"]
    num_addr = config["num_addr"]
    image_folder = config["image_folder"]
    ks_ = config["ks"]
    En = Env(repetition=1)

    H_rand = History(max_size= N)
    H2_rand = History(max_size=N)
    rand = RANDOM(N = N,E = En, H = H_rand, H2 = H2_rand,max_=max_len)

    try:
        with open(f"{folder1module}/history_rand_N_{N}_{0}.pkl", "rb") as f:
            sample_rand = pickle.load(f)
            content_random = sample_rand["memory_perf"]
    except:
        print("start random exploration")
        rand()
        print("done")
        #save results
        H_rand.save_pickle(f"{folder1module}/history_rand_N_{N}")
        with open(f"{folder1module}/history_rand_N_{N}_{0}.pkl", "rb") as f:
            sample_rand = pickle.load(f)
            content_random = sample_rand["memory_perf"]
    ks = ks_
    for k in ks:
        print(f"start: k = {k}, N={N}")
        Pi = OptimizationPolicyWeak(k=k,mutation_rate=mutation_rate,max_len=max_len)
        H_imgep = History(max_size=N)
        imgep = WeakIMGEP(N,N_init, En,H_imgep,Pi, max_len = max_len)
        imgep.take(sample_rand,N_init)
        imgep()
        H_imgep.save_pickle(f"{folder}/history_weak_{k}_N_{N}")
        print(f"done")
#    N = 10000
    ks = [1,2,3,4]
    file = lambda k,N: f"{folder}/history_weak_{k}_N_{N}_0.pkl" 
    for k_moins_un,name in [(k,file(k,N)) for k in ks]:
        with open(name, "rb") as f:
            sample = pickle.load(f)
            content_imgep = sample["memory_perf"]
        file_names = [f"{image_folder}/comp_ratios_{k_moins_un}_{N}",f"{image_folder}/comp_times_k{k_moins_un}_{N}",f"{image_folder}/comp_count_{k_moins_un}_{N}"]
        comparaison3(content_random, content_imgep, name = file_names, title=f"weak imgep miss ratios k = {k_moins_un}, {N} iterations",label_algo = "weak imgep")
    ks = [1,2,3,4]
    #exit()
    for k in ks:
        diversity_time_iteration2(content_random,
                                [("weak imgep",k,f"{folder}/history_weak_{k}_N_{N}_0.pkl")],
                                title = f"comparaison_time_diversity_{k}",
                                folder = image_folder)
    for k in [1,2,3,4]:
        with open(f"{folder}/history_weak_{k}_N_{N}_0.pkl", "rb") as f:
            sample = pickle.load(f)
            content_imgep = sample["memory_perf"] 
        comparaison_ratios_iterations(("random",content_random),
                                     ("weak imgep",content_imgep),
                                     name = f"{image_folder}/comp_ratios_iteration_{k}_{N}", k = k)
        comparaison_ratios_global_iterations(("random",content_random),
                                            ("weak_imgep",content_imgep),
                                            name = f"{image_folder}/comp_global_ratios_iteration_{k}_{N}", k = k)
