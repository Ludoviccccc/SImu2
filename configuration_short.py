import json
import numpy as np
if __name__ == "__main__":
    num_addr = 20
    N = int(10000)
    N_init = 1000
    max_len = 50
    periode = 5
    num_bank = 4
    mutation_rate = .1
    num_iteration = 5 #has to be small compared to N
    #modules =   ["time"]
    min_instr = 5
    modules = [{"type":"time_vector","bins":list(np.linspace(0,1000,21))}]
    modules +=   [{"type":"miss_bank","bank":j,"bins":list(np.linspace(0,1,21))} for j in range(num_bank)]
    modules +=  [{"type":"diff_ratios_bank","bank":j,"bins":list(np.linspace(0,1,21))} for j in range(num_bank)]

    modules +=  [{"type":"time_diff","core":core,"bins":list(np.linspace(0,1000,21))} for core in range(2)]
    modules +=  [{"type":"miss_count", "bank":bank,"core":core,"bins":list(np.linspace(0,20,20))} for bank in range(num_bank) for core in [None,0,1]]
    dict_modules = [{"type":"miss_ratios","bank":bank, "core":core,"bins":list(np.linspace(0,1,21))} for core in [None, 0,1] for bank in range(num_bank)]
    dict_times = [{"type":"time", "core":core,"single":single,"bins":list(np.linspace(0,1000,21))} for core in range(2) for single in [True, False]]

    ratios_detailled = [{"type":"miss_ratios_detailled","bank":bank,"core":core,"row":row,"bins":list(np.linspace(0,1,21))} for core in [None,0,1] for bank in range(num_bank) for row in range((num_addr//16)+1)]
    cache_ratios = [{"type":"shared_cache_miss_ratio","bins":list(np.linspace(0,1,21)),"addr":addr} for addr in range(num_addr)]+[{"type":"cache_miss_ratio","level":f"L{j}","core":i,"bins":list(np.linspace(0,1,21)),"addr":addr} for j in [1,2,3] for i in [0,1] for addr in range(num_addr)]
    ks = [2]
    modules = modules + dict_modules + ratios_detailled + dict_times+cache_ratios
    #modules = [{"type":"miss_ratios_global_time"}]

    folder = "all_data/data_short3"
    config = {"N_init":N_init,
              "N":N,
              "periode":periode,
              "mutation_rate":mutation_rate,
              "max_len":max_len,
              "min_len":min_instr,
              "num_addr":num_addr,
              "num_bank":num_bank,
              "folder":folder,
              "ks":ks,
              "num_iteration": num_iteration}
    with open(f"{folder}/config.json","w") as f:
        json.dump(config, f)
    modules_dict = {"modules":modules}

    with open(f"{folder}/modules.json","w") as f:
        json.dump(modules_dict, f)
