import pickle
from visualisation.comp import comparaison
import json
import sys
import os
import numpy as np
if __name__=="__main__":
    ##############################
    folder = ["all_data/data_large","all_data/svd_results"]
    N = int(10000)
    N_init = 1000
    ks = [2]
    num_bank = 4
    num_addr = 20
    file_mix = lambda k,N: f"history_weak_{k}_N_{N}_0.pkl"
    file_imgep_no_ir = lambda k,N: f"history_kNN_{k}_N_{N}_no_lp_0.pkl"
    file_imgep_svd = lambda k,N: f"history_kNN_{k}_N_{N}_svd_0.pkl"
    files = []

    files +=[{"folder":folder[0],"file":file_imgep_no_ir(k,N),"name":f"imgep no ir","k":k,"N":N,"type":"imgep_no_ir"} for k in ks]
    files +=[{"folder":folder[1],"file":file_imgep_svd(k,N),"name":f"imgep svd","k":k,"N":N,"type":"imgep_svd"} for k in ks]
    random = {"folder":folder[0],
            "N":N,
            "file":f"history_rand_N_{N}_0.pkl",
            "name": "random"}
    ##############################
    with open(os.path.join(random["folder"],random["file"]),"rb") as f:
        sample = pickle.load(f)
    content_random = sample["memory_perf"]


    contents_ = []
    for data in files:
        with open(os.path.join(data["folder"],data["file"]), "rb") as f:
            sample = pickle.load(f)
            content_imgep = sample["memory_perf"]
            contents_.append((data["name"],content_imgep,data["k"]))
    comparaison(content_random, 
                contents_,
                name="comparaison")
