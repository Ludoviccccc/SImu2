import json
if __name__ == "__main__":
    folder = ["all_data/data_short5","all_data/data_short5","all_data/data_weak"]
    folder1module = "all_data/data_1module"
    image_folder ="all_images/image_short5"
    N = int(10000)
    N_init = 1000
    ks = [3]
    num_bank = 4
    num_addr = 20
    file_mix = lambda k,N: f"history_weak_{k}_N_{N}_0.pkl"
    file_imgep_ir = lambda k,N: f"history_kNN_{k}_N_{N}_lp_0.pkl"
    file_imgep_no_ir = lambda k,N: f"history_kNN_{k}_N_{N}_no_lp_0.pkl"
    files = []
#    files +=[{"folder":folder[2],"file":file_mix(k,N),"name":f"mix_programs k={k},N={N}","k":k,"N":N,"type":"mix_programs"} for k in ks]
    files +=[{"folder":folder[0],"file":file_imgep_ir(k,N),"name":f"imgep_ir k={k},N={N}","k":k,"N":N,"type":"imgep_ir"} for k in ks]
    files +=[{"folder":folder[1],"file":file_imgep_no_ir(k,N),"name":f"imgep_no_ir k={k},N={N}","k":k,"N":N,"type":"imgep_no_ir"} for k in ks]

    random = {"folder":folder[0],
                "N":N,
                "file":f"history_rand_N_{N}_0.pkl",
                "name": "random"}
    config = {"files":files,
              "N_init":N_init,
              "N":N,
              "image_folder":image_folder,
              "random":random,
              "num_bank": num_bank,
              "num_addr":num_addr,
              "ks":ks}

    with open(f"{image_folder}/config_plots.json","w") as f:
        json.dump(config, f)
