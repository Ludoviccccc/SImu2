import json
import os
if __name__ == "__main__":
    folder = ["all_data/svd_results","all_data/svd_results","all_data/data_weak"]
    #folder = ["all_data/data10bis","all_data/data10bis","all_data/data_weak"]
    #image_folder ="all_images/image10bis"
    image_folder ="all_images/svd"
    N = int(10000)
    N_init = 1000
    ks = [1,2,3,5]
    ks2 = [1,2,3,5,10]
    ks2 = []
    num_bank = 4
    num_addr = 20
    file_mix = lambda k,N: f"history_weak_{k}_N_{N}_0.pkl"
    file_imgep_ir = lambda k,N: f"history_kNN_{k}_N_{N}_lp_0.pkl"
    file_imgep_svd = lambda k,N: f"history_kNN_{k}_N_{N}_svd_0.pkl"
    file_imgep_svd2 = lambda k,N: f"history_kNN_{k}_N_{N}_svd_1.pkl"
    files = []
#    files +=[{"folder":folder[2],"file":file_mix(k,N),"name":f"mix_programs k={k},N={N}","k":k,"N":N,"type":"mix_programs"} for k in ks]
#    files +=[{"folder":folder[0],"file":file_imgep_ir(k,N),"name":f"imgep_ir k={k},N={N}","k":k,"N":N,"type":"imgep_ir"} for k in ks]
    files +=[{"folder":folder[1],"file":file_imgep_svd(k,N),"name":f"imgep_svd k={k},N={N}","k":k,"N":N,"type":"imgep_svd"} for k in ks]
    files +=[{"folder":folder[1],"file":file_imgep_svd2(k,N),"name":f"imgep_svd k={k},N={N}","k":k,"N":N,"type":"imgep_svd"} for k in ks2]
    if os.path.exists(os.path.join("../",image_folder))==False:
        os.system(f"mkdir {image_folder}")
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

    with open(f"../{image_folder}/config_plots.json","w") as f:
        json.dump(config, f)
