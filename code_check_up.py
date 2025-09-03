import pickle
#from visualisation.comp import comparaison
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from exploration.imgep.OptimizationPolicy import OptimizationPolicykNN
from exploration.history import History
from exploration.imgep.imgep import IMGEP
from exploration.env.func import Env
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
    #files +=[{"folder":folder[1],"file":file_imgep_svd(k,N),"name":f"imgep svd","k":k,"N":N,"type":"imgep_svd"} for k in ks]
    random = {"folder":folder[0],
            "N":N,
            "file":f"history_rand_N_{N}_0.pkl",
            "name": "random"}
    ##############################
    with open(os.path.join(random["folder"],random["file"]),"rb") as f:
        sample_random = pickle.load(f)
    content_random = sample_random["memory_perf"]
    content_random_pgr = sample_random["memory_program"]


    contents_ = []
    for data in files:
        with open(os.path.join(data["folder"],data["file"]), "rb") as f:
            sample_imgep = pickle.load(f)
        content_imgep = sample_imgep["memory_perf"]
        content_imgep_pgr = sample_imgep["memory_program"]
        contents_.append((data["name"],content_imgep,data["k"]))
        with open(os.path.join(data["folder"],'modules.json'), "rb") as f:
            modules_imgep = json.load(f)
    modules_imgep = modules_imgep['modules']
    Pi = OptimizationPolicykNN(k=1)
    H_imgep = History(10000) 
    H_rand = History(10000) 
    lenghts_core0_imgep = [len(seq) for seq in content_imgep_pgr["core0"]]
    lenghts_core0_random = [len(seq) for seq in content_random_pgr["core0"]]
    imgep_imgep = IMGEP(N,N_init,None,H_imgep,None,None,None,None)
    imgep_rand = IMGEP(N,N_init,None,H_rand,None,None,None,None)
    imgep_imgep.take(sample_imgep,N)
    imgep_rand.take(sample_random,N)


    mutual_times = np.stack((H_imgep.memory_perf['time_core0_together'],H_imgep.memory_perf['time_core1_together']))
    iso_vs_mut_core0 = np.stack((H_imgep.memory_perf['time_core0_alone'],H_imgep.memory_perf['time_core0_together']))

    closest = Pi.feature2closest_code(mutual_times,np.array([300,500]))
    closest_inter = Pi.feature2closest_code(iso_vs_mut_core0,np.array([1050,3000]))
    print("closest", closest)
    #print(content_imgep.keys())
    print('core0 isolation',content_imgep['time_core0_alone'][closest_inter[0]],
          'core0 together', content_imgep['time_core0_together'][closest_inter[0]],
          'core1 isolation',content_imgep['time_core1_alone'][closest_inter[0]],
          'core1 together', content_imgep['time_core1_together'][closest_inter[0]])
    print(len(content_imgep_pgr['core0'][closest_inter[0]]))
    print(len(content_imgep_pgr['core1'][closest_inter[0]]))
    #plt.figure()
    #plt.plot(np.arange(10001),lenghts_core0_imgep, label="imgep")
    #plt.plot(np.arange(10000),lenghts_core0_random, label="random", alpha=.5)
    #plt.legend()
    #plt.show()
    print("max len pgr random", max(lenghts_core0_random))
    print("max len pgr imgep", max(lenghts_core0_imgep))
    print("min len pgr random", min(lenghts_core0_random))
    print("min len pgr imgep", min(lenghts_core0_imgep))
    E = Env()

    
    print('core0',content_imgep_pgr['core0'][closest_inter[0]])
    print('core1',content_imgep_pgr['core1'][closest_inter[0]])
    obs = E({'core0':[[]],'core1': [content_imgep_pgr['core1'][closest_inter[0]]]})
    print(obs['time_core0_alone'])

    idx = [j for j,a in enumerate(iso_vs_mut_core0.transpose()) if np.abs(a[0]-a[1])>1500]
    print(f"{len(idx)} programs avec des allongements en temps de plus de 2000 cycles")
    def hist_addr_accesses(idx,content_pgrm,num_addr=20):
        '''
        outputs np.ndarray of shape (len(idx),2,num_addr) with histogram the addresses accesses
        '''
        out = np.zeros((len(idx),2,num_addr))
        for i,id_ in enumerate(idx):
            for j in range(len(content_pgrm['core0'][i])):
                out[i,0,content_pgrm['core0'][id_][j]['addr']-1]+=1
            for j in range(len(content_pgrm['core1'][i])):
                out[i,1,content_pgrm['core1'][id_][j]['addr']-1]+=1
        return out
    print(content_imgep_pgr['core0'][5])
    hist_ = hist_addr_accesses(idx,content_imgep_pgr) 
    print(hist_.shape)
    def plot_distrib(hist_,title,name):
        plt.figure(figsize=(20,12))
        for j in range(hist_.shape[-1]):
            plt.plot(hist_[:,0,j],label=f'addr = {j+1}')
        plt.title(title,fontsize=32)
        plt.xlabel("program idx",fontsize=32)
        plt.ylabel("nb of accesses",fontsize=32)
        plt.legend(fontsize=18)
        plt.savefig(name,bbox_inches = 'tight',pad_inches = 0)
        plt.show()
    plot_distrib(hist_,"programs with execution time increase larger than 1500 cycles core 0","increase_core0")
    hist2_ = hist_addr_accesses(range(N),content_imgep_pgr) 
    plot_distrib(hist2_,"accesses of programs to addresses core0","accesses")
