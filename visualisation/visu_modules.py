from exploration.history import History
from exploration.imgep.features import Features as F
from exploration.imgep.intrinsic_reward import eval_diversity
import matplotlib.pyplot as plt
import os
def visu_modules(content:dict,modules:list[dict],folder:str):
    Feat = F()
    for j,module in enumerate(modules):
        feature = Feat.data2feature(content, module)
        print("feature type",module["type"])
        print("feature shape", feature.shape)
        print("div", eval_diversity(feature,module))
        if feature.ndim==1:
            div_list = [eval_diversity(feature[:j],module) for j in range(0,len(feature),200)]
        if feature.ndim==2:
            div_list = [eval_diversity(feature[:,:j],module) for j in range(0,feature.shape[1],200)]
        if j<=4:
            print(div_list)
        plt.figure()
        if feature.ndim==1:
            plt.plot(range(0,len(feature),200),div_list)
        else:
            plt.plot(range(0,feature.shape[1],200),div_list)

        plt.title(module["type"])
        os.path.join
        plt.savefig(os.path.join(folder,f"{j}"))
        plt.close()

