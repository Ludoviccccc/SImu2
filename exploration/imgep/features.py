import numpy as np
class Features:
    def __init__(self):
        pass
    def data2feature(self,stats:dict,module:str,slice_=slice(None))->np.ndarray:
        assert type(module)==dict,f"wrong type:{type(module)}"
        if module["type"]=="miss_ratios":
            bank = module["bank"]
            core = module["core"]
            if core!=None:
                out = np.array(stats[f"miss_ratios_core{core}"])[slice_,bank]
            else:
                out = np.array(stats[f"miss_ratios"])[:,bank]
        elif module["type"]=="time_vector":
            out = np.stack((stats["time_core0_alone"][slice_],
                            stats["time_core1_alone"][slice_],
                            stats["time_core0_together"][slice_],
                            stats["time_core1_together"][slice_],
                            ))
    
        elif module["type"]=="miss_bank":
            bank = module["bank"]
            out = np.stack((np.array(stats["miss_ratios_core0"])[slice_,bank],
                            np.array(stats["miss_ratios_core1"])[:slice_,bank],
                                  np.array(stats["miss_ratios"])[:slice_,bank]))
        elif module["type"]=="diff_ratios_bank":
            bank = module["bank"]
            out = np.stack((np.array(stats["diff_ratios_core0"])[:slice_,bank],
                            np.array(stats["diff_ratios_core1"])[:slice_,bank]))
        elif module["type"]=="diff_ratios_detailled":
            bank = module["bank"]
            row = module["row"]
            core = module["core"]
            out = np.array(stats[f"miss_ratios_core{core}_detailled"])[slice_,row,bank] - np.array(stats[f"miss_ratios_detailled"])[slice_,row,bank]
        elif module["type"]=="time":
            core = module["core"]
            single = module["single"]
            if single:
                out = stats[f"time_core{core}_alone"]
            else:
                out = stats[f"time_core{core}_together"]
        elif module["type"]=="miss_count":
            bank = module["bank"]
            core = module["core"]
            if core==None:
                out = np.array(stats["miss_count"])[slice_,bank]
            else:
                out = np.array(stats[f"miss_count_core{core}"])[slice_,bank]
        elif module["type"]=="miss_ratios_detailled":
            core = module["core"] 
            bank = module["bank"]
            row = module["row"]
            if core!=None:
                out = np.array(stats[f"miss_ratios_core{core}_detailled"])[slice_,row,bank]
            else:
                out = np.array(stats[f"miss_ratios_detailled"])[slice_,row,bank]
        elif module["type"]=="time_diff":
            core = module["core"]
            out = np.array(stats[f"diff_time{core}"][slice_])

        elif module["type"]=="miss_ratios_global_time": 
            out = np.stack((np.array(stats["miss_ratios_global"][slice_]),
                           np.array(stats["miss_ratios_global0"][slice_]),
                           np.array(stats["miss_ratios_global1"][slice_]),
                           stats["time_core0_alone"][slice_],
                           stats["time_core1_alone"][slice_],
                           stats["time_core0_together"][slice_],
                           stats["time_core1_together"][slice_]))
        elif module["type"]=="miss_ratios_global":
            out = np.stack((np.array(stats["miss_ratios_global"][slice_]),
                           np.array(stats["miss_ratios_global0"][slice_]),
                           np.array(stats["miss_ratios_global1"][slice_])))
        elif module["type"]=="cache_miss_ratio":
            core = module["core"]
            level = module["level"]
            out = stats[f"core{core}_{level}_cache_miss"][slice_,module["addr"]]
            #print("out", np.array(out))
            #exit()
        elif module["type"]=="shared_cache_miss_ratio":
            out = np.array(stats[f"shared_cache_miss"][slice_])
            #print("out", out.shape)
            out = out[slice_,module["addr"]]
        else:
            TypeError(f"module {module} unknown")
        return np.array(out[slice_])
