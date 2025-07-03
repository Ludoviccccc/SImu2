import numpy as np
class Features:
    def __init__(self):
        pass
    def data2feature(self,stats:dict,module:str)->np.ndarray:
        #if module=="time":
        #    out = np.stack((stats["time_core0_alone"],
        #                    stats["time_core1_alone"],
        #                    stats["time_core0_together"],
        #                    stats["time_core1_together"],
        #                    ))
        #elif module in [f"miss_bank_{j}" for j in range(self.num_bank)]:
        #    out = np.stack((np.array(stats["miss_ratios_core0"])[:,int(module[-1])],
        #                    np.array(stats["miss_ratios_core1"])[:,int(module[-1])],
        #                    np.array(stats["miss_ratios"])[:,int(module[-1])]))
        #elif module in [f"diff_ratios_bank_{j}" for j in range(self.num_bank)]:
        #    out = np.stack((np.array(stats["diff_ratios_core0"])[:,int(module[-1])],
        #        np.array(stats["diff_ratios_core1"])[:,int(module[-1])]))
        if type(module)==dict: 
            if module["type"]=="miss_ratios":
                bank = module["bank"]
                core = module["core"]
                if core!=None:
                    out = np.array(stats[f"miss_ratios_core{core}"])[:,bank]
                else:
                    out = np.array(stats[f"miss_ratios"])[:,bank]
            elif module["type"]=="time_vector":
                out = np.stack((stats["time_core0_alone"],
                                stats["time_core1_alone"],
                                stats["time_core0_together"],
                                stats["time_core1_together"],
                                ))
    
            elif module["type"]=="miss_bank":
                bank = module["bank"]
                out = np.stack((np.array(stats["miss_ratios_core0"])[:,bank],
                                np.array(stats["miss_ratios_core1"])[:,bank],
                                      np.array(stats["miss_ratios"])[:,bank]))
            elif module["type"]=="diff_ratios_bank":
                bank = module["bank"]
                out = np.stack((np.array(stats["diff_ratios_core0"])[:,bank],
                np.array(stats["diff_ratios_core1"])[:,bank]))
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
                    out = np.array(stats["miss_count"])[:,bank]
                else:
                    out = np.array(stats[f"miss_count_core{core}"])[:,bank]
                    #out  = np.stack((np.array(stats["miss_count_core0"])[:,bank],
                    #                np.array(stats["miss_count_core1"])[:,bank],
                    #                np.array(stats["miss_count"])[:,bank]))
            elif module["type"]=="miss_ratios_detailled":
                core = module["core"] 
                bank = module["bank"]
                row = module["row"]
                if core!=None:
                    out = np.array(stats[f"miss_ratios_core{core}_detailled"])[:,row,bank]
                else:
                    out = np.array(stats[f"miss_ratios_detailled"])[:,row,bank]
            elif module["type"]=="time_diff":
                core = module["core"]
                out = np.array(stats[f"diff_time{core}"])

            elif module["type"]=="miss_ratios_global_time": 
                out = np.stack((np.array(stats["miss_ratios_global"]),
                               np.array(stats["miss_ratios_global0"]),
                               np.array(stats["miss_ratios_global1"]),
                               stats["time_core0_alone"],
                                stats["time_core1_alone"],
                                stats["time_core0_together"],
                                stats["time_core1_together"]))
            elif module["type"]=="miss_ratios_global":
                out = np.stack((np.array(stats["miss_ratios_global"]),
                               np.array(stats["miss_ratios_global0"]),
                               np.array(stats["miss_ratios_global1"])))

        else:
            print(f"module {module} unknown")
            exit()
        return np.array(out)
