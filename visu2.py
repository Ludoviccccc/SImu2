import numpy as np
import matplotlib.pyplot as plt
def diversity(data:[np.ndarray,np.ndarray],bins:[np.ndarray, np.ndarray]):
    H,_,_ = np.histogram2d(data[0],data[1],bins)
    divers = np.sum(H>0)
    return divers
def comparaison3(content_random, content_imgep = None, name = None, title = None,label_algo = "imgep",num_bank=4,num_row = 2):
    fig, axs = plt.subplots(num_row*num_bank,4, figsize = (28*num_row//2,20), layout='constrained')
    for j in range(num_bank):
        for row in range(num_row):
            bins = np.arange(-1.0,1.0,0.05)
            axs[num_bank*row+j,0].hist(content_imgep["miss_ratios_detailled"][:,row,j] - content_imgep["miss_ratios_core0_detailled"][:,row,j],bins=bins,alpha = .5, label=label_algo)
            axs[num_bank*row+j,0].hist(content_random["miss_ratios_detailled"][:,row,j] - content_random["miss_ratios_core0_detailled"][:,row,j],  bins=bins,alpha = .5, label="random")
            axs[num_bank*row+j,0].set_xlabel(f"ratio[bank{j+1},row{row},(S_0,S_1)] - ratio[bank{j+1},row{row},(S_0,)]")
            axs[num_bank*row+j,0].set_title("row miss hits ratio difference")
            axs[num_bank*row+j,0].legend()
            axs[num_bank*row+j,1].hist(content_imgep["miss_ratios"][:,j] - content_imgep["miss_ratios_core1"][:,j],bins=bins,alpha = .5, label=label_algo)
            axs[num_bank*row+j,1].hist(content_random["miss_ratios"][:,j] - content_random["miss_ratios_core1"][:,j],  bins=bins,alpha = .5, label="random")
            axs[num_bank*row+j,1].set_xlabel(f"ratio[bank{j+1},row{row},(S_0,S_1)] - ratio[bank{j+1},row{row},(,S_1)]")
            axs[num_bank*row+j,1].set_title("row miss hits ratio difference")
            axs[num_bank*row+j,1].legend()

            diversity_ratio_random = diversity([content_random["miss_ratios_core0_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j]], [bins, bins])
            diversity_ratio_imgep = diversity([content_imgep["miss_ratios_core0_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j]], [bins, bins])
            axs[num_bank*row+j,2].scatter(content_imgep["miss_ratios_core0_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j],label=label_algo, alpha = .5)
            axs[num_bank*row+j,2].scatter(content_random["miss_ratios_core0_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j],  label="random", alpha = .5)
            axs[num_bank*row+j,2].set_xlabel("miss ratio alone (S_0,)")
            axs[num_bank*row+j,2].set_ylabel("(S_0,S_1)")
            axs[num_bank*row+j,2].axline(xy1=(0, 0), slope=1, color='r', lw=2)
            axs[num_bank*row+j,2].set_title(f"bank {j+1}, row {row}, imgep:{diversity_ratio_imgep}, rand:{diversity_ratio_random}")
            axs[num_bank*row+j,2].legend()
            axs[num_bank*row+j,2].set_xticks(np.linspace(0,1,11))
            axs[num_bank*row+j,2].set_yticks(np.linspace(0,1,11))
            axs[num_bank*row+j,2].grid()


            diversity_ratio_random = diversity([content_random["miss_ratios_core1_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j]], [bins, bins])
            diversity_ratio_imgep = diversity([content_imgep["miss_ratios_core1_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j]], [bins, bins])
            axs[num_bank*row + j,3].scatter(content_imgep["miss_ratios_core1_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j],label=label_algo, alpha=.5)
            axs[num_bank*row + j,3].scatter(content_random["miss_ratios_core1_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j],  label="random", alpha=.5)
            axs[num_bank*row + j,3].set_xlabel("miss ratio alone (,S_1)")
            axs[num_bank*row + j,3].set_ylabel("(S_0,S_1)")
            axs[num_bank*row + j,3].axline(xy1=(0, 0), slope=1, color='r', lw=2)
            axs[num_bank*row + j,3].set_title(f"bank {j+1}, row {row}, imgep:{diversity_ratio_imgep}, rand:{diversity_ratio_random}")
            axs[num_bank*row + j,3].legend()
            axs[num_bank*row + j,3].set_xticks(np.linspace(0,1,11))
            axs[num_bank*row + j,3].set_yticks(np.linspace(0,1,11))
            axs[num_bank*row + j,3].grid()
    if title:
        fig.suptitle(title[0],fontsize = 20)
    if name:
        plt.savefig(name[0])
    plt.close()

    #bins = np.linspace(0,1000,21)
    bins = np.arange(0,max(np.max(content_imgep["time_core0_alone"]),np.max(content_imgep["time_core0_together"])),50)
    diversity_time_rand = diversity([content_random["time_core0_alone"],content_random["time_core0_together"]], [bins, bins])
    diversity_time_imgep = diversity([content_imgep["time_core0_alone"],content_imgep["time_core0_together"]], [bins, bins])

    fig, axs = plt.subplots(3,2, figsize = (15,10), layout='constrained')
    axs[0,0].scatter(content_imgep["time_core0_alone"],content_imgep["time_core0_together"], label="imgep", alpha = .5)
    axs[0,0].scatter(content_random["time_core0_alone"],content_random["time_core0_together"], label="random", alpha = .5)
    axs[0,0].axline(xy1=(0, 0), slope=1, color='r', lw=2)
    axs[0,0].set_xlabel("time_core0_alone")
    axs[0,0].set_ylabel("time_core0_together")
    axs[0,0].legend()
    axs[0,0].set_xticks(bins)
    axs[0,0].set_yticks(bins)
    axs[0,0].grid()
    axs[0,0].set_title(f"imgep:{diversity_time_imgep}, rand:{diversity_time_rand}")

    bins = np.arange(1,max(np.max(content_imgep["time_core1_alone"]),np.max(content_imgep["time_core1_together"])),50)

    diversity_time_rand = diversity([content_random["time_core1_alone"],content_random["time_core1_together"]], [bins, bins])
    diversity_time_imgep = diversity([content_imgep["time_core1_alone"],content_imgep["time_core1_together"]], [bins, bins])

    axs[0,1].scatter(content_imgep["time_core1_alone"],content_imgep["time_core1_together"], alpha = .5, label="imgep")
    axs[0,1].scatter(content_random["time_core1_alone"],content_random["time_core1_together"], alpha = .5, label="random")
    axs[0,1].axline(xy1=(0, 0), slope=1, color='r', lw=2)
    axs[0,1].set_xlabel("time_core1_alone")
    axs[0,1].set_ylabel("time_core1_together")
    axs[0,1].legend()
    axs[0,1].set_xticks(bins)
    axs[0,1].set_yticks(bins)
    axs[0,1].grid()
    axs[0,1].set_title(f"imgep:{diversity_time_imgep}, rand:{diversity_time_rand}")


    bins_hist= np.linspace(-200,1000,25)
    bins = np.linspace(0,1000,21)
    axs[1,0].hist(content_imgep["time_core0_together"] - content_imgep["time_core0_alone"], bins=bins_hist,alpha=.5, label="imgep")
    axs[1,0].hist(content_random["time_core0_together"] - content_random["time_core0_alone"], bins=bins_hist,alpha=.5, label="random")
    axs[1,0].set_xlabel("time[together] - time[alone]")
    axs[1,0].legend()


    axs[1,1].hist(content_imgep["time_core1_together"]-content_imgep["time_core1_alone"],   bins=bins_hist,alpha = .5, label="imgep")
    axs[1,1].hist(content_random["time_core1_together"]-content_random["time_core1_alone"], bins=bins_hist,alpha = .5, label="random")
    axs[1,1].set_xlabel("time[together] - time[alone]")
    axs[1,1].legend()




    
    bins = np.arange(0,max(np.max(content_imgep["time_core0_together"]),np.max(content_imgep["time_core1_together"])),50)
    diversity_time_rand = diversity([content_random["time_core0_together"],content_random["time_core1_together"]], [bins, bins])
    diversity_time_imgep = diversity([content_imgep["time_core0_together"],content_imgep["time_core1_together"]],[bins, bins])
    axs[2,0].scatter(content_imgep["time_core0_together"],content_imgep["time_core1_together"], label="imgep", alpha = .5)
    axs[2,0].scatter(content_random["time_core0_together"],content_random["time_core1_together"], label="random", alpha=.5)
    axs[2,0].set_xlabel("time_core0_together")
    axs[2,0].set_ylabel("time_core1_together")
    axs[2,0].legend()
    axs[2,0].set_xticks(bins)
    axs[2,0].set_yticks(bins)
    axs[2,0].grid()
    axs[2,0].set_title(f"imgep:{diversity_time_imgep}, rand:{diversity_time_rand}")
    if title:
        fig.suptitle(title[1],fontsize = 20)

    if name:
        plt.savefig(name[1])
    plt.close()


def comparaison_ratios_iterations(contents:list[tuple], name = None,k = None):
    plt.figure()
    fig, axs = plt.subplots(8,1, figsize = (25,20), layout='constrained')

    bins = np.arange(-1.0,1.0,0.05)
    for j in range(4):
        for row in range(2):
            for label, content in contents:
                ll = len(content["miss_ratios_core0_detailled"])
                diversity_ratio = [diversity([content["miss_ratios_core0_detailled"][:k,row,j],  content["miss_ratios_detailled"][:k,row,j]], [bins, bins]) for k in range(0,ll+1,100)]
                axs[j+row*4].plot(range(0,ll+1,100),diversity_ratio, label=label)
                axs[j+row*4].set_xlabel("iteration",fontsize=18)
                axs[j+row*4].set_ylabel("diversity",fontsize=18)
                axs[j+row*4].legend()
                axs[j+row*4].set_title(f"Mutual Vs Isolation bank {j},row {row}", fontsize=20)
    if name:
        plt.savefig(name)
    plt.close()

        #plt.show()
def comparaison_ratios_global_iterations(list_args:list[tuple], name = None,k = None):
    plt.figure(figsize = (25,20))
    bins = np.arange(-1.0,1.0,0.05)
    for label, content in list_args:
        ll = len(content["miss_ratios_global0"])
        print("together",len(content["miss_ratios_global0"]))
        print("0 alone",len(content["miss_ratios_global"]))
        diversity_ratio = [diversity([content["miss_ratios_global0"][:k],  content["miss_ratios_global"][:k]], [bins, bins]) for k in range(0,ll+1,100)]
        #print("k", k)
        print("name", name)
        plt.plot(range(0,ll+1,100),diversity_ratio, label=label)
        plt.xlabel("iteration",fontsize=18)
        plt.ylabel("diversity",fontsize=18)
        plt.legend()
        plt.title(f"Global miss ratio diversity, Mutual Vs Isolation", fontsize=20)
    if name:
        plt.savefig(name)
    plt.close()
    
