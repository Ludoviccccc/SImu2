import matplotlib.pyplot as plt
import numpy as np
import os
def diversity(data:[np.ndarray,np.ndarray],bins:[np.ndarray, np.ndarray]):
    H,_,_ = np.histogram2d(data[0],data[1],bins)
    divers = np.sum(H>0)
    return divers
def comparaison(content_random,contents,name:str=None):
    bins_hist= np.linspace(-200,1000,25)
    fig = plt.figure(figsize = (18,12))#, layout='constrained')

    bins = np.arange(0,3000,50)
    diversity_time_imgep = {"core0":[diversity([content_imgep[1]["time_core0_alone"],content_imgep[1]["time_core0_together"]],[bins, bins]) for content_imgep in contents]}

    diversity_time_imgep["core1"] = [diversity([content_imgep[1]["time_core1_alone"],content_imgep[1]["time_core1_together"]],[bins, bins]) for content_imgep in contents]

    diversity_time_rand = {}
    diversity_time_rand['core0'] = diversity([content_random["time_core0_alone"],content_random["time_core0_together"]], [bins, bins])
    diversity_time_rand['core1'] = diversity([content_random["time_core1_alone"],content_random["time_core1_together"]], [bins, bins])

    diversity_time_rand_together = diversity([content_random["time_core0_together"],content_random["time_core1_together"]], [bins, bins])

    ax1 = plt.subplot(321)
    ax1.axline(xy1=(0, 0), slope=1, color='r', lw=2)
    ax1.set_xlabel("time core 0 alone",fontsize=14)
    ax1.set_ylabel("time core 0 mutual", fontsize=14)
    ax1.set_xticks(bins,minor=True)
    ax1.set_yticks(bins,minor=True)
    ax1.grid(which='minor')

    ax1.set_title(f"{contents[0][0]}:{diversity_time_imgep['core0'][0]}, {contents[1][0]}:{diversity_time_imgep['core0'][1]},rand:{diversity_time_rand['core0']}",fontsize = 18)

    ax2 = plt.subplot(322)
    ax2.axline(xy1=(0, 0), slope=1, color='r', lw=2)
    ax2.set_xlabel("time core 1 alone",fontsize=14)
    ax2.set_ylabel("time core 1 mutual",fontsize=14)
    ax2.set_xticks(bins,minor=True)
    ax2.set_yticks(bins,minor=True)
    ax2.grid(which='minor')
    ax2.set_title(f"{contents[0][0]}:{diversity_time_imgep['core1'][0]}, {contents[1][0]}:{diversity_time_imgep['core1'][1]},rand:{diversity_time_rand['core1']}",fontsize=18)
    ax3 = plt.subplot(323)
    ax3.set_xlabel("time[mutual] - time[alone]",fontsize=14)

    ax4 = plt.subplot(324)
    ax4.set_xlabel("time[mutual] - time[alone]",fontsize=14)

    ax5 = plt.subplot(313)
    ax5.set_xlabel("time core 0 mutual",fontsize=14)
    ax5.set_ylabel("time core 1 mutual",fontsize=14)
    bins = np.arange(1,max(np.max(contents[0][1]["time_core1_alone"]), np.max(contents[1][1]["time_core1_alone"]),np.max(contents[0][1]["time_core1_together"]),np.max(contents[1][1]["time_core1_together"])),50)

    diversity_time_imgeps = [diversity([content_imgep[1]["time_core0_together"],content_imgep[1]["time_core1_together"]], [bins, bins]) for content_imgep in contents]
    ax5.set_xticks(bins,minor=True)
    ax5.set_yticks(bins,minor=True)
    ax5.grid(which='minor')
    ax5.set_title(f"no ir:{diversity_time_imgeps[0]}, svd:{diversity_time_imgeps[1]},rand:{diversity_time_rand_together}",fontsize=18)

    for j,content in enumerate(contents):
        ax1.scatter(content[1]["time_core0_alone"],content[1]["time_core0_together"], label=content[0], alpha = .5)
        ax2.scatter(content[1]["time_core1_alone"],content[1]["time_core1_together"], alpha = .5, label= content[0])
        ax3.hist(content[1]["time_core0_together"] - content[1]["time_core0_alone"], bins=bins_hist,alpha=.5, label=content[0])
        ax4.hist(content[1]["time_core1_together"] - content[1]["time_core1_alone"], bins=bins_hist,alpha=.5, label=content[0])
        ax5.scatter(content[1]["time_core0_together"],content[1]["time_core1_together"], label=content[0], alpha = .5)

    ax1.scatter(content_random["time_core0_alone"],content_random["time_core0_together"], label="random", alpha = .5)
    ax2.scatter(content_random["time_core1_alone"],content_random["time_core1_together"], alpha = .5, label="random")
    ax3.hist(content_random["time_core0_together"] - content_random["time_core0_alone"], bins=bins_hist,alpha=.5, label="random")
    ax4.hist(content_random["time_core1_together"]-content_random["time_core1_alone"], bins=bins_hist,alpha = .5, label="random")
    ax5.scatter(content_random["time_core0_together"],content_random["time_core1_together"], label="random", alpha=.5)
    ax1.legend(fontsize=15)
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()

    if name:
        k = 0
        while os.path.isfile(f"{name}_{k}.png"):
            k+=1
        plt.savefig(f"{name}_{k}.png",bbox_inches = 'tight',pad_inches = 0)
    plt.close()
    plt.show()
#def ratio_plot(content_random, contents,name:str=None):
#    fig, axs = plt.subplots(num_row*num_bank,4, figsize = (28*num_row//2,20), layout='constrained')
#    for j in range(num_bank):
#        for row in range(num_row):
#            bins = np.arange(-1.0,1.0,0.05)
#            axs[num_bank*row+j,0].hist(content_random["miss_ratios_detailled"][:,row,j] - content_random["miss_ratios_core0_detailled"][:,row,j],  bins=bins,alpha = .5, label="random")
#            axs[num_bank*row+j,0].set_xlabel(f"ratio[bank{j+1},row{row},(S_0,S_1)] - ratio[bank{j+1},row{row},(S_0,)]")
#            axs[num_bank*row+j,0].set_title("row miss hits ratio difference")
#            axs[num_bank*row+j,0].legend()
#
#            axs[num_bank*row+j,1].hist(content_random["miss_ratios_detailled"][:,row,j] - content_random["miss_ratios_core1_detailled"][:,row,j],  bins=bins,alpha = .5, label="random")
#            axs[num_bank*row+j,1].set_xlabel(f"ratio[bank{j+1},row{row},(S_0,S_1)] - ratio[bank{j+1},row{row},(,S_1)]")
#            axs[num_bank*row+j,1].set_title("row miss hits ratio difference")
#            axs[num_bank*row+j,1].legend()
#
#            diversity_ratio_random = diversity([content_random["miss_ratios_core0_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j]], [bins, bins])
#            axs[num_bank*row+j,2].scatter(content_random["miss_ratios_core0_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j],  label="random", alpha = .5)
#            axs[num_bank*row+j,2].set_xlabel("miss ratio alone (S_0,)")
#            axs[num_bank*row+j,2].set_ylabel("(S_0,S_1)")
#            axs[num_bank*row+j,2].axline(xy1=(0, 0), slope=1, color='r', lw=2)
#            axs[num_bank*row+j,2].set_title(f"bank {j+1}, row {row}, imgep:{diversity_ratio_imgep}, rand:{diversity_ratio_random}")
#            axs[num_bank*row+j,2].legend()
#            axs[num_bank*row+j,2].set_xticks(np.linspace(0,1,11))
#            axs[num_bank*row+j,2].set_yticks(np.linspace(0,1,11))
#            axs[num_bank*row+j,2].grid()
#
#
#            diversity_ratio_random = diversity([content_random["miss_ratios_core1_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j]], [bins, bins])
#            #diversity_ratio_imgep = diversity([content_imgep["miss_ratios_core1_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j]], [bins, bins])
#            axs[num_bank*row + j,3].scatter(content_random["miss_ratios_core1_detailled"][:,row,j],  content_random["miss_ratios_detailled"][:,row,j],  label="random", alpha=.5)
#            axs[num_bank*row + j,3].set_xlabel("miss ratio alone (,S_1)")
#            axs[num_bank*row + j,3].set_ylabel("(S_0,S_1)")
#            axs[num_bank*row + j,3].axline(xy1=(0, 0), slope=1, color='r', lw=2)
#            axs[num_bank*row + j,3].set_title(f"bank {j+1}, row {row}, imgep:{diversity_ratio_imgep}, rand:{diversity_ratio_random}")
#            axs[num_bank*row + j,3].legend()
#            axs[num_bank*row + j,3].set_xticks(np.linspace(0,1,11))
#            axs[num_bank*row + j,3].set_yticks(np.linspace(0,1,11))
#            axs[num_bank*row + j,3].grid()
#
#            for content in contents:
#                diversity
#                axs[num_bank*row+j,0].hist(content[1]["miss_ratios_detailled"][:,row,j] - content[1]["miss_ratios_core0_detailled"][:,row,j],  bins=bins,alpha = .5, label=content[0])
#                axs[num_bank*row+j,1].hist(content[1]["miss_ratios_detailled"][:,row,j] - content[1]["miss_ratios_core1_detailled"][:,row,j],  bins=bins,alpha = .5, label=content[0])
#                axs[num_bank*row+j,2].scatter(content[1]["miss_ratios_core0_detailled"][:,row,j],  content[1]["miss_ratios_detailled"][:,row,j], alpha = .5,  label=content[0])
#                axs[num_bank*row + j,3].scatter(content[1]["miss_ratios_core1_detailled"][:,row,j],  content[1]["miss_ratios_detailled"][:,row,j], alpha=.5, label=content[0])
#                diversity_ratio_imgep = diversity([content_imgep["miss_ratios_core0_detailled"][:,row,j],  content_imgep["miss_ratios_detailled"][:,row,j]], [bins, bins])
#    if name:
#        k = 0
#        while os.path.isfile(f"{name}_{k}.png"):
#            k+=1
#        plt.savefig(f"{name}_{k}.png",bbox_inches = 'tight',pad_inches = 0)
#    plt.close()
#    plt.show()
