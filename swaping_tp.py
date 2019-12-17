import numpy as np
import matplotlib.pyplot as plt
import module
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import json
import os
import matplotlib.colors as mcolors
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

main_path = "/media/zezhou/Seagate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/datafterlinearshift/tplasmid/"
cleanmode = 1 # 1-cleaned data. The roi.json and tot_file_clean.json files have been saved in ./data folder.
              # 0-raw data. Experimental data before shift to zero.
# Read files
handle1, tot_file = module.bashload(main_path)
handle1, tot_vector = module.bashvector(handle1)
handle1, tot_vec_overlay = module.bashoverlay(handle1)

# Data clean
if cleanmode == 0:
    handle1, roi = module.bashroi(handle1) # ROI selection
    handle1, tot_file_clean = module.bashclean(handle1) # Delete points ouside ROI and attach mask to handle1.
    handle1, tot_file_shift = module.allshift(handle1) # Shift data to zero according to YOYO-3 channel
elif cleanmode == 1:
    os.chdir(main_path+'/data')
    tot_file_clean = json.load(open('tot_file_clean.json')) # data is saved in list format
    for filename in tot_file_clean:
        tot_file_clean[filename] = np.array(tot_file_clean[filename]) # Changing format to array
    handle1.tot_file_shift = tot_file_clean

# Cleaned data re-calculate
handle1, tot_vector_clean = module.bashvector(handle1, mode='clean')
handle1, tot_vec_overlay_clean = module.bashoverlay(handle1, mode='clean')
handle1, tot_pos_overlay_shift = module.bashoverlay(handle1, mode='clean', set='position')

def swappingplt():
    dataset = ['ecc0', 'ecc06','ecc08','ecc09']
    tot_vector = handle1.tot_vec_overlay_clean
    a = 6.4 # figure size
    # Create a figure and axis
    fig = plt.figure(figsize=[a, a/1.3333333])

    ind = 0
    bins = [6, 7, 6, 6]
    maxrange = [300, 350, 350, 1000]
    maxtime = np.array(maxrange)/17
    legend = ['Ecc=0', 'Ecc=0.6', 'Ecc=0.8', 'Ecc=0.9']
    width = np.array(maxrange)/np.array(bins)*0.9 # bar width
    tot_fit = []
    for item in dataset:
        r = ind//2
        c = ind%2
        ax = fig.add_axes([0.1+c*0.4, 0.6-r*0.4, 0.3, 0.3])
        data = tot_vector[item+'_delx']
        state, time = module.swapcounter(data)
        hist, bin_edge = np.histogram(time, bins=bins[ind], range = [0, maxrange[ind]])
        ax.bar(bin_edge[:-1], hist, align = 'center', width=width[ind], color = 'r', edgecolor = 'k' )

        # Exp fitting
        if item == 'ecc0':
            fitx_x = bin_edge[:4]
            fit_para = np.polyfit(fitx_x[:4], np.log(hist[:4]), deg=int(1)) # TODO: confinm. Zero has an effect on fitting.
        elif item == 'ecc06':
            fitx_x = bin_edge[:4]
            fit_para = np.polyfit(fitx_x[:4], np.log(hist[:4]), deg=int(1))
        elif item == 'ecc09':
            fitx_x = bin_edge[:4]
            fit_para = np.polyfit(fitx_x[:4], np.log(hist[:4]), deg=int(1))
            ax.set_ylim(top=30)
        else:
            fitx_x = bin_edge[:-1]
            fit_para = np.polyfit(fitx_x, np.log(hist), deg=int(1))
        tot_fit.append(fit_para)
        # plot exponential fitting
        line_fit = ax.plot(fitx_x, np.exp(fit_para[0] * fitx_x + fit_para[1]), '--', color='k')

        ax.legend(['fitting', legend[ind]])
        xticklabel = np.arange(0,maxtime[ind],np.floor(maxtime[ind]/3))
        xtick = xticklabel*17
        ax.set_xticks(xtick)
        ax.set_xticklabels(xticklabel)
        ax.set_yscale('log')
        ax.set_xlabel('Time(s)', fontsize=13)
        ax.set_ylabel('Counts', fontsize=13)
        ind += 1

    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.2,0.2,0.6,0.6])
    time = []
    for item in tot_fit:
        time.append(-1/item[0]/17)
    x = [0,1,2,3]
    y = [0,10,20]
    ax.plot(x, time, 's', ms=13)
    ax.set_xlabel('Eccentricity', fontsize=25)
    ax.set_ylabel(r'$\tau$(s)', fontsize=25)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(['0', '0.3', '0.6', '0.9'])
    ax.tick_params(labelsize=20)
    plt.show()

swappingplt()