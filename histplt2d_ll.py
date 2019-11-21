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

main_path = "/media/zezhou/Seagate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/llambda/"
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

################################## Visualization #########################################

###############
# histogram2d #
###############
# Details of the plot data are in function. The unlisted data are directly from datahandle.
##########################
# lambda-lambda
def ecc0_ll():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc0_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc0_1_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc0_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc0_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc0_1_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc0_1_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc0_2_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc0_2_y3y'])
    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc0:' + str(np.mean(x)))
    return x, y
def ecc03_ll():
    # ecc03
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc03_1_y1x'])
    y = np.append(y, -handle1.tot_file_shift['ecc03_1_y1y'])                                # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc03_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_3_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_4_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_5_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_6_y1y'])

    x = np.append(x, handle1.tot_file_shift['ecc03_1_y3x'])
    y = np.append(y, -handle1.tot_file_shift['ecc03_1_y3y'])  # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc03_2_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_2_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_3_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_3_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_4_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_4_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_5_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_5_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc03_6_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc03_6_y3y'])

    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc03:' + str(np.mean(x)))
    return x, y
def ecc06_ll():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc06_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_1_y1y'])  # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc06_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_3_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_4_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_5_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_6_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_7_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_7_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_8_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_8_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_9_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_9_y1y'])

    x = np.append(x, handle1.tot_file_shift['ecc06_1_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_1_y3y'])  # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc06_2_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_2_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_3_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_3_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_4_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_4_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_5_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_5_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_6_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_6_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_7_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_7_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_8_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_8_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_9_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_9_y3y'])
    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc06:' + str(np.mean(x)))
    return x, y
def ecc09_ll():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc09_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_1_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_3_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_4_y1y'])                             # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc09_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_5_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_6_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_7_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_7_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_8_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_8_y1y'])
    # x = np.append(x, -handle1.tot_file_shift['ecc09_9_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_9_y1y'])

    x = np.append(x, handle1.tot_file_shift['ecc09_1_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_1_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_2_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_2_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_3_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_3_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_4_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_4_y3y'])  # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc09_5_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_5_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_6_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_6_y3y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_7_y3x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_7_y3y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_8_y3x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_8_y3y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_9_y3x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_9_y3y'])
    return x, y
def ecc095_ll():
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_1_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_2_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_2_y1y'])

    x1 = np.append(x1, -handle1.tot_file_shift['ecc095_1_y3x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_1_y3y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_2_y3x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_2_y3y'])
    return  x1, y1
def ecc098_ll():
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_1_y1x']) #1
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_1_y1y'])

    x1 = np.append(x1, handle1.tot_file_shift['ecc098_1_y3x'])  # 1
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_1_y3y'])
    # x1 = x1 - np.mean(x1)
    print('tot_pos_overlay_shift ecc098:'+str(np.mean(x1)))
    return x1, y1
def ecc0995_ll():
# ecc0995(T4-plasmid)
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_1_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_2_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_2_y1y'])

    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_1_y3x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_1_y3y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_2_y3x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_2_y3y'])

    return x1, y1

## Cavity overlay
def cav_ecc06():
    a, b = module.ellipse_para(ecc=0)
    n = 1000 # sampling pts
    t = np.linspace(0, 2*np.pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y

def llhist():
    ###Data setup##########
    x0, y0 = ecc0_ll()
    x1, y1 = ecc03_ll()
    x2, y2 = ecc06_ll()
    x3, y3 = ecc09_ll()
    x4, y4 = ecc095_ll()
    x5, y5 = ecc098_ll()
    x6, y6 = ecc0995_ll()
    X = [x0, x1, x2, x3] # Input data array
    Y = [y0, y1, y2, y3]
    # savename = 'Ecc0995_p.eps'


    savename_tot = ['Ecc0_ll.png', 'Ecc03_ll.png', 'Ecc06_ll.png', 'Ecc09_ll.png']
    figtitle_tot = ['Plasmid position distribution Ecc=0', 'Plasmid position distribution Ecc=0.3',
            'Plasmid position distribution Ecc=0.6', 'Plasmid position distribution Ecc=0.9']

    savepath = '/media/zezhou/Se' \
           'agate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/Plots/llambda/hist/'

    xlim_tot = [10, 10, 10, 10, 10, 10, 10]
    scalebar_tot = np.array(xlim_tot)/10*6.25 # Keep length of the scale bar the same

    # Annotation text
    marker_tot = ['Eccentricity=0', 'Eccentricity=0.3', 'Eccentricity=0.6',
              'Eccentricity=0.9']
    scalelabel_tot = ['1um', '1um', '1um', '1um', '1um', '1um', '1um']

    ###
    bins = 80
    cmap = 'YlOrRd'
    fontsize = 10
    labelsize = 7.5
    cb_fontsize = 7.5
    ###############
    a = 6
    fig2 = plt.figure(figsize=[a, a/1.3333]) # figs8ize=[6.4, 4.8]

    #### Do NOT change here. This section changes color ONLY ####
    vmax = 0
    vmin = 1000
    ############################
    for i in range(4):
        x = X[i]
        y = Y[i]
        xlim = xlim_tot[i]

        # Rescaling
        h = np.histogram2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], density=True)
        ## scale the color
        sh = h[0][int(bins/2-10):int(bins/2+10), int(bins/2-4):int(bins/2+4)]
        mi = np.min(sh) # smallest probability of the center square
        ma = np.max(h[0]) #largest
        ####################
        if vmin>mi:
            vmin = mi
        if vmax<ma:
            vmax = ma

    for i in range(4):
        x = X[i]
        y = Y[i]
        xlim = xlim_tot[i]
        figtitle = figtitle_tot[i]
        savename = savename_tot[i]
        # scale = 6.4/4.8
        # ax2 = fig2.add_subplot(3,3,i+1) # Axes location.
        # c = i%3
        # r = i//3
        width = 0.17
        ax2 = fig2.add_axes([0, 0.7-i*width*1.333333, width, width*1.333333])
        # 2d hist

        # individual plot: uncomment this section for individual colorbar plot
        # h, xedges, yedges, img = ax2.hist2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], cmap=cmap, vmin=np.min(sh),
        #                                     vmax=np.max(h[0]), density=True, norm=mcolors.PowerNorm(0.7))

        # Universal plot:
        h, xedges, yedges, img = ax2.hist2d(y, x, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], cmap=cmap,
                                            vmin=vmin, vmax=vmax, density=True, norm=mcolors.PowerNorm(0.7))
        # colorbar: uncomment this section for individual colorbar
        # norm = colors.Normalize(vmin=np.min(h), vmax=np.max(h))
        # cb = fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax2)
        # cb.set_label('Probability', fontsize = fontsize, rotation = -90, horizontalalignment = 'center', verticalalignment = 'bottom')

        # axes label
        ax2.set_xlabel(r'Position($\mu$m)', fontsize = fontsize)
        ax2.set_ylabel(r'Position($\mu$m)', fontsize = fontsize)

        # tick location and tick label
        xtick_temp = np.arange(0, xlim, step = 6.25)
        xtick_temp = np.concatenate((-xtick_temp, xtick_temp))
        xtick_temp = np.sort(xtick_temp)
        xtick_temp = np.delete(xtick_temp, len(xtick_temp)/2 - 1 )
        xticks = list(xtick_temp)
        xticklabel = []
        for ticks in xticks:
            xticklabel.append(str(ticks/6.25))
        ax2.set_xticks(xticks)
        ax2.set_yticks(xticks)

        # Tick label. Uncomment below to show unit in um
        ax2.set_xticklabels(xticklabel, fontsize = labelsize)
        ax2.set_yticklabels(xticklabel, fontsize = labelsize)

        # Axis switch
        ax2.axis('off')
        # Grid
        # ax2.grid(b=True, ls = '--', dash_capstyle='round')

        # Scale bar

        # Cavity overlay
        # ax2.plot(x, y, '-')

        # Title
        # ax2.set_title(figtitle, fontsize = fontsize)

        # text comment
        mark = marker_tot[i]
        ax2.text(x = -xlim*0.9, y=xlim*0.8, s=mark, fontsize = labelsize)

        # scale bar location
        sc = scalebar_tot[i]
        rect = patches.Rectangle((xlim*0.3, -xlim*0.75), width = sc, height=sc/8, fill=True, color = 'black')
        ax2.add_patch(rect)
        # scale bar label
        scalelabel = scalelabel_tot[i]
        ax2.text(x=xlim * 0.5, y=-xlim * 0.95, s=scalelabel, fontsize=labelsize)
        # fig2.suptitle(r'$\lambda$-DNA distribution in different cavities', fontsize=20, x=0.3)

    # Universal colorbar
    cax = fig2.add_axes([0.2, 0.3, 0.025, 0.5])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, orientation = 'vertical')
    cb.ax.tick_params(labelsize=12)
    cb.set_label('Probability', fontsize = cb_fontsize+5 , horizontalalignment = 'center', rotation=-90, labelpad=15)

    os.chdir(savepath)
    plt.show()
    return
def denplt():
    # Density plot. The bottom two figures
    bins = 80
    cmap = 'YlOrRd'
    fontsize = 13
    labelsize = 7.5
    cb_fontsize = 7.5
    ###############
    width = 0.5
    a = 6
    fig2 = plt.figure(figsize=[a, a / 1.3333])  # figs8ize=[6.4, 4.8]
    ax3 = fig2.add_axes([0.15, 0.1, width, width/1.3333])
    ax4 = fig2.add_axes([0.15, 0.1+ width/1.3333+0.2, width, width/1.3333])

    dataset = [ecc0_ll(), ecc03_ll(), ecc06_ll(), ecc09_ll()]
    r_density = []
    r_edge = []
    theta_density = []
    theta_edge = []
    ### Density calculation ###
    for file in dataset:
        x, y = file
        handle1 = module.densitycal(handle1, dataset='position', bins=bins, x=x, y=y, debug=True)
        r_density.append(handle1.tot_density_hist['test_density'])
        r_edge.append(handle1.tot_density_hist['test_edge'])
        theta_density.append(handle1.tot_density_hist['test_degdensity'])
        theta_edge.append(handle1.tot_density_hist['test_degedge'])

    ### Plot1 set-up ###
    legend = ['ecc0', 'ecc0.3', 'ecc0.6', 'ecc0.9']
    ## Counting total data number ##
    n_datapts = []
    for file in dataset:
        x, y = file
        n_datapts.append(len(x))
    ################################

    ## Radius/Theta density plot
    for i in range(len(dataset)):
        ### Overlap -pi-0 with 0-pi.
        tmp = theta_density[i] #current density
        tmp[-1] = tmp[0]# off-set the edge(otherwise the boundry will be zero)
        ind = int(3*len(tmp)/4) # off-set the density to -pi/2--3pi/2
        tmp = np.concatenate((tmp[ind:], tmp[:ind]))
        mid_ind = int(len(tmp)/2) ##prepare reverse
        tmp_swap = tmp[:mid_ind][::-1]+tmp[mid_ind:] ## adding tmp(theta)+tmp(pi-theta)
        ############################
        ax3.plot(theta_edge[i][:mid_ind]+np.pi/2, tmp_swap) ##shift x ticks to -pi/2 to pi/2. 0 is center
        ax4.plot(r_edge[i], r_density[i])

    ax3.legend(legend, loc=1)
    ax4.set_ylabel('Probability density', fontsize=fontsize)
    ax4.set_xlabel('Effective radius', fontsize=fontsize)
    ax3.set_xlabel('Theta(rad)', fontsize=fontsize)
    ax3.set_ylabel('Probability density', fontsize=fontsize)
    return



def swappingplt():
    tot_vector = handle1.tot_vec_overlay_clean
    module.swapcounter(handle1.tot_vec_overlay_clean['ecc09_delx'], bins=8)
    module.swapcounter(handle1.tot_vec_overlay_clean['ecc09_delx'], bins=8)
# test
# a,b = module.ellipse_para(ecc=0.995)
# print('a is:'+str(a))
# print('b is:'+str(b))
# x, y = ecc0995_ll()
# ax5=fig2.add_axes([0.5,0.5,0.5,0.5])
# a = 6.25*a
# b = 6.25*b
# r = np.linspace(0,1,5)
# theta = np.linspace(-np.pi, np.pi, 1000)
# ax5.plot(x, y, '+')
# for i in r:
#     ax5.plot(a*np.cos(theta)*np.sqrt(i), b*np.sin(theta)*np.sqrt(i))

