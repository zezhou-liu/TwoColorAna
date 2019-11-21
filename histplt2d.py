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

main_path = "/media/zezhou/Seagate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/datafterlinearshift/tplasmid"
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
    handle1, tot_file_shift = module.bashshift(handle1) # Shift data to zero according to YOYO-3 channel
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
# T4-Plasmid
def ecc0_tp():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc0_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc0_1_y1y'])
    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc0:' + str(np.mean(x)))
    return x, y
def ecc06_tp():
    # ecc03
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc06_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_1_y1y'])                                # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc06_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_3_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_4_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_5_y1y'])
    x = np.append(x, -handle1.tot_file_shift['ecc06_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_6_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc06_7_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc06_7_y1y'])
    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc06:' + str(np.mean(x)))
    return x, y
def ecc08_tp():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc08_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_1_y1y'])  # Plot detail. DO NOT delete.!!!!
    x = np.append(x, handle1.tot_file_shift['ecc08_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_3_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_4_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_5_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_6_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_7_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_7_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc08_8_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc08_8_y1y'])
    # x = x - np.mean(x)
    print('tot_pos_overlay_shift ecc08:' + str(np.mean(x)))
    return x, y
def ecc09_tp():
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_file_shift['ecc09_1_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_1_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_3_y1y'])
    x = np.append(x, -handle1.tot_file_shift['ecc09_4_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_4_y1y'])                             # Plot detail. DO NOT delete.!!!!
    x = np.append(x, -handle1.tot_file_shift['ecc09_5_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_5_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_6_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_6_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_7_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_7_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc09_8_y1x'][:2000])
    y = np.append(y, handle1.tot_file_shift['ecc09_8_y1y'][:2000])
    x = np.append(x, -handle1.tot_file_shift['ecc09_8_y1x'][-2000:])
    y = np.append(y, handle1.tot_file_shift['ecc09_8_y1y'][-2000:])
    x = np.append(x, handle1.tot_file_shift['ecc09_9_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc09_9_y1y'])
    return x, y
def ecc095_tp():
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_1_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_2_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_2_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_3_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_3_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_4_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_4_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_5_y1x'][:3000])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_5_y1y'][:3000])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc095_5_y1x'][3000:])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_5_y1y'][3000:])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_6_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_6_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_7_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_7_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_8_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_8_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_9_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_9_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_10_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_10_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_11_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_11_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_12_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_12_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_13_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_13_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_14_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_14_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc095_15_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc095_15_y1y'])
    return  x1, y1
def ecc098_tp():
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_1_y1x']) #1
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_2_y1x']) #1
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_2_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_3_y1x']) #1
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_3_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_4_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_4_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_5_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_5_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_6_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_6_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_8_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_8_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_9_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_9_y1y'])
    # x1 = x1 - np.mean(x1)
    print('tot_pos_overlay_shift ecc098:'+str(np.mean(x1)))
    return x1, y1
def ecc0995_tp():
# ecc0995(T4-plasmid)
    x1 = np.array([])
    y1 = np.array([])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_1_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_2_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_2_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_3_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_3_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_4_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_4_y1y'])
    # x1 = np.append(x1, handle1.tot_file_shift['ecc0995_5_y1x']) # This set goes to the end. TODO: Check original video. Why is the alignment different?
    # y1 = np.append(y1, handle1.tot_file_shift['ecc0995_5_y1y'])
    # x1 = np.append(x1, handle1.tot_file_shift['ecc0995_6_y1x'])
    # y1 = np.append(y1, handle1.tot_file_shift['ecc0995_6_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_7_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_7_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_8_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_8_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_9_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_9_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_10_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_10_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_11_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_11_y1y'])
    x1 = np.append(x1, -handle1.tot_file_shift['ecc0995_12_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_12_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_13_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_13_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_14_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_14_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_15_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_15_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc0995_16_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc0995_16_y1y'])
    return x1, y1

##
def ecc0995_lp():
# ecc0995(lambda-plasmid)
    x = np.array([])
    y = np.array([])
    x = np.append(x, handle1.tot_pos_overlay_shift['ecc0995_y1x'])
    y = np.append(y, handle1.tot_pos_overlay_shift['ecc0995_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc0995_1_y1x']-4.5)
    y = np.append(y, handle1.tot_file_shift['ecc0995_1_y1y'])
    x = np.append(x, -handle1.tot_file_shift['ecc0995_2_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc0995_2_y1y'])
    x = np.append(x, handle1.tot_file_shift['ecc0995_3_y1x'])
    y = np.append(y, handle1.tot_file_shift['ecc0995_3_y1y'])
    return x, y

## Cavity overlay
def cav_ecc06():
    a, b = module.ellipse_para(ecc=0)
    n = 1000 # sampling pts
    t = np.linspace(0, 2*np.pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y

###Data setup##########
x0, y0 = ecc0_tp()
x1, y1 = ecc06_tp()
x2, y2 = ecc08_tp()
x3, y3 = ecc09_tp()
x4, y4 = ecc095_tp()
x5, y5 = ecc098_tp()
x6, y6 = ecc0995_tp()
X = [x0, x1, x2, x3, x4, x5, x6] # Input data array
Y = [y0, y1, y2, y3, y4, y5, y6]
# savename = 'Ecc0995_p.eps'
savename_tot = ['Ecc0_p.png', 'Ecc06_p.png', 'Ecc08_p.png', 'Ecc09_p.png',
            'Ecc095_p.png', 'Ecc098_p.png', 'Ecc0995_p.png']
figtitle_tot = ['Plasmid position distribution Ecc=0', 'Plasmid position distribution Ecc=0.6',
            'Plasmid position distribution Ecc=0.8', 'Plasmid position distribution Ecc=0.9',
            'Plasmid position distribution Ecc=0.95', 'Plasmid position distribution Ecc=0.98',
            'Plasmid position distribution Ecc=0.995']

savepath = '/media/zezhou/Se' \
           'agate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/Plots/tplasmid/hist/'
xlim_tot = [10, 10, 10, 12, 12, 16, 16]
scalebar_tot = np.array(xlim_tot)/10*6.25 # Keep length of the scale bar the same

# Annotation text
marker_tot = ['Eccentricity=0', 'Eccentricity=0.6', 'Eccentricity=0.8',
              'Eccentricity=0.9', 'Eccentricity=0.95', 'Eccentricity=0.98', 'Eccentricity=0.995']
scalelabel_tot = ['1um', '1um', '1um', '1.2um', '1.2um', '1.6um', '1.6um']

###
bins = 80
cmap = 'YlOrRd'
fontsize = 20
labelsize = 15
cb_fontsize = 15
###############
a = 12
fig2 = plt.figure(figsize=[a, a/1.3333]) # figs8ize=[6.4, 4.8]

#### Do NOT change here. This section changes color ONLY ####
vmax = 0
vmin = 1000
############################
for i in range(7):
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
for i in range(7):
    x = X[i]
    y = Y[i]
    xlim = xlim_tot[i]
    figtitle = figtitle_tot[i]
    savename = savename_tot[i]
    # scale = 6.4/4.8
    ax2 = fig2.add_subplot(3,3,i+1) # Axes location. Right now it's an easy going version.
    # 2d hist

    # individual plot: uncomment this section for individual colorbar plot
    # h, xedges, yedges, img = ax2.hist2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], cmap=cmap, vmin=np.min(sh),
    #                                     vmax=np.max(h[0]), density=True, norm=mcolors.PowerNorm(0.7))

    # Universal plot:
    h, xedges, yedges, img = ax2.hist2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], cmap=cmap,
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
fig2.suptitle('Plasmid distribution in different cavities', fontsize=20)

# Universal colorbar
cax = fig2.add_axes([0.4, 0.2, 0.5, 0.05])
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cb = fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, orientation = 'horizontal')
cb.set_label('Probability', fontsize = cb_fontsize , horizontalalignment = 'center')

os.chdir(savepath)
# plt.savefig('histall_uni.png')
plt.show()
