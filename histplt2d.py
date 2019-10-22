import numpy as np
import matplotlib.pyplot as plt
import module
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import json
import os

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
    return handle1.tot_pos_overlay_shift['ecc0_y1x'], handle1.tot_pos_overlay_shift['ecc0_y1y']
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
    return x, y
def ecc08_tp():
    return handle1.tot_pos_overlay_shift['ecc08_y1x'], handle1.tot_pos_overlay_shift['ecc08_y1y']
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
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_1_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_1_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_2_y1x'])
    y1 = np.append(y1, handle1.tot_file_shift['ecc098_2_y1y'])
    x1 = np.append(x1, handle1.tot_file_shift['ecc098_3_y1x'])
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

###Data setup####
x, y = ecc0995_tp()
xlim = 20
bins = 100
cmap = 'YlOrRd'
###############


# scale = 6.4/4.8
fig2 = plt.figure() # figsize=[6.4, 4.8]
ax2 = fig2.add_axes([0.15,0.15,0.7,0.7]) # Axes location

# Rescaling
h = np.histogram2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], density=True)
## scale the color
sh = h[0][int(bins/2-10):int(bins/2+10), int(bins/2-10):int(bins/2+10)]
####################

# 2d hist
h, xedges, yedges, img = ax2.hist2d(x, y, bins=[bins, bins], range=[[-xlim, xlim], [-xlim, xlim]], cmap=cmap, vmin=np.min(sh), vmax=np.max(h[0]), density=True)
norm = colors.Normalize(vmin = np.min(h), vmax = np.max(h))
# colorbar
cb = fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax2)
cb.set_label('Probability', fontsize = 15, rotation = -90, horizontalalignment = 'center', verticalalignment = 'bottom')

# axes label
ax2.set_xlabel(r'Position($\mu$m)', fontsize = 15)
ax2.set_ylabel(r'Position($\mu$m)', fontsize = 15)

# tick location and tick label
xtick_temp = np.arange(0, xlim + 1, step = 6.25/2)
xtick_temp = np.concatenate((-xtick_temp, xtick_temp))
xtick_temp = np.sort(xtick_temp)
xtick_temp = np.delete(xtick_temp, len(xtick_temp)/2 - 1 )
xticks = list(xtick_temp)
xticklabel = []
for ticks in xticks:
    xticklabel.append(str(ticks/6.25))
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabel, fontsize = 10)
ax2.set_yticks(xticks)
ax2.set_yticklabels(xticklabel, fontsize = 10)

# Title
ax2.set_title('T4 position distribution. Ecc=0.995')

plt.show()