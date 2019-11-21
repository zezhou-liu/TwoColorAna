import numpy as np
import matplotlib.pyplot as plt
import module
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

## Density plot ##
## We define the "effective radius" as the RHS of elliptical equation. 0<r<1. Therefore.

### Geometry parameters ###
ecc = 0
a = 1
b = 1

### Plot parameters ###
bins = 50
titlefontsize = 20
labelsize = 15
cb_fontsize = 15
ticksize = 15
figuresize = 20

# data set-up
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

### Batch data setup
dataset = [ecc0_tp(), ecc06_tp(), ecc08_tp(), ecc09_tp(), ecc095_tp(), ecc098_tp(), ecc0995_tp()]
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
fig = plt.figure(figsize=[figuresize, figuresize/1.333])
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
legend = ['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9', 'ecc0.95', 'ecc0.98', 'ecc0.995']
## Counting total data number ##
n_datapts = []
for file in dataset:
    x, y = file
    n_datapts.append(len(x))
################################

## Radius/Theta density plot
for i in range(len(dataset)):
    ax1.plot(r_edge[i], r_density[i]/n_datapts[i])
    ### Overlap -pi-0 with 0-pi.
    tmp = theta_density[i]/n_datapts[i] #current density
    tmp[-1] = tmp[0]# off-set the edge(otherwise the boundry will be zero)
    ind = int(3*len(tmp)/4) # off-set the density to -pi/2--3pi/2
    tmp = np.concatenate((tmp[ind:], tmp[:ind]))
    mid_ind = int(len(tmp)/2) ##prepare reverse
    tmp_swap = tmp[:mid_ind][::-1]+tmp[mid_ind:] ## adding tmp(theta)+tmp(pi-theta)
    ############################
    ax2.plot(theta_edge[i][:mid_ind]+np.pi/2, tmp_swap) ##shift x ticks to -pi/2 to pi/2. 0 is center
ax2.legend(legend)


## ecc0.995 region
a,b = module.ellipse_para(ecc=0.995)
print('a is:'+str(a))
print('b is:'+str(b))
x, y = ecc0995_tp()

a = 6.25*a
b = 6.25*b
r = np.linspace(0,1,5)
theta = np.linspace(-np.pi, np.pi, 1000)
ax3.plot(x, y, '+')
for i in r:
    ax3.plot(a*np.cos(theta)*np.sqrt(i), b*np.sin(theta)*np.sqrt(i))
plt.show()