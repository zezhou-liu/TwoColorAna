import numpy as np
import matplotlib.pyplot as plt
import module
import matplotlib.cm as cm
import matplotlib.colors as colors
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
    tot_file_clean = json.load(open('tot_file_clean.json'))
    for filename in tot_file_clean:
        tot_file_clean[filename] = np.array(tot_file_clean[filename])
    handle1.tot_file_shift = tot_file_clean

# Cleaned data re-calculate
handle1, tot_vector_clean = module.bashvector(handle1, mode='clean')
handle1, tot_vec_overlay_clean = module.bashoverlay(handle1, mode='clean')
handle1, tot_pos_overlay_shift = module.bashoverlay(handle1, mode='clean', set='position')

# Save data

# Outdated visulization script. Please refer Visulization.py script for ploting.
def oldvisul():
    # handle1 = module.densitycal(handle1, bins = 10, ecc=0)

    # Visualization

    #################
    # Position distribution plot #
    #################
    # xlim = 12
    # fig = plt.figure() # figsize=[6.4, 4.8]
    # scale = 6.4/4.8
    # ax = fig.add_axes([0.2,0.2,0.6,0.6])
    # ax.plot(handle1.tot_pos_overlay_shift['ecc09_y1x'], handle1.tot_pos_overlay_shift['ecc09_y1y'], '+')
    # ax.plot(handle1.tot_pos_overlay_shift['ecc09_y3x'], handle1.tot_pos_overlay_shift['ecc09_y3y'], '+')
    # ax.legend(['Plasmid', 'T4'])
    # ax.set_title('DNA position distribution confined in ecc09 cavity', fontsize = 15)
    # ax.set_xlim([-xlim, xlim])
    # ax.set_ylim([-xlim/scale, xlim/scale])
    #
    # xtick_temp = 2*np.arange(-0.5*xlim, 0.5*xlim + 1)
    # xticks = list(xtick_temp)
    # xticklabel = []
    # for ticks in xticks:
    #     xticklabel.append(str(ticks/6.25))
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabel, fontsize = 10)
    #
    # ytick_temp = 2*np.arange(np.floor(-0.5*xlim/scale), np.ceil(0.5*xlim/scale))
    # yticks = list(ytick_temp)
    # yticklabel = []
    # for ticks in yticks:
    #     yticklabel.append(str(ticks/6.25))
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabel, fontsize = 10)
    #
    # ax.set_xlabel(r"X-position($\mu$m)", fontsize = 15)
    # ax.set_ylabel(r"Y-position($\mu$m)", fontsize = 15)

    ###############
    # histogram2d #
    ###############
    # ecc09(t4-plasmid)
    # x = handle1.tot_file_shift['ecc09_1_y1x']
    # y = handle1.tot_file_shift['ecc09_1_y1y']
    # x = np.append(x, handle1.tot_file_shift['ecc09_2_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_2_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_3_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_3_y1y'])
    # x = np.append(x, -handle1.tot_file_shift['ecc09_4_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_4_y1y'])                             # Plot detail. DO NOT delete.!!!!
    # x = np.append(x, -handle1.tot_file_shift['ecc09_5_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_5_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_6_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_6_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_7_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_7_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc09_8_y1x'][:2000])
    # y = np.append(y, handle1.tot_file_shift['ecc09_8_y1y'][:2000])
    # x = np.append(x, -handle1.tot_file_shift['ecc09_8_y1x'][-2000:])
    # y = np.append(y, handle1.tot_file_shift['ecc09_8_y1y'][-2000:])
    # x = np.append(x, handle1.tot_file_shift['ecc09_9_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc09_9_y1y'])

    # handle1 = module.densitycal(handle1, dataset='position',x=x, y=y, ecc=0.9, bins=50, debug="True")

    # ecc03
    # x = np.array([])
    # y = np.array([])
    # # x = np.append(x, handle1.tot_file_shift['ecc03_1_y1x'])
    # # y = np.append(y, handle1.tot_file_shift['ecc03_1_y1y'])                                # Plot detail. DO NOT delete.!!!!
    # # x = np.append(x, handle1.tot_file_shift['ecc03_2_y1x'])
    # # y = np.append(y, handle1.tot_file_shift['ecc03_2_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc03_3_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc03_3_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc03_4_y1x'])
    # y = np.append(y, -handle1.tot_file_shift['ecc03_4_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc03_5_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc03_5_y1y'])
    # # x = np.append(x, handle1.tot_file_shift['ecc03_6_y1x'])
    # # y = np.append(y, handle1.tot_file_shift['ecc03_6_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc03_7_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc03_7_y1y'])
    #
    # xlim = 10
    # scale = 6.4/4.8
    # fig2 = plt.figure() # figsize=[6.4, 4.8]
    # ax2 = fig2.add_axes([0.15,0.15,0.7,0.7])
    # h, xedges, yedges, img = ax2.hist2d(x, y, bins=[80, 80],
    #                                    range=[[-xlim, xlim], [-xlim, xlim]])
    # norm = colors.Normalize(vmin = np.amin(h), vmax = np.amax(h))
    # cb = fig2.colorbar(cm.ScalarMappable(norm=norm), ax = ax2)
    # cb.set_label('Counts', fontsize = 15, rotation = -90, horizontalalignment = 'center', verticalalignment = 'bottom')
    # ax2.set_xlabel(r'Position($\mu$m)', fontsize = 15)
    # ax2.set_ylabel(r'Position($\mu$m)', fontsize = 15)
    #
    # xtick_temp = 2*np.arange(-0.5*xlim, 0.5*xlim + 1)
    # xticks = list(xtick_temp)
    # xticklabel = []
    # for ticks in xticks:
    #     xticklabel.append(str(ticks/6.25))
    # ax2.set_xticks(xticks)
    # ax2.set_xticklabels(xticklabel, fontsize = 10)
    # ax2.set_yticks(xticks)
    # ax2.set_yticklabels(xticklabel, fontsize = 10)
    # ax2.set_title('T4 position distribution. Ecc=0.3')
    # plt.show()

    # ecc0995(T4-plasmid)
    # x1 = np.array([])
    # y1 = np.array([])
    # x1 = np.append(x1, handle1.tot_file_shift['ecc0995_1_y1x'])
    # y1 = np.append(y1, handle1.tot_file_shift['ecc0995_1_y1y'])
    # x1 = np.append(x1, handle1.tot_file_shift['ecc0995_2_y1x'])
    # y1 = np.append(y1, handle1.tot_file_shift['ecc0995_2_y1y'])
    # x1 = np.append(x1, handle1.tot_file_shift['ecc0995_3_y1x'])
    # y1 = np.append(y1, handle1.tot_file_shift['ecc0995_3_y1y'])

    # ecc0995(lambda-plasmid)
    # x = handle1.tot_pos_overlay_shift['ecc0995_y1x']
    # y = handle1.tot_pos_overlay_shift['ecc0995_y1y']
    # x = np.array([])
    # y = np.array([])
    # x = np.append(x, handle1.tot_file_shift['ecc0995_1_y1x']-4.5)
    # y = np.append(y, handle1.tot_file_shift['ecc0995_1_y1y'])
    # x = np.append(x, -handle1.tot_file_shift['ecc0995_2_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc0995_2_y1y'])
    # x = np.append(x, handle1.tot_file_shift['ecc0995_3_y1x'])
    # y = np.append(y, handle1.tot_file_shift['ecc0995_3_y1y'])
    #
    # xlim = 15
    # scale = 6.4/4.8
    # fig2 = plt.figure() # figsize=[6.4, 4.8]
    # ax2 = fig2.add_axes([0.15,0.15,0.7,0.7])
    # h, xedges, yedges, img = ax2.hist2d(x, y, bins=[80, 80],
    #                                    range=[[-xlim, xlim], [-xlim, xlim]])
    # norm = colors.Normalize(vmin = np.amin(h), vmax = np.amax(h))
    # cb = fig2.colorbar(cm.ScalarMappable(norm=norm), ax = ax2)
    # cb.set_label('Counts', fontsize = 15, rotation = -90, horizontalalignment = 'center', verticalalignment = 'bottom')
    # ax2.set_xlabel(r'Position($\mu$m)', fontsize = 15)
    # ax2.set_ylabel(r'Position($\mu$m)', fontsize = 15)
    #
    # xtick_temp = 2*np.arange(-0.5*xlim, 0.5*xlim + 1)
    # xticks = list(xtick_temp)
    # xticklabel = []
    # for ticks in xticks:
    #     xticklabel.append(str(ticks/6.25))
    # ax2.set_xticks(xticks)
    # ax2.set_xticklabels(xticklabel, fontsize = 10)
    # ax2.set_yticks(xticks)
    # ax2.set_yticklabels(xticklabel, fontsize = 10)
    # ax2.set_title('T4 position distribution. Ecc=0.995')
    # plt.show()

    ###################
    # Density profile #
    ###################
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    # ax.plot(handle1.tot_density_hist['test_edge'], handle1.tot_density_hist['test_density'])
    # ax.plot(handle1.tot_density_hist['ecc06_edge'], handle1.tot_density_hist['ecc06_density_y1'])
    # ax.plot(handle1.tot_density_hist['ecc08_edge'], handle1.tot_density_hist['ecc08_density_y1'])
    # ax.plot(handle1.tot_density_hist['ecc09_edge'], handle1.tot_density_hist['ecc09_density_y1'])
    # ax.legend(['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9'])
    # ax.set_xlabel('Normalized radius', fontsize=15)
    # ax.set_ylabel(r'Density$(pts/pixel^2)$', fontsize=15)
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(handle1.tot_density_hist['test_degedge'], handle1.tot_density_hist['test_degdensity'])
    # plt.show()

    # lambda-plasmid
    # handle1 = module.densitycal(handle1, bins=60)
    # density_test = module.densitycal(handle1, bins=60,x=x,y=y,ecc=0.995, debug=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    #
    # ax.plot(handle1.tot_density_hist['ecc0_edge'], handle1.tot_density_hist['ecc0_density_y1']/len(handle1.tot_pos_overlay_shift['ecc0_y1x']))
    # ax.plot(handle1.tot_density_hist['ecc06_edge'], handle1.tot_density_hist['ecc06_density_y1']/len(handle1.tot_pos_overlay_shift['ecc06_y1x']))
    # ax.plot(handle1.tot_density_hist['ecc08_edge'], handle1.tot_density_hist['ecc08_density_y1']/len(handle1.tot_pos_overlay_shift['ecc08_y1x']))
    # ax.plot(handle1.tot_density_hist['ecc09_edge'], handle1.tot_density_hist['ecc09_density_y1']/len(handle1.tot_pos_overlay_shift['ecc09_y1x']))
    # ax.plot(density_test['test_edge'], density_test['test_density']/np.sum(density_test['test_density'][1:]))
    # ax.legend(['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9', 'ecc0.995'])
    # ax.set_xlabel('Normalized radius', fontsize=15)
    # ax.set_ylabel('Probability', fontsize=15)
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(handle1.tot_density_hist['ecc0_degedge'], handle1.tot_density_hist['ecc0_degdensity_y1'])
    # ax2.plot(handle1.tot_density_hist['ecc06_degedge'], handle1.tot_density_hist['ecc06_degdensity_y1'])
    # ax2.plot(handle1.tot_density_hist['ecc08_degedge'], handle1.tot_density_hist['ecc08_degdensity_y1'])
    # ax2.plot(handle1.tot_density_hist['ecc09_degedge'], handle1.tot_density_hist['ecc09_degdensity_y1'])
    # ax2.plot(density_test['test_degedge'], density_test['test_degdensity'])
    # ax2.legend(['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9', 'ecc0.995'])
    # ax2.set_xlabel('Theta(rad)', fontsize=15)
    # ax2.set_ylabel(r'Density$(pts/pixel^2)$', fontsize=15)
    # plt.show()

    # T4-plasmid
    handle1 = module.densitycal(handle1, bins=50)
    density_test = module.densitycal(handle1, bins=50, x=x, y=y, ecc=0.9, debug=True)
    density_test1 = module.densitycal(handle1, bins=50, x=x1, y=y1, ecc=0.995, debug=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)

    ax.plot(handle1.tot_density_hist['ecc0_edge'],
            handle1.tot_density_hist['ecc0_density_y1'] / len(handle1.tot_pos_overlay_shift['ecc0_y1x']))
    ax.plot(handle1.tot_density_hist['ecc06_edge'],
            handle1.tot_density_hist['ecc06_density_y1'] / len(handle1.tot_pos_overlay_shift['ecc06_y1x']))
    ax.plot(handle1.tot_density_hist['ecc08_edge'],
            handle1.tot_density_hist['ecc08_density_y1'] / len(handle1.tot_pos_overlay_shift['ecc08_y1x']))
    ax.plot(density_test['test_edge'], density_test['test_density'] / len(handle1.tot_pos_overlay_shift['ecc09_y1x']))
    ax.plot(density_test1['test_edge'], density_test1['test_density'] / np.sum(density_test1['test_density'][1:]))
    ax.legend(['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9', 'ecc0.995'])
    ax.set_xlabel('Normalized radius', fontsize=15)
    ax.set_ylabel('Probability', fontsize=15)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(handle1.tot_density_hist['ecc0_degedge'], handle1.tot_density_hist['ecc0_degdensity_y1'])
    ax2.plot(handle1.tot_density_hist['ecc06_degedge'], handle1.tot_density_hist['ecc06_degdensity_y1'])
    ax2.plot(handle1.tot_density_hist['ecc08_degedge'], handle1.tot_density_hist['ecc08_degdensity_y1'])
    ax2.plot(density_test['test_degedge'], density_test['test_degdensity'])
    ax2.plot(density_test1['test_degedge'], density_test1['test_degdensity'])
    ax2.legend(['ecc0', 'ecc0.6', 'ecc0.8', 'ecc0.9', 'ecc0.995'])
    ax2.set_xlabel('Theta(rad)', fontsize=15)
    ax2.set_ylabel(r'Density$(pts/pixel^2)$', fontsize=15)
    plt.show()


    def temperr():
        ecc09_delx = np.concatenate([handle.tot_vector['ecc09_1_delx'], handle.tot_vector['ecc09_2_delx'],
                                     handle.tot_vector['ecc09_3_delx'], handle.tot_vector['ecc09_4_delx'],
                                     handle.tot_vector['ecc09_5_delx'], handle.tot_vector['ecc09_6_delx'],
                                     handle.tot_vector['ecc09_7_delx'], -handle.tot_vector['ecc09_8_delx'],
                                     handle.tot_vector['ecc09_9_delx']])

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        # ax.hist2d(handle.tot_vec_overlay['ecc09_delx'], handle.tot_vec_overlay['ecc09_dely'], bins = [80, 80], range = [[-10, 10], [-10, 10]])
        # ax.hist2d(np.concatenate([handle.tot_vec_overlay['ecc09_delx'], -handle.tot_vec_overlay['ecc09_delx']]), np.concatenate([handle.tot_vec_overlay['ecc09_dely'],handle.tot_vec_overlay['ecc09_dely']]), bins = [130, 130])
        h, xedges, yedges, img = ax.hist2d(ecc09_delx, handle.tot_vec_overlay['ecc09_dely'], bins=[70, 70],
                                           range=[[-10, 10], [-10, 10]])
        # h, xedges, yedges, img = ax.hist2d(handle.tot_vec_overlay['ecc06_delx'], handle.tot_vec_overlay['ecc06_dely'], bins = [70, 70], range = [[-10, 10], [-10, 10]])
        norm = colors.Normalize(vmin=np.amin(h), vmax=np.amax(h))
        cb = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cb.set_label('Counts', fontsize=15, rotation=-90, horizontalalignment='center', verticalalignment='bottom')
        ax.set_xlabel('Position(pixel)', fontsize=15)
        ax.set_ylabel('Position(pixel)', fontsize=15)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(handle.tot_file['ecc09_1_y3x'], handle.tot_file['ecc09_1_y3y'], '+')
        ax2.plot(handle.tot_file['ecc09_2_y3x'], handle.tot_file['ecc09_2_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_3_y3x'], handle.tot_file['ecc09_3_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_4_y3x'], handle.tot_file['ecc09_4_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_5_y3x'], handle.tot_file['ecc09_5_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_6_y3x'], handle.tot_file['ecc09_6_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_7_y3x'], handle.tot_file['ecc09_7_y3y'], '+')
        # ax2.plot(handle.tot_file['ecc09_8_y3x'], handle.tot_file['ecc09_8_y3y'], '+')
        # ax2.legend(['1', '2'])
        # ax2.set_title('T4 shift', fontsize=15)
        # ax2.set_xlabel('Position(pixel)', fontsize=15)
        # ax2.set_ylabel('Position(pixel)', fontsize=15)
        #
        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(1, 1, 1)
        # ax3.plot(handle.tot_file['ecc09_1_y1x'], handle.tot_file['ecc09_1_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_2_y1x'], handle.tot_file['ecc09_2_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_3_y1x'], handle.tot_file['ecc09_3_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_4_y1x'], handle.tot_file['ecc09_4_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_5_y1x'], handle.tot_file['ecc09_5_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_6_y1x'], handle.tot_file['ecc09_6_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_7_y1x'], handle.tot_file['ecc09_7_y1y'], '+')
        # ax3.plot(handle.tot_file['ecc09_8_y1x'], handle.tot_file['ecc09_8_y1y'], '+')
        # ax3.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
        # ax3.set_title('Plasmid shift', fontsize=15)
        # ax3.set_xlabel('Position(pixel)', fontsize=15)
        # ax3.set_ylabel('Position(pixel)', fontsize=15)
        # plt.show()
        # return
    return