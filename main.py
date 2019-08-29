import module
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

################################################################################
main_path = 'D:/McGillResearch/2019Manuscript_Analysis/Analysis/tplasmid_all'
h, tot_file = module.bashload(main_path)
print('There is '+str(len(tot_file))+' files in total.')
h, tot_vector = module.bashvector(h)
fig = plt.figure()
llim = -7.5
hlim = 10
center = 11.5
a = 6
b = 4
fontsize = 15
size = [0.13, 0.1, 0.8, 0.8]
ax1 = fig.add_axes(size)

# line = ax1.plot(0.9*h.tot_vector['ecc06_1_delx'], h.tot_vector ['ecc06_1_dely'], '+')
ax1.plot(h.tot_vector['ecc06_1_delx'], h.tot_vector ['ecc06_1_dely'], '+')
ax1.plot(h.tot_vector['ecc06_2_delx'], h.tot_vector ['ecc06_2_dely'], '+')
ax1.plot(h.tot_vector['ecc06_3_delx'], h.tot_vector ['ecc06_3_dely'], '+')
ax1.set_ylim([llim, hlim])
ax1.set_xlim([llim, hlim])
ax1.grid(b=True)
ax1.set_xlabel('Position(pixels)', fontsize = fontsize )
ax1.set_ylabel('Position(pixels)', fontsize = fontsize )
ax1.legend(['ecc06'], loc = 'upper left')
ax2 = fig.add_axes([0.73, 0.1, 0.2, 0.8])
projy, edge = np.histogram(np.concatenate([h.tot_vector['ecc06_1_dely'], h.tot_vector ['ecc06_2_dely'], h.tot_vector ['ecc06_3_dely']]), range = (llim, hlim),bins = 30)
width = np.zeros(len(edge[0:-1]))
x = np.linspace(-6, 6, len(edge[0:-1]))
y = b*np.sqrt(1-(x/a)**2)
ax2.plot(np.divide(projy, y), edge[0:-1])
ax2.tick_params(axis='both', bottom=False, top=True, labelbottom=False, right=False, left=False, labelleft=False, labeltop=True, direction='in', pad = -15)
ax2.set_ylabel('Density(#/pix2)', fontsize = fontsize )
ax2.yaxis.set_label_position('right')
plt.show()
################### delta x,y calculation #############
# h, tot_vector = module.bashvector(h)
# plt.plot(tot_vector.get('ecc0_1_delx'), tot_vector.get('ecc0_1_dely'), "+")

# ################### over lay videos sharing the same cavity shape #######
# tot_vec_overlay = module.bashoverlay(tot_vector)
# # plt.plot(tot_vec_overlay['ecc06_delx'],tot_vec_overlay['ecc06_dely'],'+')
# # plt.plot(tot_vec_overlay['ecc03_delx'], tot_vec_overlay['ecc03_dely'], '+')
# # plt.legend(['6','3'])
# ################### calculate the free energy landscape #########
# tot_free, sep = module.bashfree(tot_vec_overlay, type = "o")
# # plt.plot(tot_free['ecc095_bins'], tot_free['ecc095_F'])

# ################### plot free energy landscape ################
# ax = module.bashfreeplt(tot_free)
# unit = 0.16 #160um/pixel
# ax.set_xlim([0,14])
# xticks = ax.get_xticks()
# cov_xticks = xticks * unit
# ax.set_xticklabels(cov_xticks)
#
# ax.set_xlabel(r'X-separation($/mu$m)', fontsize = 15)
# ax.set_ylabel(r'Free energy($/mathrm{k_B}$T)', fontsize = 15)
#
# ax.tick_params(axis ='both', labelsize=13)
# ##################################################
# plt.show()
