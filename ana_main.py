import numpy as np
import matplotlib.pyplot as plt
import module
import matplotlib.cm as cm
import matplotlib.colors as colors
main_path = "D:/McGillResearch/2019Manuscript_Analysis/Analysis/tplasmid"

# Read files
handle, tot_file = module.bashload(main_path)
handle, tot_vector = module.bashvector(handle)
handle, tot_vec_overlay = module.bashoverlay(handle)

ecc09_delx = np.concatenate([handle.tot_vector['ecc09_1_delx'], handle.tot_vector['ecc09_2_delx'],
                             handle.tot_vector['ecc09_3_delx'], handle.tot_vector['ecc09_4_delx'],
                             handle.tot_vector['ecc09_5_delx'], handle.tot_vector['ecc09_6_delx'],
                             handle.tot_vector['ecc09_7_delx'], -handle.tot_vector['ecc09_8_delx'],
                             handle.tot_vector['ecc09_9_delx']])

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# ax.hist2d(handle.tot_vec_overlay['ecc09_delx'], handle.tot_vec_overlay['ecc09_dely'], bins = [80, 80], range = [[-10, 10], [-10, 10]])
# ax.hist2d(np.concatenate([handle.tot_vec_overlay['ecc09_delx'], -handle.tot_vec_overlay['ecc09_delx']]), np.concatenate([handle.tot_vec_overlay['ecc09_dely'],handle.tot_vec_overlay['ecc09_dely']]), bins = [130, 130])
# h, xedges, yedges, img = ax.hist2d(ecc09_delx, handle.tot_vec_overlay['ecc09_dely'], bins = [70, 70], range = [[-10, 10], [-10, 10]])
h, xedges, yedges, img = ax.hist2d(handle.tot_vec_overlay['ecc06_delx'], handle.tot_vec_overlay['ecc06_dely'], bins = [70, 70], range = [[-10, 10], [-10, 10]])
norm = colors.Normalize(vmin = np.amin(h), vmax = np.amax(h))
cb = fig.colorbar(cm.ScalarMappable(norm=norm), ax = ax)
cb.set_label('Counts', fontsize = 15, rotation = -90, horizontalalignment = 'center', verticalalignment = 'bottom')
ax.set_xlabel('Position(pixel)', fontsize = 15)
ax.set_ylabel('Position(pixel)', fontsize = 15)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(handle.tot_vec_overlay['ecc06_delx'],handle.tot_vec_overlay['ecc06_dely'], '+')
plt.show()