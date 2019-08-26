import module
import matplotlib.pyplot as plt

################################################################################
main_path = 'D:/McGillResearch/2019Manuscript_Analysis/Analysis/groupmeeting_dataset'
tot_file = module.bashload(main_path)
print('There is '+str(len(tot_file))+' files in total.')

################### delta x,y calculation #############
tot_vector = module.bashvector(tot_file)
# plt.plot(tot_vector.get('ecc0_1_delx'), tot_vector.get('ecc0_1_dely'), "+")

################### over lay videos sharing the same cavity shape #######
tot_vec_overlay = module.bashoverlay(tot_vector)
# plt.plot(tot_vec_overlay['ecc06_delx'],tot_vec_overlay['ecc06_dely'],'+')
# plt.plot(tot_vec_overlay['ecc03_delx'], tot_vec_overlay['ecc03_dely'], '+')
# plt.legend(['6','3'])
################### calculate the free energy landscape #########
tot_free, sep = module.bashfree(tot_vec_overlay, type = "o")
# plt.plot(tot_free['ecc095_bins'], tot_free['ecc095_F'])

################### plot free energy landscape ################
ax = module.bashfreeplt(tot_free)
unit = 0.16 #160um/pixel
ax.set_xlim([0,14])
xticks = ax.get_xticks()
cov_xticks = xticks * unit
ax.set_xticklabels(cov_xticks)

ax.set_xlabel(r'X-separation($/mu$m)', fontsize = 15)
ax.set_ylabel(r'Free energy($/mathrm{k_B}$T)', fontsize = 15)

ax.tick_params(axis ='both', labelsize=13)
##################################################
plt.show()
