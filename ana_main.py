import numpy as np
import matplotlib.pyplot as plt
import module

main_path = "D:/McGillResearch/2019Manuscript_Analysis/Analysis/tplasmid"

# Read files
handle, tot_file = module.bashload(main_path)
handle, tot_vector = module.bashvector(handle)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(handle.tot_vector['ecc06_6_delx'], handle.tot_vector['ecc06_6_dely'], '+')
plt.show()