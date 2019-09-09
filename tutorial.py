import numpy as np
import matplotlib.pyplot as plt
import module
# This simple tutorial aims to train you on the TCA. More functions will be implemented in the future.

main_path = ""

# Read files
h, tot_file = module.bashload(main_path)
# Calculate delx,y
h, tot_vector = module.bashvector(h)
# Calculate overlay
h, tot_vec_overlay = module.bashoverlay(h)
# Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(h.tot_vec_overlay['ecc03_delx'], h.tot_vec_overlay['ecc03_dely'], '+')
plt.show()