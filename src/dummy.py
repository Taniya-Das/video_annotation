
import matplotlib
matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import numpy as np


ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linestyle = 'dotted')
plt.show(block=True)
plt.savefig('file_name.png')