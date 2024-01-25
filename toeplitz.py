import numpy as np
import matplotlib.pyplot as plt

N = 20
A = np.eye(N,k=0) + np.eye(N,k=1) * 0.5 - 0.45 * np.eye(N,k=-1)
Ainv = np.linalg.inv(A)

fig,axs = plt.subplots(1,2)
ax = axs[0]
ax.imshow(A)

ax = axs[1]
ax.imshow(Ainv)

fig.savefig('plots/test_plots/toe.png')
