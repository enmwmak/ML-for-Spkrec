import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-2.0:2.0:60j, -2.0:2.0:60j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu1 = np.array([0.5, 0.5])
cov1 = np.array([[0.2, 0.1],[0.1, 0.2]])
mu2 = np.array([-0.5, -0.5])
cov2 = np.array([[0.2, -0.1],[-0.1, 0.2]])
mu3 = np.array([1, -1])
cov3 = np.array([[0.1, -0.1],[-0.1, 0.2]])

z1 = multivariate_normal.pdf(xy, mean=mu1, cov=cov1)
z2 = multivariate_normal.pdf(xy, mean=mu2, cov=cov2)
z3 = multivariate_normal.pdf(xy, mean=mu3, cov=cov3)
z = z1 + z2 + z3

# Reshape back to a (60, 60) grid.
z = z.reshape(x.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z)
#ax.plot_wireframe(x,y,z)

plt.show()