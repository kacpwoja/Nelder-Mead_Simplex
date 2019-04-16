import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

# Definitions
start = [-2, -1]
end = [2, 3]

simplex_fun = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
# -----------
x = np.linspace(start[0], end[0])
y = np.linspace(start[1], end[1])
x, y = np.meshgrid(x, y)

xx = np.array([x, y])

z = simplex_fun(xx)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx[0], xx[1], z, linewidth=0, cmap=cm.jet)

plt.show()