import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import nmsimplex as nm

# Definitions
start = [-10, -10]
end = [10, 10]

simplex_fun = lambda x, y: x**2+y**2
# -----------
x = np.linspace(start[0], end[0])
y = np.linspace(start[1], end[1])
x, y = np.meshgrid(x, y)

z = simplex_fun(x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x,y,z, linewidth=0, cmap=cm.jet)
plt.show()