import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import pi, e
import nmsimplex as nm

def ackley(x):
    a = -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))
    b = -np.exp(0.5*(np.cos(2*pi*x[0])+np.cos(2*pi*x[1])))
    return  a + b + e + 20

def draw(simplex, i, xy, z):
    plt.figure()
    ax_1 = plt.axes(projection='3d')
    ax_1.set_xlim(-4, 4)
    ax_1.set_ylim(-4, 4)
    ax_1.set_zlim(0, 12)
    ax_1.plot_surface(xy[0], xy[1], z, linewidth=0, cmap=cm.jet, alpha=0.3)
    plt.title("NM Simplex - 3D Plot")
    simplex = np.transpose(simplex)
    p = Poly3DCollection([list(zip(simplex[0], simplex[1], simplex_fun(simplex)))])
    p.set_color('k')
    ax_1.add_collection3d(p)
    plt.plot(simplex[0], simplex[1], simplex_fun(simplex), 'kx')
    plt.savefig('./figs3D/simplex_%d.png'%i, format='png', dpi = 200)


# Definitions
start = [-4, -4]
end = [4, 4]

simplex_fun = ackley
max_iterations = 10
# -----------
x = np.linspace(start[0], end[0])
y = np.linspace(start[1], end[1])
x, y = np.meshgrid(x, y)
xy = np.array([x, y])
z = simplex_fun(xy)

# Plots: 3D
plt.figure(1)
ax_1 = plt.axes(projection='3d')
ax_1.set_xlim(-4, 4)
ax_1.set_ylim(-4, 4)
ax_1.set_zlim(0, 12)
ax_1.plot_surface(xy[0], xy[1], z, linewidth=0, cmap=cm.jet)
plt.title("NM Simplex - 3D Plot")
plt.savefig("./figs3D/function.png", format='png', dpi=200)


simplex = [[-3, 3, -1], [-3, 1, 3]]
simplex = np.array(simplex)
simplex = np.transpose(simplex)[simplex_fun(simplex).argsort()]
draw(simplex, 0, xy, z)

i = 1
while i <= max_iterations:
    simplex = nm.nm_step(simplex, simplex_fun)
    draw(simplex, i, xy, z)
    i += 1