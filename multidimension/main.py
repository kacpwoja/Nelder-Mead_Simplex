import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import pi, e
import nmsimplex as nm

def rastrigin_n(x, a=10, dim=2):
    f = 0
    for i in range(dim):
        f += x[i]**2 - a*np.cos(2*pi*x[i])
    return a*dim + f

def rastrigin(x, a=10):
    return a*2 + x[0]**2 + x[1]**2 - a*np.cos(2*pi*x[0]) - a*np.cos(2*pi*x[1])

def rosenbrock(x, a=1, b=100):
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

def ackley(x):
    a = -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))
    b = -np.exp(0.5*(np.cos(2*pi*x[0])+np.cos(2*pi*x[1])))
    return  a + b + e + 20

# Definitions
start = [-5.12, -5.12]
end = [5.12, 5.12]

simplex_fun = rastrigin_n
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
ax_1.plot_surface(xy[0], xy[1], z, linewidth=0, cmap=cm.jet)
plt.title("NM Simplex - 3D Plot")
# ax_1.add_collection3d(Poly3DCollection([list(zip(simplex[0], simplex[1], simplex_fun(simplex)))]))

# Plots: 2D
plt.figure(2)
ax_2 = plt.contourf(xy[0], xy[1], z)
plt.title("NM Simplex - Contour Plot")


# Init
# simplex = nm.init_simplex(2, start, end, simplex_fun)
# Manual
simplex = [[-4, -4, 4], [-4, 4, 4]]
simplex = np.array(simplex)
simplex = np.transpose(simplex)[simplex_fun(simplex).argsort()]

# Plot
simplex_T = np.transpose(simplex)
simplex_Tx = np.append(simplex_T[0], simplex_T[0][0])
simplex_Ty = np.append(simplex_T[1], simplex_T[1][0])
# Add dots
# plt.figure(1)
# plt.plot(simplex_T[0], simplex_T[1], simplex_fun(simplex_T), 'x', label='init simplex')
plt.figure(2)
plt.plot(simplex_Tx, simplex_Ty, label='init simplex')

# NM algorithm
i = 1
while i <= max_iterations:
    simplex = nm.nm_step(simplex, simplex_fun)
    simplex_T = np.transpose(simplex)
    simplex_Tx = np.append(simplex_T[0], simplex_T[0][0])
    simplex_Ty = np.append(simplex_T[1], simplex_T[1][0])
    # plt.figure(1)
    # plt.plot(simplex_T[0], simplex_T[1], simplex_fun(simplex_T), 'x', label='iteration %d' %i)
    plt.figure(2)
    plt.plot(simplex_Tx, simplex_Ty, label='iteration %d' %i)
    i += 1
print("Found minimum:")
print(simplex[0])

# plt.figure(1)
# plt.legend()
plt.figure(2)
#plt.legend()
plt.show()