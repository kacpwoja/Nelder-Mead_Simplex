import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nmsimplex as nm

def rosenbrock(x, a=1, b=100):
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

# Definitions
start = [-2, -1]
end = [2, 3]

simplex_fun = rosenbrock
max_iterations = 1000
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
simplex = [[0, 1.5, -1], [2.5, -0.5, 0]]
simplex = np.array(simplex)
simplex = np.transpose(simplex)[simplex_fun(simplex).argsort()]
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
plt.legend()
plt.show()