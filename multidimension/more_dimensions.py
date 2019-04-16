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

# Definitions
dimension = 111
start = np.ones((dimension,))*-5.12
end = np.ones((dimension,))*5.12

simplex_fun = lambda x: rastrigin_n(x, dim=dimension)
max_iterations = 300000
# -----------

# Init
simplex = nm.init_simplex(dimension, start, end, simplex_fun)

# Manual
# simplex = [[-4, -4, 4], [-4, 4, 4]]
# simplex = np.array(simplex)
# simplex = np.transpose(simplex)[simplex_fun(simplex).argsort()]

# Random
# simplex = np.random.rand(dimension+1, dimension)*10.24 - 5.12
# -----------

print("First Simplex:")
print(simplex)

# NM algorithm
i = 1
while i <= max_iterations and simplex_fun(np.transpose(simplex[0])) != 0:
    simplex = nm.nm_step(simplex, simplex_fun)
    i += 1
print("Found minimum:")
print(simplex[0])
print("Iterations:")
print(i)