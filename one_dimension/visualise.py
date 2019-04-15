import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import nmsimplex as nm

def fun(x):
    return 3*x**4 + 4*x**3 - 6*x**2 + 4*x + 7

def draw(simplex, i):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.plot(simplex, simplex_fun(simplex), 'x',)
    ax.add_collection(PatchCollection([Polygon(np.transpose(np.array([simplex, simplex_fun(simplex)])))], alpha = 0.4))
    plt.title("NM simplex - one dimension - visualisation")
    fig.savefig('./figs/simplex_%d.png'%i, format='png', dpi = 200)


# Definitions
start = -3
end = 2

simplex_fun = fun
simplex_size = 4
max_iterations = 15
# -----------

x = np.linspace(start, end)
y = simplex_fun(x)


simplex = nm.init_simplex(simplex_size, start, end, fun)
draw(simplex, 0)

i = 1
while i <= max_iterations:
    simplex = nm.nm_step(simplex, simplex_fun)
    draw(simplex, i)
    i += 1