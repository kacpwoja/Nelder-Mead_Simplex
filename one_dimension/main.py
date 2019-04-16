import numpy as np
import matplotlib.pyplot as plt
import nmsimplex as nm

def fun(x):
    return 3*x**4 + 4*x**3 - 6*x**2 + 4*x + 7

# Definitions
start = -3
end = 2

simplex_fun = fun
max_iterations = 15
# -----------

x = np.linspace(start, end)
y = simplex_fun(x)

plt.plot(x, y, label='function')

simplex = nm.init_simplex(start, end, fun)
plt.plot(simplex, simplex_fun(simplex), 'x', label='init simplex')

i = 1
while i <= max_iterations:
    simplex = nm.nm_step(simplex, simplex_fun)
    plt.plot(simplex, simplex_fun(simplex), 'x', label='iteration %d' %i)
    i += 1
print("Found minimum:")
print(simplex[0])

plt.title("NM simplex - one dimension")
plt.legend()

plt.show()