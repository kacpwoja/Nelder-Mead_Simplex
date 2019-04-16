import numpy as np

def init_simplex(dimension, start, end, function):
    if len(start) != dimension or len(end) != dimension:
        raise Exception("Interval doesn't match dimension")
    simplex = []
    for i in range(dimension):
        simplex.append(np.linspace(start[i], end[i], dimension+1))
    simplex = np.array(simplex)
    return np.transpose(simplex)[function(simplex).argsort()]

def nm_step(simplex, function, reflection = 1, expansion = 2, contraction = 0.5, shrinkage = 0.5):
    n = len(simplex) - 1

    # 1. Order
    f = function(np.transpose(simplex))
    order = f.argsort()
    simplex = simplex[order]
    f = f[order]

    # 2. Reflect
    x_centroid = sum(simplex[:n])/n
    x_r = (1+reflection)*x_centroid - reflection*simplex[n]
    f_r = function(x_r)

    if f_r >= f[0] and f_r < f[n-1]:
        return reorder_nonshrink(simplex, x_r, f, f_r)

    # 3. Expand
    if f_r < f[0]:
        x_e = (1+reflection*expansion)*x_centroid - reflection*expansion*simplex[n]
        f_e = function(x_e)

        if f_e < f_r:
            return reorder_nonshrink(simplex, x_e, f, f_e)
        else:
            return reorder_nonshrink(simplex, x_r, f, f_r)

    # 4. Contract
    # a. Outside
    if f_r >= f[n-1] and f_r < f[n]:
        x_c = (1+reflection*contraction)*x_centroid - reflection*contraction*simplex[n]
        f_c = function(x_c)

        if f_c <= f_r:
            return reorder_nonshrink(simplex, x_c, f, f_c)
        else:
            return shrink(simplex, function, shrinkage)

    # b. Inside
    if f_r >= f[n]:
        x_cc = (1-contraction)*x_centroid + contraction*simplex[n]
        f_cc = function(x_cc)

        if f_cc < f[n]:
            return reorder_nonshrink(simplex, x_cc, f, f_cc)
        else:
            return shrink(simplex, function, shrinkage)

    raise Exception("How did you get here?")

def reorder_nonshrink(simplex, new_vertex, f_simplex, f_new_vertex):
    l_list = []
    for l in range(len(simplex)):
        if f_new_vertex < f_simplex[l]:
            l_list.append(l)

    l = max(l_list)

    return np.insert(simplex, max(l_list), new_vertex, 0)[:-1]

def shrink(simplex, function, shrinkage):
    for i in range(1, len(simplex)):
        simplex[i] = simplex[1] + shrinkage*(simplex[i]-simplex[1])
    return simplex[function(np.transpose(simplex)).argsort()]