########## DERIVATIVE FUNCTIONS ##########

import numpy as np
from numba import njit, prange
# function for accessing elements outside of the computation domain 
# boundaries[0] and boundaries[-1] specifies the value that the function should take to the left and to the right
# of the computational domain, respectively
@njit
def safe_get(f, i, boundaries = None):
    if boundaries is not None:
        if i < 0:
            return boundaries[0]
        elif i >= len(f):
            return boundaries[1]
        else:
            return f[i]
    else:
        if i >= len(f):
            return f[-1]
        elif i < 0:
            return f[0]
        else:
            return f[i]

# central difference scheme
@njit(parallel = True)
def d(f, h, n = 1, boundaries = None):
    if n == 0:
        return f
    elif n == 1:
        coeffs = [[-1, -1/2], [1, 1/2]]
    elif n == 2:
        coeffs = [[-1, 1], [0, -2], [1, 1]]
    elif n == 3:
        coeffs = [[-2, -1/2], [-1, 1], [1, -1], [2, 1/2]]
    elif n == 4:
        coeffs = [[-2, 1], [-1, -4], [0, 6], [1, -4], [2, 1]]
    nx = len(f)
    r = np.zeros(nx)
    for i in prange(nx):
        for c in coeffs:
            loc = c[0]
            val = c[1]
            r[i] += safe_get(f, i + loc, boundaries) * val
    return r/(h ** n)

# integration function
@njit(parallel = True)
def integrate(f, dx):
    return np.sum(f)*dx