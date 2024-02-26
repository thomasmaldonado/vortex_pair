### DERIVATIVE DEFINITION ###

import numpy as np
from numba import njit

# central difference scheme
# h = grid spacing, n = order of derivative 
# boundary_left and boundary_right represent function values one cell to the left and right of the domain
# axis = 0 for derivative with respect to 0th coordinate, axis = 1 for derivative with respect to 1st coordinate
@njit(cache = True)
def d(f, h, n, boundary_left, boundary_right, axis = 0):
    if axis == 1:
        f = f.transpose()
    boundary_left = np.reshape(boundary_left, (1, f.shape[1]))
    boundary_right = np.reshape(boundary_right, (1, f.shape[1]))
    f_b = np.concatenate((boundary_left, f, boundary_right))
    f_b = f_b.transpose()
    f_b_shape = f_b.shape
    f_b = f_b.flatten()
    if n == 1:
        padded = (-(1/2)*np.roll(f_b, 1) + (1/2)*np.roll(f_b, -1))/h**n
    elif n == 2:
        padded = (np.roll(f_b, 1) -2*f_b + np.roll(f_b, -1))/h**n
    f_b = np.reshape(padded, f_b_shape)
    f_b = f_b.transpose()[1:-1,:]
    if axis == 1:
        f_b = f_b.transpose()
    return f_b