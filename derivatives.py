########## DERIVATIVE FUNCTIONS ##########

import numpy as np
from numba import njit
# function for accessing elements outside of the computation domain 
# boundaries[0] and boundaries[-1] specifies the value that the function should take to the left and to the right
# of the computational domain, respectively
# central difference scheme 
# axis = axis along which derivatives are taken

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

# integration function
@njit(parallel = True)
def integrate(f, dx):
    return np.sum(f)*dx