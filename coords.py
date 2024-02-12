import numpy as np
from numba import njit, prange

@njit
def M(u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return np.array([[M_00,M_01],[M_10,M_11]])

@njit
def BP2cart(Fu, Fv, u, v):
    Fx, Fy = np.dot(M(u,v), np.array([Fu, Fv]))
    return Fx, Fy

@njit
def cart2BP(Fx, Fy, u, v):
    Fu, Fv = np.dot(M(u,v).transpose(), np.array([Fx, Fy]))
    return Fu, Fv
