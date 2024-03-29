### COORDINATE TRANSFORMATIONS ###

from sympy import symbols, Function, diff, ln, solve
from sympy.utilities import lambdify
from numba import njit
import numpy as np
# vp represents primed coordinate on the domain (0, 1)
v, vp, a, j = symbols('v v_p a j')

# require v(0) = 0, v(1) = infty
v_of_vp= ln((1+vp)/(1-vp))

# invert mapping and compute all possible derivatives up to second order
vp_of_v = solve(v_of_vp - v, vp)[0]

dv_dvp = v_of_vp.diff(vp).doit().simplify()
d2v_dvp2 = v_of_vp.diff(vp,2).doit().simplify()

dvp_dv = vp_of_v.diff(v).doit().simplify()
d2vp_dv2 = vp_of_v.diff(v, 2).doit().simplify()

args = [vp, a, j]
v_of_vp_lambdified = lambdify(args, v_of_vp)
dv_dvp_lambdified = lambdify(args, dv_dvp)
d2v_dvp2_lambdified = lambdify(args, d2v_dvp2)

args = [v, a, j]
vp_of_v_lambdified = lambdify(args, vp_of_v)
dvp_dv_lambdified = lambdify(args, dvp_dv)
d2vp_dv2_lambdified = lambdify(args, d2vp_dv2)

v_of_vp_lambdified = njit(v_of_vp_lambdified)
dv_dvp_lambdified = njit(dv_dvp_lambdified)
d2v_dvp2_lambdified = njit(d2v_dvp2_lambdified)

vp_of_v_lambdified = njit(vp_of_v_lambdified)
dvp_dv_lambdified = njit(dvp_dv_lambdified)
d2vp_dv2_lambdified = njit(d2vp_dv2_lambdified)

# convert a vector with bipolar components (Fu, Fv) at coordinates (u,v) to cartesian coordinates (Fx, Fy)
@njit
def BP2cart(Fu, Fv, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return Fu * M_00 + Fv * M_01, Fu * M_10 + Fv * M_11

# convert a vector with cartesian components (Fx, Fy) at coordinates (u,v) to bipolar coordinates (Fu, Fv)
@njit
def cart2BP(Fx, Fy, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

# convert a vector with cartesian components (Fx, Fy) at coordinates (u,infty) to bipolar coordinates (Fu, Fv)
@njit
def cart2BPinfinity(Fx, Fy, u):
    M_00 = -np.sin(u)
    M_01 = -np.cos(u)
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11
