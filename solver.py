### BOUNDARY VALUE PROBLEM SOLVER ###

from analytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified
from coords import v_of_vp_lambdified, dvp_dv_lambdified, d2vp_dv2_lambdified, dv_dvp_lambdified, BP2cart, cart2BP, cart2BPinfinity
from derivatives import d
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from files import load, save
from time import time
import sys

# command line arguments
A_idx = int(sys.argv[1])
K = float(sys.argv[2])
N = float(sys.argv[3])
NU = int(sys.argv[4])
NV = int(sys.argv[5])
file = 'data/' + sys.argv[6] + '.npy'

# default scipy tolerance
tol = 1.49012e-08 

# define separation between vortices (=2A) and background charge density (=J)
As = np.linspace(0, 2*K, 101)[1:]
A = As[A_idx]
J = -4/K**4

#construct bipolar coordinates
us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

du = us[1]-us[0]
dvp = vps[1]-vps[0]

dvp_dv = np.zeros((NU, NV))
d2vp_dv2 = np.zeros((NU, NV))

# define coordinate transformation based on conformal mapping defined in coords.py
NUNV = NU*NV
for j, v in enumerate(vs):
    args = (v, A, J)
    dvp_dv[:,j] = dvp_dv_lambdified(*args)
    d2vp_dv2[:,j] = d2vp_dv2_lambdified(*args)

# define derivatives (note the chain rule employed in the d_dv function that enables conformal mapping)
@njit
def d_du(f, n):
    boundary_left = np.ascontiguousarray(f[-1,:])
    boundary_right = np.ascontiguousarray(f[0,:])
    return d(f, du, n, boundary_left, boundary_right, axis = 0)

@njit
def d_dvp(f, n, boundary_left, boundary_right):
    return d(f, dvp, n, boundary_left, boundary_right, axis = 1)

@njit
def d_dv(f, n, boundary_left, boundary_right):
    if n == 0:
        return f
    df_dvp = d_dvp(f, 1, boundary_left, boundary_right)
    if n == 1:
        return df_dvp * dvp_dv
    df2_dx2 = d_dvp(f, 2, boundary_left, boundary_right)
    return df2_dx2 * dvp_dv**2 + df_dvp * d2vp_dv2

@njit
def dV_du(V, n):
    return d_du(V, n)

@njit
def dFu_du(Fu, n):
    return d_du(Fu, n)

@njit
def dFv_du(Fv, n):
    return d_du(Fv, n)

@njit
def dF_du(Fu, Fv, n):
    return d_du(Fu, n), d_du(Fv, n)

@njit
def dC_du(C, n):
    return d_du(C, n)

@njit
def dV_dv(V, n):
    boundary_left = np.ascontiguousarray(V[:,0])
    boundary_left[0] = -1
    boundary_right = np.full(NU, np.mean(V[:,-1]))
    return d_dv(V, n, boundary_left, boundary_right)

@njit
def dF_dv(Fu, Fv, n = 1):
    boundary_left_u = np.ascontiguousarray(Fu[:,0])
    boundary_left_u[0] = 0
    boundary_left_v = np.zeros(NU)
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vs[-1])
    avg_x = np.mean(xs)
    avg_y = np.mean(ys)
    boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, us)
    Fu_result = d_dv(Fu, n, boundary_left_u, boundary_right_u)
    Fv_result = d_dv(Fv, n, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@njit
def dC_dv(C, n = 1):
    boundary_left = np.ascontiguousarray(C[:,0])
    boundary_left[0] = np.sqrt(-J)
    boundary_right = np.zeros(np.shape(C)[0])
    return d_dv(C, n, boundary_left, boundary_right)

# helper function to unpack and reshape the solutions
@njit
def unpack(V_Fu_Fv_C):
    V = np.reshape(V_Fu_Fv_C[0:NUNV], (NU,NV))
    Fu = np.reshape(V_Fu_Fv_C[NUNV:2*NUNV], (NU,NV))
    Fv = np.reshape(V_Fu_Fv_C[2*NUNV:3*NUNV], (NU,NV))
    C = np.reshape(V_Fu_Fv_C[3*NUNV:],(NU,NV))
    return V, Fu, Fv, C

# define function whose root we seek
@njit(parallel = True)
def f(V_Fu_Fv_C, N):
    # unpack input and calculate derivatives
    V, Fu, Fv, C = unpack(V_Fu_Fv_C)
    eq0_V, eq0_Fu, eq0_Fv, eq0_C = np.zeros((NU, NV)), np.zeros((NU, NV)), np.zeros((NU, NV)), np.zeros((NU, NV))
    V_u = dV_du(V, n = 1)
    V_v = dV_dv(V, n = 1)
    V_uu = dV_du(V, n = 2)
    V_uv = dV_du(V_v, n = 1)
    V_vv = dV_dv(V, n = 2)
    Fu_u, Fv_u = dF_du(Fu, Fv, n = 1)
    Fu_v, Fv_v = dF_dv(Fu, Fv, n = 1)
    Fu_uu, Fv_uu = dF_du(Fu, Fv, n = 2)
    Fu_uv, Fv_uv = dF_du(Fu_v, Fv_v, n = 1)
    Fu_vv, Fv_vv = dF_dv(Fu, Fv, n = 2)
    C_u = dC_du(C, n = 1)
    C_v = dC_dv(C, n = 1)
    C_uu = dC_du(C, n = 2)
    C_uv = dC_du(C_v, n = 1)
    C_vv = dC_dv(C, n = 2)
    
    # evaluate sympy lambdified expressions at each point in space and return result
    args = np.zeros(29)
    for i in prange(NU):
        u = us[i]
        for j in prange(NV):
            v = vs[j]
            args = (V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j], Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j], Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j], C[i,j], C_u[i,j], C_v[i,j], C_uu[i,j], C_uv[i,j], C_vv[i,j], u, v, A, J, N)
            eq0_V[i,j] = eq0_V_lambdified(*args)
            eq0_Fu[i,j] = eq0_Fu_lambdified(*args)
            eq0_Fv[i,j] = eq0_Fv_lambdified(*args)
            eq0_C[i,j] = eq0_C_lambdified(*args)
    result = np.append(eq0_V.flatten(), eq0_Fu.flatten())
    result = np.append(result, eq0_Fv.flatten())
    result = np.append(result, eq0_C.flatten())
    return result

# reduced function whose root yields the electrostatic solution
def f_electrostatic(V_C):
    V, C = V_C[0:NUNV], V_C[NUNV:], 
    Fu, Fv = np.zeros(NUNV), np.zeros(NUNV)
    V_Fu_Fv_C = np.append(V, Fu)
    V_Fu_Fv_C = np.append(V_Fu_Fv_C, Fv)
    V_Fu_Fv_C = np.append(V_Fu_Fv_C, C)
    result = f(V_Fu_Fv_C, 0)
    eq0_V = result[0:NUNV]
    eq0_C = result[-NUNV:]
    return np.append(eq0_V, eq0_C)

# use initial guess for the electrostatic problem given by the bulk solution and perform Newton's method
x0_electrostatic = np.zeros(2*NUNV, dtype = float)
x0_electrostatic[0:NUNV] = -1
x0_electrostatic[-NUNV:] = np.sqrt(-J)
start = time()
solution_electrostatic, infodict, ier, mesg = fsolve(f_electrostatic, x0_electrostatic, full_output = True, xtol = tol)
end = time()
print(0, ier, mesg, end - start)
if ier != 1:
    raise Exception('Electrostatic solution not found')


# reduced function whose root yields the magnetostatic solution
def f_magnetostatic(V_Fu_Fv_C):
    return f(V_Fu_Fv_C, N)

# use initial guess for the magnetostatic problem given by the electrostatic solution and perform Newton's method
x0_magnetostatic = np.zeros(4*NUNV)
x0_magnetostatic[0:NUNV] = solution_electrostatic[0:NUNV]
x0_magnetostatic[-NUNV:] = solution_electrostatic[-NUNV:]
start = time()
solution, infodict, ier, mesg = fsolve(f_magnetostatic, x0_magnetostatic, full_output = True, xtol = tol)
print(N, ier, mesg, end - start)
end = time()

# take derivatives of solution for post-processing
V, Fu, Fv, C = unpack(solution)
V_u = dV_du(V, n = 1)
V_v = dV_dv(V, n = 1)
V_uu = dV_du(V, n = 2)
V_uv = dV_du(V_v, n = 1)
V_vv = dV_dv(V, n = 2)
Fu_u, Fv_u = dF_du(Fu, Fv, n = 1)
Fu_v, Fv_v = dF_dv(Fu, Fv, n = 1)
Fu_uu, Fv_uu = dF_du(Fu, Fv, n = 2)
Fu_uv, Fv_uv = dF_du(Fu_v, Fv_v, n = 1)
Fu_vv, Fv_vv = dF_dv(Fu, Fv, n = 2)
C_u = dC_du(C, n = 1)
C_v = dC_dv(C, n = 1)
C_uu = dC_du(C, n = 2)
C_uv = dC_du(C_v, n = 1)
C_vv = dC_dv(C, n = 2)

# calculate electric and magnetic fields
Eu, Ev, B = np.zeros((NU,NV)), np.zeros((NU,NV)), np.zeros((NU,NV))
for i, u in enumerate(us):
    for j, v in enumerate(vs):
        E_args = [V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j], u, v, A, N]
        Eu[i,j], Ev[i,j] = Eu_lambdified(*E_args), Ev_lambdified(*E_args)
        B_args = [Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j]]
        B_args.extend([Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j]])
        B_args.extend([u, v, A, N])
        B[i,j] = B_lambdified(*B_args)

# calculate electric, magnetic, and hydraulic energy densities
# note the bulk energy density (-J) is subtracted from the hydraulic energy density for normalization
EED = (Eu**2 + Ev**2)/2
MED = (B**2)/2
HED = C**2 * V**2 + J

# calculate area elements
dA = np.zeros((NU, NV))
for i in range(NU):
    u = us[i]
    for j in range(NV):
        v = vs[j]
        vp = vps[j]
        h = A / (np.cosh(v)-np.cos(u))
        args = (vp, A, J)
        dv = dv_dvp_lambdified(*args) * dvp
        dA[i,j] = du*dv * h**2

# calculate electric, magnetic, and hydraulic energies
EE = np.sum(EED * dA)
ME = np.sum(MED * dA)
HE = np.sum((HED)*dA)

# save solution
save(file, A, K, N, NU, NV, ier, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED)