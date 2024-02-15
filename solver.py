from analytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified
from derivatives import d
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from files import load, save
from time import time
import sys

A = float(sys.argv[1])
K = float(sys.argv[2])
N = float(sys.argv[3])
NU = int(sys.argv[4])
NX = NV = int(sys.argv[5])
STEPS = int(sys.argv[6])
file = sys.argv[7] + '.npy'

Ns = np.linspace(0, N, STEPS + 1)

for N in Ns:
    J = -4/K**4
    
    us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
    xs = np.linspace(0, A, NX+2)[1:-1]

    du = us[1]-us[0]
    dx = xs[1]-xs[0]

    dx_dv = np.zeros((NU, NX))
    dx2_dv2 = np.zeros((NU, NX))

    # define coordinate transformation
    NUNV = NU*NV
    vs = np.log((A+xs)/(A-xs))
    for j, v in enumerate(vs):
        dx_dv[:,j] = A / (1+np.cosh(v))
        dx2_dv2[:,j] = -A * np.sinh(v) / (1 + np.cosh(v))**2

    @njit
    def d_du(f, n):
        boundary_left = np.ascontiguousarray(f[-1,:])
        boundary_right = np.ascontiguousarray(f[0,:])
        return d(f, du, n, boundary_left, boundary_right, axis = 0)

    @njit
    def d_dx(f, n, boundary_left, boundary_right):
        return d(f, dx, n, boundary_left, boundary_right, axis = 1)

    @njit
    def d_dv(f, n, boundary_left, boundary_right):
        if n == 0:
            return f
        df_dx = d_dx(f, 1, boundary_left, boundary_right)
        if n == 1:
            return df_dx * dx_dv
        df2_dx2 = d_dx(f, 2, boundary_left, boundary_right)
        return df2_dx2 * dx_dv**2 + df_dx * dx2_dv2

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
    def BP2cart(Fu, Fv, u, v):
        M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
        M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
        M_10 = - M_01
        M_11 = M_00
        return Fu * M_00 + Fv * M_01, Fu * M_10 + Fv * M_11

    @njit
    def cart2BP(Fx, Fy, u, v):
        M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
        M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
        M_10 = - M_01
        M_11 = M_00
        MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
        return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

    @njit
    def cart2BPinfinity(Fx, Fy, u):
        M_00 = -np.sin(u)
        M_01 = -np.cos(u)
        M_10 = - M_01
        M_11 = M_00
        MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
        return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

    @njit
    def dV_dv(V, n):
        boundary_left = np.ascontiguousarray(V[:,0])
        boundary_left[0] = -1
        boundary_right = np.full(NU, np.mean(V[:,-1]))
        return d_dv(V, n, boundary_left, boundary_right)

    @njit(parallel = True)
    def dF_dv(Fu, Fv, n = 1):
        boundary_left_u = np.ascontiguousarray(Fu[:,0])
        boundary_left_u[0] = N / A
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

    @njit
    def unpack(V_Fu_Fv_C):
        V = np.reshape(V_Fu_Fv_C[0:NUNV], (NU,NV))
        Fu = np.reshape(V_Fu_Fv_C[NUNV:2*NUNV], (NU,NV))
        Fv = np.reshape(V_Fu_Fv_C[2*NUNV:3*NUNV], (NU,NV))
        C = np.reshape(V_Fu_Fv_C[3*NUNV:],(NU,NV))
        return V, Fu, Fv, C


    @njit(parallel = True)
    def f(V_Fu_Fv_C):
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

    @njit
    def f_reduced(V_C):
        V, C = V_C[0:NUNV], V_C[NUNV:], 
        Fu, Fv = np.zeros(NUNV), np.zeros(NUNV)
        V_Fu_Fv_C = np.append(V, Fu)
        V_Fu_Fv_C = np.append(V_Fu_Fv_C, Fv)
        V_Fu_Fv_C = np.append(V_Fu_Fv_C, C)
        result =  f(V_Fu_Fv_C)
        eq0_V = result[0:NUNV]
        eq0_C = result[-NUNV:]
        return np.append(eq0_V, eq0_C)
    
    if N == 0:
        x0_reduced = np.zeros(2*NUNV, dtype = float)
        x0_reduced[0:NUNV] = -1
        x0_reduced[-NUNV:] = np.sqrt(-J)
        start = time()
        solution_reduced, infodict, ier, mesg = fsolve(f_reduced, x0_reduced, full_output = True)
        end = time()
        solution = np.zeros(4*NUNV)
        solution[0:NUNV] = solution_reduced[0:NUNV]
        solution[-NUNV:] = solution_reduced[-NUNV:]
    else:
        start = time()
        solution, infodict, ier, mesg = fsolve(f, x0, full_output = True)
        end = time()
    print(N, ier, mesg, end - start)
    x0 = solution
    V, Fu, Fv, C = unpack(x0)
    """
    plt.imshow(V, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(Fu, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(Fv, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(C, cmap='hot', interpolation='nearest')
    plt.show()
    """
    if ier != 1:
        raise Exception(mesg)
save(file, A, K, N, NU, NV, ier, solution)