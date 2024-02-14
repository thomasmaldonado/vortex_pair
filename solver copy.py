from analytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified
from derivatives import d
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from files import load, save
import sys

a = 1/2
k = 1/np.sqrt(2)
k = 0.7
J = -4/k**4
Nf = 1
nu = 20 # even
nv = 20



@njit(cache = True)
def d_du(f, h, n, boundary_left, boundary_right):
    return d(f, h, n, boundary_left, boundary_right, axis = 0)




nx = nv
Xs = np.linspace(0, a, nv+2)[1:-1]
Vs = np.log((a+Xs)/(a-Xs))
#Vs = np.linspace(0, 2, nv + 1)[1:]
Us = np.linspace(0, 2*np.pi, nu + 1)[:-1]
dx = Xs[1]-Xs[0]
du = Us[1] - Us[0]
dv = Vs[1] - Vs[0]

dx_dv = np.zeros((nu, nv))
dx2_dv2 = np.zeros((nu, nv))
for j, v in enumerate(Vs):
    dx_dv[:,j] = a / (1+np.cosh(v))
    dx2_dv2[:,j] = -a * np.sinh(v) / (1 + np.cosh(v))**2

@njit
def d_dx(f, h, n, boundary_left, boundary_right):
    return d(f, h, n, boundary_left, boundary_right, axis = 1)

@njit
def d_dv(f, h, n, boundary_left, boundary_right):
    if n == 0:
        return f
    df_dx = d_dx(f, dx, 1, boundary_left, boundary_right)
    if n == 1:
        return df_dx * dx_dv
    df2_dx2 = d_dx(f, dx, 2, boundary_left, boundary_right)
    return df2_dx2 * dx_dv**2 + df_dx * dx2_dv2

@njit
def dfunc_du(func, n):
    boundary_left = np.ascontiguousarray(func[-1,:])
    boundary_right = np.ascontiguousarray(func[0,:])
    return d_du(func, du, n, boundary_left, boundary_right)

@njit
def dV_du(V, n):
    return dfunc_du(V, n)

@njit
def dFu_du(Fu, n):
    return dfunc_du(Fu, n)

@njit
def dFv_du(Fv, n):
    return dfunc_du(Fv, n)

@njit
def dF_du(Fu, Fv, n):
    return dfunc_du(Fu, n), dfunc_du(Fv, n)

@njit
def dC_du(C, n):
    return dfunc_du(C, n)







#dv = np.log((a+x_min)/(a-x_min))

for N in np.linspace(0, 1, 5):
    print(N)
    
    @njit
    def M(u,v):
        M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
        M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
        M_10 = - M_01
        M_11 = M_00
        return 

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
        boundary_right = np.full(np.shape(V)[0], np.mean(V[:,-1]))
        return d_dv(V, dv, n, boundary_left, boundary_right)

    @njit
    def dF_dv(Fu, Fv, n = 1):
        boundary_left_u = np.ascontiguousarray(Fu[:,0])
        boundary_left_u[0] = N / a
        boundary_left_v = np.zeros(nu)
        xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], Us, Vs[-1])
        avg_x = np.mean(xs)
        avg_y = np.mean(ys)
        boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, Us)
        Fu_result = d_dv(Fu, dv, n, boundary_left_u, boundary_right_u)
        Fv_result = d_dv(Fv, dv, n, boundary_left_v, boundary_right_v)
        return Fu_result, Fv_result

    @njit
    def dC_dv(C, n = 1):
        boundary_left = np.ascontiguousarray(C[:,0])
        boundary_left[0] = np.sqrt(-J)
        boundary_right = np.zeros(np.shape(C)[0])
        return d_dv(C, dv, n, boundary_left, boundary_right)


    @njit(parallel = True)
    def f(V_Fu_Fv_C):
        V = np.reshape(V_Fu_Fv_C[0:(nu*nv)], (nu,nv))
        Fu = np.reshape(V_Fu_Fv_C[(nu*nv):2*(nu*nv)], (nu,nv))
        Fv = np.reshape(V_Fu_Fv_C[2*(nu*nv):3*(nu*nv)], (nu,nv))
        C = np.reshape(V_Fu_Fv_C[3*(nu*nv):],(nu,nv))
        eq0_V = np.zeros((nu, nv))
        eq0_Fu = np.zeros((nu, nv))
        eq0_Fv = np.zeros((nu, nv))
        eq0_C = np.zeros((nu, nv))
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
        # define arguments for the lambdified expression from sympy and fill the result array
        args = np.zeros(29)
        for i in prange(nu):
            u = Us[i]
            for j in prange(nv):
                v = Vs[j]
                args = (V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j], Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j], Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j], C[i,j], C_u[i,j], C_v[i,j], C_uu[i,j], C_uv[i,j], C_vv[i,j], u, v, a, J, N)
                eq0_V[i,j] = eq0_V_lambdified(*args)
                eq0_Fu[i,j] = eq0_Fu_lambdified(*args)
                eq0_Fv[i,j] = eq0_Fv_lambdified(*args)
                eq0_C[i,j] = eq0_C_lambdified(*args)
        result = np.append(eq0_V.flatten(), eq0_Fu.flatten())
        result = np.append(result, eq0_Fv.flatten())
        result = np.append(result, eq0_C.flatten())
        return result


    # solve nonlinear problem

    def f_reduced(V_C):
        V, C = V_C[0:nu*nv], V_C[nu*nv:], 
        Fu, Fv = np.zeros(nu*nv), np.zeros(nu*nv)
        V_Fu_Fv_C = np.concatenate([V,Fu,Fv,C])
        result =  f(V_Fu_Fv_C)
        eq0_V = result[0:nu*nv]
        eq0_C = result[-nu*nv:]
        return np.concatenate([eq0_V, eq0_C])

    if not N:
        print('Electrostatic')
        x0_reduced = np.zeros(2*nu*nv)
        x0_reduced[0:nu*nv] = -1
        x0_reduced[-nu*nv:] = np.sqrt(-J)
        solution, infodict, ier, mesg = fsolve(f_reduced, x0_reduced, full_output = True)
        print(mesg)
        V = np.reshape(solution[0:(nu*nv)], (nu,nv))
        Fu = np.zeros((nu,nv))
        Fv = np.zeros((nu,nv))
        C = np.reshape(solution[-(nu*nv):], (nu,nv))
    else:
        V0 = V
        Fu0 = Fu
        Fv0 = Fv
        C0 = C
        plt.imshow(V0, interpolation='nearest')
        plt.show()
        plt.imshow(Fu0, interpolation='nearest')
        plt.show()
        plt.imshow(Fv0, interpolation='nearest')
        plt.show()
        plt.imshow(C0, interpolation='nearest')
        plt.show()
        x0 = np.concatenate([V0.flatten(), Fu0.flatten(), Fv0.flatten(), C0.flatten()])
        solution, infodict, ier, mesg = fsolve(f, x0, full_output = True)
        print(mesg)
        V = np.reshape(solution[0:(nu*nv)], (nu,nv))
        Fu = np.reshape(solution[(nu*nv):2*(nu*nv)], (nu,nv))
        Fv = np.reshape(solution[2*(nu*nv):3*(nu*nv)], (nu,nv))
        C = np.reshape(solution[3*(nu*nv):], (nu,nv))
    
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(Fu, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(Fv, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(C, cmap='hot', interpolation='nearest')
plt.show()
