from sympy import symbols, Function, cos, sin, diff
from sympy.utilities import lambdify
from sympy.vector import CoordSys3D, is_solenoidal, curl, Del
from sympy.functions.elementary.hyperbolic import cosh, sinh

# define symbols and functions
V = Function('V')
Fu = Function('Fu')
Fv = Function('Fv')
C = Function('C')

V_syms = symbols('V, V_u, V_v, V_uu, V_uv, V_vv')
Fu_syms = symbols('Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv')
Fv_syms = symbols('Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv')
C_syms = symbols('C, C_u, C_v, C_uu, C_uv, C_vv')

u, v, a, j, n= symbols('u v a j n')

# helper function to turn functions into symbols
def symbolify(expr, fun, syms):
    expr = expr.subs(fun.diff(v,2), syms[-1])
    expr = expr.subs(fun.diff(u).diff(v), syms[-2])
    expr = expr.subs(fun.diff(u,2), syms[-3])
    expr = expr.subs(fun.diff(v), syms[-4])
    expr = expr.subs(fun.diff(u), syms[-5])
    expr = expr.subs(fun, syms[-6])
    return expr

from sympy.vector import CoordSys3D
BP = CoordSys3D('BP')
u_hat, v_hat, z_hat = BP.i, BP.j, BP.k

h = a/(cosh(v)-cos(u))

def gradient(f):
    return (diff(f, u)*BP.i + diff(f, v)*BP.j)/h

def div(F):
    Fu = F.dot(u_hat)
    Fv = F.dot(v_hat)
    return (diff(Fu*h, u) + diff(Fv*h, v)) / h**2

def curl(F):
    Fu = F.dot(u_hat)
    Fv = F.dot(v_hat)
    Fz = F.dot(z_hat)
    curl_u = u_hat * diff(Fz, v) / h
    curl_v = - v_hat * diff(Fz, u) / h 
    curl_z = z_hat * (diff(h * Fv, u) - diff(h * Fu, v)) / h**2
    return curl_u + curl_v + curl_z

def laplacian(f):
    return div(gradient(f))

Au = Fu(u,v) - n * cosh(v) / a
Av = Fv(u,v)

#eq0_Fu = (1/h)*diff((1/h**2)*(diff(h*Fv(u,v), u) - diff(h * (Fu(u,v) - n*cosh(v)/a), v)),v) + C(u,v)**2*(Fu(u,v) - n*cosh(v)/a)
#eq0_Fu = eq0_Fu.doit().simplify()
eq0_V = (-laplacian(V(u,v)) + C(u,v)**2 * V(u,v) - j).doit()
eq0_F = (curl(curl(Au*u_hat + Av*v_hat).simplify()) + C(u,v)**2 * (Au*u_hat + Av*v_hat)).doit().expand()
eq0_Fu = eq0_F.dot(u_hat)
eq0_Fv = eq0_F.dot(v_hat)
eq0_C = (-laplacian(C(u,v)) + (1 - V(u,v)**2 + Au**2 + Av**2) * C(u,v)).doit()

"""
# ELECTROSTATIC REDUCTION
eq0_V = eq0_V.subs(As(s,t), 0).subs(At(s,t), 0).doit()
eq0_As = eq0_As.subs(As(s,t), 0).subs(At(s,t), 0).doit()
eq0_At = eq0_At.subs(As(s,t), 0).subs(At(s,t), 0).doit()
#eq0_C = eq0_C.subs(As(s,t), cosh(t)/a).subs(At(s,t), 0).doit()
eq0_C = eq0_C.subs(As(s,t), 0).subs(At(s,t), 0).doit()
"""

eq0_V = symbolify(eq0_V, V(u,v), V_syms)
eq0_V = symbolify(eq0_V, C(u,v), C_syms)

eq0_Fu = symbolify(eq0_Fu, Fu(u,v), Fu_syms)
eq0_Fu = symbolify(eq0_Fu, Fv(u,v), Fv_syms)
eq0_Fu = symbolify(eq0_Fu, C(u,v), C_syms)

eq0_Fv = symbolify(eq0_Fv, Fu(u,v), Fu_syms)
eq0_Fv = symbolify(eq0_Fv, Fv(u,v), Fv_syms)
eq0_Fv = symbolify(eq0_Fv, C(u,v), C_syms)

eq0_C = symbolify(eq0_C, V(u,v), V_syms)
eq0_C = symbolify(eq0_C, Fu(u,v), Fu_syms)
eq0_C = symbolify(eq0_C, Fv(u,v), Fv_syms)
eq0_C = symbolify(eq0_C, C(u,v), C_syms)

#print('V comp')
#display(eq0_V)
#print('Fu comp')
#display(eq0_Fu)
#print('Fv comp')
#display(eq0_Fv)
#print('C comp')
#display(eq0_C)

B = curl(Au*u_hat+Av*v_hat).dot(z_hat).simplify()
B = symbolify(B, Fu(u,v), Fu_syms)
B = symbolify(B, Fv(u,v), Fv_syms)
#print('B')
#display(B)

args = list(V_syms)
args.extend(Fu_syms)
args.extend(Fv_syms)
args.extend(C_syms)
args.extend([u, v, a, j, n])

#print(args)
#print(len(args))

eq0_V_lambdified = lambdify(args, eq0_V, 'numpy')
eq0_Fu_lambdified = lambdify(args, eq0_Fu, 'numpy')
eq0_Fv_lambdified = lambdify(args, eq0_Fv, 'numpy')
eq0_C_lambdified = lambdify(args, eq0_C, 'numpy')

B_args = list(Fu_syms)
B_args.extend(Fv_syms)
B_args.extend([u,v,a,n])
#print(B_args)
#print(len(B_args))
B_lambdified = lambdify(B_args, B)

from numba import njit, jit, typeof, prange
eq0_V_lambdified = njit(eq0_V_lambdified)
eq0_Fu_lambdified = njit(eq0_Fu_lambdified)
eq0_Fv_lambdified = njit(eq0_Fv_lambdified)
eq0_C_lambdified = njit(eq0_C_lambdified)

from matplotlib import pyplot as plt
from time import time
import numpy as np
from scipy.optimize import fsolve

# central difference scheme 
# axis = axis along which derivatives are taken

@njit(cache = True)
def d_du(f, h, n, boundary_left, boundary_right):
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
    f_b = f_b.transpose()
    return f_b[1:-1,:]

@njit(cache = True)
def d_dv(f, h, n, boundary_left, boundary_right):
    return d_du(f.transpose(), h, n, boundary_left, boundary_right).transpose()

a = 10
k = 1/np.sqrt(2)
J = -4/k**4
N = 1
nu = 20 # even
nv = 40
#dv = np.log((a+x_min)/(a-x_min))

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

@njit(cache = True)
def d_dx(f, h, n, boundary_left, boundary_right):
    return d_du(f.transpose(), h, n, boundary_left, boundary_right).transpose()

@njit(cache = True)
def d_dv(f, h, n, boundary_left, boundary_right):
    if n == 0:
        return f
    df_dx = d_dx(f, dx, 1, boundary_left, boundary_right)
    if n == 1:
        return df_dx * dx_dv
    df2_dx2 = d_dx(f, dx, 2, boundary_left, boundary_right)
    return df2_dx2 * dx_dv**2 + df_dx * dx2_dv2

dAs = np.zeros((nu, nv))
for i, u in enumerate(Us):
    for j, v in enumerate(Vs):
        h = a / (np.cosh(v) - np.cos(u))
        dAs[i,j] = h**2 * du * dv
print(np.max(dAs))


radii = a/np.sinh(Vs)
print(radii)

@njit(cache = True)
def dfunc_du(func, n):
    boundary_left = np.ascontiguousarray(func[-1,:])
    boundary_right = np.ascontiguousarray(func[0,:])
    return d_du(func, du, n, boundary_left, boundary_right)

@njit(cache = True)
def dV_du(V, n):
    return dfunc_du(V, n)

@njit(cache = True)
def dFu_du(Fu, n):
    return dfunc_du(Fu, n)

@njit(cache = True)
def dFv_du(Fv, n):
    return dfunc_du(Fv, n)

@njit(cache = True)
def dF_du(Fu, Fv, n):
    return dfunc_du(Fu, n), dfunc_du(Fv, n)

@njit(cache = True)
def dC_du(C, n):
    return dfunc_du(C, n)

@njit(cache = True)
def M(u,v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return 

@njit(cache = True)
def BP2cart(Fu, Fv, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return Fu * M_00 + Fv * M_01, Fu * M_10 + Fv * M_11
    #return np.dot(M(u,v), np.array([Fu, Fv]))

@njit(cache = True)
def cart2BP(Fx, Fy, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11
    #return np.dot(M(u,v).transpose(), np.array([Fx, Fy]))

@njit(cache = True)
def cart2BPinfinity(Fx, Fy, u):
    M_00 = -np.sin(u)
    M_01 = -np.cos(u)
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11


@njit(cache = True)
def dV_dv(V, n):
    boundary_left = np.ascontiguousarray(V[:,0])
    boundary_left[0] = -1
    boundary_right = np.full(np.shape(V)[0], np.mean(V[:,-1]))
    return d_dv(V, dv, n, boundary_left, boundary_right)

@njit(parallel = True, cache = True)
def dF_dv(Fu, Fv, n = 1):
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], Us, Vs[-1])
    avg_x = np.mean(xs)
    avg_y = np.mean(ys)
    boundary_left_u = np.ascontiguousarray(Fu[:,0])
    boundary_left_u[0] = N / a
    boundary_left_v = np.zeros(nu)
    #boundary_right_u, boundary_right_v =  cart2BP(avg_x, avg_y, Us, Vs[-1]+dv)
    boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, Us)
    Fu_result = d_dv(Fu, dv, n, boundary_left_u, boundary_right_u)
    Fv_result = d_dv(Fv, dv, n, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@njit(cache = True)
def dC_dv(C, n = 1):
    boundary_left = np.ascontiguousarray(C[:,0])
    boundary_left[0] = np.sqrt(-J)
    boundary_right = np.zeros(np.shape(C)[0])
    return d_dv(C, dv, n, boundary_left, boundary_right)


@njit(parallel = True, cache = True)
def f(V_Fu_Fv_C):
    #atime = time()
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
    #btime = time()
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
    #ctime = time()
    #print('b-a', btime - atime)
    #print('c-b', ctime - btime)
    #print(atime, btime, ctime)
    #print('ab', btime - atime)
    #print('bc', ctime - btime)
    return result


@njit(cache = True)
def parta(V_Fu_Fv_C):
    V = np.reshape(V_Fu_Fv_C[0:(nu*nv)], (nu,nv))
    Fu = np.reshape(V_Fu_Fv_C[(nu*nv):2*(nu*nv)], (nu,nv))
    Fv = np.reshape(V_Fu_Fv_C[2*(nu*nv):3*(nu*nv)], (nu,nv))
    C = np.reshape(V_Fu_Fv_C[3*(nu*nv):],(nu,nv))
    V_u = dV_du(V, n = 1)
    V_v = dV_dv(V, n = 1)
    V_uu = dV_du(V, n = 2)
    V_uv = dV_du(V_v, n = 1)
    V_vv = dV_dv(V, n = 2)
    #a2 = time()
    Fu_u, Fv_u = dF_du(Fu, Fv, n = 1)
    #a3 = time()
    Fu_v, Fv_v = dF_dv(Fu, Fv, n = 1)
    #a4 = time()
    Fu_uu, Fv_uu = dF_du(Fu, Fv, n = 2)
    #a5 = time()
    Fu_uv, Fv_uv = dF_du(Fu_v, Fv_v, n = 1)
    #a6 = time()
    Fu_vv, Fv_vv = dF_dv(Fu, Fv, n = 2)
    #a7 = time()
    C_u = dC_du(C, n = 1)
    C_v = dC_dv(C, n = 1)
    C_uu = dC_du(C, n = 2)
    C_uv = dC_du(C_v, n = 1)
    C_vv = dC_dv(C, n = 2)
    #a4 = time()
    #print('2', a3-a2) 
    #print('3', a4-a3) 
    #print('4', a5-a4) 
    #print('5', a6-a5) 
    #print('6', a7 - a6)
    return V, V_u, V_v, V_uu, V_uv, V_vv, Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv, Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv, C, C_u, C_v, C_uu, C_uv, C_vv

@njit(parallel = True, cache = True)
def partb(derivatives):
    V, V_u, V_v, V_uu, V_uv, V_vv, Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv, Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv, C, C_u, C_v, C_uu, C_uv, C_vv = derivatives
    eq0_V = np.zeros((nu, nv))
    eq0_Fu = np.zeros((nu, nv))
    eq0_Fv = np.zeros((nu, nv))
    eq0_C = np.zeros((nu, nv))
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
    #ctime = time()
    #print('b-a', btime - atime)
    #print('c-b', ctime - btime)
    #print(atime, btime, ctime)
    #print('ab', btime - atime)
    #print('bc', ctime - btime)
    return result

def g(V_Fu_Fv_C):
    timea = time()
    derivatives = parta(V_Fu_Fv_C)
    timeb = time()
    result = partb(derivatives)
    timec = time()
    print('part a', timeb-timea)
    print('part b', timec-timeb)
    return result
"""
def g(V_Fu_Fv_C):
    start = time()
    result = f(V_Fu_Fv_C)
    end = time()
    print(start, end)
    return result
"""

# solve nonlinear problem
x0 = np.zeros(4*nu*nv, dtype = float)
x0[0:(nu*nv)] = -1
Fu0 = np.zeros((nu,nv))
for j, v in enumerate(Vs):
    Fu0[:, j] = N * np.cosh(v) / a
x0[(nu*nv):2*(nu*nv)] = Fu0.flatten()
x0[3*(nu*nv):] = np.sqrt(-J)
start = time()
solution, infodict, ier, mesg = fsolve(f, x0, full_output = True)
end = time()
print(mesg)
print(end-start)

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