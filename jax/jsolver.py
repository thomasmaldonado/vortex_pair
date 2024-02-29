### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from janalytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified
from jderivatives import d1x, d2x, d1y, d2y
from jcoords import BP2cart, cart2BP, cart2BPinfinity
import numpy as np
import jax.numpy as jnp
from jax import jit 
from jax import vmap, pmap, grad, jacfwd, jacrev
import jax 
from jax.numpy import linalg as jla
from matplotlib import pyplot as plt
import time 
from functools import partial
from files import save
#from jaxopt import ScipyRootFinding, Broyden
jax.config.update('jax_platform_name', 'cpu')
# from matplotlib import pyplot as plt
# from scipy.optimize import fsolve
# from files import load, save
# from time import time
sys.path.append('../')
from coords import v_of_vp_lambdified, dvp_dv_lambdified, d2vp_dv2_lambdified, dv_dvp_lambdified


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

vs, dvp_dv, d2vp_dv2 = jnp.array(vs), jnp.array(dvp_dv), jnp.array(d2vp_dv2)

################

# define derivatives (note the chain rule employed in the d_dv function that enables conformal mapping)
def _d_du(f, du, n):
    boundary_left = f[-1,:]
    boundary_right = f[0,:]
    if n == 1:
        return d1x(f, du, boundary_left, boundary_right)
    if n == 2:
        return d2x(f, du, boundary_left, boundary_right)

@jit
def d_du1(f, du):
    return _d_du(f, du, n=1)

@jit
def d_du2(f, du):
    return _d_du(f, du, n=2)

@jit
def _d_dvp1(f, dvp, boundary_left, boundary_right):
    return d1y(f, dvp, boundary_left, boundary_right)

@jit
def _d_dvp2(f, dvp, boundary_left, boundary_right):
    return d2y(f, dvp, boundary_left, boundary_right)

@jit
def d_dv1(f, dvp_dv, dvp, boundary_left, boundary_right):
    df_dvp = _d_dvp1(f, dvp, boundary_left, boundary_right)
    return df_dvp * dvp_dv

@jit
def d_dv2(f, dvp_dv, d2vp_dv2, dvp, boundary_left, boundary_right):
    df_dvp = _d_dvp1(f, dvp, boundary_left, boundary_right)
    d2f_dvp2 = _d_dvp2(f, dvp, boundary_left, boundary_right)
    return d2f_dvp2 * dvp_dv**2 + df_dvp * d2vp_dv2

@jit
def dV_dv1(V, dvp_dv, dvp):
    boundary_left = jnp.pad(V[1:,0], (1,0), constant_values=(-1, -1))
    boundary_right = jnp.full(V.shape[0], jnp.mean(V[:,-1]))
    return d_dv1(V, dvp_dv, dvp, boundary_left, boundary_right)

@jit
def dV_dv2(V, dvp_dv, d2vp_dv2, dvp):
    boundary_left = jnp.pad(V[1:,0], (1,0), constant_values=(-1, -1))
    boundary_right = jnp.full(V.shape[0], jnp.mean(V[:,-1]))
    return d_dv2(V, dvp_dv, d2vp_dv2, dvp, boundary_left, boundary_right)

@jit
def dC_dv1(C, J, dvp_dv, dvp):
    boundary_left = jnp.pad(C[1:,0], (1,0), constant_values=(jnp.sqrt(-J), jnp.sqrt(-J)))
    boundary_right = jnp.zeros(C.shape[0])
    return d_dv1(C, dvp_dv, dvp, boundary_left, boundary_right)

@jit
def dC_dv2(C, J, dvp_dv, d2vp_dv2, dvp):
    boundary_left = jnp.pad(C[1:,0], (1,0), constant_values=(jnp.sqrt(-J), jnp.sqrt(-J)))
    boundary_right = jnp.zeros(C.shape[0])
    return d_dv2(C, dvp_dv, d2vp_dv2, dvp, boundary_left, boundary_right)

@jit
def dF_dv1(Fu, Fv, us, vsm1, dvp_dv, dvp):
    boundary_left_u = jnp.pad(Fu[1:,0], (1,0), constant_values=(0, 0))
    boundary_left_v = jnp.zeros(Fv.shape[0])
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vsm1)
    avg_x = jnp.mean(xs)
    avg_y = jnp.mean(ys)
    boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, us)
    Fu_result = d_dv1(Fu, dvp_dv, dvp, boundary_left_u, boundary_right_u)
    Fv_result = d_dv1(Fv, dvp_dv, dvp, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@jit
def dF_dv2(Fu, Fv, us, vsm1, dvp_dv, d2vp_dv2, dvp):
    boundary_left_u = jnp.pad(Fu[1:,0], (1,0), constant_values=(0, 0))
    boundary_left_v = jnp.zeros(Fv.shape[0])
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vsm1)
    avg_x = jnp.mean(xs)
    avg_y = jnp.mean(ys)
    boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, us)
    Fu_result = d_dv2(Fu, dvp_dv, d2vp_dv2, dvp, boundary_left_u, boundary_right_u)
    Fv_result = d_dv2(Fv, dvp_dv, d2vp_dv2, dvp, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

# helper function to unpack and reshape the solutions
def _unpack(V_Fu_Fv_C, NU, NV):
    NUNV = NU*NV
    V = jnp.reshape(V_Fu_Fv_C[0:NUNV], (NU,NV))
    Fu = jnp.reshape(V_Fu_Fv_C[NUNV:2*NUNV], (NU,NV))
    Fv = jnp.reshape(V_Fu_Fv_C[2*NUNV:3*NUNV], (NU,NV))
    C = jnp.reshape(V_Fu_Fv_C[3*NUNV:],(NU,NV))
    return V, Fu, Fv, C

@jit
def unpack(V_Fu_Fv_C):
    return _unpack(V_Fu_Fv_C, NU, NV)

# [V_vv, V_uu, V, j, C, v, a, u]
# first map v (which corresponds to j in loop, so axis 1 of V_vv etc), then map u (which corresponds to i in loop, so axis 0 of V_vv etc)
# eq0_V_func = vmap(eq0_V_lambdified, in_axes=(1, 1, 1, None, 1, 0, None, None))
# eq0_V_func = vmap(vmap(eq0_V_lambdified, in_axes=(0, 0, 0, None, 0, 0, None, None)), in_axes=(0, 0, 0, None, 0, None, None, 0))
vv, uu = jnp.meshgrid(vs, us)

# define function whose root we seek
def f(V_Fu_Fv_C, du, dvp_dv, dvp, vv, uu, J, A, N):
    V, Fu, Fv, C = unpack(V_Fu_Fv_C)
    V_uu = d_du2(V, du)
    V_vv = dV_dv2(V, dvp_dv, d2vp_dv2, dvp)
    Fu_u, Fv_u = d_du1(Fu, du), d_du1(Fv, du)
    Fu_v, Fv_v = dF_dv1(Fu, Fv, us, vv[0, -1], dvp_dv, dvp)
    Fv_uu = d_du2(Fv, du)
    Fu_uv, Fv_uv = d_du1(Fu_v, du), d_du1(Fv_v, du)
    Fu_vv, _ = dF_dv2(Fu, Fv, us, vv[0, -1], dvp_dv, d2vp_dv2, dvp)
    C_uu = d_du2(C, du)
    C_vv = dC_dv2(C, J, dvp_dv, d2vp_dv2, dvp)

    eq0_V = jnp.ravel(eq0_V_lambdified(V_vv, V_uu, V, J, C, vv, A, uu))
    eq0_Fu = jnp.ravel(eq0_Fu_lambdified(C, N, uu, Fv_v, Fv_uv, vv, Fu_vv, Fv_u, A, Fu))
    eq0_Fv = jnp.ravel(eq0_Fv_lambdified(vv, Fu_u, Fu_v, Fv_uu, C, Fv, A, Fu_uv, uu))
    eq0_C = jnp.ravel(eq0_C_lambdified(vv, Fu, C_uu, N, V, C, C_vv, Fv, A, uu))
    return jnp.concatenate((eq0_V, eq0_Fu, eq0_Fv, eq0_C))

@jit
def f_opt(V_Fu_Fv_C):
    return f(V_Fu_Fv_C, du, dvp_dv, dvp, vv, uu, J, A, N)

# reduced function whose root yields the electrostatic solution
def f_electrostatic(V_C, NU, NV, du, dvp_dv, dvp, vv, uu, J, A):
    NUNV = NU*NV
    V = jnp.reshape(V_C[0:NUNV], (NU,NV))
    C = jnp.reshape(V_C[NUNV:], (NU,NV))
    V_uu = d_du2(V, du)
    V_vv = dV_dv2(V, dvp_dv, d2vp_dv2, dvp)
    C_uu = d_du2(C, du)
    C_vv = dC_dv2(C, J, dvp_dv, d2vp_dv2, dvp)

    eq0_V = jnp.ravel(eq0_V_lambdified(V_vv, V_uu, V, J, C, vv, A, uu))
    eq0_C = jnp.ravel(eq0_C_lambdified(vv, 0, C_uu, 0, V, C, C_vv, 0, A, uu))
    return jnp.concatenate((eq0_V, eq0_C))

@jit 
def f_electrostatic_opt(V_C):
    return f_electrostatic(V_C, NU, NV, du, dvp_dv, dvp, vv, uu, J, A)

# # use initial guess for the electrostatic problem given by the bulk solution and perform Newton's method
jax.config.update("jax_enable_x64", True)

x0_electrostatic = np.zeros(2*NUNV)
x0_electrostatic[0:NUNV] = -1
x0_electrostatic[-NUNV:] = np.sqrt(-J)
x0_electrostatic = jnp.array(x0_electrostatic, dtype = jnp.float64)

def newton(f, x_0, tol=1e-4, max_iter=30):
    """
    A multivariate Newton root-finding routine.

    """
    x = x_0
    f_jac = jit(jax.jacobian(f))

    @jit
    def q(x):
        " Updates the current guess. "
        return x - jla.solve(f_jac(x), f(x))
    
    error = tol + 1
    n = 0
    while error > tol:
        n += 1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        error = jla.norm(x - y)
        x = y
        jax.debug.print('iteration {n}, error = {error}', n=n, error=error)
    return x

# electrostatic_solver = Broyden(f_electrostatic_opt, jit=True, verbose=False, maxiter=1000, maxls=15)

# electrostatic_solver = ScipyRootFinding('hybr', tol=tol, 
#                                         optimality_fun=f_electrostatic_opt, jit=True, use_jacrev=False)
# params, info = electrostatic_solver.run(init_params = x0_electrostatic) 

start = time.time()
VC = newton(f_electrostatic_opt, x0_electrostatic)
# state = electrostatic_solver.run(init_params = x0_electrostatic) 
# VC = state.params

# params, info = electrostatic_solver.run(init_params = x0_electrostatic) 
end = time.time()
print("Elapsed time for electrostatic solution: ", end - start)

NUNV = NU*NV
V = jnp.reshape(VC[0:NUNV], (NU,NV))
C = jnp.reshape(VC[NUNV:], (NU,NV))


Fu = jnp.zeros(NUNV, dtype = jnp.float32)
Fv = jnp.zeros(NUNV, dtype = jnp.float32)
x0_magnetostatic = jnp.concatenate((VC[0:NUNV], Fu, Fv, VC[-NUNV:]))
print(x0_magnetostatic.dtype)
start = time.time()
solution = newton(f_opt, x0_magnetostatic)
end = time.time()
print("Elapsed time for magnetostatic solution: ", end - start)

V, Fu, Fv, C = unpack(solution)
V_u = d_du1(V, du)
V_v = dV_dv1(V, dvp_dv, dvp)
V_uu = d_du2(V, du)
V_uv = d_du1(V_v, du)
V_vv = dV_dv2(V, dvp_dv, d2vp_dv2, dvp)
Fu_u, Fv_u = d_du1(Fu, du), d_du1(Fv, du)
Fu_v, Fv_v = dF_dv1(Fu, Fv, us, vs[-1], dvp_dv, dvp)
Fu_uu, Fv_uu = d_du2(Fu, du), d_du2(Fv, du)
Fu_uv, Fv_uv = d_du1(Fu_v, du), d_du1(Fv_v, du)
Fu_vv, Fv_vv = dF_dv2(Fu, Fv, us, vs[-1], dvp_dv, d2vp_dv2, dvp)
C_u = d_du1(C, du)
C_v = dC_dv1(C, J, dvp_dv, dvp)
C_uu = d_du2(C, du)
C_uv = d_du1(C_v, du)
C_vv = dC_dv2(C, J, dvp_dv, d2vp_dv2, dvp)

Eu = Eu_lambdified(uu, vv, A, V_u)
Ev = Ev_lambdified(V_v, vv, A, uu)
B = B_lambdified(vv, Fv_u, Fu_v, Fu, Fv, A, uu)

EED = (Eu**2 + Ev**2)/2
MED = (B**2)/2
HED = C**2 * V**2 + J

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
dA = jnp.array(dA)
EE = np.sum(EED * dA)
ME = np.sum(MED * dA)
HE = np.sum(HED*dA)

# save solution
save(file, A, K, N, NU, NV, 1, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED)




plt.imshow(V)
plt.savefig('testV')
plt.clf() 
plt.imshow(Fu)
plt.savefig('testFu')
plt.clf() 
plt.imshow(Fv)
plt.savefig('testFv')
plt.clf() 
plt.imshow(C)
plt.savefig('testC')
plt.clf()
plt.imshow(Eu)
plt.savefig('testEu')
plt.clf()
plt.imshow(Ev)
plt.savefig('testEv')
plt.clf()
plt.imshow(B)
plt.savefig('testB')
plt.clf()
exit()


if __name__ == '__main__':
    import solver 
    import time
    
    JIT_BEFORE = 1 

    @jit
    def get_jax_Ds(f, f2):
        d_du1_jax = d_du1(f, du)
        d_du2_jax = d_du2(f, du)
        dV_dv1_jax = dV_dv1(f, dvp_dv, dvp)
        dV_dv2_jax = dV_dv2(f, dvp_dv, d2vp_dv2, dvp)
        dC_dv1_jax = dC_dv1(f, J, dvp_dv, dvp)
        dC_dv2_jax = dC_dv2(f, J, dvp_dv, d2vp_dv2, dvp)
        dF_dv1_jax = dF_dv1(f, f2, us, vs[-1], dvp_dv, dvp)
        dF_dv2_jax = dF_dv2(f, f2, us, vs[-1], dvp_dv, d2vp_dv2, dvp)
        return d_du1_jax, d_du2_jax, dV_dv1_jax, dV_dv2_jax, dC_dv1_jax, dC_dv2_jax, dF_dv1_jax, dF_dv2_jax

    def D_test(output = True):
        f = np.random.rand(NU, NV)
        f2 = np.random.rand(NU, NV)

        t = time.time()
        d_du1_jax, d_du2_jax, dV_dv1_jax, dV_dv2_jax, dC_dv1_jax, dC_dv2_jax, dF_dv1_jax, dF_dv2_jax = get_jax_Ds(f, f2)
        if output: print("Elapsed time for jax functions: ", time.time()-t)

        t = time.time()
        d_du1_np = solver.d_du(f, 1)
        d_du2_np = solver.d_du(f, 2)
        dV_dv1_np = solver.dV_dv(f, 1)
        dV_dv2_np = solver.dV_dv(f, 2)
        dC_dv1_np = solver.dC_dv(f, 1)
        dC_dv2_np = solver.dC_dv(f, 2)
        dF_dv1_np = solver.dF_dv(f, f2, 1)
        dF_dv2_np = solver.dF_dv(f, f2, 2)
        if output: print("Elapsed time for original functions: ", time.time()-t)

        assert np.allclose(d_du1_jax, d_du1_np, atol=1e-6)
        assert np.allclose(d_du2_jax, d_du2_np, atol=1e-6)
        assert np.allclose(dV_dv1_jax, dV_dv1_np, atol=1e-6) 
        assert np.allclose(dV_dv2_jax, dV_dv2_np, atol=1e-6)
        assert np.allclose(dC_dv1_jax, dC_dv1_np, atol=1e-6)
        assert np.allclose(dC_dv2_jax, dC_dv2_np, atol=1e-6)
        assert np.allclose(dF_dv1_jax, dF_dv1_np, atol=1e-6)
        assert np.allclose(dF_dv2_jax, dF_dv2_np, atol=1e-6)

        if output: print("Passed!")

    def ftest(output = True):
        V_Fu_Fv_C = np.random.rand(4*NUNV)
        V_Fu_Fv_C_jax = jnp.array(V_Fu_Fv_C)

        t = time.time()
        result_jax = f_opt(V_Fu_Fv_C_jax)
        result_static_jax = f_electrostatic_opt(V_Fu_Fv_C_jax[0:2*NUNV])
        if output: print("Elapsed time for jax function: ", time.time()-t)

        t = time.time()
        result_np  = solver.f(V_Fu_Fv_C, N)
        result_static_np = solver.f_electrostatic(V_Fu_Fv_C[0:2*NUNV])
        if output: print("Elapsed time for original function: ", time.time()-t)

        assert np.allclose(result_jax, result_np, atol=1e-5)
        assert np.allclose(result_static_jax, result_static_np, atol=1e-5)

        if output: print("Passed!")

        jacfnc = jacrev(f_opt)
        jac = jacfnc(V_Fu_Fv_C_jax)
        print(jac.shape)

    if JIT_BEFORE:
        D_test(False)
        ftest(False)

    D_test()
    ftest()

    


exit()
# reduced function whose root yields the magnetostatic solution
def f_magnetostatic(V_Fu_Fv_C):
    return f(V_Fu_Fv_C, N)

# use initial guess for the magnetostatic problem given by the electrostatic solution and perform Newton's method
x0_magnetostatic = np.zeros(4*NUNV)
x0_magnetostatic[0:NUNV] = solution_electrostatic[0:NUNV]
x0_magnetostatic[-NUNV:] = solution_electrostatic[-NUNV:]
start = time()
solution, infodict, ier, mesg = fsolve(f_magnetostatic, x0_magnetostatic, full_output = True, xtol = tol)
end = time()
print(N, ier, mesg, end - start)

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


# @jit
# def dV_du1(V, du):
#     return d_du(V, du, n=1)

# @jit
# def dV_du2(V, du):
#     return d_du(V, du, n=2)

# @jit
# def dFu_du(Fu, n):
#     return d_du(Fu, n)

# @njit
# def dFv_du(Fv, n):
#     return d_du(Fv, n)

# @njit
# def dF_du(Fu, Fv, n):
#     return d_du(Fu, n), d_du(Fv, n)

# @njit
# def dC_du(C, n):
#     return d_du(C, n)
