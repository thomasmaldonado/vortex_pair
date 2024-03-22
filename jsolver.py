### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter, NL, NR, NU, NV
from janalytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified
from jderivatives import d_dx1, d_dx2, d_dy1, d_dy2
from jcoords import BP2cart, cart2BP, cart2BPinfinity, v_of_vp_lambdified, dvp_dv1_lambdified, dvp_dv2_lambdified, dv_dvp1_lambdified
import numpy as np
import jax.numpy as jnp
from jax import jit 
import jax
from jax.numpy import linalg as jla
import time 
from files import save, load
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

# command line arguments
K_idx = int(sys.argv[1])
A_idx = int(sys.argv[2])
outputfile = 'data/' + sys.argv[3] + '.npy'
try:
    inputfile = 'data/' + sys.argv[4] + '.npy'
except:
    inputfile = None

# define separation between vortices (=2A) and background charge density (=J)
K = K_func(K_idx, A_idx)
A = A_func(K_idx, A_idx)
J = -4/K**4
N = NR
if np.abs(NL) != np.abs(NR):
    raise Exception("Only winding numbers of equal magnitude are supported")
samesign = (NL / NR == 1)

#construct bipolar coordinates
us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

du = us[1]-us[0]
dvp = vps[1]-vps[0]
vv, uu = jnp.meshgrid(vs, us)
hh = A / (np.cosh(vv)-np.cos(uu))
xx, yy = hh*np.sinh(vv), hh*np.sin(uu)

du = ((jnp.roll(uu, -1, axis = 0) - jnp.roll(uu, 1, axis = 0))%(2*jnp.pi))/2
dvp = dvp*jnp.ones((NU,NV))

# define coordinate transformation based on conformal mapping defined in coords.py
NUNV = NU*NV
dvp_dv1 = np.zeros((NU, NV))
dvp_dv2 = np.zeros((NU, NV))
for j, v in enumerate(vs):
    args = (v, A, J)
    dvp_dv1[:,j] = dvp_dv1_lambdified(*args)
    dvp_dv2[:,j] = dvp_dv2_lambdified(*args)
vs, dvp_dv1, dvp_dv2 = jnp.array(vs), jnp.array(dvp_dv1), jnp.array(dvp_dv2)

# define derivatives (note the chain rule employed in the d_dv1 and d_dv2 functions that enables conformal mapping)
@jit
def d_du1(f):
    boundary_left = f[-1,:]
    boundary_right = f[0,:]
    return d_dx1(f, du, boundary_left, boundary_right)

@jit
def d_du2(f):
    boundary_left = f[-1,:]
    boundary_right = f[0,:]
    return d_dx2(f, du, boundary_left, boundary_right)

@jit
def d_dv1(f, boundary_left, boundary_right):
    df_dvp1 = d_dy1(f, dvp, boundary_left, boundary_right)
    return df_dvp1 * dvp_dv1

@jit
def d_dv2(f, boundary_left, boundary_right):
    df_dvp1 = d_dy1(f, dvp, boundary_left, boundary_right)
    df_dvp2 = d_dy2(f, dvp, boundary_left, boundary_right)
    return df_dvp2 * dvp_dv1**2 + df_dvp1 * dvp_dv2

@jit
def dV_dv1(V):
    boundary_left = jnp.pad(V[1:,0], (1,0), constant_values=(-1, -1))
    boundary_right = jnp.full(V.shape[0], jnp.mean(V[:,-1]))
    return d_dv1(V, boundary_left, boundary_right)

@jit
def dV_dv2(V):
    boundary_left = jnp.pad(V[1:,0], (1,0), constant_values=(-1, -1))
    boundary_right = jnp.full(V.shape[0], jnp.mean(V[:,-1]))
    return d_dv2(V, boundary_left, boundary_right)

@jit
def dC_dv1(C):
    boundary_left = jnp.pad(C[1:,0], (1,0), constant_values=(jnp.sqrt(-J), jnp.sqrt(-J)))
    boundary_right = jnp.zeros(C.shape[0])
    return d_dv1(C, boundary_left, boundary_right)

@jit
def dC_dv2(C):
    boundary_left = jnp.pad(C[1:,0], (1,0), constant_values=(jnp.sqrt(-J), jnp.sqrt(-J)))
    boundary_right = jnp.zeros(C.shape[0])
    return d_dv2(C, boundary_left, boundary_right)

if samesign:
    @jit
    def boundary_left_F(Fu, Fv):
        boundary_left_u = (N/A) * (1-jnp.cos(us))
        boundary_left_v = Fv[:,0] #jnp.pad(Fv[1:,0], (1,0), constant_values=(0, 0))
        return boundary_left_u, boundary_left_v
else:
    @jit
    def boundary_left_F(Fu, Fv):
        boundary_left_u = Fu[:,0] #jnp.pad(Fu[1:,0], (1,0), constant_values=(0, 0))
        boundary_left_v = jnp.zeros(Fv.shape[0])
        return boundary_left_u, boundary_left_v

@jit
def dF_dv1(Fu, Fv):
    boundary_left_u, boundary_left_v = boundary_left_F(Fu, Fv)
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vs[-1])
    boundary_right_u, boundary_right_v =  cart2BPinfinity(jnp.mean(xs), jnp.mean(ys), us)
    Fu_result = d_dv1(Fu, boundary_left_u, boundary_right_u)
    Fv_result = d_dv1(Fv, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@jit
def dF_dv2(Fu, Fv):
    boundary_left_u, boundary_left_v = boundary_left_F(Fu, Fv)
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vs[-1])
    boundary_right_u, boundary_right_v =  cart2BPinfinity(jnp.mean(xs), jnp.mean(ys), us)
    Fu_result = d_dv2(Fu, boundary_left_u, boundary_right_u)
    Fv_result = d_dv2(Fv, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

# helper functions to pack/unpack and reshape the solutions
@jit
def pack_electrostatic(V, C):
    return jnp.array(jnp.concatenate((jnp.ravel(V), jnp.ravel(C))))

@jit
def pack_magnetostatic(V, Fu, Fv, C):
    return jnp.array(jnp.concatenate((jnp.ravel(V), jnp.ravel(Fu), jnp.ravel(Fv), jnp.ravel(C))))

@jit
def unpack_electrostatic(V_C):
    V = jnp.reshape(V_C[0:NUNV], (NU,NV))
    C = jnp.reshape(V_C[NUNV:], (NU,NV))
    return V, C

@jit
def unpack_magnetostatic(V_Fu_Fv_C):
    V = jnp.reshape(V_Fu_Fv_C[0:NUNV], (NU,NV))
    Fu = jnp.reshape(V_Fu_Fv_C[NUNV:2*NUNV], (NU,NV))
    Fv = jnp.reshape(V_Fu_Fv_C[2*NUNV:3*NUNV], (NU,NV))
    C = jnp.reshape(V_Fu_Fv_C[3*NUNV:],(NU,NV))
    return V, Fu, Fv, C

# define function whose root yields the electrostatic solution (NL = NR = 0)
@jit
def f_electrostatic(V_C):
    V, C = unpack_electrostatic(V_C)
    V_uu = d_du2(V)
    V_vv = dV_dv2(V)
    C_uu = d_du2(C)
    C_vv = dC_dv2(C)
    eq0_V = eq0_V_lambdified(V_vv, V_uu, V, J, C, vv, A, uu)
    eq0_C = eq0_C_lambdified(vv, 0, C_uu, 0, V, C, C_vv, 0, A, uu)
    return pack_electrostatic(eq0_V, eq0_C)

# define function whose root yields the magnetostatic solution
@jit
def f_magnetostatic(V_Fu_Fv_C):
    V, Fu, Fv, C = unpack_magnetostatic(V_Fu_Fv_C)
    V_uu = d_du2(V)
    V_vv = dV_dv2(V)
    Fu_u, Fv_u = d_du1(Fu), d_du1(Fv)
    Fu_v, Fv_v = dF_dv1(Fu, Fv)
    Fv_uu = d_du2(Fv)
    Fu_uv, Fv_uv = d_du1(Fu_v), d_du1(Fv_v)
    Fu_vv, _ = dF_dv2(Fu, Fv)
    C_uu = d_du2(C)
    C_vv = dC_dv2(C)
    eq0_V = eq0_V_lambdified(V_vv, V_uu, V, J, C, vv, A, uu)
    eq0_Fu = eq0_Fu_lambdified(C, N, uu, Fv_v, Fv_uv, vv, Fu_vv, Fv_u, A, Fu)
    eq0_Fv = eq0_Fv_lambdified(vv, Fu_u, Fu_v, Fv_uu, C, Fv, A, Fu_uv, uu)
    eq0_C = eq0_C_lambdified(vv, Fu, C_uu, N, V, C, C_vv, Fv, A, uu)
    return pack_magnetostatic(eq0_V, eq0_Fu, eq0_Fv, eq0_C)

# Newton's method for a function f, initial guess x_0, tolerance tol, and maximum number of iterations max_iter
def newton(f, x_0, tol=tol, max_iter=max_iter):
    x = x_0
    f_jac = jit(jax.jacobian(f))
    #@jit
    def q(x):
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

try:
    # use input file solution as initial guess for magnetostatic problem
    _, _, _, _, _, _, _, _, _, _, _, _, V0, Fu0, Fv0, C0, _, _, _, _, _, _, _ = load(inputfile)
except:
    # use bulk solution as initial guess for the electrostatic problem and perform Newton's method
    V0 = jnp.full((NU, NV), -1)
    C0 = jnp.full((NU, NV), jnp.sqrt(-J))
    x0 = pack_electrostatic(V0, C0)
    start = time.time()
    electrostatic_solution = newton(f_electrostatic, x0)
    end = time.time()
    print("Elapsed time for electrostatic solution: ", end - start)

    # use electrostatic solution as initial guess for magnetostatic problem
    V0, C0 = unpack_electrostatic(electrostatic_solution)
    Fu0 , Fv0 = jnp.zeros((NU, NV)), jnp.zeros((NU, NV))

# perform Newton's method 
x0 = pack_magnetostatic(V0, Fu0, Fv0, C0)
start = time.time()
magnetostatic_solution = newton(f_magnetostatic, x0)
end = time.time()
print("Elapsed time for magnetostatic solution: ", end - start)

start_processing = time.time()

# begin post-processing 
V, Fu, Fv, C = unpack_magnetostatic(magnetostatic_solution)

# calculate supercurrent
Au = Fu - (N/A) * (np.cosh(vv) - np.cos(uu))
Av = Fv

J0 = -C**2 * V
Ju = -C**2 * Au
Jv = -C**2 * Av

# calculate energy densities
V_u = d_du1(V)
V_v = dV_dv1(V)
Fu_u, Fv_u = d_du1(Fu), d_du1(Fv)
Fu_v, Fv_v = dF_dv1(Fu, Fv)

Eu = Eu_lambdified(uu, vv, A, V_u)
Ev = Ev_lambdified(V_v, vv, A, uu)

def get_B():
    _, vps_mesh = jnp.meshgrid(us, vps, indexing = 'ij')
    dv_mesh = dv_dvp1_lambdified(vps_mesh, A, J) * dvp
    Jx, Jy = BP2cart(Ju, Jv, uu, vv)
    measure = hh**2*du*dv_mesh/(2*jnp.pi)

    if samesign:
        bl_Ju, bl_Jv = jnp.zeros(NU), jnp.pad(Jv[1:,0], (1,0), constant_values=(0, 0))
    else:
        bl_Ju, bl_Jv = jnp.pad(Ju[1:,0], (1,0), constant_values=(0, 0)), jnp.zeros(NU)
    bl_Jx, bl_Jy = BP2cart(bl_Ju, bl_Jv, us, 0)
    bl_h = jnp.pad(A / (jnp.cosh(0) - jnp.cos(us[1:])), (1,0), constant_values=(0, 0))
    bl_measure = bl_h**2 * du * dv_dvp1_lambdified(0, A, J) * dvp / (2*jnp.pi)

    if samesign:
        Ju_minus, Jv_minus = -Ju, Jv
    else:
        Ju_minus, Jv_minus = Ju, -Jv
    Jx_minus, Jy_minus = BP2cart(Ju_minus, Jv_minus, uu, -vv)
    measure_minus = measure

    @jit
    def B_func(u,v):
        h = A / (jnp.cosh(v) - jnp.cos(u))
        x, y = h*jnp.sinh(v), h*jnp.sin(u)
        rx = x - xx
        ry = y - yy
        integrand = (Jx*ry - Jy*rx)/(rx**2 + ry**2)*measure
        summed = jnp.sum(integrand, where = ((u!=uu) | (v!=vv)))

        bl_x, bl_y = bl_h*jnp.sinh(0), bl_h*jnp.sin(us)
        bl_rx, bl_ry = x - bl_x, y - bl_y
        bl_integrand = ((bl_Jx*bl_ry - bl_Jy*bl_rx)/(bl_rx**2 + bl_ry**2)*bl_measure)
        summed += jnp.sum(bl_integrand, where = (us!=0)) # + (bl_integrand[1] + bl_integrand[-1])/2

        xx_minus = -xx
        yy_minus = yy
        rx_minus = x - xx_minus
        ry_minus = y - yy_minus
        integrand_minus = (Jx_minus*ry_minus - Jy_minus*rx_minus)/(rx_minus**2 + ry_minus**2)*measure_minus
        summed += jnp.sum(integrand_minus)

        return summed

    B = np.zeros((NU,NV))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            B[i,j] = B_func(u,v)
    return B

B = get_B()

EED = (Eu**2 + Ev**2)/2
MED = (B**2)/2
HED = C**2 * V**2 + J
TED = EED + MED + HED


_, vps_mesh = jnp.meshgrid(us, vps, indexing = 'ij')
dv = dv_dvp1_lambdified(vps_mesh, A, J) * dvp
dA = du*dv*hh**2
EE = np.sum(EED * dA)
ME = np.sum(MED * dA)
HE = np.sum(HED * dA)
TE = EE + ME + HE

# save solution
save(outputfile, K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED)
summary = 'Saved: '
for x in sys.argv[1:]:
    summary += x + ' '
print(summary)

flux = B*dA
where_flux = np.ones((NU,NV))
where_flux[0,0] = 0
print('flux without 00', jnp.sum(B*dA, where = jnp.array(where_flux)))
print('full flux:', jnp.sum(B*dA))

end_processing = time.time()
print(end_processing - start_processing)