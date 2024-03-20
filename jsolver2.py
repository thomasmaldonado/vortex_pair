### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter
from janalytics2 import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified
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
NL = int(sys.argv[3])
NR = int(sys.argv[4])
NU = int(sys.argv[5])
NV = int(sys.argv[6])
outputfile = 'data/' + sys.argv[7] + '.npy'
try:
    inputfile = 'data/' + sys.argv[8] + '.npy'
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
        boundary_left_u = jnp.full(NU, N)
        boundary_left_v = Fv[:,0] #=jnp.pad(Fv[1:,0], (1,0), constant_values=(0, 0))
        return boundary_left_u, boundary_left_v
else:
    @jit
    def boundary_left_F(Fu, Fv):
        boundary_left_u = Fu[:,0] # jnp.pad(Fu[1:,0], (1,0), constant_values=(N, 0))
        boundary_left_v = jnp.zeros(NU)
        return boundary_left_u, boundary_left_v

dA0 = jnp.pi*(A/jnp.sinh(vs[-1]))**2
hm1 = A/(jnp.cosh(vs[-1]) - jnp.cos(us))

@jit
def boundary_right_F(Fu, Fv):
    dFu_du = (jnp.roll(Fu[:,-1],-1) - jnp.roll(Fu[:,-1],1))/(2*du)
    dFv_du = (jnp.roll(Fv[:,-1],-1) - jnp.roll(Fv[:,-1],1))/(2*du)
    int_Fu = jnp.sum(Fu[:,-1])*du
    int_Fv = jnp.sum(Fv[:,-1])*du
    dFv_dv = -(dFu_du - int_Fv*hm1**2/dA0)
    dFu_dv = dFv_du - int_Fu*hm1**2/dA0
    dv = dv_dvp1_lambdified(vps[-1], A, J) * dvp
    boundary_right_u = dFu_dv*2*dv + Fu[:,-2]
    boundary_right_v = dFv_dv*2*dv + Fv[:,-2]
    return boundary_right_u, boundary_right_v

def dF_dv1(Fu, Fv):
    boundary_left_u, boundary_left_v = boundary_left_F(Fu, Fv)
    boundary_right_u, boundary_right_v = boundary_right_F(Fu, Fv)
    Fu_result = d_dv1(Fu, boundary_left_u, boundary_right_u)
    Fv_result = d_dv1(Fv, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@jit
def dF_dv2(Fu, Fv):
    boundary_left_u, boundary_left_v = boundary_left_F(Fu, Fv)
    boundary_right_u, boundary_right_v = boundary_right_F(Fu, Fv)
    Fu_result = d_dv2(Fu, boundary_left_u, boundary_right_u)
    Fv_result = d_dv2(Fv, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

if samesign:
    @jit
    def boundary_left_B(B):
        boundary_left = jnp.pad(B[1:,0], (1,0), constant_values=(0, 0))
        return boundary_left
else:
    @jit
    def boundary_left_B(B):
        boundary_left = jnp.zeros(B.shape[0])
        return boundary_left


@jit
def dB_dv1(B):
    boundary_left = boundary_left_B(B)
    boundary_right = jnp.full(B.shape[0], jnp.mean(B[:,-1]))
    return d_dv1(B, boundary_left, boundary_right)

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
    eq0_Fu = eq0_Fu_lambdified(C, N, uu, Fv_v, Fv_uv, vv, Fu_vv, Fv_u, A, Fu, Fv, Fu_v)
    eq0_Fv = eq0_Fv_lambdified(vv, Fu_u, Fu_v, Fv_uu, C, Fv, A, Fu_uv, uu, Fu, N, Fv_u)
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

# begin post-processing 
V, Fu, Fv, C = unpack_magnetostatic(magnetostatic_solution)

# calculate supercurrent

h = A / (jnp.cosh(vv)-np.cos(uu))


Au = Fu/h - (N/h)
Av = Fv/h

J0 = C**2 * V
Ju = C**2 * Au
Jv = C**2 * Av

# calculate energy densities
V_u = d_du1(V)
V_v = dV_dv1(V)
Fu_u, Fv_u = d_du1(Fu), d_du1(Fv)
Fu_v, Fv_v = dF_dv1(Fu, Fv)

Eu = Eu_lambdified(uu, vv, A, V_u)
Ev = Ev_lambdified(V_v, vv, A, uu)

B = B_lambdified(vv, Fv_u, Fu_v, Fu, Fv, A, uu, N)

from matplotlib import pyplot as plt
plt.imshow(Fu/h)
plt.show()
plt.imshow(Fv/h)
plt.show()
plt.imshow(B)
plt.show()

EED = (Eu**2 + Ev**2)/2
MED = (B**2)/2
HED = C**2 * V**2 + J
TED = EED + MED + HED

# calculate energies
dA = np.zeros((NU, NV))
for i in range(NU):
    u = us[i]
    for j in range(NV):
        v = vs[j]
        vp = vps[j]
        h = A / (np.cosh(v)-np.cos(u))
        args = (vp, A, J)
        dv = dv_dvp1_lambdified(*args) * dvp
        dA[i,j] = du*dv * h**2
        
dA = jnp.array(dA)
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
exit()

flux = np.zeros(NV)
for i, v in enumerate(vs):
    h = A / (np.cosh(v)-np.cos(us))
    flux[i] = np.sum(Au[:, i] * h * du) + 2*np.pi*N
print(flux[0])

from matplotlib import pyplot as plt

vhalf = v_of_vp_lambdified(vps[-1] - dvp/2, A, J)
dvhalf = dv_dvp1_lambdified(vps[-1] - dvp/2, A, J) * dvp

ab = (Fu[:,-2] + np.roll(Fu[:,-2],-1))/2
bc = np.roll((Fv[:,-2] + Fv[:,-1])/2,-1)
cd = (Fu[:,-1] + np.roll(Fu[:,-1],-1))/2
da = (Fv[:,-2] + Fv[:,-1])/2

def arclength_du(u1, u2, v):
    indef = lambda u: 2*A*np.arctan(np.tan(u/2)/np.tanh(v/2))/np.sinh(v)
    return np.abs(indef(u2)-indef(u1))

def arclength_dv(v1, v2, u):
    indef = lambda v: -2*A*np.arctan(np.tan(u/2)/np.tanh(v/2))/np.sin(v)
    return np.abs(indef(v2)-indef(v1))

hab = A/(np.cosh(vs[-2])-(np.cos(us) + du/2))
hbc = np.roll(A/(np.cosh(vhalf)-(np.cos(us))),-1)
hcd = A/(np.cosh(vs[-1])-(np.cos(us) + du/2))
hda = A/(np.cosh(vhalf)-(np.cos(us)))

sab = arclength_du(us, np.roll(us, -1), vs[-2])
sbc = arclength_dv(vs[-2], vs[-1], np.roll(us,-1))
scd = arclength_du(us, np.roll(us, -1), vs[-1])
sda = arclength_dv(vs[-2], vs[-1], us)

plt.plot(sab)
plt.show()
plt.plot(hab*du)
plt.show()
print(hab, hbc, hcd, hda)
plt.plot(hab*du - sab)
plt.show()


flux0 = ab*hab*du + bc*hbc*dvhalf - cd*hcd*du - da*hda*dvhalf
fluxnew = ab*sab+ bc*sbc - cd*scd - da*sda

hhalf = A/(np.cosh(vhalf)-np.cos(us + du/2))
dAhalf = hhalf**2 * du * dvhalf

plt.plot(flux0)
plt.title('flux')
plt.show()
plt.plot(flux0/dAhalf)
plt.title('B')
plt.show()
#Au0 = - (N/(2*A))*(jnp.cos(us) - 1)
#flux = np.concatenate([np.array([2*np.pi*N]), flux])
#plt.plot(flux)
#plt.show()

args = (1-dvp/2, A, J)
core_v = v_of_vp_lambdified(*args)
core_area = np.pi * (A / np.sinh(core_v))**2
core_B = flux[-1]


areas = np.pi * (A / np.sinh(vs))**2
print(flux[-1] / areas[-1])
"""
@jit
def f_B(B):
    B = jnp.reshape(B[0:NUNV], (NU,NV))
    B_v = dB_dv1(B)
    eq0_Bu = eq0_Bu_lambdified(vv, Ju, A, uu, B_v)
    return jnp.ravel(eq0_Bu)

x0 = jnp.zeros(NU*NV)

start = time.time()
B = jnp.reshape(newton(f_B, x0), (NU, NV))
end = time.time()
print("Elapsed time for magnetic field solution: ", end - start)

"""
Fu_uv, Fv_uv = d_du1(Fu_v), d_du1(Fv_v)
Fv_uu = d_du2(Fv)
Fu_vv, _ = dF_dv2(Fu, Fv)

from matplotlib import pyplot as plt
dB_du = dB_du_lambdified(A, uu, Fu_uv, vv, Fu_u, Fv_u, Fv, Fv_uu, Fu_v, Fu)
dB_dv = dB_dv_lambdified(A, uu, vv, Fv_u, Fv, Fu_vv, Fv_v, Fv_uv, Fu_v, Fu)
plt.imshow(dB_du)
plt.show()
plt.plot(dB_du[:,-1])
plt.title('dB_du sympy')
plt.show()
plt.imshow(dB_dv)
plt.show()
dB_dvp = np.zeros((NU,NV))
for i, vp in enumerate(vps):
    args = (vp, A, J)
    dv_dvp = dv_dvp1_lambdified(*args)
    dB_dvp[:,i] = dB_dv[:,i] * dv_dvp
B_m1 = 0
print(np.sum((NV+1)*dvp))
B_p1 = np.sum(dB_dvp[0,:]*dvp) #+ B_lambdified(vs[0], Fv_u[0,0], Fu_v[0,0], Fu[0,0], Fv[0,0], A, us[0], N)
B = np.zeros((NU,NV))
for i in range(NU):
    for j in range(NV):
        B[i,j] = B_p1 - np.sum(dB_dvp[i,j:]*dvp)
plt.imshow(B)
plt.show()
#plt.imshow(dB_dvp)
plt.show()

from janalytics import Ju_lambdified,Jv_lambdified,curl_B_u_lambdified,curl_B_v_lambdified
V_u = d_du1(V)
V_v = dV_dv1(V)
Fu_u, Fv_u = d_du1(Fu), d_du1(Fv)
Fu_v, Fv_v = dF_dv1(Fu, Fv)
Ju = Ju_lambdified(C, uu, A, Fu, vv, N)
Jv = Jv_lambdified(C, uu, A, Fv, N)
Fv_uu = d_du2(Fv)
Fu_uv, Fv_uv = d_du1(Fu_v), d_du1(Fv_v)
curl_B_v = curl_B_v_lambdified(uu, Fv_uu, Fv_u, v, Fu_v, Fv, A, Fu, Fu_u, Fu_uv)
#curl_B_u_args = [u, Fv_u, v, Fv_v, Fu_v, Fv, a, Fu, Fv_uv, Fu_vv]
#curl_B_v_args = [u, Fv_uu, Fv_u, v, Fu_v, Fv, a, Fu, Fu_u, Fu_uv]


from matplotlib import pyplot as plt

solution_error = f_magnetostatic(magnetostatic_solution)
_, _, Fv_error, _ = unpack_magnetostatic(solution_error)
plt.plot(Fv_error[:,-1])
plt.title('error')
plt.show()

plt.plot(Jv[:,-1])
plt.title('Jv')
plt.show()

plt.plot(curl_B_v[:,-1])
plt.title('curl(B)_v')
plt.show()

B = B_lambdified(vv, Fv_u, Fu_v, Fu, Fv, A, uu, N)
plt.plot(B[:,-1])
plt.title('B')
plt.show()


h = A / (np.cosh(v)-np.cos(u))
Au0 = Au[:,0]
count = 0
for i, u in enumerate(us):
    print(i)
    h = A / (np.cosh(vs[0])-np.cos(u))
    count += h * Au0[i]
print(count)



dB_du = d_du1(B)
dB_du_over_h = np.zeros((NU,NV))
for i in range(NU):
    u = us[i]
    for j in range(NV):
        v = vs[j]
        h = A / (np.cosh(v)-np.cos(u))
        dB_du_over_h[i,j] = dB_du[i,j]/h
from matplotlib import pyplot as plt
plt.plot(dB_du_over_h[:,-1])
plt.title('dB_du/h')
plt.show()
plt.plot(dB_du[:,-1])
plt.show()
