### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter
from janalytics3 import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified
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
h = A / (np.cosh(vv)-np.cos(uu))

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
print(dA0)
vhalf = v_of_vp_lambdified(1-dvp/2, A, J)
dvhalf = dv_dvp1_lambdified(1-dvp/2, A, J) * dvp
hhalf = hm1 = A/(jnp.cosh(vhalf) - jnp.cos(us))
#dA0 = jnp.sum(hhalf**2*du*dvhalf)
print(dA0)
hm1 = A/(jnp.cosh(vs[-1]) - jnp.cos(us))


@jit
def boundary_right_F(Fu, Fv):
    dFu_du_m2 = (jnp.roll(Fu[:,-2],-1) - jnp.roll(Fu[:,-2],1))/(2*du)
    dFv_du_m2 = (jnp.roll(Fv[:,-2],-1) - jnp.roll(Fv[:,-2],1))/(2*du)
    dFu_dvp_m2 = (Fu[:,-1] - Fu[:,-3])/(2*dvp)
    dFv_dvp_m2 = (Fv[:,-1] - Fv[:,-3])/(2*dvp)
    dFu_dv_m2 = dFu_dvp_m2*dvp_dv1_lambdified(vs[-2], A, J)
    dFv_dv_m2 = dFv_dvp_m2*dvp_dv1_lambdified(vs[-2], A, J)
    hm2 = h[:,-2]
    div_m2 = (dFu_du_m2 + dFv_dv_m2)/hm2**2
    curl_m2 = (dFv_du_m2 - dFu_dv_m2)/hm2**2

    div_right = jnp.sum(Fv[:,-1])*du / dA0
    curl_right = jnp.sum(Fu[:,-1])*du / dA0

    div_m1 = (div_m2 + div_right)/2
    curl_m1 = (curl_m2 + curl_right)/2


    dFu_du_m1 = (jnp.roll(Fu[:,-1],-1) - jnp.roll(Fu[:,-1],1))/(2*du)
    dFv_du_m1 = (jnp.roll(Fv[:,-1],-1) - jnp.roll(Fv[:,-1],1))/(2*du)

    dFv_dv_m1 = div_m1*hm1**2 - dFu_du_m1
    dFu_dv_m1 = dFv_du_m1 - curl_m1*hm1**2 
    
    dFv_dvp_m1 = dFv_dv_m1*dv_dvp1_lambdified(vps[-1], A, J)
    dFu_dvp_m1 = dFu_dv_m1*dv_dvp1_lambdified(vps[-1], A, J)

    boundary_right_u = dFu_dvp_m1*2*dvp + Fu[:,-2]
    boundary_right_v = dFv_dvp_m1*2*dvp + Fu[:,-2]

    return boundary_right_u, boundary_right_v



@jit
def boundary_right_F(Fu, Fv):
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vs[-1])
    avg_xs, avg_ys = jnp.sum(xs)/jnp.sum(hm1), jnp.sum(ys)/jnp.sum(hm1)
    fu_infty, fv_infty = cart2BPinfinity(avg_xs, avg_ys, us)
    dv = dv_dvp1_lambdified(vs[-1], A, J) * dvp
    fum2, fvm2 = Fu[:,-2] / h[:,-2], Fv[:,-2] / h[:,-2]
    dfu_dv, dfv_dv = (fu_infty - fum2)/(2*dv), (fv_infty - fvm2)/(2*dv)
    dFu_dv= hm1*dfu_dv - hm1*Fu[:,-1]*jnp.sinh(vs[-1])/A
    dFv_dv= hm1*dfv_dv - hm1*Fu[:,-1]*jnp.sinh(vs[-1])/A
    boundary_right_u = dFu_dv*2*dv + Fu[:,-2]
    boundary_right_v = dFv_dv*2*dv + Fv[:,-2]
    return boundary_right_u, boundary_right_v
@jit
def boundary_right_F(Fu, Fv):
    return jnp.zeros(NU), jnp.zeros(NU)


@jit
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
def dB_dv(B):
    boundary_left = boundary_left_B(B)
    boundary_right = jnp.full(NU, jnp.mean(B[:,-1]))
    return d_dv1(B, boundary_left, boundary_right)


@jit
def _apply_boundaries(f, boundary_left, boundary_right):
    bl, br = jnp.reshape(boundary_left, (f.shape[0], 1)), jnp.reshape(boundary_right, (f.shape[0], 1))
    return jnp.concatenate((bl, f, br), axis = 1)
us_dual = us + du/2
vps_dual = jnp.linspace(dvp/2, 1-dvp/2, NV+1)
vs_dual = v_of_vp_lambdified(vps_dual, A, J)
vv_dual, uu_dual = jnp.meshgrid(vs_dual, us_dual)
h_dual = A / (np.cosh(vv_dual)-np.cos(uu_dual))
dv_dual = dv_dvp1_lambdified(vv_dual, A, J) * dvp
dA_dual = h_dual**2 * du * dv_dual

dv = dv_dvp1_lambdified(vv, A, J) * dvp

@jit
def B_func(Fu, Fv):
    boundary_left_Fu, boundary_left_Fv = boundary_left_F(Fu,Fv)
    boundary_right_Fu, boundary_right_Fv = boundary_right_F(Fu,Fv)
    Fu_pad = _apply_boundaries(Fu, boundary_left_Fu, boundary_right_Fu)
    Fv_pad = _apply_boundaries(Fv, boundary_left_Fv, boundary_right_Fv)
    line_0 = Fu_pad[:,:-1]*du
    line_1 = (jnp.roll(Fv_pad, -1, axis = 0)[:,:-1])*dv_dual
    line_2 = -jnp.roll(jnp.roll(Fu_pad, -1, axis = 0), -1, axis = 1)[:,:-1]*du
    line_3 = -jnp.roll(Fv_pad, -1, axis = 0)[:,:-1]*dv_dual
    line_int = line_0+line_1+line_2+line_3
    B = line_int / dA_dual
    return B

@jit
def eq0_F(Fu, Fv, C):
    B = B_func(Fu, Fv)
    dB_dv = (jnp.roll(B,-1, axis = 0) - B)[:,0:-1] / dv
    dB_du = (jnp.roll(B,-1, axis = 1) - B)[:,0:-1] / du
    hAu = Fu - N
    hAv = Fv
    return dB_dv + C**2*hAu, -dB_du+C**2*hAv


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
    eq0_Fu = eq0_Fu_lambdified(C, N, uu, Fv_v, Fv_uv, vv, Fu_vv, Fv_u, A, Fu, Fu_v)
    eq0_Fv = eq0_Fv_lambdified(vv, Fu_u, Fu_v, Fv_uu, C, Fv, A, Fu_uv, uu, Fv_u)
    #eq0_Fu, eq0_Fv = eq0_F(Fu, Fv, C)
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
    Fu0 = jnp.full((NU, NV), N)    

# perform Newton's method 
x0 = pack_magnetostatic(V0, Fu0, Fv0, C0)
start = time.time()
magnetostatic_solution = newton(f_magnetostatic, x0)
end = time.time()
print("Elapsed time for magnetostatic solution: ", end - start)

# begin post-processing 
V, Fu, Fv, C = unpack_magnetostatic(magnetostatic_solution)

# calculate supercurrent
h = A / (np.cosh(vv)-np.cos(uu))
Au = Fu/h - N/h
Av = Fv/h


J0 = C**2 * V
Ju = C**2 * Au
Jv = C**2 * Av

"""
@jit
def dF_dv1(Fu, Fv):
    boundary_left_u, boundary_left_v = boundary_left_F(Fu, Fv)
    boundary_right_u, boundary_right_v = jnp.zeros(NU), jnp.zeros(NU)
    Fu_result = d_dv1(Fu, boundary_left_u, boundary_right_u)
    Fv_result = d_dv1(Fv, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result
"""
# calculate energy densities
V_u = d_du1(V)
V_v = dV_dv1(V)
Fu_u, Fv_u = d_du1(Fu), d_du1(Fv)
Fu_v, Fv_v = dF_dv1(Fu, Fv)

Eu = Eu_lambdified(uu, vv, A, V_u)
Ev = Ev_lambdified(V_v, vv, A, uu)


B = B_lambdified(vv, Fv_u, Fu_v, Fu, Fv, A, uu)
dFu_dv, _ = dF_dv1(Fu, Fv)
#B = (d_du1(Fv) - dFu_dv)/h**2

from matplotlib import pyplot as plt
Verror, Fuerror, Fverror, Cerror = unpack_magnetostatic(f_magnetostatic(magnetostatic_solution))
plt.imshow(Verror)
plt.show()
plt.imshow(Fuerror)
plt.show()
plt.imshow(Fverror)
plt.show()
plt.imshow(Cerror)
plt.show()

plt.plot(Fu[:,-1])
plt.show()
plt.plot(Fv[:,-1])
plt.show()

#plt.imshow(Fu[:,7*NV//8:])
#plt.show()
#plt.imshow(Fu[:,3*NV//4:])
#plt.show()

plt.imshow(Fu/h)
plt.title('Fu/h')
plt.show()
plt.imshow(Fv/h)
plt.title('Fv/h')
plt.show()
h = A / (np.cosh(vv)-np.cos(uu))
plt.imshow(B[:,1:(NV//2)])
plt.title('B')
plt.show()
plt.imshow(((d_du1(Fv) - dFu_dv)/h**2)[:,:(NV//2)])
plt.show()
plt.imshow(B_func(Fu, Fv)[:,:(NV//2)])
plt.show()
plt.imshow(Ju)
plt.title('Ju')
plt.show()
plt.imshow(Jv)
plt.title('Jv')
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

from matplotlib import pyplot as plt
plt.plot(Ju[:,0])
plt.show()
plt.plot(Jv[:,0])
plt.show()
plt.plot(Fu[:,0])
plt.title("Fu0")
plt.show()
plt.plot(B[:,0])
plt.show()
plt.plot(B[NU//2, :])
plt.show()
# save solution
save(outputfile, K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED)
summary = 'Saved: '
for x in sys.argv[1:]:
    summary += x + ' '
print(summary)
exit()

from matplotlib import pyplot as plt
h = A / (np.cosh(vv)-np.cos(uu))
flux_cumulative = jnp.sum(h*Fu*du, axis = 0)
if samesign:
    flux_cumulative_left = 2*np.pi*N
else:
    x = (Fu[:,0]*h[:,0])[1:]
    test = jnp.concatenate((jnp.array([(x[0]+x[-1])/2]), x))
    plt.plot(test)
    plt.show()
    Fu0 = Fu[0]
    flux_cumulative_left = flux_cumulative[0]
dvs = dv_dvp1_lambdified(vps, A, J) * dvp
flux_cumulative_right = 0
flux_cumulative = jnp.concatenate((jnp.array([flux_cumulative_left]), flux_cumulative, jnp.array([flux_cumulative_right])))
flux_density = (flux_cumulative - jnp.roll(flux_cumulative, -1))[0:-2] / dvs
print(jnp.sum(flux_density*dvs))
plt.plot(flux_density)
plt.show()