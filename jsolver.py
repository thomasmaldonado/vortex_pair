### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter
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
hh = A / (np.cosh(vv)-np.cos(uu))
xx, yy = hh*np.sinh(vv), hh*np.sin(uu)

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
        boundary_left_v = jnp.pad(Fv[1:,0], (1,0), constant_values=(0, 0))
        return boundary_left_u, boundary_left_v
else:
    @jit
    def boundary_left_F(Fu, Fv):
        boundary_left_u = jnp.pad(Fu[1:,0], (1,0), constant_values=(0, 0))
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

# begin post-processing 
V, Fu, Fv, C = unpack_magnetostatic(magnetostatic_solution)

# calculate supercurrent
Au = Fu - (N/A) * (np.cosh(vv) - np.cos(uu))
Av = Fv

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

B = B_lambdified(vv, Fv_u, Fu_v, Fu, Fv, A, uu)
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


from matplotlib import pyplot as plt

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

if samesign:
    @jit
    def boundary_left_hF(hFu, hFv):
        boundary_left_u = jnp.full(NU, N)
        boundary_left_v = hFv[:,0] #=jnp.pad(Fv[1:,0], (1,0), constant_values=(0, 0))
        return boundary_left_u, boundary_left_v
else:
    @jit
    def boundary_left_hF(hFu, hFv):
        boundary_left_u = hFu[:,0] # jnp.pad(Fu[1:,0], (1,0), constant_values=(N, 0))
        boundary_left_v = jnp.zeros(NU)
        return boundary_left_u, boundary_left_v

@jit
def boundary_right_hF(hFu,hFv):
    return jnp.zeros(NU), jnp.zeros(NV)
@jit
def B_func(hFu, hFv):
    boundary_left_hFu, boundary_left_hFv = boundary_left_hF(hFu,hFv)
    boundary_right_hFu, boundary_right_hFv = boundary_right_hF(hFu,hFv)
    hFu_pad = _apply_boundaries(hFu, boundary_left_hFu, boundary_right_hFu)
    hFv_pad = _apply_boundaries(hFv, boundary_left_hFv, boundary_right_hFv)
    line_0 = hFu_pad[:,:-1]*du
    line_1 = (jnp.roll(hFv_pad, -1, axis = 0)[:,:-1])*dv_dual
    line_2 = -jnp.roll(jnp.roll(hFu_pad, -1, axis = 0), -1, axis = 1)[:,:-1]*du
    line_3 = -jnp.roll(hFv_pad, -1, axis = 0)[:,:-1]*dv_dual
    line_int = line_0+line_1+line_2+line_3
    B = line_int / dA_dual
    return B



def B_func2():
    h = A / (np.cosh(vv)-np.cos(uu))
    dvs = dv_dvp1_lambdified(vps, A, J) * dvp
    hFu, hFv = h*Fu, h*Fv
    boundary_left_hFu, boundary_left_hFv = boundary_left_hF(hFu,hFv)
    boundary_right_hFu, boundary_right_hFv = boundary_right_hF(hFu,hFv)
    hFu_pad = _apply_boundaries(hFu, boundary_left_hFu, boundary_right_hFu)
    hFv_pad = _apply_boundaries(hFv, boundary_left_hFv, boundary_right_hFv)
    flux = jnp.sum(hFu_pad*du, axis = 0)
    #dflux_dv = d_dv1(flux[1:-1], flux[0], flux[-1])
    dflux_dv = ((jnp.roll(flux, -1) - jnp.roll(flux, 1))[1:-1]/(2*dvs))
    dB_du = h*C**2*Av
    BuminusB0 = np.zeros((NU, NV))
    for i in range(NU):
        BuminusB0[i,:] = jnp.sum(dB_du[0:i,:]*du)
    B0 = (dflux_dv - jnp.sum(BuminusB0 * h**2*du, axis = 0))/jnp.sum(h**2*du, axis = 0)
    Bu = np.zeros((NU,NV))
    for i in range(NU):
        Bu[i,:] = BuminusB0[i,:] + B0[:]
    return Bu
#plt.imshow(B_func2())
#plt.title('B_func2')
#plt.show()
#plt.imshow(B)
#plt.title('B lambdified')
#plt.show()

hFu, hFv = h*Fu, h*Fv

plt.plot(hFu[:,0])
plt.title('hFu0')
plt.show()

#plt.imshow(B_func(hFu, hFv)[:,:(NV//2)])
#plt.show()


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
    summed += jnp.sum((bl_Jx*bl_ry - bl_Jy*bl_rx)/(bl_rx**2 + bl_ry**2)*bl_measure, where = (us != 0))

    return summed

B = np.zeros((NU,NV))
for i, u in enumerate(us):
    for j, v in enumerate(vs):
        B[i,j] = B_func(u,v)

def BS():
    h = A / (jnp.cosh(vv)-jnp.cos(uu))
    dvs = dv_dvp1_lambdified(vps, A, J) * dvp
    Jx, Jy = BP2cart(Ju, Jv, uu, vv)
    xx, yy = h*np.sinh(vv), h*np.sin(uu)

    B = np.zeros((NU,NV))

    for i, u in enumerate(us):
        print(i)
        for j, v in enumerate(vs):
            for iprime, uprime in enumerate(us):
                for jprime, vprime in enumerate(vs):
                        if i == iprime and j == jprime:
                            pass
                        else:
                            rx = xx[i,j] - xx[iprime, jprime]
                            ry = yy[i,j] - yy[iprime, jprime]
                            B[i,j] += (1/(2*np.pi))*(Jx[iprime, jprime]*ry - Jy[iprime, jprime]*rx)/(rx**2 + ry**2) * h[iprime,jprime]**2*du*dvs[jprime]
                            #B[i,j] += (Jx[iprime, jprime]*ry - Jy[iprime, jprime]*rx)/(rx**2 + ry**2) * measure[iprime, jprime]

                #rx = xx[i,j] - 0
                #ry = yy[i,j] - yy[iprime, 0]
                #B[i,j] += ((Jx[iprime, jprime]*ry - Jy[iprime, jprime]*rx)/((rx**2 + ry**2)**(3/2))) * h[iprime,jprime]**2*du*dvs[jprime]

    return B




plt.imshow(B)
plt.title('Biot Savart')
plt.show()
#plt.imshow(BS())
#plt.title('Biot Savart unvectorized')
#plt.show()

EED = (Eu**2 + Ev**2)/2
MED = (B**2)/2
plt.imshow(MED)
plt.show()
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

#dA = jnp.array(dA)
dA = hh**2*du*dv_mesh
EE = np.sum(EED * dA)
ME = np.sum(MED * dA)
HE = np.sum(HED * dA)
TE = EE + ME + HE
flux = B*dA

where_flux = np.ones((NU,NV))
where_flux[0,0] = 0
print(jnp.sum(B*dA, where = jnp.array(where_flux)))
flux = jnp.sum(B*dA, axis = 0)
plt.imshow(B*dA)
plt.title
plt.show()
plt.plot(flux)
plt.show()
print('flux (excluding vp = dvp): ', jnp.sum(flux[1:]))

# save solution
save(outputfile, K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED)
summary = 'Saved: '
for x in sys.argv[1:]:
    summary += x + ' '
print(summary)
exit()

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