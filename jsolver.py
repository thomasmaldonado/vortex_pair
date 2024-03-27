### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter, NL, NR, NU, NV
from janalytics import eq0_V_lambdified, eq0_C_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, B_lambdified, Eu_lambdified, Ev_lambdified, FuMinusAu_lambdified, FvMinusAv_lambdified
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
K = 1
A = 1/10
J = -4/K**4
print(NL, NR)
N = NR
if np.abs(NL) != np.abs(NR):
    raise Exception("Only winding numbers of equal magnitude are supported")
samesign = (NL / NR == 1)

#construct bipolar coordinates
tol = 1e-8
NU = 40
NV = 41 # ODD
NUNV = NU*NV
if NV % 2 == 0:
    raise Exception('Only odd NV supported')
mask = jnp.arange(0, NV) != (NV-1)//2

max_up = 10*K/2
ups = jnp.linspace(0, max_up, NU+2)[1:-1]
dup = ups[1]-ups[0]

max_u = jnp.arccosh((A+max_up)/A)
#us = jnp.linspace(0, max_u, NU+2)[1:-1]
us = jnp.arccosh((A + ups)/A)

#max_u = jnp.arcsinh(10*K/A)
vs = jnp.linspace(-jnp.pi/2, jnp.pi/2, NV+2)[1:-1]
du = us[1]-us[0]
dv = vs[1]-vs[0]
uu, vv = jnp.meshgrid(us, vs, indexing = 'ij')
xx, yy = A*jnp.cosh(uu)*jnp.cos(vv), A*jnp.sinh(uu)*jnp.sin(vv)

dup_du1 = A*jnp.sinh(uu)
dup_du2 = A*jnp.cosh(uu)


# define coordinate transformation based on conformal mapping defined in coords.py
# define derivatives (note the chain rule employed in the d_dv1 and d_dv2 functions that enables conformal mapping)

#d_du1 = d_dx1
#d_du2 = d_dx2
#d_dv1 = d_dy1
#d_dv2 = d_dy2

@jit
def d_du1(f, boundary_left, boundary_right):
    df_dup1 = d_dx1(f, dup, boundary_left, boundary_right)
    return df_dup1 * dup_du1
@jit
def d_du2(f, boundary_left, boundary_right):
    df_dup1 = d_dx1(f, dup, boundary_left, boundary_right)
    df_dup2 = d_dx2(f, dup, boundary_left, boundary_right)
    return df_dup2 * dup_du1**2 + df_dup1 * dup_du2


#@jit
#def d_du1(f, boundary_left, boundary_right):
#    return d_dx1(f, du, boundary_left, boundary_right)
#@jit
#def d_du2(f, boundary_left, boundary_right):
#    return d_dx2(f, du, boundary_left, boundary_right)

@jit
def d_dv1(f, boundary_left, boundary_right):
    return d_dy1(f, dv, boundary_left, boundary_right)
@jit
def d_dv2(f, boundary_left, boundary_right):
    return d_dy2(f, dv, boundary_left, boundary_right)

# V boundaries
@jit
def left_V(V):
    return (V[0,:] + jnp.flip(V[0,:]))/2
@jit
def right_V(V):
    return jnp.full(NV, -1)
@jit
def bottom_V(V):
    return V[:,0]
@jit
def top_V(V):
    return V[:,-1]

# Fu boundaries
@jit
def left_Fu(Fu):
    bl = (Fu[0,:] - jnp.flip(Fu[0,:]))/2
    return bl # *mask + (1-mask)*Fu[0,(NV-1)//2]
@jit
def right_Fu(Fu):
    return FuMinusAu_lambdified(A, vs, max_u, N)
@jit
def bottom_Fu(Fu):
    return Fu[:,0]
@jit
def top_Fu(Fu):
    return Fu[:,-1]

# Fv boundaries
@jit
def left_Fv(Fv):
    bl = (Fv[0,:] - jnp.flip(Fv[0,:]))/2
    return bl # *mask + (1-mask)*Fv[0,(NV-1)//2]
@jit
def right_Fv(Fv):
    return FvMinusAv_lambdified(A, vs, max_u, N)
@jit
def bottom_Fv(Fv):
    return jnp.zeros(NU)
@jit
def top_Fv(Fv):
    return jnp.zeros(NU)

# C boundaries
@jit
def left_C(C):
    return mask * (C[0,:] + jnp.flip(C[0,:]))/2
@jit
def right_C(C):
    return jnp.full(NV, jnp.sqrt(-J))
@jit
def bottom_C(C):
    return C[:,0]
@jit
def top_C(C):
    return C[:,-1]

# V derivatives
@jit
def dV_du1(V):
    return d_du1(V, left_V(V), right_V(V))
@jit
def dV_du2(V):
    return d_du2(V, left_V(V), right_V(V))
@jit
def dV_dv1(V):
    return d_dv1(V, bottom_V(V), top_V(V))
@jit
def dV_dv2(V):
    return d_dv2(V, bottom_V(V), top_V(V))

# Fu derivatives
@jit
def dFu_du1(Fu):
    return d_du1(Fu, left_Fu(Fu), right_Fu(Fu))
@jit
def dFu_du2(Fu):
    return d_du2(Fu, left_Fu(Fu), right_Fu(Fu))
@jit
def dFu_dv1(Fu):
    return d_dv1(Fu, bottom_Fu(Fu), top_Fu(Fu))
@jit
def dFu_dv2(Fu):
    return d_dv2(Fu, bottom_Fu(Fu), top_Fu(Fu))

# Fv derivatives
@jit
def dFv_du1(Fv):
    return d_du1(Fv, left_Fv(Fv), right_Fv(Fv))
@jit
def dFv_du2(Fv):
    return d_du2(Fv, left_Fv(Fv), right_Fv(Fv))
@jit
def dFv_dv1(Fv):
    return d_dv1(Fv, bottom_Fv(Fv), top_Fv(Fv))
@jit
def dFv_dv2(Fv):
    return d_dv2(Fv, bottom_Fv(Fv), top_Fv(Fv))

# C derivatives
@jit
def dC_du1(C):
    return d_du1(C, left_C(C), right_C(C))
@jit
def dC_du2(C):
    return d_du2(C, left_C(C), right_C(C))
@jit
def dC_dv1(C):
    return d_dv1(C, bottom_C(C), top_C(C))
@jit
def dC_dv2(C):
    return d_dv2(C, bottom_C(C), top_C(C))

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
    V_uu = dV_du2(V)
    V_vv = dV_dv2(V)
    C_uu = dC_du2(C)
    C_vv = dC_dv2(C)
    eq0_V = eq0_V_lambdified(J, C, V_uu, V_vv, uu, V, vv, A)
    eq0_C = eq0_C_lambdified(C_uu, C, vv, V, 0, 0, C_vv, uu, 0, A)
    return pack_electrostatic(eq0_V, eq0_C)

# define function whose root yields the magnetostatic solution
@jit
def f_magnetostatic(V_Fu_Fv_C):
    V, Fu, Fv, C = unpack_magnetostatic(V_Fu_Fv_C)
    V_uu = dV_du2(V)
    V_vv = dV_dv2(V)
    Fu_u, Fv_u = dFu_du1(Fu), dFv_du1(Fv)
    Fu_v, Fv_v = dFu_dv1(Fu), dFv_dv1(Fv)
    Fu_uv, Fv_uv = dFu_dv1(Fu_u), dFv_dv1(Fv_u)
    Fu_vv = dFu_dv2(Fu)
    Fv_uu = dFv_du2(Fv)
    C_uu = dC_du2(C)
    C_vv = dC_dv2(C)
    eq0_V = eq0_V_lambdified(J, C, V_uu, V_vv, uu, V, vv, A)
    eq0_Fu = eq0_Fu_lambdified(Fu_v, Fv_v, C, Fv_u, Fu_vv, vv, N, Fu, uu, Fv_uv, Fv, A)
    eq0_Fv = eq0_Fv_lambdified(Fu_uv, Fu_v, Fu_u, C, Fv_u, Fv_uu, vv, N, Fu, uu, Fv, A)
    eq0_C = eq0_C_lambdified(C_uu, C, vv, V, N, Fu, C_vv, uu, Fv, A)
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
    #x0 = pack_electrostatic(V0, C0)
    #start = time.time()
    #electrostatic_solution = newton(f_electrostatic, x0)
    #end = time.time()
    #print("Elapsed time for electrostatic solution: ", end - start)

    # use electrostatic solution as initial guess for magnetostatic problem
    from matplotlib import pyplot as plt
    from scipy.interpolate import LinearNDInterpolator

    #V0, C0 = unpack_electrostatic(electrostatic_solution)
    #plt.imshow(V0)
    #plt.show()
    #plt.imshow(C0)
    #plt.show()
    #plot_cart((A*(jnp.cosh(uu)-jnp.cos(vv))), 'r')
    #V0 = jnp.full((NU, NV), -1)
    #C0 = jnp.full((NU, NV), jnp.sqrt(-J))
    Fu0, Fv0 = FuMinusAu_lambdified(A, vv, uu, N), FvMinusAv_lambdified(A, vv, uu, N)
    #Fu0 , Fv0 = jnp.zeros((NU, NV)), jnp.zeros((NU, NV))

# perform Newton's method 
x0 = pack_magnetostatic(V0, Fu0, Fv0, C0)
start = time.time()
magnetostatic_solution = newton(f_magnetostatic, x0)
end = time.time()
print("Elapsed time for magnetostatic solution: ", end - start)

start_processing = time.time()

# begin post-processing 
V, Fu, Fv, C = unpack_magnetostatic(magnetostatic_solution)
plt.imshow(V)
plt.savefig('V')
plt.close()
plt.imshow(Fu)
plt.savefig('Fu')
plt.close()
plt.imshow(Fv)
plt.savefig('Fv')
plt.close()
plt.imshow(C)
plt.savefig('C')
plt.close()

# plot in cartesian space
def plot_cart(func, label):
    #save_file = 'data/' + sys.argv[1] + '_' + label + '_cart.png'
    #xs = []
    #ys = []
    #zs = []
    xs = jnp.ravel(xx)
    ys = jnp.ravel(yy)
    zs = jnp.ravel(func)
    xs = jnp.concatenate((xs, -xs))
    ys = jnp.concatenate((ys, ys))
    zs = jnp.concatenate((zs, zs))
    #for i, u in enumerate(us):
    #    for j, v in enumerate(vs):
    #        h = A / (np.cosh(v)-np.cos(u))
    #        x = h*np.sinh(v)
    #        y = h*np.sin(u)
    #        xs.append(x)
    #        ys.append(y)
    #        zs.append(func[i,j])
    #        h = A / (np.cosh(-v)-np.cos(u))
    #        x = h*np.sinh(-v)
    #        y = h*np.sin(u)
    #        xs.append(x)
    #        ys.append(y)
    #        zs.append(func[i,j])

    NX = 1000
    NY = 1000
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    #bulk_val = 0
    #if label == 'V':
    #    bulk_val = -1
    #if label == 'C':
    #    bulk_val = np.sqrt(-J)
    #if label == 'J0':
    #    bulk_val = -J
    #for x in [min_x, max_x]:
    #    for y in [min_y, max_y]:
    #        xs.append(x)
    #        ys.append(y)
    #        zs.append(bulk_val)
    X = np.linspace(min_x, max_x, NX)
    Y = np.linspace(min_y, max_y, NY)
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(list(zip(xs, ys)), zs)
    Z = interp(X,Y)
    plt.pcolormesh(X,Y,Z, cmap = 'hot', shading = 'auto')
    plt.colorbar()
    #plt.axis('equal')
    #plt.title(label)
    plt.savefig(label)
    plt.close()
plt.scatter(vs, left_Fu(Fu))
plt.title('left Fu')
plt.savefig('left_Fu')
plt.close()
plt.scatter(vs, left_Fv(Fv))
plt.title('left Fv')
plt.savefig('left_Fv')
plt.close()

plot_cart(V, 'V_cart')
plot_cart(Fu, 'Fu_cart')
plot_cart(Fv, 'Fv_cart')
plot_cart(C, 'C_cart')
Fu_u, Fv_u = dFu_du1(Fu), dFv_du1(Fv)
Fu_v, Fv_v = dFu_dv1(Fu), dFv_dv1(Fv)
B = B_lambdified(Fu_v, Fv_u, vv, Fu, uu, Fv, A)
plt.imshow(B)
plt.savefig('B')
plt.close()
plot_cart(B, 'B_cart')
exit()
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
    vmin = v_of_vp_lambdified(dvp, A, J)
    bl_Jx, bl_Jy = BP2cart(bl_Ju, bl_Jv, us, vmin)

    bl_h = A / (jnp.cosh(vmin) - jnp.cos(us))
    bl_measure = bl_h**2 * du * dv_dvp1_lambdified(dvp, A, J) * dvp / (2*jnp.pi)
    if samesign:
        Ju_minus, Jv_minus = -Ju, Jv
    else:
        Ju_minus, Jv_minus = Ju, -Jv
    Jx_minus, Jy_minus = BP2cart(Ju_minus, Jv_minus, uu, -vv)
    measure_minus = measure

    def B_func(u,v):
        h = A / (jnp.cosh(v) - jnp.cos(u))
        x, y = h*jnp.sinh(v), h*jnp.sin(u)
        rx = x - xx
        ry = y - yy
        integrand = (Jx*ry - Jy*rx)/(rx**2 + ry**2)*measure
        summed = jnp.sum(integrand, where = ((u!=uu) | (v!=vv)))

        bl_x, bl_y = bl_h*jnp.sinh(vmin), bl_h*jnp.sin(us)
        bl_rx, bl_ry = x - bl_x, y - bl_y
        bl_integrand = ((bl_Jx*bl_ry - bl_Jy*bl_rx)/(bl_rx**2 + bl_ry**2)*bl_measure)

        summed += jnp.sum(bl_integrand) # jnp.sum(bl_integrand, where = (us!=0)) # + (bl_integrand[1] + bl_integrand[-1])/2

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

from single_solver import solve
EE_single, ME_single, HE_single, TE_single = solve(K, NL+NR)
EE, ME, HE, TE = EE/EE_single, ME/ME_single, HE/HE_single, TE/TE_single

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

