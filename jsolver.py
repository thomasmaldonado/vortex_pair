### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
from params import K_func, A_func, tol, max_iter, NL, NR, NU, NV
from janalytics import eq0_V_lambdified, eq0_Fx_lambdified, eq0_Fy_lambdified, eq0_C_lambdified, B_lambdified, Ex_lambdified, Ey_lambdified
from jderivatives import d_dx1, d_dx2, d_dy1, d_dy2
from jcoords import BP2cart, cart2BP, cart2BPinfinity, v_of_vp_lambdified, dvp_dv1_lambdified, dvp_dv2_lambdified, dv_dvp1_lambdified
import numpy as np
import jax.numpy as jnp
from jax import jit 
import jax
from jax.numpy import linalg as jla
import time 
from files import save, load
#jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

tol = 1e-5

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
NX = 40
NY = 41
xs_pad = jnp.linspace(0, 1, NX+2) * 5*K
ys_pad = jnp.linspace(-1,1, NY+2) * 5*K
xs = xs_pad[1:-1]
ys = ys_pad[1:-1]
dx = xs[1]-xs[0]
dy = ys[1]-ys[0]
yy, xx = jnp.meshgrid(ys, xs)
yy_pad, xx_pad = jnp.meshgrid(ys_pad, xs_pad)
A = K/4
A = 10*dx
#A = 2*dx
#A = 10*dx
A_LOC = int(A / dx) - 1
print(A_LOC)
raveled_loc = NX*((NY-1)//2) + A_LOC

from matplotlib import pyplot as plt
plt.imshow(xx)
plt.show()
# define derivatives (note the chain rule employed in the d_dv1 and d_dv2 functions that enables conformal mapping)

@jit
def left_V(V):
    return V[0,:]
@jit
def right_V(V):
    return jnp.full(V.shape[1], -1)
@jit
def bottom_V(V):
    return jnp.full(V.shape[0], -1)
@jit
def top_V(V):
    return jnp.full(V.shape[0], -1)
@jit
def left_C(C):
    return C[0,:]
@jit
def right_C(C):
    return jnp.full(C.shape[1], jnp.sqrt(-J))
@jit
def bottom_C(C):
    return jnp.full(C.shape[0], jnp.sqrt(-J))
@jit
def top_C(C):
    return jnp.full(C.shape[0], jnp.sqrt(-J))
@jit
def AxminusFx(x,y):
    return N*y/((x-A)**2+y**2)
@jit
def AyminusFy(x,y):
    return -N*(x-A)/((x-A)**2+y**2)

if samesign:
    @jit
    def left_Fx(Fx):
        Ax1 = AxminusFx(xs[0],ys) + Fx[0,:]
        Ax0 = Ax1
        Fx0 = Ax0 - AxminusFx(0, ys)
        return Fx0
    @jit
    def left_Fy(Fy):
        return (-N*A)/(A**2+ys**2)
else:
    @jit
    def left_Fx(Fx):
        return (-N*ys)/(A**2+ys**2)
    @jit
    def left_Fy(Fy):
        Ay1 = AyminusFy(xs[0],ys) + Fy[0,:]
        Ay0 = Ay1
        Fy0 = Ay0 - AyminusFy(0, ys)
        return Fy0
@jit
def right_Fx(Fx):
    return (-N*ys) / ((xs_pad[-1]-A)**2 + ys**2)
@jit
def right_Fy(Fx):
    return (N*(xs_pad[-1]-A)) / ((xs_pad[-1]-A)**2 + ys**2)
@jit
def bottom_Fx(Fx):
    return (-N*ys_pad[0]) / ((xs-A)**2 + ys_pad[0]**2)
@jit
def bottom_Fy(Fx):
    return (N*(xs-A)) / ((xs-A)**2 + ys_pad[0]**2)
@jit
def top_Fx(Fx):
    return (-N*ys_pad[-1]) / ((xs-A)**2 + ys_pad[-1]**2)
@jit
def top_Fy(Fx):
    return (N*(xs-A)) / ((xs-A)**2 + ys_pad[-1]**2)

@jit
def dV_dx1(V):
    return d_dx1(V, dx, left_V(V), right_V(V))
@jit
def dV_dx2(V):
    return d_dx2(V, dx, left_V(V), right_V(V))
@jit
def dV_dy1(V):
    return d_dy1(V, dy, bottom_V(V), top_V(V))
@jit
def dV_dy2(V):
    return d_dy2(V, dy, bottom_V(V), top_V(V))

@jit
def dFx_dx1(Fx):
    return d_dx1(Fx, dx, left_Fx(Fx), right_Fx(Fx))
@jit
def dFx_dx2(Fx):
    return d_dx2(Fx, dx, left_Fx(Fx), right_Fx(Fx))
@jit
def dFx_dy1(Fx):
    return d_dy1(Fx, dy, left_Fx(Fx), right_Fx(Fx))
@jit
def dFx_dy2(Fx):
    return d_dy2(Fx, dy, left_Fx(Fx), right_Fx(Fx))

@jit
def dFy_dx1(Fy):
    return d_dx1(Fy, dx, left_Fy(Fy), right_Fy(Fy))
@jit
def dFy_dx2(Fy):
    return d_dx2(Fy, dx, left_Fy(Fy), right_Fy(Fy))
@jit
def dFy_dy1(Fy):
    return d_dy1(Fy, dy, left_Fy(Fy), right_Fy(Fy))
@jit
def dFy_dy2(Fy):
    return d_dy2(Fy, dy, left_Fy(Fy), right_Fy(Fy))

@jit
def dC_dx1(C):
    return d_dx1(C, dx, left_C(C), right_C(C))
@jit
def dC_dx2(C):
    return d_dx2(C, dx, left_C(C), right_C(C))
@jit
def dC_dy1(C):
    return d_dy1(C, dy, bottom_C(C), top_C(C))
@jit
def dC_dy2(C):
    return d_dy2(C, dy, bottom_C(C), top_C(C))

# helper functions to pack/unpack and reshape the solutions
@jit
def pack_electrostatic(V, C):
    C_plugged = jnp.ravel(C)
    C_unplugged = jnp.concatenate((C_plugged[0:raveled_loc], C_plugged[raveled_loc+1:]))
    return jnp.array(jnp.concatenate((jnp.ravel(V), jnp.ravel(C_unplugged))))

@jit
def pack_magnetostatic(V, Fu, Fv, C):
    return jnp.array(jnp.concatenate((jnp.ravel(V), jnp.ravel(Fu), jnp.ravel(Fv), jnp.ravel(C))))

@jit
def unpack_electrostatic(V_C):
    V = jnp.reshape(V_C[0:NX*NY], (NX,NY))
    C_unplugged = V_C[NX*NY:]
    C_plugged = jnp.concatenate((C_unplugged[0:raveled_loc], jnp.array([0]), C_unplugged[raveled_loc:]))
    C = jnp.reshape(C_plugged, (NX,NY))
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
    V_xx = dV_dx2(V)
    V_yy = dV_dy2(V)
    C_xx = dC_dx2(C)
    C_yy = dC_dy2(C)
    eq0_V = eq0_V_lambdified(V_yy, J, C, V, V_xx)
    eq0_C = eq0_C_lambdified(C, C_yy, xx, C_xx, 0, yy, V, A, 0, 0)
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
    V0 = jnp.full((NX, NY), -1)
    C0 = jnp.full((NX, NY), jnp.sqrt(-J))
    x0 = pack_electrostatic(V0, C0)
    print(x0.shape)
    from matplotlib import pyplot as plt
    V0, C0 = unpack_electrostatic(x0)
    plt.imshow(V0)
    plt.show()
    plt.imshow(C0)
    plt.show()


    start = time.time()
    electrostatic_solution = newton(f_electrostatic, x0)
    end = time.time()

    print("Elapsed time for electrostatic solution: ", end - start)

    # use electrostatic solution as initial guess for magnetostatic problem
    V0, C0 = unpack_electrostatic(electrostatic_solution)
    from matplotlib import pyplot as plt
    plt.imshow(V0)
    plt.show()
    plt.imshow(C0)
    plt.show()
    from scipy.interpolate import LinearNDInterpolator
    def plot_cart(func, label):
        save_file = 'data/' + sys.argv[1] + '_' + label + '_cart.png'
        xs_func = []
        ys_func = []
        zs_func = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                xs_func.append(x)
                ys_func.append(y)
                zs_func.append(func[i,j])
                xs_func.append(-x)
                ys_func.append(y)
                zs_func.append(func[i,j])

        NX_interp = 1000
        NY_interp = 1000
        min_x, max_x = -5*K, 5*K
        min_y, max_y = -5*K, 5*K
        X = np.linspace(min_x, max_x, NX_interp)
        Y = np.linspace(min_y, max_y, NY_interp)
        X, Y = np.meshgrid(X, Y)
        interp = LinearNDInterpolator(list(zip(xs_func, ys_func)), zs_func)
        Z = interp(X,Y)
        plt.pcolormesh(X,Y,Z, cmap = 'hot', shading = 'auto')
        plt.colorbar()
        plt.axis('equal')
        plt.title(label)
        plt.savefig(save_file)
        plt.close()

    plot_cart(V0, 'V')
    plot_cart(C0, 'C')











    exit()
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

