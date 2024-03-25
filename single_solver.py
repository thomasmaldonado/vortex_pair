### BOUNDARY VALUE PROBLEM SOLVER ###
import sys 
#from params import K_func, A_func, tol, max_iter
from single_coords import r_of_rp_lambdified, drp_dr1_lambdified, drp_dr2_lambdified, dr_drp1_lambdified
from single_analytics import eq0_V_lambdified, eq0_F_lambdified, eq0_C_lambdified, B_lambdified, E_lambdified
from single_derivatives import d_dx1, d_dx2
import numpy as np
import jax.numpy as jnp
from jax import jit 
import jax
from jax.numpy import linalg as jla
import time 
#from files import save, load
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

# command line arguments

def solve(K, N):
    J = -4/K**4
    tol = 1e-6
    max_iter = 30

    MAX_R = 10*K**2
    nr = 1000

    rs = jnp.linspace(0, MAX_R, nr+2)[1:-1]
    dr = rs[1]-rs[0]

    @jit
    def d_dr1(f, boundary_left, boundary_right):
        return d_dx1(f, dr, boundary_left, boundary_right)

    @jit
    def d_dr2(f, boundary_left, boundary_right):
        return d_dx2(f, dr, boundary_left, boundary_right)


    @jit
    def dC_dr1(C):
        return d_dr1(C, 0, jnp.sqrt(-J))
    @jit
    def dC_dr2(C):
        return d_dr2(C, 0, jnp.sqrt(-J))
    @jit
    def dF_dr1(F):
        return d_dr1(F, 0, N/MAX_R)
    @jit
    def dF_dr2(F):
        return d_dr2(F, 0, N/MAX_R)
    @jit
    def dV_dr1(V):
        return d_dr1(V, V[0], -1)
    @jit
    def dV_dr2(V):
        return d_dr2(V, V[0], -1)

    # helper functions to pack/unpack and reshape the solutions
    @jit
    def pack_electrostatic(V, C):
        return jnp.array(jnp.concatenate((V, C)))
    @jit
    def pack_magnetostatic(V, F, C):
        return jnp.array(jnp.concatenate((V, F, C)))

    @jit
    def unpack_electrostatic(V_C):
        return V_C[0:nr], V_C[nr:]

    @jit
    def unpack_magnetostatic(V_F_C):
        return V_F_C[0:nr], V_F_C[nr:2*nr], V_F_C[2*nr:]

    # define function whose root yields the electrostatic solution (NL = NR = 0)
    @jit
    def f_electrostatic(V_C):
        V, C = unpack_electrostatic(V_C)
        V_1 = dV_dr1(V)
        V_2 = dV_dr2(V)
        C_1 = dC_dr1(C)
        C_2 = dC_dr2(C)
        eq0_V = eq0_V_lambdified(C, V_2, J, V, rs, V_1)
        eq0_C = eq0_C_lambdified(0, 0, C_2, C, V, C_1, rs)
        return pack_electrostatic(eq0_V, eq0_C)

    # define function whose root yields the magnetostatic solution
    @jit
    def f_magnetostatic(V_F_C):
        V, F, C = unpack_magnetostatic(V_F_C)
        V_1 = dV_dr1(V)
        V_2 = dV_dr2(V)
        C_1 = dC_dr1(C)
        C_2 = dC_dr2(C)
        F_1 = dF_dr1(F)
        F_2 = dF_dr2(F)
        eq0_V = eq0_V_lambdified(C, V_2, J, V, rs, V_1)
        eq0_F = eq0_F_lambdified(F_1, C, N, F_2, rs, F)
        eq0_C = eq0_C_lambdified(F, N, C_2, C, V, C_1, rs)
        return pack_magnetostatic(eq0_V, eq0_F, eq0_C)

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
        V0 = jnp.full(nr, -1)
        C0 = jnp.full(nr, jnp.sqrt(-J))
        x0 = pack_electrostatic(V0, C0)
        start = time.time()
        electrostatic_solution = newton(f_electrostatic, x0)
        end = time.time()
        print("Elapsed time for electrostatic solution: ", end - start)

        # use electrostatic solution as initial guess for magnetostatic problem
        V0, C0 = unpack_electrostatic(electrostatic_solution)
        F0 = jnp.zeros(nr)
        F0 = N/rs

    # perform Newton's method 
    x0 = pack_magnetostatic(V0, F0, C0)
    start = time.time()
    magnetostatic_solution = newton(f_magnetostatic, x0)
    end = time.time()
    print("Elapsed time for magnetostatic solution: ", end - start)

    start_processing = time.time()

    # begin post-processing 
    V, F, C = unpack_magnetostatic(magnetostatic_solution)
    #from matplotlib import pyplot as plt
    #plt.scatter(rs, V)
    #plt.xlim(0,5*K**2)
    #plt.title('V')
    #plt.show()
    #plt.scatter(rs, F)
    #plt.title('F')
    #plt.xlim(0,5*K**2)
    #plt.show()
    #plt.scatter(rs, C)
    #plt.xlim(0,5*K**2)
    #plt.title('C')
    #plt.show()



    # calculate supercurrent
    A = F - N/rs

    J0 = -C**2 * V
    Ju = -C**2 * A

    # calculate energy densities
    V_1 = dV_dr1(V)
    E = E_lambdified(V_1)
    F_1 = dF_dr1(F)
    B = B_lambdified(F, F_1, rs)

    EED = (E**2)/2
    MED = (B**2)/2
    HED = C**2 * V**2 + J
    TED = EED + MED + HED

    #plt.plot(EED)
    #plt.title('EED')
    #plt.show()
    #plt.plot(MED)
    #plt.title('MED')
    #plt.show()
    #plt.plot(HED)
    #plt.title('HED')
    #plt.show()
    #plt.close()


    dA = 2*np.pi*rs*dr
    EE = jnp.sum(EED * dA)
    ME = jnp.sum(MED * dA)
    HE = jnp.sum(HED * dA)
    TE = EE + ME + HE

    #print(HED[-1])
    #print(C[-1], V[-1], J)
    print(EE, ME, HE, TE)
    # save solution
    #save(outputfile, K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED)
    #summary = 'Saved: '
    #for x in sys.argv[1:]:
    #    summary += x + ' '
    #print(summary)

    flux = B*dA
    print('full flux single:', jnp.sum(B*dA))
    return EE, ME, HE, TE