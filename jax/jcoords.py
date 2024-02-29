### COORDINATE TRANSFORMATIONS ###
from jax import jit 
import jax.numpy as jnp
from sympy import symbols, ln, solve
from sympy.utilities import lambdify
import jax
jax.config.update('jax_platform_name', 'cpu')

# vp represents primed coordinate on the domain (0, 1)
v, vp, a, j = symbols('v v_p a j')

# require v(0) = 0, v(1) = infty
v_of_vp= ln((1+vp)/(1-vp))

# invert mapping and compute all possible derivatives up to second order
vp_of_v = solve(v_of_vp - v, vp)[0]

dv_dvp = v_of_vp.diff(vp).doit().simplify()
d2v_dvp2 = v_of_vp.diff(vp,2).doit().simplify()

dvp_dv = vp_of_v.diff(v).doit().simplify()
d2vp_dv2 = vp_of_v.diff(v, 2).doit().simplify()

args = [vp, a, j]
v_of_vp_lambdified = jit(lambdify(args, v_of_vp, modules='jax'))
dv_dvp_lambdified = jit(lambdify(args, dv_dvp, modules='jax'))
d2v_dvp2_lambdified = jit(lambdify(args, d2v_dvp2, modules='jax'))

args = [v, a, j]
vp_of_v_lambdified = jit(lambdify(args, vp_of_v, modules='jax'))
dvp_dv_lambdified = jit(lambdify(args, dvp_dv, modules='jax'))
d2vp_dv2_lambdified = jit(lambdify(args, d2vp_dv2, modules='jax'))

# convert a vector with bipolar components (Fu, Fv) at coordinates (u,v) to cartesian coordinates (Fx, Fy)
@jit
def BP2cart(Fu, Fv, u, v):
    M_00 = -jnp.sin(u)*jnp.sinh(v) / (jnp.cosh(v)-jnp.cos(u))
    M_01 = (1 - jnp.cos(u)*jnp.cosh(v)) / (jnp.cosh(v)-jnp.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return Fu * M_00 + Fv * M_01, Fu * M_10 + Fv * M_11

# convert a vector with cartesian components (Fx, Fy) at coordinates (u,v) to bipolar coordinates (Fu, Fv)
@jit
def cart2BP(Fx, Fy, u, v):
    M_00 = -jnp.sin(u)*jnp.sinh(v) / (jnp.cosh(v)-jnp.cos(u))
    M_01 = (1 - jnp.cos(u)*jnp.cosh(v)) / (jnp.cosh(v)-jnp.cos(u))
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

# convert a vector with cartesian components (Fx, Fy) at coordinates (u,infty) to bipolar coordinates (Fu, Fv)
@jit
def cart2BPinfinity(Fx, Fy, u):
    M_00 = -jnp.sin(u)
    M_01 = -jnp.cos(u)
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

if __name__ == '__main__':
    import sys 
    sys.path.append('../')
    import coords 
    import numpy as np
    import time 

    print("Running unit tests on coordinates")
    JIT_BEFORE = 1
    LAMBDIFY_TEST = 1
    COORD_TEST = 1

    @jit
    def get_lambdified(args):
        t1 = v_of_vp_lambdified(*args)
        t2 = dv_dvp_lambdified(*args)
        t3 = d2v_dvp2_lambdified(*args)
        t4 = vp_of_v_lambdified(*args)
        t5 = dvp_dv_lambdified(*args)
        t6 = d2vp_dv2_lambdified(*args)
        return t1, t2, t3, t4, t5, t6
    
    def lambdify_test(args, output = True):
        t = time.time()
        t1, t2, t3, t4, t5, t6 = get_lambdified(args)
        if output: print("Elapsed time for jax functions: ", time.time()-t)
        
        t = time.time()
        t1_og = coords.v_of_vp_lambdified(*args)
        t2_og = coords.dv_dvp_lambdified(*args)
        t3_og = coords.d2v_dvp2_lambdified(*args)
        t4_og = coords.vp_of_v_lambdified(*args)
        t5_og = coords.dvp_dv_lambdified(*args)
        t6_og = coords.d2vp_dv2_lambdified(*args)
        if output: print("Elapsed time for original functions: ", time.time()-t)

        assert np.isclose(t1, t1_og)
        assert np.isclose(t2, t2_og)
        assert np.isclose(t3, t3_og)
        assert np.isclose(t4, t4_og)
        assert np.isclose(t5, t5_og)
        assert np.isclose(t6, t6_og)

        if output: print("Passed lambdified tests.")
    
    @jit
    def get_coords(Fu, Fv, u, v):
        t1 = BP2cart(Fu, Fv, u, v)
        t2 = cart2BP(t1[0], t1[1], u, v)
        t3 = cart2BPinfinity(t1[0], t1[1], u)
        return t1, t2, t3
    
    def coord_test(Fu, Fv, u, v, output = True):
        if output: print("Testing coordinate transformations: ")
        t = time.time()
        t1, t2, t3 = get_coords(Fu, Fv, u, v)
        if output: print("Elapsed time for jax functions: ", time.time()-t)

        t = time.time()
        t1_og = coords.BP2cart(Fu, Fv, u, v)
        t2_og = coords.cart2BP(t1_og[0], t1_og[1], u, v)
        t3_og = coords.cart2BPinfinity(t1_og[0], t1_og[1], u)
        if output: print("Elapsed time for original functions: ", time.time()-t)

        assert np.isclose(t1, t1_og).all()
        assert np.isclose(t2, t2_og).all()
        assert np.isclose(t3, t3_og).all()

        if output: print("Passed coordinate tests.")

    if JIT_BEFORE:
        args = [0.5, 1, 1]
        u = np.random.random()
        v = np.random.random()
        Fu = np.random.random()
        Fv = np.random.random()

        lambdify_test(args, False)
        coord_test(Fu, Fv, u, v, False)

    if LAMBDIFY_TEST:
        args = [0.5, 1, 1]
        lambdify_test(args)

    if COORD_TEST:
        assert LAMBDIFY_TEST
        u = np.random.random()
        v = np.random.random()
        Fu = np.random.random()
        Fv = np.random.random()

        coord_test(Fu, Fv, u, v, True)

