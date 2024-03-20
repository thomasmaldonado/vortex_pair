### COORDINATE TRANSFORMATIONS ###
from jax import jit 
import jax.numpy as jnp
from sympy import symbols, ln, solve, sqrt, sinh
from sympy.utilities import lambdify
import jax
#jax.config.update('jax_platform_name', 'cpu')

# vp represents primed coordinate on the domain (0, 1)
v, vp, a, j = symbols('v v_p a j')

# require v(0) = 0, v(1) = infty
v_of_vp = ln((1+vp)/(1-vp))
#k = (-4/j)**(1/4)

#c1,c2=1,1
#R = c1*k*(1 - vp) + c2*(k*(1 - vp))**2
#x=a/R
#v_of_vp = ((1/2)*ln((x + sqrt(1 + x**2))**2))

# invert mapping and compute all possible derivatives up to second order
vp_of_v = solve(v_of_vp - v, vp)[0]
#vp_of_v = (1 + 2*k - sqrt(4*a + c1*sinh(v))/sqrt(c1*sinh(v)))/(2*k)
#vp_of_v = (c1 + 2*c2*k - sqrt(4*a*c2 + c1**2 *sinh(v))/sqrt(sinh(v)))/(2*c2*k)

#vp_of_v = 1 - ln((-a - sinh(v))**(1/3)/(-sinh(v))**(1/3))/ln(k)

dv_dvp1 = v_of_vp.diff(vp).doit()
dv_dvp2 = v_of_vp.diff(vp,2).doit()

dvp_dv1 = vp_of_v.diff(v).doit()
dvp_dv2 = vp_of_v.diff(v, 2).doit()

args = [vp, a, j]
v_of_vp_lambdified = jit(lambdify(args, v_of_vp, modules='jax'))
dv_dvp1_lambdified = jit(lambdify(args, dv_dvp1, modules='jax'))
dv_dvp2_lambdified = jit(lambdify(args, dv_dvp2, modules='jax'))

args = [v, a, j]
vp_of_v_lambdified = jit(lambdify(args, vp_of_v, modules='jax'))
dvp_dv1_lambdified = jit(lambdify(args, dvp_dv1, modules='jax'))
dvp_dv2_lambdified = jit(lambdify(args, dvp_dv2, modules='jax'))

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