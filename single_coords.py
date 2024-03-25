### COORDINATE TRANSFORMATIONS ###
from jax import jit 
import jax.numpy as jnp
from sympy import symbols, ln, solve, sqrt, sinh, pi, tan
from sympy.utilities import lambdify
import jax
#jax.config.update('jax_platform_name', 'cpu')

# vp represents primed coordinate on the domain (0, 1)
r, rp, j = symbols('r r_p j')

# require v(0) = 0, v(1) = infty
#r_of_rp = ln((1+rp)/(1-rp))
k = (-4/j)**(1/4)
r_of_rp = 10*k**2 *(tan(rp*pi/2))**2
#r = R*tan(rp*pi/2)

#c1,c2=1,1
#R = c1*k*(1 - vp) + c2*(k*(1 - vp))**2
#x=a/R
#v_of_vp = ((1/2)*ln((x + sqrt(1 + x**2))**2))

# invert mapping and compute all possible derivatives up to second order
rp_of_r = solve(r_of_rp - r, rp)[0]
#vp_of_v = (1 + 2*k - sqrt(4*a + c1*sinh(v))/sqrt(c1*sinh(v)))/(2*k)
#vp_of_v = (c1 + 2*c2*k - sqrt(4*a*c2 + c1**2 *sinh(v))/sqrt(sinh(v)))/(2*c2*k)

#vp_of_v = 1 - ln((-a - sinh(v))**(1/3)/(-sinh(v))**(1/3))/ln(k)

dr_drp1 = r_of_rp.diff(rp).doit()
dr_drp2 = r_of_rp.diff(rp,2).doit()

drp_dr1 = rp_of_r.diff(r).doit()
drp_dr2 = rp_of_r.diff(r, 2).doit()

args = [rp, j]
r_of_rp_lambdified = jit(lambdify(args, r_of_rp, modules='jax'))
dr_drp1_lambdified = jit(lambdify(args, dr_drp1, modules='jax'))
dr_drp2_lambdified = jit(lambdify(args, dr_drp2, modules='jax'))

args = [r, j]
rp_of_r_lambdified = jit(lambdify(args, rp_of_r, modules='jax'))
drp_dr1_lambdified = jit(lambdify(args, drp_dr1, modules='jax'))
drp_dr2_lambdified = jit(lambdify(args, drp_dr2, modules='jax'))

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