from sympy import symbols, Function, diff, ln, solve
from sympy.utilities import lambdify
from numba import njit

# define symbols and functions
v, vp, a, j = symbols('v v_p a j')

#v_of_vp= ln((1+vp)/(1-vp))
k = (-4/j)**(1/4)
xi = k/2
v_of_vp = -ln(1 + 2 * a / xi)*(1+1/(vp-1))
vp_of_v = solve(v_of_vp - v, vp)[0]

dv_dvp = v_of_vp.diff(vp).doit().simplify()
d2v_dvp2 = v_of_vp.diff(vp,2).doit().simplify()

dvp_dv = vp_of_v.diff(v).doit().simplify()
d2vp_dv2 = vp_of_v.diff(v, 2).doit().simplify()

args = [vp, a, j]
v_of_vp_lambdified = lambdify(args, v_of_vp)
dv_dvp_lambdified = lambdify(args, dv_dvp)
d2v_dvp2_lambdified = lambdify(args, d2v_dvp2)

args = [v, a, j]
vp_of_v_lambdified = lambdify(args, vp_of_v)
dvp_dv_lambdified = lambdify(args, dvp_dv)
d2vp_dv2_lambdified = lambdify(args, d2vp_dv2)

v_of_vp_lambdified = njit(v_of_vp_lambdified)
dv_dvp_lambdified = njit(dv_dvp_lambdified)
d2v_dvp2_lambdified = njit(d2v_dvp2_lambdified)

vp_of_v_lambdified = njit(vp_of_v_lambdified)
dvp_dv_lambdified = njit(dvp_dv_lambdified)
d2vp_dv2_lambdified = njit(d2vp_dv2_lambdified)