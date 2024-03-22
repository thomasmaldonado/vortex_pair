### LAMBDIFICATION OF BOUNDARY VALUE PROBLEM ###

from sympy import symbols, Function, cos, sin, diff
from sympy.utilities import lambdify
from sympy.vector import CoordSys3D
from sympy.functions.elementary.hyperbolic import cosh, sinh
from jax import jit 
import jax
#jax.config.update('jax_platform_name', 'cpu')

"""
File extracts lambdified functions for numerical analysis from the equations of motion
"""
from sympy import symbols, Function, solve
from sympy.utilities import lambdify
from sympy.vector import CoordSys3D, Del
#from coordinates import R, s
#from coordinates import r as r_of_s

# define symbols and functions
j, n = symbols('j n')  # radial coordinate and kappa^2
V = Function('V') # scalar electric potential
F = Function('F')
C = Function('C')

# derivatives of functions w.r.t. radial coordinate r in the range (0, infinity)
V_r_syms = symbols('V_0 V_1 V_2')
V_0, V_1, V_2 = V_r_syms
F_r_syms = symbols('F_0 F_1 F_2')
F_0, F_1, F_2 = F_r_syms
C_r_syms = symbols('C_0 C_1 C_2')
C_0, C_1, C_2 = C_r_syms

# helper function to turn functions into symbols
def symbolify(expr, fun, var, syms):
	for i in range(len(syms)-1, -1, -1):
		expr = expr.subs(fun.diff(var, i), syms[i])
	return expr

# coordinate system 
P = CoordSys3D('P', transformation='cylindrical', variable_names=("r", "theta", "z"))
delop = Del()
A = F(P.r) - n/P.r

# functions to find the zeros of (Gauss, Ampere, Schrödinger)
eq0_V = (-delop.dot(delop(V(P.r))) + C(P.r)**2*V(P.r) - j).doit()
eq0_F = (delop.dot(delop(F(P.r))) - (F(P.r)/P.r**2) - C(P.r)**2*A).doit()
eq0_C = (delop.dot(delop(C(P.r))) - (1 - V(P.r)**2 + A**2)*C(P.r)).doit()
B = (diff(P.r * F(P.r), P.r)/P.r).doit()
E = (-diff(V(P.r), P.r)).doit()

# symbolify all
eq0_V = symbolify(eq0_V, V(P.r), P.r, V_r_syms).simplify()
eq0_V = symbolify(eq0_V, F(P.r), P.r, F_r_syms).simplify()
eq0_V = symbolify(eq0_V, C(P.r), P.r, C_r_syms).simplify()

eq0_F = symbolify(eq0_F, V(P.r), P.r, V_r_syms).simplify()
eq0_F = symbolify(eq0_F, F(P.r), P.r, F_r_syms).simplify()
eq0_F = symbolify(eq0_F, C(P.r), P.r, C_r_syms).simplify()

eq0_C = symbolify(eq0_C, V(P.r), P.r, V_r_syms).simplify()
eq0_C = symbolify(eq0_C, F(P.r), P.r, F_r_syms).simplify()
eq0_C = symbolify(eq0_C, C(P.r), P.r, C_r_syms).simplify()

B = symbolify(B, F(P.r), P.r, F_r_syms)

E = symbolify(E, V(P.r), P.r, V_r_syms)

eq0_V_args = [C_0, V_2, j, V_0, P.r, V_1]
eq0_F_args = [F_1, C_0, n, F_2, P.r, F_0]
eq0_C_args = [F_0, n, C_2, C_0, V_0, C_1, P.r]
B_args = [F_0, F_1, P.r]
E_args = [V_1]

# define function arguments and lambdify
eq0_V_lambdified = jit(lambdify(eq0_V_args, eq0_V, modules='jax'))
eq0_F_lambdified = jit(lambdify(eq0_F_args, eq0_F, modules='jax'))
eq0_C_lambdified = jit(lambdify(eq0_C_args, eq0_C, modules='jax'))
B_lambdified = jit(lambdify(B_args, B, modules='jax'))
E_lambdified = jit(lambdify(E_args, E, modules='jax'))

