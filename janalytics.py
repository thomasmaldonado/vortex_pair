### LAMBDIFICATION OF BOUNDARY VALUE PROBLEM ###

from sympy import symbols, Function, cos, sin, diff, sqrt
from sympy.utilities import lambdify
from sympy.vector import CoordSys3D, curl, divergence, Del
from sympy.functions.elementary.hyperbolic import cosh, sinh
from jax import jit 
import jax
#jax.config.update('jax_platform_name', 'cpu')

# define symbols and functions
u, v, a, j, n= symbols('u v a j n')

V = Function('V')
Fx = Function('Fx')
Fy = Function('Fy')
C = Function('C')

V_syms = symbols('V, V_x, V_y, V_xx, V_xy, V_yy')
V_0, V_x, V_y, V_xx, V_xy, V_yy = V_syms

Fx_syms = symbols('Fx, Fx_x, Fx_y, Fx_xx, Fx_xy, Fx_yy')
Fx_0, Fx_x, Fx_y, Fx_xx, Fx_xy, Fx_yy = Fx_syms

Fy_syms = symbols('Fy, Fy_x, Fy_y, Fy_xx, Fy_xy, Fy_yy')
Fy_0, Fy_x, Fy_y, Fy_xx, Fy_xy, Fy_yy = Fy_syms

C_syms = symbols('C, C_x, C_y, C_xx, C_xy, C_yy')
C_0, C_x, C_y, C_xx, C_xy, C_yy = C_syms  


# define bipolar coordinate system
R = CoordSys3D('R')
delop = Del()
r = sqrt((R.x-a)**2 + R.y**2)
phi_hat = (-R.y*R.i + (R.x-a)*R.j)/r
F = Fx(R.x, R.y)*R.i + Fy(R.x,R.y)*R.j
A = F - n*phi_hat/r
print(A)
# helper function to turn functions into symbols
def symbolify(expr, fun, syms):
    expr = expr.subs(fun.diff(R.y,2), syms[-1])
    expr = expr.subs(fun.diff(R.x).diff(R.y), syms[-2])
    expr = expr.subs(fun.diff(R.x,2), syms[-3])
    expr = expr.subs(fun.diff(R.y), syms[-4])
    expr = expr.subs(fun.diff(R.x), syms[-5])
    expr = expr.subs(fun, syms[-6])
    return expr

# define equations of state in the form eq0 = 0
eq0_V = (-divergence(delop((V(R.x,R.y)))) + C(R.x,R.y)**2 * V(R.x,R.y) - j).doit()
eq0_F = (curl(curl(Fx(R.x,R.y)*R.i + Fy(R.x,R.y)*R.j).simplify()) + C(R.x,R.y)**2 * A).doit().expand()
eq0_Fx = eq0_F.dot(R.i)
eq0_Fy = eq0_F.dot(R.j)
eq0_C = (-divergence(delop(C(R.x,R.y))) + (1 - V(R.x,R.y)**2 + A.dot(R.i)**2+A.dot(R.j)**2) * C(R.x,R.y)).doit()

# symbolify equations of state
eq0_V = symbolify(eq0_V, V(R.x,R.y), V_syms)
eq0_V = symbolify(eq0_V, C(R.x,R.y), C_syms)

eq0_Fx = symbolify(eq0_Fx, Fx(R.x,R.y), Fx_syms)
eq0_Fx = symbolify(eq0_Fx, Fy(R.x,R.y), Fy_syms)
eq0_Fx = symbolify(eq0_Fx, C(R.x,R.y), C_syms)

eq0_Fy = symbolify(eq0_Fy, Fx(R.x,R.y), Fx_syms)
eq0_Fy = symbolify(eq0_Fy, Fy(R.x,R.y), Fy_syms)
eq0_Fy = symbolify(eq0_Fy, C(R.x,R.y), C_syms)


eq0_C = symbolify(eq0_C, V(R.x,R.y), V_syms)
eq0_C = symbolify(eq0_C, Fx(R.x,R.y), Fx_syms)
eq0_C = symbolify(eq0_C, Fy(R.x,R.y), Fy_syms)
eq0_C = symbolify(eq0_C, C(R.x,R.y), C_syms)

print('eq0_V', eq0_V)
print('eq0_Fx', eq0_Fx)
print('eq0_Fy', eq0_Fy)
print('eq0_C', eq0_C)

# define electric and magnetic fields for future calculation of energies
B = curl(A).dot(R.k).expand().simplify()
B = symbolify(B, Fx(R.x,R.y), Fx_syms)
B = symbolify(B, Fy(R.x,R.y), Fy_syms)

E = -delop(V(R.x,R.y)).doit().simplify()
Ex = symbolify(E.dot(R.i), V(R.x,R.y), V_syms)
Ey = symbolify(E.dot(R.j), V(R.x,R.y), V_syms)

# lambdify equations of state
args = list(V_syms)
args.extend(Fx_syms)
args.extend(Fy_syms)
args.extend(C_syms)
args.extend([u, v, a, j, n])

print('eq0_V_args = ', list(eq0_V.free_symbols))
print('eq0_Fx_args = ', list(eq0_Fx.free_symbols))
print('eq0_Fy_args = ', list(eq0_Fy.free_symbols))
print('eq0_C_args = ', list(eq0_C.free_symbols))
print('B_args = ', list(B.free_symbols))
print('Ex_args = ', list(Ex.free_symbols))
print('Ey_args = ', list(Ey.free_symbols))

eq0_V_args =  [V_yy, j, C, V, V_xx]
eq0_Fx_args =  [C, R.x, Fx, R.y, a, n, Fx_yy, Fy_xy]
eq0_Fy_args =  [Fx_xy, C, R.x, R.y, a, n, Fy, Fy_xx]
eq0_C_args =  [C, C_yy, R.x, C_xx, Fx, R.y, V, a, n, Fy]
B_args =  [Fy_x, Fx_y]
Ex_args =  [V_x]
Ey_args =  [V_y]

eq0_V_lambdified = jit(lambdify(eq0_V_args, eq0_V, modules='jax'))
eq0_Fx_lambdified = jit(lambdify(eq0_Fx_args, eq0_Fx, modules='jax'))
eq0_Fy_lambdified = jit(lambdify(eq0_Fy_args, eq0_Fy, modules='jax'))
eq0_C_lambdified = jit(lambdify(eq0_C_args, eq0_C, modules='jax'))
Ex_lambdified = jit(lambdify(Ex_args, Ex, modules='jax'))
Ey_lambdified = jit(lambdify(Ey_args, Ey, modules='jax'))
B_lambdified = jit(lambdify(B_args, B, modules='jax'))