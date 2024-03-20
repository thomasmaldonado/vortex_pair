### LAMBDIFICATION OF BOUNDARY VALUE PROBLEM ###

from sympy import symbols, Function, cos, sin, diff
from sympy.utilities import lambdify
from sympy.vector import CoordSys3D
from sympy.functions.elementary.hyperbolic import cosh, sinh
from jax import jit 
import jax
#jax.config.update('jax_platform_name', 'cpu')

# define symbols and functions
u, v, a, j, n= symbols('u v a j n')

V = Function('V')
Fu = Function('Fu')
Fv = Function('Fv')
C = Function('C')

V_syms = symbols('V, V_u, V_v, V_uu, V_uv, V_vv')
V_0, V_u, V_v, V_uu, V_uv, V_vv = V_syms

Fu_syms = symbols('Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv')
Fu_0, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv = Fu_syms

Fv_syms = symbols('Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv')
Fv_0, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv = Fv_syms

C_syms = symbols('C, C_u, C_v, C_uu, C_uv, C_vv')
C_0, C_u, C_v, C_uu, C_uv, C_vv = C_syms  

# helper function to turn functions into symbols
def symbolify(expr, fun, syms):
    expr = expr.subs(fun.diff(v,2), syms[-1])
    expr = expr.subs(fun.diff(u).diff(v), syms[-2])
    expr = expr.subs(fun.diff(u,2), syms[-3])
    expr = expr.subs(fun.diff(v), syms[-4])
    expr = expr.subs(fun.diff(u), syms[-5])
    expr = expr.subs(fun, syms[-6])
    return expr

# define bipolar coordinate system
BP = CoordSys3D('BP')
u_hat, v_hat, z_hat = BP.i, BP.j, BP.k
h = a/(cosh(v)-cos(u))
Au = Fu(u,v)/h - n/h
Av = Fv(u,v)/h


# define vector calculus derivatives
def gradient(f):
    return (diff(f, u)*BP.i + diff(f, v)*BP.j)/h

def div(F):
    Fu = F.dot(u_hat)
    Fv = F.dot(v_hat)
    return (diff(Fu*h, u) + diff(Fv*h, v)) / h**2

def curl(F):
    Fu = F.dot(u_hat)
    Fv = F.dot(v_hat)
    Fz = F.dot(z_hat)
    curl_u = u_hat * diff(Fz, v) / h
    curl_v = - v_hat * diff(Fz, u) / h 
    curl_z = z_hat * (diff(h * Fv, u) - diff(h * Fu, v)) / h**2
    return curl_u + curl_v + curl_z

def laplacian(f):
    return div(gradient(f))

# define equations of state in the form eq0 = 0
eq0_V = (-laplacian(V(u,v)) + C(u,v)**2 * V(u,v) - j).doit()
eq0_F = (curl(curl(Au*u_hat + Av*v_hat).simplify()) + C(u,v)**2 * (Au*u_hat + Av*v_hat)).doit().expand()
eq0_Fu = eq0_F.dot(u_hat)
eq0_Fv = eq0_F.dot(v_hat)
eq0_C = (-laplacian(C(u,v)) + (1 - V(u,v)**2 + Au**2 + Av**2) * C(u,v)).doit()

# symbolify equations of state
eq0_V = symbolify(eq0_V, V(u,v), V_syms)
eq0_V = symbolify(eq0_V, C(u,v), C_syms)

eq0_Fu = symbolify(eq0_Fu, Fu(u,v), Fu_syms)
eq0_Fu = symbolify(eq0_Fu, Fv(u,v), Fv_syms)
eq0_Fu = symbolify(eq0_Fu, C(u,v), C_syms)

eq0_Fv = symbolify(eq0_Fv, Fu(u,v), Fu_syms)
eq0_Fv = symbolify(eq0_Fv, Fv(u,v), Fv_syms)
eq0_Fv = symbolify(eq0_Fv, C(u,v), C_syms)

eq0_C = symbolify(eq0_C, V(u,v), V_syms)
eq0_C = symbolify(eq0_C, Fu(u,v), Fu_syms)
eq0_C = symbolify(eq0_C, Fv(u,v), Fv_syms)
eq0_C = symbolify(eq0_C, C(u,v), C_syms)

# define electric and magnetic fields for future calculation of energies
B = curl(Au*u_hat+Av*v_hat).dot(z_hat).expand().simplify()
B = symbolify(B, Fu(u,v), Fu_syms)
B = symbolify(B, Fv(u,v), Fv_syms)

E = -gradient(V(u,v)).doit().simplify()
Eu = symbolify(E.dot(u_hat), V(u,v), V_syms)
Ev = symbolify(E.dot(v_hat), V(u,v), V_syms)

# lambdify equations of state
args = list(V_syms)
args.extend(Fu_syms)
args.extend(Fv_syms)
args.extend(C_syms)
args.extend([u, v, a, j, n])

eq0_V_args = [V_vv, V_uu, V, j, C, v, a, u]
eq0_Fu_args = [C, n, u, Fv_v, Fv_uv, v, Fu_vv, Fv_u, a, Fu, Fu_v] # added Fu_v
eq0_Fv_args = [v, Fu_u, Fu_v, Fv_uu, C, Fv, a, Fu_uv, u, Fv_u] # added Fv_u
eq0_C_args = [v, Fu, C_uu, n, V, C, C_vv, Fv, a, u]
Eu_args = [u, v, a, V_u]
Ev_args = [V_v, v, a, u]
B_args = [v, Fv_u, Fu_v, Fu, Fv, a, u]

eq0_V_lambdified = jit(lambdify(eq0_V_args, eq0_V, modules='jax'))
eq0_Fu_lambdified = jit(lambdify(eq0_Fu_args, eq0_Fu, modules='jax'))
eq0_Fv_lambdified = jit(lambdify(eq0_Fv_args, eq0_Fv, modules='jax'))
eq0_C_lambdified = jit(lambdify(eq0_C_args, eq0_C, modules='jax'))
Eu_lambdified = jit(lambdify(Eu_args, Eu, modules='jax'))
Ev_lambdified = jit(lambdify(Ev_args, Ev, modules='jax'))
B_lambdified = jit(lambdify(B_args, B, modules='jax'))
print(B)