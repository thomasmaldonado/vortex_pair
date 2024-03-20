### LAMBDIFICATION OF BOUNDARY VALUE PROBLEM ###

from sympy import symbols, Function, cos, sin, diff, exp
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
B = Function('B')
hFu = Function('hFu')
hFv = Function('hFv')

V_syms = symbols('V, V_u, V_v, V_uu, V_uv, V_vv')
V_0, V_u, V_v, V_uu, V_uv, V_vv = V_syms

Fu_syms = symbols('Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv')
Fu_0, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv = Fu_syms

Fv_syms = symbols('Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv')
Fv_0, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv = Fv_syms

hFu_syms = symbols('hFu, hFu_u, hFu_v, hFu_uu, hFu_uv, hFu_vv')
hFu_0, hFu_u, hFu_v, hFu_uu, hFu_uv, hFu_vv = Fu_syms

hFu_syms = symbols('hFv, hFv_u, hFv_v, hFv_uu, hFv_uv, hFv_vv')
hFv_0, hFv_u, hFu_v, hFv_uu, hFv_uv, hFv_vv = Fu_syms

C_syms = symbols('C, C_u, C_v, C_uu, C_uv, C_vv')
C_0, C_u, C_v, C_uu, C_uv, C_vv = C_syms  

B_syms = symbols('B, B_u, B_v, B_uu, B_uv, B_vv')
B_0, B_u, B_v, B_uu, B_uv, B_vv = B_syms  
Ju, Jv = symbols('Ju, Jv')

"""
B_syms = symbols('B, B_u, B_v, B_uu, B_uv, B_vv')
Eu_syms = symbols('Eu, Eu_u, Eu_v, Eu_uu, Eu_uv, Eu_vv')
Ev_syms = symbols('Ev, Ev_u, Ev_v, Ev_uu, Ev_uv, Ev_vv')
"""

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

Au = Fu(u,v)/h - (n/h)
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
    return (curl_u + curl_v + curl_z).doit()

def laplacian(f):
    return div(gradient(f))

# define equations of state in the form eq0 = 0
eq0_V = (-laplacian(V(u,v)) + C(u,v)**2 * V(u,v) - j).doit()
eq0_F = (curl(curl(Au*u_hat + Av*v_hat)) + C(u,v)**2 * (Au*u_hat + Av*v_hat)).doit().expand()
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
"""
eq0_B =(curl(B(u,v) * z_hat) - (Ju * u_hat + Jv * v_hat))
eq0_Bu = eq0_B.dot(u_hat)
eq0_Bv = eq0_B.dot(v_hat)
eq0_Bu = symbolify(eq0_Bu, B(u,v), B_syms)
eq0_Bv = symbolify(eq0_Bv, B(u,v), B_syms)
"""

B = curl(Au*u_hat+Av*v_hat).dot(z_hat)
B = symbolify(B, Fu(u,v), Fu_syms)
B = symbolify(B, Fv(u,v), Fv_syms)

E = -gradient(V(u,v))
Eu = symbolify(E.dot(u_hat), V(u,v), V_syms)
Ev = symbolify(E.dot(v_hat), V(u,v), V_syms)

# lambdify equations of state
args = list(V_syms)
args.extend(Fu_syms)
args.extend(Fv_syms)
args.extend(C_syms)
args.extend([u, v, a, j, n])

#print(eq0_V.free_symbols)
#print(eq0_Fu.free_symbols)
#print(eq0_Fv.free_symbols)
#print(eq0_C.free_symbols)

eq0_V_args = [V_vv, V_uu, V_0, j, C_0, v, a, u]
eq0_Fu_args = [C_0, n, u, Fv_v, Fv_uv, v, Fu_vv, Fv_u, a, Fu_0, Fv_0, Fu_v] # added Fv, Fu_v
eq0_Fv_args = [v, Fu_u, Fu_v, Fv_uu, C_0, Fv_0, a, Fu_uv, u, Fu_0, n, Fv_u] # added Fu, n, Fv_u
eq0_C_args = [v, Fu_0, C_uu, n, V_0, C_0, C_vv, Fv_0, a, u]

Eu_args = [u, v, a, V_u]
Ev_args = [V_v, v, a, u]
B_args = [v, Fv_u, Fu_v, Fu_0, Fv_0, a, u, n] # added n
eq0_Bu_args = [v, Ju, a, u, B_v]

def check(args, expr):
    for x in args:
        if x not in expr.free_symbols:
            print('Extraneous ' + str(x))
    for x in expr.free_symbols:
        if x not in args:
            print('Mising ' + str(x))
""""
print('eq0_V')
check(eq0_V_args, eq0_V)
print('Eq0_Fu')
check(eq0_Fu_args, eq0_Fu)
print('Eq0_Fv')
check(eq0_Fv_args, eq0_Fv)
print('Eq0_C')
check(eq0_C_args, eq0_C)
print('Eu')
check(Eu_args, Eu)
print('Ev')
check(Ev_args, Ev)
print('B')
check(B_args, B)
"""

eq0_V_lambdified = jit(lambdify(eq0_V_args, eq0_V, modules='jax'))
eq0_Fu_lambdified = jit(lambdify(eq0_Fu_args, eq0_Fu, modules='jax'))
eq0_Fv_lambdified = jit(lambdify(eq0_Fv_args, eq0_Fv, modules='jax'))
eq0_C_lambdified = jit(lambdify(eq0_C_args, eq0_C, modules='jax'))
Eu_lambdified = jit(lambdify(Eu_args, Eu, modules='jax'))
Ev_lambdified = jit(lambdify(Ev_args, Ev, modules='jax'))
B_lambdified = jit(lambdify(B_args, B, modules='jax'))

""""
#eq0_Bu_lambdified = jit(lambdify(eq0_Bu_args, eq0_Bu, modules='jax'))
print('B')
print(B.simplify())
print('done')


Ju = C(u,v)**2 * Au
Jv = C(u,v)**2 * Av
B = curl(Fu(u,v)*u_hat+Fv(u,v)*v_hat)
curl_B_u = curl(B).dot(u_hat)
curl_B_v = curl(B).dot(v_hat).doit()

Ju = symbolify(Ju, C(u,v), C_syms)
Ju = symbolify(Ju, Fu(u,v), Fu_syms)
Jv = symbolify(Jv, C(u,v), C_syms)
Jv = symbolify(Jv, Fv(u,v), Fv_syms)
curl_B_u = symbolify(curl_B_u, Fu(u,v), Fu_syms)
curl_B_u = symbolify(curl_B_u, Fv(u,v), Fv_syms)
curl_B_v = symbolify(curl_B_v, Fu(u,v), Fu_syms)
curl_B_v = symbolify(curl_B_v, Fv(u,v), Fv_syms)

print(curl_B_v)
Ju_args = [C, u, a, Fu, v, n]
Jv_args = [C, u, a, Fv, n]
curl_B_u_args = [u, Fv_u, v, Fv_v, Fu_v, Fv, a, Fu, Fv_uv, Fu_vv]
curl_B_v_args = [u, Fv_uu, Fv_u, v, Fu_v, Fv, a, Fu, Fu_u, Fu_uv]
Ju_lambdified = jit(lambdify(Ju_args, Ju, modules='jax'))
Jv_lambdified = jit(lambdify(Jv_args, Jv, modules='jax'))
curl_B_u_lambdified = jit(lambdify(curl_B_u_args, curl_B_u, modules='jax'))
curl_B_v_lambdified = jit(lambdify(curl_B_v_args, curl_B_v, modules='jax'))
print(Jv.free_symbols)


print(Ju.free_symbols)
print(Jv.free_symbols)
print(curl_B_u.free_symbols)
print(curl_B_v.free_symbols)

print('Ju')
print(Ju)
print('Jv')
print(Jv)
print('curl_B_u')
print(curl_B_u)
print('curl_B_v')
print(curl_B_v)

dB_du = B.dot(z_hat).diff(u)
dB_dv = B.dot(z_hat).diff(v)
dB_du = symbolify(dB_du, Fu(u,v), Fu_syms)
dB_du = symbolify(dB_du, Fv(u,v), Fv_syms)
dB_dv = symbolify(dB_dv, Fu(u,v), Fu_syms)
dB_dv = symbolify(dB_dv, Fv(u,v), Fv_syms)
print(dB_du.free_symbols)
print(dB_dv.free_symbols)
dB_du_args = [a, u, Fu_uv, v, Fu_u, Fv_u, Fv, Fv_uu, Fu_v, Fu]
dB_dv_args = [a, u, v, Fv_u, Fv, Fu_vv, Fv_v, Fv_uv, Fu_v, Fu]
dB_du_lambdified = jit(lambdify(dB_du_args, dB_du, modules='jax'))
dB_dv_lambdified = jit(lambdify(dB_dv_args, dB_dv, modules='jax'))
print('dB_du')
print(dB_du.simplify())
"""