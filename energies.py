from files import load
import sys
import numpy as np
from derivatives import d
from analytics import Eu_lambdified, Ev_lambdified, B_lambdified, Eu_h_lambdified, Ev_h_lambdified, B_h_lambdified
from coords import v_of_vp_lambdified, dv_dvp_lambdified, d2v_dvp2_lambdified, dvp_dv_lambdified, d2vp_dv2_lambdified
from numba import njit
from matplotlib import pyplot as plt
file = 'data/' + sys.argv[1] + '.npy'
A, K, N, NU, NV, ier, solution = load(file)
J = -4/K**4
NUNV = NU*NV

V = np.reshape(solution[0:NUNV], (NU,NV))
Fu = np.reshape(solution[NUNV:2*NUNV], (NU,NV))
Fv = np.reshape(solution[2*NUNV:3*NUNV], (NU,NV))
C = np.reshape(solution[3*NUNV:],(NU,NV))


us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

du = us[1]-us[0]
dvp = vps[1]-vps[0]

dvp_dv = np.zeros((NU, NV))
d2vp_dv2 = np.zeros((NU, NV))

# define coordinate transformation
NUNV = NU*NV
for j, v in enumerate(vs):
    args = (v, A, J)
    dvp_dv[:,j] = dvp_dv_lambdified(*args)
    d2vp_dv2[:,j] = d2vp_dv2_lambdified(*args)

@njit
def d_du(f, n):
    boundary_left = np.ascontiguousarray(f[-1,:])
    boundary_right = np.ascontiguousarray(f[0,:])
    return d(f, du, n, boundary_left, boundary_right, axis = 0)

@njit
def d_dvp(f, n, boundary_left, boundary_right):
    return d(f, dvp, n, boundary_left, boundary_right, axis = 1)

@njit
def d_dv(f, n, boundary_left, boundary_right):
    if n == 0:
        return f
    df_dvp = d_dvp(f, 1, boundary_left, boundary_right)
    if n == 1:
        return df_dvp * dvp_dv
    df2_dx2 = d_dvp(f, 2, boundary_left, boundary_right)
    return df2_dx2 * dvp_dv**2 + df_dvp * d2vp_dv2

@njit
def dV_du(V, n):
    return d_du(V, n)

@njit
def dFu_du(Fu, n):
    return d_du(Fu, n)

@njit
def dFv_du(Fv, n):
    return d_du(Fv, n)

@njit
def dF_du(Fu, Fv, n):
    return d_du(Fu, n), d_du(Fv, n)

@njit
def dC_du(C, n):
    return d_du(C, n)

@njit
def BP2cart(Fu, Fv, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    return Fu * M_00 + Fv * M_01, Fu * M_10 + Fv * M_11

@njit
def cart2BP(Fx, Fy, u, v):
    M_00 = -np.sin(u)*np.sinh(v) / (np.cosh(v)-np.cos(u))
    M_01 = (1 - np.cos(u)*np.cosh(v)) / (np.cosh(v)-np.cos(u))
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

@njit
def cart2BPinfinity(Fx, Fy, u):
    M_00 = -np.sin(u)
    M_01 = -np.cos(u)
    M_10 = - M_01
    M_11 = M_00
    MT_00, MT_01, MT_10, MT_11 = M_00, M_10, M_01, M_11
    return Fx * MT_00 + Fy * MT_01, Fx * M_10 + Fy * M_11

@njit
def dV_dv(V, n):
    boundary_left = np.ascontiguousarray(V[:,0])
    boundary_left[0] = -1
    boundary_right = np.full(NU, np.mean(V[:,-1]))
    return d_dv(V, n, boundary_left, boundary_right)

@njit(parallel = True)
def dF_dv(Fu, Fv, N, n = 1):
    boundary_left_u = np.ascontiguousarray(Fu[:,0])
    boundary_left_u[0] = 0 #N / A
    boundary_left_v = np.zeros(NU)
    xs, ys = BP2cart(Fu[:,-1],  Fv[:,-1], us, vs[-1])
    avg_x = np.mean(xs)
    avg_y = np.mean(ys)
    boundary_right_u, boundary_right_v =  cart2BPinfinity(avg_x, avg_y, us)
    Fu_result = d_dv(Fu, n, boundary_left_u, boundary_right_u)
    Fv_result = d_dv(Fv, n, boundary_left_v, boundary_right_v)
    return Fu_result, Fv_result

@njit
def dC_dv(C, n = 1):
    boundary_left = np.ascontiguousarray(C[:,0])
    boundary_left[0] = np.sqrt(-J)
    boundary_right = np.zeros(np.shape(C)[0])
    return d_dv(C, n, boundary_left, boundary_right)

V_u = dV_du(V, n = 1)
V_v = dV_dv(V, n = 1)
V_uu = dV_du(V, n = 2)
V_uv = dV_du(V_v, n = 1)
V_vv = dV_dv(V, n = 2)
Fu_u, Fv_u = dF_du(Fu, Fv, n = 1)
Fu_v, Fv_v = dF_dv(Fu, Fv, N, n = 1)
Fu_uu, Fv_uu = dF_du(Fu, Fv, n = 2)
Fu_uv, Fv_uv = dF_du(Fu_v, Fv_v, n = 1)
Fu_vv, Fv_vv = dF_dv(Fu, Fv, N, n = 2)
C_u = dC_du(C, n = 1)
C_v = dC_dv(C, n = 1)
C_uu = dC_du(C, n = 2)
C_uv = dC_du(C_v, n = 1)
C_vv = dC_dv(C, n = 2)

Eu, Ev, B = np.zeros((NU,NV)), np.zeros((NU,NV)), np.zeros((NU,NV))


for i in range(len(us)):
    u = us[i]
    for j in range(len(vs)):
        v = vs[j]
        E_args = [V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j], u, v, A, N]
        Eu[i,j], Ev[i,j] = Eu_lambdified(*E_args), Ev_lambdified(*E_args)
        B_args = [Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j]]
        B_args.extend([Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j]])
        B_args.extend([u, v, A, N])
        B[i,j] = B_lambdified(*B_args)

electric_energy_density = (Eu**2 + Ev**2)/2
magnetic_energy_density = (B**2)/2
hydraulic_energy_density = C**2 * V**2
bulk_energy_density = - J

save_file_E = 'data/' + sys.argv[1] + '_E.png'
save_file_B = 'data/' + sys.argv[1] + '_B.png'
save_file_H = 'data/' + sys.argv[1] + '_H.png'

plt.imshow(electric_energy_density, cmap='hot', interpolation='nearest')
plt.savefig(save_file_E)
plt.imshow(magnetic_energy_density, cmap='hot', interpolation='nearest')
plt.savefig(save_file_B)
plt.imshow(hydraulic_energy_density - bulk_energy_density, cmap='hot', interpolation='nearest')
plt.savefig(save_file_H)

dA = np.zeros((NU, NV))
for i in range(NU):
    u = us[i]
    for j in range(NV):
        v = vs[j]
        vp = vps[j]
        h = A / (np.cosh(v)-np.cos(u))
        args = (vp, A, J)
        dv = dv_dvp_lambdified(*args) * dvp
        dA[i,j] = du*dv * h**2
"""
dA_no_h = np.zeros((NU,NV))
Eu_h = np.zeros((NU,NV))
Ev_h = np.zeros((NU,NV))
B_h = np.zeros((NU,NV))
for i in range(NU):
    u = us[i]
    for j in range(NV):
        v = vs[j]
        vp = vps[j]
        #h = A / (np.cosh(v)-np.cos(u))
        args = (vp, A, J)
        dv = dv_dvp_lambdified(*args) * dvp
        dA_no_h[i,j] = du*dv

        E_args = [V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j], u, v, A, N]
        Eu_h[i,j], Ev_h[i,j] = Eu_h_lambdified(*E_args), Ev_h_lambdified(*E_args)
        B_args = [Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j]]
        B_args.extend([Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j]])
        B_args.extend([u, v, A, N])
        B_h[i,j] = B_h_lambdified(*B_args)
"""
#plt.imshow((Eu_h**2 + Ev_h ** 2) * dA_no_h / 2)
#plt.savefig('data/' + sys.argv[1] + '_electric.png')
#plt.imshow(dA_no_h*(V_u**2 + V_v**2)/2)
#plt.show()
#electric_energy = np.sum((Eu_h**2 + Ev_h**2)* dA_no_h / 2)
#magnetic_energy = np.sum(B_h**2 * dA_no_h / 2)
electric_energy = np.sum(electric_energy_density * dA)
magnetic_energy = np.sum(magnetic_energy_density * dA)
hydraulic_energy = np.sum((hydraulic_energy_density - bulk_energy_density)*dA)
total_energy = electric_energy + magnetic_energy + hydraulic_energy
arr = np.array([A, K, N, electric_energy, magnetic_energy, hydraulic_energy, total_energy])
np.save('data/' + sys.argv[1] + '_energies.npy', arr)