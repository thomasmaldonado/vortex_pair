from files import load
import sys
import numpy as np
from derivatives import d
from analytics import Eu_lambdified, Ev_lambdified, B_lambdified, Eu_h_lambdified, Ev_h_lambdified, B_h_lambdified
from coords import v_of_vp_lambdified, dv_dvp_lambdified, d2v_dvp2_lambdified, dvp_dv_lambdified, d2vp_dv2_lambdified
from numba import njit
from matplotlib import pyplot as plt
file = 'data/' + sys.argv[1] + '.npy'
A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, total_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = load(file)

save_file_E = 'data/' + sys.argv[1] + '_E.png'
save_file_B = 'data/' + sys.argv[1] + '_B.png'
save_file_H = 'data/' + sys.argv[1] + '_H.png'

plt.imshow(electric_energy_density, cmap='hot', interpolation='nearest')
plt.savefig(save_file_E)
plt.imshow(magnetic_energy_density, cmap='hot', interpolation='nearest')
plt.savefig(save_file_B)
plt.imshow(hydraulic_energy_density, cmap='hot', interpolation='nearest')
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