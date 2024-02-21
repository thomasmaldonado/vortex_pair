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