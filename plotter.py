from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np

file = 'data/' + sys.argv[1] + '.npy'
save_file_V = 'data/' + sys.argv[1]+ '_V' + '.png'
save_file_Fu = 'data/' + sys.argv[1]+ '_Fu' + '.png'
save_file_Fv = 'data/' + sys.argv[1]+ '_Fv' + '.png'
save_file_C = 'data/' + sys.argv[1]+ '_C' + '.png'

A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, total_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = load(file)
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.savefig(save_file_V)
plt.imshow(Fu, cmap='hot', interpolation='nearest')
plt.savefig(save_file_Fu)
plt.imshow(Fv, cmap='hot', interpolation='nearest')
plt.savefig(save_file_Fv)
plt.imshow(C, cmap='hot', interpolation='nearest')
plt.savefig(save_file_C)