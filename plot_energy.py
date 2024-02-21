import numpy as np
from matplotlib import pyplot as plt
from files import load

numpoints = 100
electric = []
magnetic = []
hydraulic = []
total = []
indices = []
separations = []
for i in range(numpoints):
    file = 'data/' +  str(i) + '_energies.npy'
    try:
        A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, total_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = load(file)
        indices.append(i)
        separations.append(2*A)
        electric.append(electric_energy)
        magnetic.append(magnetic_energy)
        hydraulic.append(hydraulic_energy)
        total.append(total_energy)
    except:
        continue

plt.scatter(separations, electric)
plt.ylabel('Electric energy')
plt.xlabel('Separation')
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('electric_energy')
plt.close()
plt.scatter(separations, magnetic)
plt.ylabel('Magnetic energy')
plt.xlabel('Separation')
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('magnetic_energy')
plt.close()
plt.scatter(separations, hydraulic)
plt.ylabel('Hydraulic energy')
plt.xlabel('Separation')
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('hydraulic_energy')
plt.close()
plt.scatter(separations, total)
plt.ylabel('Total energy')
plt.xlabel('Separation')
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('total_energy')
