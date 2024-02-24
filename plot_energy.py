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
hydro_inf = []
for i in range(numpoints):
    file = 'data/' +  str(i) + '.npy'
    try:
        A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, total_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = load(file)
        indices.append(i)
        separations.append(2*A)
        electric.append(electric_energy)
        magnetic.append(magnetic_energy)
        hydraulic.append(hydraulic_energy)
        total.append(total_energy)
        hydro_inf.append(hydraulic_energy_density[0,0])
    except:
        continue

electromagnetic = np.array(electric) + np.array(magnetic)

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
plt.close()
plt.scatter(separations, electromagnetic, label = 'Electromagnetic')
plt.scatter(separations, hydraulic, label = 'Hydraulic')
plt.scatter(separations, total, label = 'Total')
plt.ylabel('Energy')
plt.xlabel('Separation')
plt.legend()
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('electromagnetic_energy')
plt.close()

hydro_d = []
electro_d = []
total_d = []
for i in range(1,len(hydraulic)):
    hydro_d.append((hydraulic[i] - hydraulic[i-1])/(separations[i] - separations[i-1]))
    electro_d.append((electromagnetic[i] - electromagnetic[i-1])/(separations[i] - separations[i-1]))
    total_d.append(hydro_d[-1] + electro_d[-1])
plt.scatter(separations[1:], electro_d, label = 'Electromagnetic')
plt.scatter(separations[1:], hydro_d, label = 'Hydraulic')
plt.scatter(separations[1:], total_d, label = 'Total')
plt.axhline(0)
plt.ylim(-1e-9,1e-9)
plt.ylabel('d Energy / d Separation')
plt.xlabel('Separation')
plt.legend()
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('derivatives')
plt.close()





plt.scatter(separations, hydro_inf, label = 'Hydro inf')
plt.yscale('log')
plt.ylabel('Energy')
plt.xlabel('Separation')
plt.legend()
plt.title(r'$\kappa =$' + f'{K}')
plt.savefig('hydro_inf')
plt.close()
