import numpy as np
from matplotlib import pyplot as plt

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
        A, K, N, E, B, H, T = np.load(file)
        indices.append(i)
        separations.append(2*A)
        electric.append(E)
        magnetic.append(B)
        hydraulic.append(H)
        total.append(T)
    except:
        continue

#print(electric)
#fig, ax = plt.subplots()
#ax.scatter(indices, electric)
#ax.set_yscale('log')
#plt.show()
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
