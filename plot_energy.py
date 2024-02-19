import numpy as np
from matplotlib import pyplot as plt

numpoints = 100
electric = []
magnetic = []
hydraulic = []
total = []
indices = []
for i in range(numpoints):
    file = str(i) + '_energies.npy'
    try:
        E, B, H, T = np.load(file)
        indices.append(i)
        electric.append(E)
        magnetic.append(B)
        hydraulic.append(H)
        total.append(T)
    except:
        continue
    print(E)

#print(electric)
#fig, ax = plt.subplots()
#ax.scatter(indices, electric)
#ax.set_yscale('log')
#plt.show()
plt.scatter(indices, magnetic)
plt.show()
plt.scatter(indices, hydraulic)
plt.ylim(20, 40)
plt.show()
plt.scatter(indices, total)
plt.show()
