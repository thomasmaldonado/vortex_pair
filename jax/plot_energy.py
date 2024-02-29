import numpy as np
from matplotlib import pyplot as plt
from files import load

numpoints = 50
separations = []
EEs, MEs, HEs, TEs = [], [], [], []
for i in range(numpoints):
    file = 'data/' +  str(i) + '.npy'
    try:
        A, K, N, NU, NV, ier, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED = load(file)
        TE = EE + ME + HE
        separations.append(2*A)
        EEs.append(EE)
        MEs.append(ME)
        HEs.append(HE)
        TEs.append(TE)
    except:
        continue

energies = [EEs, MEs, HEs, TEs]
labels = ['EE', 'ME', 'HE', 'TE']

for i in range(4):
    plt.scatter(separations, energies[i])
    plt.ylabel(labels[i])
    plt.xlabel('Separation')
    plt.title(r'$\kappa =$' + f'{K}')
    plt.tight_layout()
    plt.savefig('data/' + labels[i] + '.png')
    plt.close()