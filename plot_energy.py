import numpy as np
from matplotlib import pyplot as plt
from files import load

numpoints = 40
separations = []
EEs, MEs, HEs, TEs = [], [], [], []
for i in range(numpoints):
    file = 'data/' +  str(i) + '.npy'
    try:
        K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED = load(file)
        N = NR
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