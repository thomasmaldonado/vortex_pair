import numpy as np
from matplotlib import pyplot as plt
from files import load
from params import num_Ks, num_As
from matplotlib import cm

colors = plt.cm.plasma(np.linspace(0,1,num_Ks))


fig, axes = plt.subplots()

for k_idx in range(num_Ks):
    print(k_idx)
    separations = []
    EEs, MEs, HEs, TEs = [], [], [], []
    for a_idx in range(num_As):
        file = 'data/' + str(k_idx) + '/' + str(a_idx) + '.npy'
        try:
            K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED = load(file)
            separations.append(2*A/K)
            EEs.append(EE)
            MEs.append(ME)
            HEs.append(HE)
            TEs.append(TE)
        except:
            pass
    try:
        energies = [EEs, MEs, HEs, TEs]
        TEs = np.array(TEs)
        min_TE, max_TE = np.min(TEs), np.max(TEs)
        TEs = (TEs - min_TE) / (max_TE-min_TE)
        axes.plot(separations, TEs, c = colors[k_idx])
    except:
        pass
plt.savefig('energies.png')
"""
labels = ['EE', 'ME', 'HE', 'TE']
for i in range(4):
    axes[i].set_ylabel(labels[i])
    axes[i].set_xlabel('Separation')
plt.savefig('energies.png')
"""
