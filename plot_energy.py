import numpy as np
from matplotlib import pyplot as plt
from files import load
from params import num_Ks, num_As, K_func
from matplotlib import cm


colors = plt.cm.plasma(np.linspace(0,1,num_Ks))

def normalize(Es):
    Es = np.array(Es)
    min_E, max_E = np.min(Es), np.max(Es)
    Es = (Es - min_E) / (max_E-min_E)
    return Es

fig, axes = plt.subplots(4, sharex=True)
labels = ['EE', 'ME', 'HE', 'TE']
for idx, ax in enumerate(axes):
    ax.set_ylabel(labels[idx])
    ax.set_yticks([])

for k_idx in range(num_Ks-1,1,-1):
    separations = []
    EEs, MEs, HEs, TEs = [], [], [], []
    for a_idx in range(num_As):
        file = 'data/' + str(k_idx) + '/' + str(a_idx) + '.npy'
        try:
            K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED = load(file)
            separations.append(4*A/K)
            EEs.append(EE)
            MEs.append(ME)
            HEs.append(HE)
            TEs.append(TE)
        except:
            pass
    def normalize(Es):
        Es = np.array(Es)
        min_E, max_E = np.min(Es), np.max(Es)
        Es = (Es - min_E) / (max_E-min_E)
        return Es
    energies = [EEs, MEs, HEs, TEs]
    for idx, Es in enumerate(energies):
        try:
            axes[idx].plot(separations, normalize(Es), c = colors[k_idx])
        except:
            pass

plt.subplots_adjust(hspace=0)
axes[-1].set_xlabel('Separation / ' + r'$\xi$')


import matplotlib as mpl
cax = plt.axes([.9, 0.1, 0.025, 0.8])
cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=K_func(0,0), vmax=K_func(num_Ks-1,0))
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', label=r'$\kappa$')
plt.savefig('energies.png')