### SOLUTION VISUALIZATION ###

from matplotlib import pyplot as plt
from files import load
from coords import v_of_vp_lambdified
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# load data and construct coordinate system
file = 'data/' + sys.argv[1] + '.npy'
A, K, N, NU, NV, ier, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED = load(file)
J = -4/K**4
us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

# define list of functions + labels to be plotted
funcs = [V, Fu, Fv, C, EED, MED, HED]
labels = ['V', 'Fu', 'Fv', 'C', 'EED', 'MED', 'HED']

# plot in (u, vp) space
def plot(func, label):
    save_file = 'data/' + sys.argv[1] + '_' + label + '.png'
    plt.imshow(func, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(save_file)
    plt.close()

# plot in cartesian space
def plot_cart(func, label):
    save_file = 'data/' + sys.argv[1] + '_' + label + '_cart.png'
    xs = []
    ys = []
    zs = []
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            h = A / (np.cosh(v)-np.cos(u))
            x = h*np.sinh(v)
            y = h*np.sin(u)
            xs.append(x)
            ys.append(y)
            zs.append(func[i,j])
            h = A / (np.cosh(-v)-np.cos(u))
            x = h*np.sinh(-v)
            y = h*np.sin(u)
            xs.append(x)
            ys.append(y)
            zs.append(func[i,j])
    NX = 1000
    NY = 1000
    min_x, max_x = -10*K, 10*K
    min_y, max_y = -10*K, 10*K
    bulk_val = 0
    if label == 'V':
        bulk_val = -1
    elif label == 'C':
        bulk_val == np.sqrt(-J)
    print(label, bulk_val)
    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            xs.append(x)
            ys.append(y)
            zs.append(bulk_val)
    X = np.linspace(min_x, max_x, NX)
    Y = np.linspace(min_y, max_y, NY)
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(list(zip(xs, ys)), zs)
    Z = interp(X,Y)
    plt.pcolormesh(X,Y,Z, cmap = 'hot', shading = 'auto')
    plt.colorbar()
    plt.axis('equal')
    plt.title(label)
    plt.savefig(save_file)
    plt.close()

for i in range(len(funcs)):
    plot(funcs[i], labels[i])
    plot_cart(funcs[i], labels[i])