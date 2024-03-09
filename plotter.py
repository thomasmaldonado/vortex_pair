### SOLUTION VISUALIZATION ###

from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from jcoords import BP2cart

# load data and construct coordinate system
file = 'data/' + sys.argv[1] + '.npy'
K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED = load(file)
N = NR
samesign = (NL / NR == 1)
J = -4/K**4

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
    min_x, max_x = -5*K, 5*K
    min_y, max_y = -5*K, 5*K
    bulk_val = 0
    if label == 'V':
        bulk_val = -1
    if label == 'C':
        bulk_val = np.sqrt(-J)
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


def plot_vec(Fu, Fv):
    xs = []
    ys = []
    Fxs = []
    Fys = []
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            h = A / (np.cosh(v)-np.cos(u))
            x = h*np.sinh(v)
            y = h*np.sin(u)
            xs.append(x)
            ys.append(y)
            Fx, Fy = BP2cart(Fu[i,j], Fv[i,j], u, v)
            Fxs.append(Fx)
            Fys.append(Fy)
            
            h = A / (np.cosh(-v)-np.cos(u))
            x = h*np.sinh(-v)
            y = h*np.sin(u)
            xs.append(x)
            ys.append(y)
            if samesign:
                Fx, Fy = BP2cart(-Fu[i,j], Fv[i,j], u, -v)
            else:
                Fx, Fy = BP2cart(Fu[i,j], -Fv[i,j], u, -v)
            Fxs.append(Fx)
            Fys.append(Fy)
    NX = 50
    NY = 25
    min_x, max_x = -2*K, 2*K
    min_y, max_y = -K, K
    bulk_val = 0
    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            xs.append(x)
            ys.append(y)
            Fxs.append(bulk_val)
            Fys.append(bulk_val)
    X = np.linspace(min_x, max_x, NX)
    Y = np.linspace(min_y, max_y, NY)
    X, Y = np.meshgrid(X, Y)
    interp_Fx = LinearNDInterpolator(list(zip(xs, ys)), Fxs)
    interp_Fy = LinearNDInterpolator(list(zip(xs, ys)), Fys)
    FX = interp_Fx(X,Y)
    FY = interp_Fy(X,Y)
    norms = np.sqrt(FX**2 + FY**2)
    max_norm = max(norms.flatten())

    dx = max_x/NX
    dy = max_y/NY

    FX = FX * np.sqrt(dx**2 + dy**2) / max_norm
    FY = FY * np.sqrt(dx**2 + dy**2) / max_norm
    fig1, ax1 = plt.subplots()
    Q = ax1.quiver(X, Y, FX, FY, units='width', angles='xy', scale_units='xy', scale=2/3, pivot = 'middle')
    plt.xlim(-max_x, max_x)
    plt.ylim(-max_y, max_y)
    plt.savefig('pair_test.pdf')
    plt.show()

plot_vec(Ju, Jv)

for i in range(len(funcs)):
    plot(funcs[i], labels[i])
    plot_cart(funcs[i], labels[i])

