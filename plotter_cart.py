from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np
from coords import v_of_vp_lambdified
from scipy.interpolate import LinearNDInterpolator

file = 'data/' + sys.argv[1] + '.npy'
save_file_E = 'data/' + sys.argv[1]+ '_E_cart' + '.png'
save_file_B = 'data/' + sys.argv[1]+ '_B_cart' + '.png'
save_file_H = 'data/' + sys.argv[1]+ '_H_cart' + '.png'

A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, total_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = load(file)
J = -4/K**4
print(A, K, N, NU, NV, ier)
NUNV = NU*NV

us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

energy_densities = [electric_energy_density, magnetic_energy_density, hydraulic_energy_density]
save_files = [save_file_E, save_file_B, save_file_H]
titles = ['Electric energy density' , 'Magnetic energy density', 'Hydraulic energy density']
for idx, f in enumerate(energy_densities):
    f = energy_densities[idx]
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
            zs.append(f[i,j])
            h = A / (np.cosh(-v)-np.cos(u))
            x = h*np.sinh(-v)
            y = h*np.sin(u)
            xs.append(x)
            ys.append(y)
            zs.append(f[i,j])
            

    NX = 1000
    NY = 1000
    min_x, max_x = -10*K, 10*K
    min_y, max_y = -10*K, 10*K

    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            xs.append(x)
            ys.append(y)
            zs.append(0)

    X = np.linspace(min_x, max_x, NX)
    Y = np.linspace(min_y, max_y, NY)
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(list(zip(xs, ys)), zs)
    Z = interp(X,Y)
    plt.pcolormesh(X,Y,Z, shading = 'auto')
    plt.colorbar()
    plt.axis('equal')
    plt.title(titles[idx])
    plt.savefig(save_files[idx])
    plt.close()
