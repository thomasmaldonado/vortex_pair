from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np
from coords import v_of_vp_lambdified
from scipy.interpolate import LinearNDInterpolator

file = 'data/' + sys.argv[1] + '.npy'
save_file_V = 'data/' + sys.argv[1]+ '_V' + '.png'
save_file_Fu = 'data/' + sys.argv[1]+ '_Fu' + '.png'
save_file_Fv = 'data/' + sys.argv[1]+ '_Fv' + '.png'
save_file_C = 'data/' + sys.argv[1]+ '_C' + '.png'

A, K, N, NU, NV, ier, solution = load(file)
J = -4/K**4
print(A, K, N, NU, NV, ier)
NUNV = NU*NV
V = np.reshape(solution[0:NUNV],(NU, NV))
Fu = np.reshape(solution[NUNV:2*NUNV],(NU, NV))
Fv = np.reshape(solution[2*NUNV:3*NUNV],(NU, NV))
C = np.reshape(solution[3*NUNV:],(NU, NV))

us = np.linspace(0, 2*np.pi, NU + 1)[:-1]
vps = np.linspace(0, 1, NV+2)[1:-1]
args = (vps, A, J)
vs = v_of_vp_lambdified(*args)

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
        zs.append(C[i,j])
NX = 1000
NY = 1000
X = np.linspace(min(xs), max(xs), NX)
Y = np.linspace(min(ys), max(ys), NY)
X, Y = np.meshgrid(X, Y)
interp = LinearNDInterpolator(list(zip(xs, ys)), zs)
Z = interp(X,Y)
plt.pcolormesh(X,Y,Z, shading = 'auto')
plt.colorbar()
plt.axis('equal')
plt.show()
