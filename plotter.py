from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np

file = sys.argv[1] + '.npy'

A, K, N, NU, NV, ier, solution = load(file)

print(A, K, N, NU, NV, ier)
NUNV = NU*NV
print(solution)
V = np.reshape(solution[0:NUNV],(NU, NV))
Fu = np.reshape(solution[NUNV:2*NUNV],(NU, NV))
Fv = np.reshape(solution[2*NUNV:3*NUNV],(NU, NV))
C = np.reshape(solution[3*NUNV:],(NU, NV))

plt.imshow(V, cmap='hot', interpolation='nearest')
plt.savefig(sys.argv[1] + '_V')
plt.imshow(Fu, cmap='hot', interpolation='nearest')
plt.savefig(sys.argv[1] + '_Fu')
plt.imshow(Fv, cmap='hot', interpolation='nearest')
plt.savefig(sys.argv[1] + '_Fv')
plt.imshow(C, cmap='hot', interpolation='nearest')
plt.savefig(sys.argv[1] + '_C')

