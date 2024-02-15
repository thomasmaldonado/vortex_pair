from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np

file = 'data/' + sys.argv[1] + '.npy'
save_file_V = 'data/' + sys.argv[1]+ '_V' + '.png'
save_file_Fu = 'data/' + sys.argv[1]+ '_Fu' + '.png'
save_file_Fv = 'data/' + sys.argv[1]+ '_Fv' + '.png'
save_file_C = 'data/' + sys.argv[1]+ '_C' + '.png'

A, K, N, NU, NV, ier, solution = load(file)
print(A, K, N, NU, NV, ier)
NUNV = NU*NV
V = np.reshape(solution[0:NUNV],(NU, NV))
Fu = np.reshape(solution[NUNV:2*NUNV],(NU, NV))
Fv = np.reshape(solution[2*NUNV:3*NUNV],(NU, NV))
C = np.reshape(solution[3*NUNV:],(NU, NV))

plt.imshow(V, cmap='hot', interpolation='nearest')
plt.savefig(save_file_V)
plt.imshow(Fu, cmap='hot', interpolation='nearest')
plt.savefig(save_file_C)
plt.imshow(Fv, cmap='hot', interpolation='nearest')
plt.savefig(save_file_Fu)
plt.imshow(C, cmap='hot', interpolation='nearest')
plt.savefig(save_file_Fv)

