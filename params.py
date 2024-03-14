import numpy as np
import sys

num_Ks = 6
num_As = 20

tol = 1e-6
max_iter = 30

def K_func(K_idx, A_idx):
    Ks = np.logspace(1, 6, num=num_Ks, base=2)
    return Ks[K_idx]

def A_func(K_idx, A_idx):
    K = K_func(K_idx, A_idx)
    As = np.logspace(1, 2, num=num_As, base = K) / 4
    return As[A_idx]

if __name__ == '__main__':
    if sys.argv[1] == 'K' or sys.argv[1] == 'k':
        print(num_Ks)
    elif sys.argv[1] == 'A' or sys.argv[1] == 'a':
        print(num_As)
