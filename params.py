import numpy as np
import sys

num_Ks = 1
num_As = 40

NL=1 # left winding number
NR=1 # right winding number
NU=95 # number of u points (resolution)
NV=95 # number of v points (resolution)
nr = 2000 # number of r points (single vortex resolution)

tol = 1e-5
max_iter = 30

def K_func(K_idx, A_idx):
    Ks = np.logspace(0, 0, num=num_Ks, base=2)
    Ks = [1/np.sqrt(2)]
    Ks = [5]
    return Ks[K_idx]

def A_func(K_idx, A_idx):
    K = K_func(K_idx, A_idx)
    As = np.logspace(0.5, 2.5, num=num_As, base = K) / 4
    As = np.linspace(K/4,2*K, num_As)
    As = np.linspace(K**2/2, 10*K**2, num_As)
    return As[A_idx]

if __name__ == '__main__':
    if sys.argv[1] == 'K' or sys.argv[1] == 'k':
        print(num_Ks)
    elif sys.argv[1] == 'A' or sys.argv[1] == 'a':
        print(num_As)
