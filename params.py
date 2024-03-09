import numpy as np
import sys

num_Ks = 9
num_As = 20

def K_func(K_idx, A_idx):
    Ks = np.logspace(-2, 2, num=num_Ks, base=2)
    return Ks[K_idx]

def A_func(K_idx, A_idx):
    K = K_func(K_idx, A_idx)
    As = np.linspace(0, 4*K, num_As+1)[1:]
    return As[A_idx]

if __name__ == '__main__':
    if sys.argv[1] == 'K' or sys.argv[1] == 'k':
        print(num_Ks)
    elif sys.argv[1] == 'A' or sys.argv[1] == 'a':
        print(num_As)