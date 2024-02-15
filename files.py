import numpy as np

def load(file):
    arr = np.load(file)
    A, K, N, NU, NX, ier = arr[0:6]
    solution = arr[6:]
    return A, K, int(N), int(NU), int(NX), int(ier), solution

def save(file, A, K, N, NU, NX, ier, solution):
    arr = np.zeros(len(solution) + 6)
    arr[0:6] = A, K, N, NU, NX, ier
    arr[6:] = solution
    #np.save(file, arr, allow_pickle = False)
    np.save(file, arr)