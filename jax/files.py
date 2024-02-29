### DATA MANAGEMENT ###

import numpy as np

def save(file, A, K, N, NU, NV, ier, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED):
    NUNV = NU*NV
    arr = np.zeros(9 + 7 * NUNV)
    arr[0:9] = A, K, N, NU, NV, ier, EE, ME, HE
    arr[9:] = np.concatenate((V.flatten(), Fu.flatten(), Fv.flatten(), C.flatten(), EED.flatten(), MED.flatten(), HED.flatten()))
    np.save(file, arr)

def load(file):
    arr = np.load(file)
    A, K, N, NU, NV, ier, EE, ME, HE = arr[0:9]
    N, NU, NV, ier = int(N), int(NU), int(NV), int(ier)
    V, Fu, Fv, C, EED, MED, HED = np.split(arr[9:], 7)
    V = np.reshape(V, (NU, NV))
    Fu = np.reshape(Fu, (NU, NV))
    Fv = np.reshape(Fv, (NU, NV))
    C = np.reshape(C, (NU, NV))
    EED = np.reshape(EED, (NU, NV))
    MED = np.reshape(MED, (NU, NV))
    HED = np.reshape(HED, (NU, NV))
    return A, K, N, NU, NV, ier, EE, ME, HE, V, Fu, Fv, C, EED, MED, HED
