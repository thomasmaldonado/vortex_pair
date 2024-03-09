### DATA MANAGEMENT ###

import numpy as np

def save(file, K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED):
    NUNV = NU*NV
    arr = np.zeros(10 + NU + NV + 11 * NUNV)
    arr[0:10] = K, A, NL, NR, NU, NV, EE, ME, HE, TE
    arr[10:(10+NU)] = us
    arr[(10+NU):(10+NU+NV)] = vs
    arr[(10+NU+NV):] = np.concatenate([x.flatten() for x in [V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED]])
    np.save(file, arr)

def load(file):
    arr = np.load(file)
    K, A, NL, NR, NU, NV, EE, ME, HE, TE = arr[0:10]
    NL, NR, NU, NV = [int(x) for x in [NL, NR, NU, NV]]
    us = arr[10:(10+NU)]
    vs = arr[(10+NU):(10+NU+NV)]
    V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED = [np.reshape(x, (NU, NV)) for x in np.split(arr[(10+NU+NV):], 11)]
    return K, A, NL, NR, NU, NV, EE, ME, HE, TE, us, vs, V, Fu, Fv, C, J0, Ju, Jv, EED, MED, HED, TED
