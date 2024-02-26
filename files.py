### DATA MANAGEMENT ###

import numpy as np

def save(file, A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density):
    NUNV = NU*NV
    arr = np.zeros(9 + 7 * NUNV)
    arr[0:9] = A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy
    arr[9:] = np.concatenate((V.flatten(), Fu.flatten(), Fv.flatten(), C.flatten(), electric_energy_density.flatten(), magnetic_energy_density.flatten(), hydraulic_energy_density.flatten()))
    np.save(file, arr)

def load(file):
    arr = np.load(file)
    A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy = arr[0:9]
    N, NU, NV, ier = int(N), int(NU), int(NV), int(ier)
    V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density = np.split(arr[9:], 7)
    V = np.reshape(V, (NU, NV))
    Fu = np.reshape(Fu, (NU, NV))
    Fv = np.reshape(Fv, (NU, NV))
    C = np.reshape(C, (NU, NV))
    electric_energy_density = np.reshape(electric_energy_density, (NU, NV))
    magnetic_energy_density = np.reshape(magnetic_energy_density, (NU, NV))
    hydraulic_energy_density = np.reshape(hydraulic_energy_density, (NU, NV))
    return A, K, N, NU, NV, ier, electric_energy, magnetic_energy, hydraulic_energy, V, Fu, Fv, C, electric_energy_density, magnetic_energy_density, hydraulic_energy_density
