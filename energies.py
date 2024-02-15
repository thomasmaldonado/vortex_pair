from matplotlib import pyplot as plt
from files import load
import sys
import numpy as np

filenames = []

for i in range(10):
    filenames.append('data/' + str(i) + '.npy')
    A, K, N, NU, NV, ier, solution = load(file)



