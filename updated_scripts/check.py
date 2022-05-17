from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import ROOT
import pickle
suma = np.load('./summaries_ep46.npz')
# print(suma['lr'])
# print(suma['epoch'])
print(suma['valid_loss'],suma['train_loss'])
suma = np.load('./summaries_31.npz')
print('......')
print(suma['valid_loss'],suma['train_loss'])
print('....')
suma = np.load('./summaries.npz')
print(suma['valid_loss'],suma['train_loss'])
