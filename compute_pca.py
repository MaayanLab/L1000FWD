'''
Compute PCA coordinates for CD signatures.
'''

import os
import h5py
import math

import numpy as np
import pandas as pd
from sklearn import decomposition
from sqlalchemy import create_engine

f = h5py.File('data/CD_matrix_45012xlm978.h5', 'r')
mat = f['matrix']
print mat.shape
sig_ids = f['meta']['sig_ids']

batch_size = 400
ipca = decomposition.IncrementalPCA(n_components=3, batch_size=None)

n_batchs = int(math.ceil(mat.shape[0] / float(batch_size)))

for i in range(n_batchs):
	start_idx = i * batch_size
	end_idx = (i+1) * batch_size

	ipca.partial_fit(mat[start_idx:end_idx])

mat_coords = np.zeros((mat.shape[0], 3))
for i in range(n_batchs):
	start_idx = i * batch_size
	end_idx = (i+1) * batch_size

	mat_coords[start_idx:end_idx] = ipca.transform(mat[start_idx:end_idx])

np.save('data/pca_coords', mat_coords)


