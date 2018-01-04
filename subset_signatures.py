'''Utils for subseting the signatures and create visualizations
'''
import os, sys, json
import pandas as pd
import numpy as np
import h5py

from sklearn.manifold import TSNE

# from orm import UserSubset

f = h5py.File('data/download/Adjacency_matrix_LM_space_42809x42809.gctx', 'r')
mat = f['0/DATA/0/matrix']

# Load metadata from h5
adj_mat_meta_df = pd.DataFrame({
    'pert_id': f['0/META/COL/pert_id'][()],
    'cell_id': f['0/META/COL/cell_id'][()],
    'time': f['0/META/COL/pert_time'][()],
    'sig_id': f['0/META/COL/id'][()]
}).set_index('sig_id')
adj_mat_meta_df['time'] = adj_mat_meta_df['time'].astype(np.int)

def get_subset_sig_ids(user_subset):
	data = user_subset.data
	# Subset using the user_subset data
	mask = adj_mat_meta_df['pert_id'].isin(data['pert_ids']) &\
		adj_mat_meta_df['cell_id'].isin(data['cells']) &\
		adj_mat_meta_df['time'].isin(data['times'])
	meta_df_sub = adj_mat_meta_df.loc[mask]
	return meta_df_sub.index.tolist()

def subset_adj_mat(user_subset):
	sig_ids_sub = get_subset_sig_ids(user_subset)
	mask = np.in1d(adj_mat_meta_df.index, sig_ids_sub)
	# use the mask to subset the mat
	mat_sub = mat[mask, :][:, mask]
	# Convert to dist_mat
	mat_sub = 1. - mat_sub
	return mat_sub, adj_mat_meta_df.loc[mask].index.tolist()

def do_tsne(mat_sub):
	tsne = TSNE(metric='precomputed')
	return tsne.fit_transform(mat_sub)

def subset_mat_and_do_tsne(user_subset):
	mat_sub, sig_ids_sub = subset_adj_mat(user_subset)
	coords = do_tsne(mat_sub)
	return pd.DataFrame(coords, columns=['x','y'], index=sig_ids_sub)

