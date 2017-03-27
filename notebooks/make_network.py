import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from sqlalchemy import create_engine

## Load data from h5 file
import h5py
f = h5py.File('../data/CD_matrix_105939xlm978.h5', 'r')
mat = f['matrix']
sig_ids = f['meta']['sig_ids']
print mat.shape, type(mat), len(sig_ids)
print mat.dtype

## Load metadata
# Create metadata.tsv
engine = create_engine("mysql://euclid:elements@amp.pharm.mssm.edu:3306/euclid4?charset=utf8")
meta_df = pd.read_sql('optional_metadata', engine, columns=['gene_signature_fk', 'name', 'value'])
print meta_df.shape

meta_df = pd.pivot_table(meta_df, values='value', 
                         columns=['name'], index=['gene_signature_fk'], 
                         aggfunc=lambda x:x)

meta_df = meta_df.reset_index().set_index('sig_id').drop(['gene_signature_fk', 'disease', 'gene'], axis=1)
# # print meta_df.head()
# meta_df.to_csv('../data/metadata-full.tsv', sep='\t')
print meta_df.shape

# Keep only significant signatures 
meta_df['pvalue'] = meta_df['pvalue'].astype(np.float32)
# print meta_df.dtypes
# meta_df = meta_df.loc[meta_df['pvalue'] < 0.05]
meta_df = meta_df.loc[meta_df['pvalue'] < 0.9]
print meta_df.shape


# find shared sig_ids
sig_ids_shared = list(set(list(sig_ids)) & set(meta_df.index))
print '# shared sig_ids: %d' % len(sig_ids_shared)
mask_shared = np.in1d(list(sig_ids), sig_ids_shared)
sig_ids_shared = np.array(sig_ids)[mask_shared]
print 'mask_shared.shape:', mask_shared.shape

# subset and reorder the meta_df
meta_df = meta_df.ix[sig_ids_shared]
print meta_df.shape
# meta_df.head()

# Compute the pairwise cosine distance and convert to adjacency matrix
adj_mat = 1 - pairwise_distances(mat[mask_shared,:], metric='cosine',
                                n_jobs=2)
print adj_mat.shape
del mat

## remove 1's on the diagnal
adj_mat = adj_mat - np.eye(adj_mat.shape[0])
## convert negative values in adj_mat to 0's
adj_mat[adj_mat<0] = 0

# Create a undirected graph from the adj_mat by first setting values smaller than cutoff to 0's 
# to control number of edges.
percentile_cutoff = 99.9
cosine_similarity_cutoff = np.percentile(adj_mat.ravel(), percentile_cutoff)

print percentile_cutoff, cosine_similarity_cutoff

adj_mat[adj_mat<cosine_similarity_cutoff] = 0

G = nx.from_numpy_matrix(adj_mat)
del adj_mat

print G.number_of_nodes(), G.number_of_edges()

## Create a new graph only keeping the large connected components
G_new = nx.Graph()
for cc in nx.connected_component_subgraphs(G):
    if cc.number_of_nodes() > 10:
        G_new = nx.compose(G_new, cc)

del G
print G_new.number_of_nodes(), G_new.number_of_edges()

# Relabel nodes with sig_ids
d_id_sig_id = dict(zip(range(len(sig_ids_shared)), sig_ids_shared))
d_id_sig_id = {nid: sig_id for nid, sig_id in d_id_sig_id.items() if G_new.has_node(nid)}
nx.relabel_nodes(G_new, d_id_sig_id, copy=False)
print G_new.nodes()[:5]

nx.write_gml(G_new, 'Signature_Graph_%dnodes_%s.gml' % (G_new.number_of_nodes(), percentile_cutoff))


