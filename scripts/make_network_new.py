KEY = 'CD_center_LM'
percentile_cutoff = 99.9
n_jobs = 6

import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from sqlalchemy import create_engine
from pymongo import MongoClient

## Load signature metadata mongodb
client = MongoClient('mongodb://146.203.54.131:27017/')
coll = client['L1000FWD']['sigs']
cur = coll.find({}, 
                {'_id':False, 
                 'sig_id':True, 
                 'SCS_centered_by_batch':True,
                 'pert_id':True,
                 'cell_id':True,
                })

sig_meta_df = pd.DataFrame([doc for doc in cur]).set_index('sig_id')
print sig_meta_df.shape
sig_meta_df.head()


def pick_top_sigs(x):
    x = x.sort_values(ascending=True)
    n_significants = (x < 0.05).sum()
    if n_significants == 0:
        return x.head(2)
    else:
        return x.head(n_significants)


# Get the top significant signatures for each pert_id
grouped_sorted = sig_meta_df.groupby('pert_id')['SCS_centered_by_batch'].apply(pick_top_sigs)
grouped_sorted.head()

# filter out pert_id with less than 0 signatures
pert_id_counts = sig_meta_df.reset_index().groupby('pert_id')['sig_id'].count()
print pert_id_counts[:10]
pert_ids_kept = pert_id_counts[pert_id_counts > 0].index.tolist()
print 'Number of pert_id to keep: %d' % len(pert_ids_kept)

grouped_sorted = grouped_sorted[pert_ids_kept].reset_index()
print grouped_sorted.shape
n_sigs = grouped_sorted.shape[0]
print 'Number of sig_id to keep: %d' % n_sigs


def retrieve_sig_mat(sig_ids, coll, key):
    '''Retrieve signatures matrix from MongoDB'''
    # Retrieve signature matrix
    sig_mat = np.zeros((len(sig_ids), 978))
    for i, sig_id in enumerate(sig_ids): 
        doc = coll.find_one({'sig_id': sig_id}, {'_id':False, key:True})
        sig_mat[i] = doc[key]
        if i % 5000 == 0:
            print i, len(sig_ids)
    return sig_mat

## Extract signature matrix
mat = retrieve_sig_mat(grouped_sorted['sig_id'], coll, KEY)
print mat.shape


# Compute the pairwise cosine distance and convert to adjacency matrix
adj_mat = 1 - pairwise_distances(mat, metric='cosine',
                                n_jobs=n_jobs)
print adj_mat.shape

del mat

## remove 1's on the diagnal
adj_mat = adj_mat - np.eye(adj_mat.shape[0])
## convert negative values in adj_mat to 0's
adj_mat[adj_mat<0] = 0


# Create a undirected graph from the adj_mat by first setting values smaller than cutoff to 0's 
# to control number of edges.

cosine_similarity_cutoff = np.percentile(adj_mat.ravel(), percentile_cutoff)

print percentile_cutoff, cosine_similarity_cutoff

adj_mat_ = adj_mat.copy()
adj_mat_[adj_mat_<cosine_similarity_cutoff] = 0

G = nx.from_numpy_matrix(adj_mat_)
del adj_mat_, adj_mat

print G.number_of_nodes(), G.number_of_edges()

## Create a new graph only keeping the large connected components
G_new = nx.Graph()
for cc in nx.connected_component_subgraphs(G):
    if cc.number_of_nodes() > 10:
        G_new = nx.compose(G_new, cc)

del G
print G_new.number_of_nodes(), G_new.number_of_edges()

# Relabel nodes with sig_ids
d_id_sig_id = dict(zip(range(len(grouped_sorted['sig_id'])), grouped_sorted['sig_id']))
d_id_sig_id = {nid: sig_id for nid, sig_id in d_id_sig_id.items() if G_new.has_node(nid)}
nx.relabel_nodes(G_new, d_id_sig_id, copy=False)
print G_new.nodes()[:5]

nx.write_gml(G_new, '../notebooks/Signature_Graph_%s_%dnodes_%s.gml' % 
	(KEY, G_new.number_of_nodes(), percentile_cutoff))


