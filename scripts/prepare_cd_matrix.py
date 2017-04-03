import os
import json

import h5py
import numpy as np
import pandas as pd

from pymongo import MongoClient


conn = MongoClient('mongodb://146.203.54.131:27017/')


## 0. Get the genes first
coll = conn['LINCS_L1000_genes']['genes']
genes = coll.find_one({'tag': 'LINCScloud'})['geneSymbol']
genes = np.array(genes)
print len(genes), genes[:10]

# the coll for signatures
coll = conn['L1000CDS2']['cpcd-gse70138']


meta_df = pd.read_csv('../gen3va/scripts/meta_df.csv').set_index('sig_id')
pert_ids = meta_df['pert_id'].unique()
print meta_df.head()
grouped_sorted = meta_df.groupby('pert_id')['pvalue'].apply(lambda x: x.order(ascending=True).head(50))
print grouped_sorted[:50]

# filter out pert_id with less than 20 signatures
pert_id_counts = meta_df.reset_index().groupby('pert_id')['sig_id'].count()
print pert_id_counts[:10]

# pert_ids_kept = pert_id_counts[pert_id_counts > 20].index.tolist()
pert_ids_kept = pert_id_counts[pert_id_counts > 5].index.tolist()

print 'Number of pert_id to keep: %d' % len(pert_ids_kept)
print grouped_sorted.shape
grouped_sorted = grouped_sorted[pert_ids_kept].reset_index()
print grouped_sorted.shape
n_sigs = grouped_sorted.shape[0]
print 'Number of sig_id to insert: %d' % n_sigs
# print grouped_sorted[:50]
# print grouped_sorted[:50].loc[1, 'sig_id']



sig_ids = []
## write a h5 file to store the CD matrix
with h5py.File('data/CD_matrix_%sxlm978.h5' % n_sigs, 'w') as f:
	dset = f.create_dataset('matrix', shape=(n_sigs, 978), dtype=np.float32)
	meta = f.create_group('meta')

	meta['genes'] = [a.encode('utf8') for a in genes]
	
	for i in range(n_sigs):
		sig_id = grouped_sorted.loc[i, 'sig_id']
		doc = coll.find_one({'sig_id': sig_id}, {'_id':False})

		if 'chdirLm' in doc:
			row = np.array(doc['chdirLm'])
			# write row by row
			dset.write_direct(row, dest_sel=np.s_[i, :])
			sig_ids.append(sig_id)

		if i % 100 == 0:
			print i, n_sigs


	meta['sig_ids'] = sig_ids
	
	
# with h5py.File('data/CD_matrix_45012xlm978.h5', 'r') as f:
# 	mat = f['matrix']
# 	# print mat.shape
# 	# print mat[0].mean()
# 	# print mat[1].mean()
# 	# print mat[3].mean()

# 	print len(f['meta']['sig_ids'])
# 	print mat.shape

