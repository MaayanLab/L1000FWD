'''
To compute CD signatures for LINCS L1000 level3 data using 
all profiles in the same batch as controls
and add results to mongodb.
''' 
N_JOBS = 4
HELEN = False
reverse = False


import os, sys, json
import h5py
from pymongo import MongoClient
import geode

def file2list(fn, idx, sep='\t', header=False):
	"""read a file into a list"""
	l = []
	with open (fn, 'r') as f:
		if header:
			next(f)
		for line in f:
			if not line.startswith('#'):
				sl = line.strip().split(sep)
				t = sl[idx]
				l.append(t)
	return l

import numpy as np
import pandas as pd

if HELEN:
	GCTX_FILE = '../../q2norm_n1328098x22268.gctx'
	PROBES_LM1000 = file2list('../data/rid_lm1000.txt', 0)
else:
	GCTX_FILE = '/Users/zichen/Documents/Zichen_Projects/L1000_DR/data/q2norm_n1328098x22268.gctx'
	PROBES_LM1000 = file2list('/Users/zichen/Documents/bitbucket/lincs_l1000_limma/rid_lm1000.txt', 0)


print PROBES_LM1000[:5]

gctx = h5py.File(GCTX_FILE, 'r')
mat = gctx['/0/DATA/0/matrix']
print mat.shape

def slice_matrix(gctx, cids, rids):
    '''Slice the mat by cids and rids and ensure the mat 
    is ordered by cids and rids.'''    
    all_cids = gctx['/0/META/COL/id']
    c_mask = np.in1d(all_cids, cids)
    cids_subset = all_cids[c_mask].tolist()
    c_indices = np.array([cids_subset.index(id_) 
                          for id_ in cids])

    mat = gctx['/0/DATA/0/matrix']
    submat = mat[c_mask, :][c_indices, :]
    
    all_rids = gctx['/0/META/ROW/id']
    r_mask = np.in1d(all_rids, rids)
    rids_subset = all_rids[r_mask].tolist()
    r_indices = np.array([rids_subset.index(id_) 
                          for id_ in rids])
    submat = submat[:, r_mask][:, r_indices]

    return submat


client = MongoClient("mongodb://146.203.54.131:27017")
DB = client['LINCS_L1000_limma']
COLL_SIG = DB['siginfo'] # to get metadata

# Collect pert_ids and batches
cur = COLL_SIG.find({'pert_type':'trt_cp'}, 
                    {'_id':False, 'sig_id':True, 'distil_id':True})
print cur.count()

# Load the metadata for the currently visualized L1000 signatures in L1000FWD
sig_meta_df = pd.read_csv('../data/metadata-full.tsv', sep='\t').set_index('sig_id')
print sig_meta_df.shape


pert_id_include = set(sig_meta_df['pert_id'].unique())



# Retrieve sig_meta from the MongoDB
cur = COLL_SIG.find({'pert_type':'trt_cp'},
                    {'_id':False, 
                     'sig_id':True, 
                     'distil_id':True,
                    })
sig_meta_df_full = pd.DataFrame([doc for doc in cur]).set_index('sig_id')
print sig_meta_df_full.shape

cur = COLL_SIG.find({'$and': [ 
                        {'pert_type':'trt_cp'},
                        {'distil_nsample': {'$gt': 1}}
                    ]}, 
                    {'_id':False, 
                     'sig_id':True, 
                     'distil_id':True,
                     'pert_id': True,
                     'pert_dose':True,
                     'pert_time':True,
                     'cell_id':True,
                     'pert_desc':True,
                    })
print cur.count()

sig_meta_df = pd.DataFrame([doc for doc in cur if doc['pert_id'] in pert_id_include]).set_index('sig_id')
print sig_meta_df.shape
# Get batch info
sig_meta_df_full['batch'] = sig_meta_df_full.index.map(lambda x:x.split(':')[0])
print sig_meta_df_full['batch'].nunique()

sig_meta_df['batch'] = sig_meta_df.index.map(lambda x:x.split(':')[0])
print sig_meta_df['batch'].nunique()

# Get all the distil_ids
COLL_INST = DB['instinfo']
cur = COLL_INST.find({'pert_type': {'$in' : ['trt_cp', 'trt_poscon']}}, 
                     {'_id':False, 
                      'distil_id':True, 
                      'pert_id':True,
                      'det_plate':True,
                     })

distil_df = pd.DataFrame([doc for doc in cur]).set_index('distil_id')
print distil_df.shape

def distil_id_to_rna_plate(did):
    return '_'.join(did.split('_')[:4])

def distil_id_to_batch(did):
    return '_'.join(did.split('_')[:3])

# def distil_id_to_det_plate(did):
#     return '_'.join(did.split('_')[:5])

# distil_df['det_plate'] = distil_df.index.map(distil_id_to_det_plate)
distil_df['rna_plate'] = distil_df.index.map(distil_id_to_rna_plate)
distil_df['batch'] = distil_df.index.map(distil_id_to_batch)

# Make MongoDB to store those data
# client_local = MongoClient('mongodb://127.0.0.1:27017')
# db = client_local['L1000FWD']
db = client['L1000FWD']
coll = db['sigs']
coll.create_index('sig_id', unique=True)

from joblib import delayed, Parallel

def mean_center(mat, centerby):
    '''Mean center a mat based on centerby. mat is a samples x genes matrix'''
    mat_centered = np.zeros_like(mat)
    
    for group in set(centerby):
        mask = np.in1d(centerby, [group])
        mat_centered[mask] = mat[mask] - mat[mask].mean(axis=0)
    
    return mat_centered

from sklearn.decomposition import PCA
from scipy.stats import chi2
from scipy.stats.mstats import zscore

def fast_chdir(data, sampleclass):
    m1 = sampleclass == 1
    m2 = sampleclass == 2
    
    gamma = 0.5
    
    data = zscore(data)
    
    ## start to compute
    n1 = m1.sum() # number of controls
    n2 = m2.sum() # number of experiments

    ## the difference between experiment mean vector and control mean vector.
    meanvec = data[:,m2].mean(axis=1) - data[:,m1].mean(axis=1) 

    ## initialize the pca object
    pca = PCA(n_components=None)
    pca.fit(data.T)

    ## compute the number of PCs to keep
    cumsum = pca.explained_variance_ratio_ # explained variance of each PC
    keepPC = len(cumsum[cumsum > 0.001]) # number of PCs to keep
    v = pca.components_[0:keepPC].T # rotated data 
    r = pca.transform(data.T)[:,0:keepPC] # transformed data

    dd = ( np.dot(r[m1].T,r[m1]) + np.dot(r[m2].T,r[m2]) ) / float(n1+n2-2) # covariance
    sigma = np.mean(np.diag(dd)) # the scalar covariance

    shrunkMats = np.linalg.inv(gamma*dd + sigma*(1-gamma)*np.eye(keepPC))

    b = np.dot(v, np.dot(np.dot(v.T, meanvec), shrunkMats))

    b /= np.linalg.norm(b) # normalize b to unit vector
    
    return b


def compute_signatures2(sig_id, row, distil_ids_sub, mat_centered, PROBES_LM1000):
    distil_ids_pert = row['distil_id']
    # Make the sample_class
    mask_pert = np.in1d(distil_ids_sub, distil_ids_pert)
    sample_class = mask_pert.astype(int) + 1

    # Apply CD on the mean centered mat
    cd = geode.chdir(mat_centered.T, sample_class, PROBES_LM1000, 
                      calculate_sig=0, sort=False, gamma=0.5)
    # Averaging profiles after mean centering
    avg_vals = mat_centered[mask_pert].mean(axis=0)

    doc = {}
    doc['sig_id'] = sig_id
    doc['CD_center_LM_det'] = list(np.array([item[0] for item in cd], dtype=np.float64))
    doc['avg_center_LM_det'] = list(avg_vals.astype(np.float64))

    return doc

def compute_sig2_wrapper(sig_id, row, distil_ids_sub, mat_centered, PROBES_LM1000):
    try:
        doc = compute_signatures2(sig_id, row, distil_ids_sub, mat_centered, PROBES_LM1000)
    except ValueError as e:
        doc = None
        print e
        pass
    return doc

# Get all inserted document sig_ids
key = 'CD_center_LM_det'
# Get all inserted document sig_ids
inserted_sig_ids = set(coll.find({key: {'$exists': True}}).distinct('sig_id'))
print '#' * 20
print 'Number of sig_ids inserted: %d' % len(inserted_sig_ids)

sig_ids_left = list(set(sig_meta_df.index) - inserted_sig_ids)

# subset the sig_meta_df
sig_meta_df_left = sig_meta_df.ix[sig_ids_left]
all_batches = sig_meta_df_left['batch'].unique()
n_batches = len(all_batches)
print sig_meta_df_left.shape

for c, batch in enumerate(all_batches):
    sig_meta_df_sub = sig_meta_df_left.query('batch == "%s"' % batch)
    
    # all the treatment samples in this batch
    distil_ids_sub_df = distil_df.query('batch == "%s"' % batch)
    distil_ids_sub = distil_ids_sub_df.index.tolist()
    
    print c, n_batches
    print '\t', batch, sig_meta_df_sub.shape, len(distil_ids_sub)
    # Slice the matrix
    mat = slice_matrix(gctx, distil_ids_sub, PROBES_LM1000)
    print '\t', mat.shape
    # Mean center the probes by det_plate
    mat_centered = mean_center(mat, distil_ids_sub_df['det_plate'])
    
    try:
        docs = Parallel(n_jobs=N_JOBS, backend='multiprocessing', verbose=10)(\
                                      delayed(compute_sig2_wrapper)(sig_id, row, distil_ids_sub, mat_centered, PROBES_LM1000)\
                                      for sig_id, row in sig_meta_df_sub.iterrows())
        docs = filter(None, docs)
    except Exception as e:
        print e
        pass
    else:
        if len(docs) > 0:
            bulk = coll.initialize_ordered_bulk_op()
            for doc in docs:
                bulk.find({'sig_id': doc['sig_id']}).\
                    update_one({'$set': {
                        'CD_center_LM_det': doc['CD_center_LM_det'],
                        'avg_center_LM_det': doc['avg_center_LM_det']
                    }})
            bulk.execute()    
