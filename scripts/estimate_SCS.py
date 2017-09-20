'''
Estimate SCS (p-value) using mean(cosine distance) for `CD_center_LM`
'''

N_JOBS = 7
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
from sklearn.metrics import pairwise

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

def compute_average_cosine_distance(sig_mat):
    '''Compute average cosine distance given a signature matrix (n_sigs, genes)'''
    D = pairwise.cosine_distances(sig_mat)
    d = D[np.tril_indices_from(D, k=-1)]
    return d.mean()


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

distil_df['rna_plate'] = distil_df.index.map(distil_id_to_rna_plate)
distil_df['batch'] = distil_df.index.map(distil_id_to_batch)

# Make MongoDB to store those data
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

def chdir_batch(mat, sample_class, batches):
    '''Compute CD on each batch.
    '''
    cds_all = []
    for batch in set(batches):
        mask_batch = np.in1d(batches, [batch])

        if len(set(sample_class[mask_batch])) > 1:
            cd_coefs = fast_chdir(mat[:, mask_batch], sample_class[mask_batch])
            cds_all.append(cd_coefs)
    if len(cds_all) == 0:
        return None
    else:
        return np.array(cds_all)

def get_random_cosine_pdist(batches, n_batches, mat):
    sig_mat = np.zeros((n_batches, 978))
    for i, batch in enumerate(set(batches)):
        mask_batch = np.in1d(batches, [batch])
        n_samples_in_batch = mask_batch.sum()
        mat_batch = mat[mask_batch]
        sample_class_batch = np.ones(n_samples_in_batch)

        pert_idx = np.random.choice(n_samples_in_batch, 1)[0]
        sample_class_batch[pert_idx] = 2

        cd = fast_chdir(mat_batch.T, sample_class_batch)
        sig_mat[i] = cd

    mean_cosine_dist = compute_average_cosine_distance(sig_mat)
    return mean_cosine_dist

def get_null_cosine_pdist(mat, batches, n_jobs=6):
    '''Given the expression matrix of all batches, 
    compute null distribution of mean cosine distances.'''
    n = 10000
    # n = 1000
    # mean_cosine_dist_nulls = np.zeros(n)
    n_batches = len(set(batches))
    mean_cosine_dist_nulls = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=10)(\
        delayed(get_random_cosine_pdist)(batches, n_batches, mat) for permutation_idx in range(n))
    return np.array(mean_cosine_dist_nulls)


def compute_signature_mean_cosine_dist(sig_id, row, distil_ids_sub, mat_centered, plates, mean_cosine_dist_nulls):
    distil_ids_pert = row['distil_id']
    # Make the sample_class
    mask_pert = np.in1d(distil_ids_sub, distil_ids_pert)
    sample_class = mask_pert.astype(int) + 1

    # Apply CD on the mean centered mat
    cds = chdir_batch(mat_centered.T, sample_class, plates)

    mean_cosine_dist_obs = compute_average_cosine_distance(cds)

    scs = (mean_cosine_dist_nulls < mean_cosine_dist_obs).sum() / float(len(mean_cosine_dist_nulls))

    doc = {
        'sig_id': sig_id,
        'mean_cosine_dist_centered_by_batch': mean_cosine_dist_obs,
        'SCS_centered_by_batch': scs
    }
    return doc


# def compute_sig3_wrapper(sig_id, row, distil_ids_sub_df,mat, mat_centered, PROBES_LM1000):
#     try:
#         doc = compute_signatures3(sig_id, row, distil_ids_sub_df,mat, mat_centered, PROBES_LM1000)
#     except ValueError as e:
#         doc = None
#         print e
#         pass
#     return doc


# Get all inserted sig_ids with CD_center_LM
inserted_sig_ids_with_sigs = coll.find({'CD_center_LM': {'$exists':True}}).distinct('sig_id')
sig_meta_df = sig_meta_df.ix[inserted_sig_ids_with_sigs]
print sig_meta_df.shape


# Get all inserted document sig_ids with SCS_centered_by_batch
key = 'SCS_centered_by_batch'
inserted_sig_ids = set(coll.find({key: {'$exists': True}}).distinct('sig_id'))
print '#' * 20
print 'Number of sig_ids inserted with SCS_centered_by_batch: %d' % len(inserted_sig_ids)

sig_ids_left = list(set(sig_meta_df.index) - inserted_sig_ids)

# subset the sig_meta_df
sig_meta_df_left = sig_meta_df.ix[sig_ids_left]
all_batches = sig_meta_df_left['batch'].unique()
n_batches = len(all_batches)
print sig_meta_df_left.shape
if reverse:
  all_batches = all_batches[::-1]

for c, batch in enumerate(all_batches):
    sig_meta_df_sub = sig_meta_df_left.query('batch == "%s"' % batch)
    
    # all the treatment samples in this batch
    distil_ids_sub_df = distil_df.query('batch == "%s"' % batch)
    distil_ids_sub = distil_ids_sub_df.index.tolist()
    
    print c, n_batches
    print '\t', batch, sig_meta_df_sub.shape, len(distil_ids_sub)
    print '\tnumber of rna_plates:', distil_ids_sub_df['rna_plate'].nunique() 
    # Slice the matrix
    mat = slice_matrix(gctx, distil_ids_sub, PROBES_LM1000)
    print '\t', mat.shape
    # Mean center the probes
    mat_centered = mat - mat.mean(axis=0)

    # Compute the null
    plates = distil_ids_sub_df['rna_plate']
    print 'Started to estimate null mean cosine pdist'
    mean_cosine_dist_nulls = get_null_cosine_pdist(mat_centered, plates, n_jobs=N_JOBS)

    # for sig_id, row in sig_meta_df_sub.iterrows():
    #     # Calculate the observed mean_cosine_dist for each signature
    #     doc = compute_signature_mean_cosine_dist(sig_id, row, distil_ids_sub, mat_centered, plates, mean_cosine_dist_nulls)
    #     print doc
    

    docs = Parallel(n_jobs=N_JOBS, backend='multiprocessing', verbose=10)(\
                                  delayed(compute_signature_mean_cosine_dist)(sig_id, row, distil_ids_sub_df, mat_centered, plates, mean_cosine_dist_nulls)\
                                  for sig_id, row in sig_meta_df_sub.iterrows())
    docs = filter(None, docs)

    bulk = coll.initialize_ordered_bulk_op()
    for doc in docs:
        bulk.find({'sig_id': doc['sig_id']}).\
            update_one({'$set': {
                'mean_cosine_dist_centered_by_batch': doc['mean_cosine_dist_centered_by_batch'],
                'SCS_centered_by_batch': doc['SCS_centered_by_batch']
            }})
    bulk.execute()    

