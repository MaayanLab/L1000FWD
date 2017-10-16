'''
To compute CD signatures for LINCS L1000 level3 data using 
all profiles in the same batch as controls
and add results to mongodb.
''' 
N_JOBS = 6
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
from scipy import stats

if HELEN:
    GCTX_FILE = '../../q2norm_n1328098x22268.gctx'
    PROBES_LM1000 = file2list('../data/rid_lm1000.txt', 0)
    PROBES = json.load(open('../data/rid.json', 'rb'))
else:
    GCTX_FILE = '/Users/zichen/Documents/Zichen_Projects/L1000_DR/data/q2norm_n1328098x22268.gctx'
    PROBES_LM1000 = file2list('/Users/zichen/Documents/bitbucket/lincs_l1000_limma/rid_lm1000.txt', 0)
    PROBES = json.load(open('/Users/zichen/Documents/bitbucket/lincs_l1000_limma/rid.json', 'rb'))


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
DB = client['L1000FWD']
COLL_SIG = DB['sigs']

# Collect pert_ids and batches
cur = COLL_SIG.find({'SCS_centered_by_batch':{'$lt': 0.05}}, 
                    {'_id':False, 'sig_id':True, 'distil_id':True})
print cur.count()



# Retrieve sig_meta from the MongoDB
sig_meta_df_full = pd.DataFrame([doc for doc in cur]).set_index('sig_id')
print sig_meta_df_full.shape


# Get batch info
sig_meta_df_full['batch'] = sig_meta_df_full.index.map(lambda x:x.split(':')[0])
print sig_meta_df_full['batch'].nunique()


sig_meta_df_full['batch'] = sig_meta_df_full.index.map(lambda x:x.split(':')[0])
print sig_meta_df_full['batch'].nunique()

# Get all the distil_ids
COLL_INST = client['LINCS_L1000_limma']['instinfo']
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


# distil_df['det_plate'] = distil_df.index.map(distil_id_to_det_plate)
distil_df['rna_plate'] = distil_df.index.map(distil_id_to_rna_plate)
distil_df['batch'] = distil_df.index.map(distil_id_to_batch)


# Get probes2genes
COLL_GENES = client['LINCS_L1000_limma']['geneinfo']
cur = COLL_GENES.find({}, {'_id':False ,'pr_id':True, 'pr_gene_symbol':True})
probes2genes = pd.DataFrame([doc for doc in cur])
print probes2genes.shape
probes2genes = probes2genes.replace({'-666': None}).dropna(axis=0)
probes2genes = dict(zip(probes2genes['pr_id'], probes2genes['pr_gene_symbol']))
print len(probes2genes)


# Make MongoDB to store those data
db = client['L1000FWD']
coll = db['sigs']


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


def fast_chdir_with_pvals(data, sampleclass):
    cd_coefs = fast_chdir(data, sampleclass)
    zscores = stats.zscore(cd_coefs)
    pvals = stats.norm.sf(np.abs(zscores))*2 # two-sided
    return cd_coefs, pvals

def compute_signatures_full(sig_id, row, distil_ids_sub, mat_centered, PROBES, probes2genes):
    distil_ids_pert = row['distil_id']
    # Make the sample_class
    mask_pert = np.in1d(distil_ids_sub, distil_ids_pert)
    sample_class = mask_pert.astype(int) + 1

    # Apply CD on the mean centered mat
    cd_coefs, pvals = fast_chdir_with_pvals(mat_centered.T, sample_class)
    sigIdx = np.where(pvals < 0.01)[0].astype(int)
        
    # Identify significant up/down genes
    up_probes = np.array(PROBES)[(cd_coefs > 0) & (pvals< 0.01)]
    upGenes = filter(None, set([probes2genes.get(pr_id) for pr_id in up_probes]))

    dn_probes = np.array(PROBES)[(cd_coefs < 0) & (pvals< 0.01)]
    dnGenes = filter(None, set([probes2genes.get(pr_id) for pr_id in dn_probes]))
    
    doc = {}
    doc['sig_id'] = sig_id
    doc['CD_center_Full'] = list(cd_coefs.astype(np.float64))
    doc['pvalues_Full'] = list(pvals.astype(np.float64))
    doc['sigIdx'] = list(sigIdx)
    doc['upGenes'] = upGenes
    doc['dnGenes'] = dnGenes
    return doc


def compute_signatures_full_wrapper(sig_id, row, distil_ids_sub, mat_centered, PROBES, probes2genes):
    try:
        doc = compute_signatures_full(sig_id, row, distil_ids_sub, mat_centered, PROBES, probes2genes)
    except ValueError as e:
        doc = None
        print e
        pass
    return doc

# Get all inserted document sig_ids
key = 'CD_center_Full'
# Get all inserted document sig_ids
inserted_sig_ids = set(coll.find({key: {'$exists': True}}).distinct('sig_id'))
print '#' * 20
print 'Number of sig_ids inserted: %d' % len(inserted_sig_ids)

sig_ids_left = list(set(sig_meta_df_full.index) - inserted_sig_ids)

# subset the sig_meta_df
sig_meta_df_left = sig_meta_df_full.ix[sig_ids_left]
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
    mat = slice_matrix(gctx, distil_ids_sub, PROBES)
    print '\t', mat.shape
    # Mean center 
    mat_centered = mat - mat.mean(axis=0)
    
    try:
        docs = Parallel(n_jobs=N_JOBS, backend='multiprocessing', verbose=10)(\
                                      delayed(compute_signatures_full_wrapper)(sig_id, row, distil_ids_sub, mat_centered, PROBES, probes2genes)\
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
                        'CD_center_Full': doc['CD_center_Full'],
                        'pvalues_Full': doc['pvalues_Full'],
                        'sigIdx': doc['sigIdx'],
                        'upGenes': doc['upGenes'],
                        'dnGenes': doc['dnGenes'],
                    }})
            bulk.execute()


