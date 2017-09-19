import os, sys, json
from itertools import combinations
import h5py
from pymongo import MongoClient
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
np.random.seed(10)

from joblib import delayed, Parallel

KEY = 'CD_center_LM_det'
N_JOBS = 8

def _gesa_enrichment_score(ranks_s):
    '''Calculate enrichment score from a rank ordered boolean array.
    ranks_s: np.array([0., 1., 0., 0.])
        - 1.: hits
        - 0.: misses
    '''
    n_hits = ranks_s.sum()
    n_misses = ranks_s.shape[0] - n_hits
    
    p_hit = np.cumsum(ranks_s) / n_hits
    p_miss = np.cumsum(1 - ranks_s) / n_misses
    p_diff = np.absolute(p_hit - p_miss)
    idx = np.argmax(p_diff)
    es = p_hit[idx] - p_miss[idx]
    return es
    
def gsea_score(sig1, sig2, n_sig=100):
    '''GSEA-based Kolmogorov-Smirnov statsitics.
    n_sig: number of top ranked genes to be treated as significant
    '''
    # number of genes
    n = len(sig1)
    # rank genes in sig1 (0: most down gene, 977: most up genes)
    ranks1 = stats.rankdata(sig1) - 1 
    # identify top up/down genes in sig1
    sig1_down = ranks1 < n_sig
    sig1_up = ranks1 > (n-1-n_sig)
    # argsort sig2
    sig2_srt_idx = sig2.argsort()
    # Compute ES: sig1 as query, sig2 as ref rank
    es_up1 = _gesa_enrichment_score( sig1_up[sig2_srt_idx].astype(float) )
    es_down1 = _gesa_enrichment_score( sig1_down[sig2_srt_idx].astype(float) )
    
    # rank genes in sig2
    ranks2 = stats.rankdata(sig2) - 1
    # identify top up/down genes in sig2
    sig2_down = ranks2 < n_sig
    sig2_up = ranks2 > (n-1-n_sig)
    # argsort sig1
    sig1_srt_idx = sig1.argsort()
    # Compute ES: sig2 as query, sig1 as ref rank
    es_up2 = _gesa_enrichment_score( sig2_up[sig1_srt_idx].astype(float) )
    es_down2 = _gesa_enrichment_score( sig2_down[sig1_srt_idx].astype(float) )
    
#     print es_up1, es_down1
#     print es_up2, es_down2
    
    # es_up is using up gene set to find hits in a list ascending ordered, 
    # therefore, the desirable sign should be negative
    score = (es_down1 - es_up1 + es_down2 - es_up2) / 4. 
    return score

def cosine_sim(sig1, sig2):
    '''Cosine similarity'''
    return 1 - distance.cosine(sig1, sig2)

def correlation(sig1, sig2):
    '''Pearson correlation'''
    return 1 - distance.correlation(sig1, sig2)

def pscore(mat, func, n_jobs=1, **kwargs):
    '''mat is a signature by gene matrix, apply func to all pairwise signatures.
    Similar to pdist
    '''
    n = mat.shape[0]
    n_scores = n * (n-1) / 2
    scores = np.zeros(n_scores)
    c = 0
    if n_jobs == 1:
        for i, j in combinations(range(n), 2):
            scores[c] = func(mat[i], mat[j])
            c += 1
    else:
        scores = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=10)(
            delayed(func)(mat[i], mat[j], **kwargs) for i, j in combinations(range(n), 2))
        scores = np.array(scores)
    return scores


drug_moa_df = pd.read_csv('../../Repurposing_Hub_export.txt', sep='\t')
print drug_moa_df.shape
drug_moa_df['pert_ids'] = drug_moa_df['Id']\
    .map(lambda x: ','.join(set(['-'.join(item.split('-')[0:2]) for item in x.split(', ')])))

drug_moa_df['pert_id_count'] = drug_moa_df['pert_ids']\
    .map(lambda x: len(x.split(',')))

drug_moa_df.set_index('Name', inplace=True)

# A dict from pert_id to name
d_pert_name = {}
for name, row in drug_moa_df.iterrows():
    for pert_id in row['pert_ids'].split(','):
        d_pert_name[pert_id] = name
print len(d_pert_name)


def retrieve_signature_meta_df(coll, query):
    cur = coll.find(query, 
                    {'sig_id':True,
                     'pert_id':True,
                     '_id':False
                    })
    meta_df = pd.DataFrame([doc for doc in cur]).set_index('sig_id')
    print meta_df.shape
    
    meta_df['batch'] = meta_df.index.map(lambda x: x.split(':')[0])
    meta_df['batch_prefix'] = meta_df['batch'].map(lambda x:x.split('_')[0])
    meta_df['cell_id'] = meta_df['batch'].map(lambda x:x.split('_')[1])
    meta_df['time'] = meta_df['batch'].map(lambda x:x.split('_')[2])
    return meta_df

# CD signatures produced by Qiaonan 
client = MongoClient('mongodb://146.203.54.131:27017/')
coll = client['L1000CDS2']['cpcd-gse70138']

meta_df = retrieve_signature_meta_df(coll, {'pert_type': 'trt_cp'})
print meta_df.shape

coll_fwd = client['L1000FWD']['sigs']
meta_df_fwd = retrieve_signature_meta_df(coll_fwd, {})
print meta_df_fwd.shape

print meta_df['batch_prefix'].nunique(), meta_df_fwd['batch_prefix'].nunique()
shared_batch_prefix = set(meta_df['batch_prefix'].unique()) & set(meta_df_fwd['batch_prefix'].unique())
print len(shared_batch_prefix)

print meta_df['pert_id'].nunique(), meta_df_fwd['pert_id'].nunique()
shared_pert_ids = set(meta_df['pert_id'].unique()) & set(meta_df_fwd['pert_id'].unique())
print len(shared_pert_ids)

# Subset the two meta_df to keep only shared pert_id and batch_prefix
meta_df = meta_df.loc[meta_df['batch_prefix'].isin(shared_batch_prefix) & meta_df['pert_id'].isin(shared_pert_ids)]
print meta_df.shape

meta_df_fwd = meta_df_fwd.loc[meta_df_fwd['batch_prefix'].isin(shared_batch_prefix) & meta_df_fwd['pert_id'].isin(shared_pert_ids)]
print meta_df_fwd.shape


def retrieve_sig_mat(sig_ids, coll, key):
    '''Retrieve signatures matrix from MongoDB'''
    # Retrieve signature matrix
    sig_mat = np.zeros((len(sig_ids), 978))
    for i, sig_id in enumerate(sig_ids): 
        doc = coll.find_one({'sig_id': sig_id}, {'_id':False, key:True})
        sig_mat[i] = doc[key]
    return sig_mat

print len(d_pert_name), meta_df['pert_id'].nunique()
pert_ids_with_MoAs = set(d_pert_name) & set(meta_df['pert_id'].unique())
print len(pert_ids_with_MoAs)


meta_df = meta_df.loc[meta_df['pert_id'].isin(pert_ids_with_MoAs)]
print meta_df.shape

meta_df_fwd = meta_df_fwd.loc[meta_df_fwd['pert_id'].isin(pert_ids_with_MoAs)]
print meta_df_fwd.shape

meta_df = meta_df.sort_index()
meta_df_fwd = meta_df_fwd.sort_index()

# sample 10000 sig_ids for this benchmark
random_idx = np.random.choice(range(meta_df.shape[0]), 5000, replace=False)
meta_df_sample = meta_df.iloc[random_idx]
meta_df_fwd_sample = meta_df_fwd.iloc[random_idx]
print meta_df_sample.shape, meta_df_fwd_sample.shape


def compute_pscores(meta_df, coll, key, n_jobs=1):
    '''Given a meta_df of signatures, compute the pairwise scores and return a df.'''
    sig_ids = meta_df.index.tolist()
    sig_mat = retrieve_sig_mat(sig_ids, coll, key)
    print 'sig_mat retrieved'

    scores_es50 = pscore(sig_mat, gsea_score, n_jobs=n_jobs, n_sig=50)

    scores_cosine = pscore(sig_mat, cosine_sim, n_jobs=n_jobs)
    scores_corr = pscore(sig_mat, correlation, n_jobs=n_jobs)
    
    print 'scores computed'
    n_scores = len(scores_cosine)
    sigs_i = [None] * n_scores
    sigs_j = [None] * n_scores
    c = 0
    for sig_i, sig_j in combinations(sig_ids, 2):
        sigs_i[c] = sig_i
        sigs_j[c] = sig_j
        c += 1

    df = pd.DataFrame({
        'sig_i': sigs_i, 'sig_j': sigs_j, 
        'cosine': scores_cosine,
        'corr': scores_corr,
        'ES50': scores_es50
    })
    return df

res_scores5 = compute_pscores(meta_df_fwd_sample, coll_fwd, KEY, n_jobs=N_JOBS)
print res_scores5.shape
res_scores5.to_csv('data/signature_connectivities_sample5000_L1000FWD.%s.csv' % KEY)


