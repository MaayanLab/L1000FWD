import numpy as np
import pandas as pd
from scipy import stats

# Load null ranks from file
def load_null_ranks():
	mat = np.load('data/null_scores_mat_genes_from_Probes_10000x42809.npy')['mat']
	rankmat = np.apply_along_axis(lambda x: stats.rankdata(x, method='ordinal'), 1, mat)
	mean_ranks = rankmat.mean(axis=0)
	std_ranks = rankmat.std(axis=0)
	return mean_ranks, std_ranks

mean_ranks, std_ranks = load_null_ranks()

def get_enrich_table(res_df, 
	mean_ranks=None, std_ranks=None):
	'''res_df from SigineDMOA, compute the zscores, combined scores based on 
	mean_ranks, std_ranks.
	'''
	res_df = res_df.sort_values('scores', ascending=False)
	res_df['score_ranks'] = np.arange(res_df.shape[0]) + 1

	# sort by sig_id
	res_df = res_df.sort_index()
	res_df['zscores'] = (res_df['score_ranks'] - mean_ranks) / std_ranks
	res_df['combined_scores'] = np.log10(res_df['pvals']) * res_df['zscores']
	return res_df.drop(['score_ranks'], axis=1)

