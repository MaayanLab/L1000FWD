'''Uses random up/down gene sets to get the null distribution 
of scores for the signatures.'''

import os, sys
import pandas as pd
import numpy as np
import json, requests

RURL = 'http://146.203.54.239:31722/custom/SigineDMOA'
MONGOURI = 'mongodb://146.203.54.131:27017/L1000FWD'

class UserInput(object):
	"""The base class for GeneSets and Signature"""
	config = {"direction":"mimic","combination":False}
	headers = {'content-type':'application/json'}
	default_score = 0. # default enrichment score for an irrelevant signature

	def __init__(self, data):
		self.data = data
		self.result = None
		self.type = None
		self.rid = None

	def _enrich(self):
		'''POST to Rook API to get enriched signatures.
		'''
		self.config['method'] = self.type
		data = self.humanize()
		payload = dict(data.items() + self.config.items())
		response = requests.post(RURL, data=json.dumps(payload),headers=self.headers)
		return response.json()

	def enrich(self, df, graph_name):
		'''POST to Rook API to get enriched LJP signatures.
		df is the graph_df to subset the scores
		'''
		self.config['method'] = self.type
		data = self.humanize()
		payload = dict(data.items() + self.config.items())
		response = requests.post(RURL, data=json.dumps(payload),headers=self.headers)
		result = pd.DataFrame(response.json()).set_index('sig_ids')
		result = result.loc[df.index].reset_index()
		result_sorted = result.sort_values('scores', ascending=False)
		# Get the top N as list of records:
		topn = {
			'similar': result_sorted.iloc[:50].to_dict(orient='records'),
			'opposite': result_sorted.iloc[-50:][::-1].to_dict(orient='records'),
			}
		self.result = {
			'scores': result['scores'].tolist(), 
			'topn': topn
			}
		self.graph_name = graph_name
		return self.result

	def save(self):
		'''Save the UserInput as well as the EnrichmentResult to a document'''
		res = mongo.db.userResults.insert_one({
			'result': self.result, 
			'data': self.data, 
			'type': self.type,
			'graph_name': self.graph_name
			})
		self.rid = res.inserted_id # <class 'bson.objectid.ObjectId'>
		return str(self.rid)


class GeneSets(UserInput):
	"""docstring for GeneSets"""
	def __init__(self, up_genes, dn_genes):
		data = {'upGenes': up_genes, 'dnGenes': dn_genes}
		UserInput.__init__(self, data)
		self.type = 'geneSet'

	def json_data(self):
		'''Return an object to be encoded to json format'''
		return self.data

	def humanize(self):
		'''Convert genes to upper cases.'''
		data = self.data
		data['upGenes'] = map(lambda x:x.upper(), data['upGenes'])
		data['dnGenes'] = map(lambda x:x.upper(), data['dnGenes'])
		return data


# Get gene symbols from HGNC
genes_df = pd.read_csv('/Users/zichen/Documents/Zichen_Projects/shared_data/HGNC_symbols.txt', 
	sep='\t')
genes_df = genes_df.loc[genes_df['Locus Group'].isin(['protein-coding gene', 'non-coding RNA'])]
print genes_df.shape
all_genes = genes_df['Approved Symbol'].tolist()
print len(all_genes)

import time 
t0 = time.time()
n_perms = 10000
null_scores_mat = np.zeros((n_perms, 42809), dtype=np.float)

for i in xrange(n_perms):
	genes = np.random.choice(all_genes, 500, replace=False)
	# print len(genes)
	up_genes = genes[:250]
	dn_genes = genes[250:]
	gene_set = GeneSets(up_genes, dn_genes)
	result = gene_set._enrich()
	# print len(result)
	# print result.keys()
	# print len(result['sig_ids'])
	srt_idx = np.argsort(result['sig_ids'])
	scores = np.array(result['scores'])[srt_idx]
	# print result['sig_ids'][:5]
	null_scores_mat[i] = scores
	if i % 500 == 0:
		print i, n_perms

tt = time.time()
print tt - t0 # 3.82(s) for 10 
print null_scores_mat[:5, :2]
np.savez(open('../data/null_scores_mat_%dx%d.npy' % null_scores_mat.shape, 'w'), 
	mat=null_scores_mat)

# null_scores_mat = np.load('null_scores_mat_10x42809.npy')['mat']
# print null_scores_mat[:5, :2]



