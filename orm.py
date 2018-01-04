import os
import json, requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import create_engine
from bson.objectid import ObjectId
# from pymongo import MongoClient
from flask_pymongo import PyMongo

mongo = PyMongo()

RURL = os.environ['RURL']
MONGOURI = os.environ['MONGOURI']
# client = MongoClient(MONGOURI)
# DB = client['DMOA']
# COLL_RES = DB['userResults']

MYSQLURI = os.environ['MYSQLURI']
engine = create_engine(MYSQLURI)

def encode_rare_categories(df, colname, max=19):
	'''
	Encode rare categories in a df as 'RARE'
	'''
	frequent_categories = set(df[colname].value_counts()[:max].index)
	def _encode_rare(element):
		if element in frequent_categories:
			return element
		else:
			return 'rare'

	df[colname] = df[colname].apply(_encode_rare)
	return df


def load_graphs_meta():
	'''Load and preprocess the meta for graphs in the `graphs` collection.
	'''
	graph_names = mongo.db.graphs.distinct('name')
	graphs = {
		'cells':[
			{'name': 'Signature_Graph_CD_center_LM_sig-only_16848nodes.gml.cyjs', 
			'display_name': 'All cell lines', 'cell': 'all',
			'n_sigs': 16848
			}
		],
		'agg': [
			{'name': 'graph_pert_cell_12894nodes_99.9.gml.cyjs', 'display_name': 'Aggregated by drugs and cells'},
			{'name': 'kNN_5_layout', 'display_name': 'Aggregated by drugs (kNN-FWD)'},
			{'name': 'threshold_99.5', 'display_name': 'Aggregated by drugs (thresholding-FWD)'},
		],
	}

	# process graphs for individual cells
	for graph_name in graph_names:
		doc = mongo.db.graphs.find_one({'name': graph_name}, 
			{'sig_ids':True, '_id':False})
		
		n_sigs = len(doc['sig_ids'])
		rec = {
			'name': graph_name,
			'n_sigs': n_sigs
		}
		if graph_name.endswith('-tSNE_layout.csv'):
			cell = graph_name.split('-')[0]
			display_name = '%s (tSNE)' % cell
			rec['cell'] = cell
			rec['display_name'] = display_name
			graphs['cells'].append(rec)			
		elif graph_name.endswith('kNN_5'):
			cell = graph_name.split('_')[0]
			display_name = '%s (FWD)' % cell
			rec['display_name'] = display_name
			rec['cell'] = cell
			graphs['cells'].append(rec)
	return graphs

def load_predicted_MOAs(filename, name='predicted_MOA'):
	# Load pert level MOA predictions
	# drug_moa_preds_df = pd.read_csv('data/agg_MOA_preds_at_pert_level_XGB680.csv').set_index('pert_id')
	drug_moa_preds_df = pd.read_csv(filename).set_index('pert_id')
	classes = drug_moa_preds_df.columns.values
	max_idx = drug_moa_preds_df.values.argmax(axis=1)
	predicted_MOAs = pd.DataFrame({name: classes[max_idx]},
		index=drug_moa_preds_df.index)
	return predicted_MOAs


def load_drug_meta_from_db():
	drug_meta_df = pd.read_sql_query('''
		SELECT drug_repurposedb2.pert_id, drug_repurposedb2.pert_iname AS pert_desc,
		most_frequent_dx_rx.most_frequent_rx, most_frequent_dx_rx.most_frequent_dx, 
		drug_repurposedb2.Phase, drug_repurposedb2.MOA
		FROM most_frequent_dx_rx
		RIGHT JOIN drug_repurposedb2
		ON drug_repurposedb2.pert_id=most_frequent_dx_rx.pert_id
		''', engine, 
		index_col='pert_id')
	print drug_meta_df.shape

	# # Borrow the info from pert_ids with the same pert_desc
	# for col in ['most_frequent_rx','most_frequent_dx','Phase', 'MOA']:
	# 	d = drug_meta_df[['pert_desc', col]].dropna(axis=0)
	# 	d = dict(zip(d['pert_desc'], d[col]))
	# 	drug_meta_df[col] = [d.get(name, None) for name in drug_meta_df['pert_desc']]

	drug_chem_df = pd.read_sql_table('drug_scaffolds_sub', engine, index_col='pert_id')	
	drug_meta_df = drug_meta_df.merge(drug_chem_df, left_index=True, right_index=True, how='left')

	predicted_MOAs = load_predicted_MOAs('data/agg_MOA_preds_at_pert_level_XGB680.csv')
	drug_meta_df = drug_meta_df.merge(predicted_MOAs, left_index=True, right_index=True
		, how='left')

	predicted_MOAs_GE = load_predicted_MOAs('data/agg_MOA_preds_at_pert_level_kNN15_GE.csv',
		name='predicted_MOA_GE')
	drug_meta_df = drug_meta_df.merge(predicted_MOAs_GE, left_index=True, right_index=True,
		how='left')
	return drug_meta_df

def load_drug_synonyms_from_db(meta_df, graph_df):
	# Load synonyms for drugs
	drug_synonyms = pd.read_sql_table('drug_synonyms', engine, columns=['pert_id', 'Name'])
	print drug_synonyms.shape
	# Keep only the pert_id that are in the graph
	pert_ids_in_graph = meta_df.loc[graph_df.index]['pert_id'].unique()
	print 'Number of unique pert_id in graph:', len(pert_ids_in_graph)
	drug_synonyms = drug_synonyms.loc[drug_synonyms['pert_id'].isin(pert_ids_in_graph)]
	# Add pert_id itself as name
	drug_synonyms = drug_synonyms.append(
		pd.DataFrame({'pert_id': pert_ids_in_graph, 'Name': pert_ids_in_graph})
		)
	drug_synonyms.drop_duplicates(inplace=True)
	print drug_synonyms.shape
	return drug_synonyms


def load_signature_meta_from_db(collection_name, query={}, drug_meta_df=None):
	projection = {
		'_id': False,
		'sig_id': True,
		'pert_id': True,
		'cell_id': True,
		'pert_dose': True,
		'pert_time': True,
		'SCS_centered_by_batch':True,
		# sigs_pert_cell and sigs_pert
		'avg_pvalue': True,
		'n_signatures_aggregated': True,
		'avg_time': True,
		'avg_dose': True,
		'mean_cosine_dist': True,
		'n_cells': True,
		}
	coll = mongo.db[collection_name]
	cur = coll.find(query, projection)
	print cur.count()
	meta_df = pd.DataFrame.from_records([doc for doc in cur]).set_index('sig_id')
	if collection_name == 'sigs_pert':
		meta_df['pert_id'] = meta_df.index

	meta_df = meta_df.rename(index=str, columns={'cell_id':'cell','pert_dose':'dose'})
	if drug_meta_df is not None:
		meta_df = meta_df.merge(drug_meta_df, 
			left_on='pert_id', 
			right_index=True,
			how='left'
			)
	meta_df.fillna('unknown', inplace=True)
	meta_df.replace(['unannotated', '-666'], 'unknown', inplace=True)
	print meta_df.shape
	return meta_df


def _minmax_scaling(arr):
	scl = MinMaxScaler((-10, 10))
	arr = scl.fit_transform(arr.reshape(-1, 1))
	return arr[:, 0]

from sklearn import cluster
def _cluster(x, y, clstr):
	data = np.zeros((len(x), 2))
	data[:, 0] = x
	data[:, 1] = y
	clstr.fit(data)
	return clstr.labels_


def load_graph_from_db(graph_name, drug_meta_df=None):
	# Find the graph by name
	graph_doc = mongo.db.graphs.find_one({'name': graph_name}, {'_id':False})
	graph_df = pd.DataFrame({
		'sig_ids': graph_doc['sig_ids'],
		'x': graph_doc['x'],
		'y': graph_doc['y'],
		}).set_index('sig_ids')
	graph_df.index.name = 'sig_id'
	# Scale the x, y 
	graph_df['x'] = _minmax_scaling(graph_df['x'].values)
	graph_df['y'] = _minmax_scaling(graph_df['y'].values)
	# graph_df['DBSCAN'] = _cluster(graph_df['x'].values, graph_df['y'].values,
	# 	clstr = cluster.DBSCAN(min_samples=15, eps=0.35))
	graph_df['DBSCAN'] = _cluster(graph_df['x'].values, graph_df['y'].values,
		clstr = cluster.DBSCAN(min_samples=15, eps=0.33))
	graph_df['KMeans'] = _cluster(graph_df['x'].values, graph_df['y'].values, 
		cluster.KMeans(n_clusters=30))
	# graph_df['SpectralClustering'] = _cluster(graph_df['x'].values, graph_df['y'].values, 
	# 	cluster.SpectralClustering(n_clusters=20))
	# graph_df['Birch'] = _cluster(graph_df['x'].values, graph_df['y'].values, 
	# 	cluster.Birch(n_clusters=20))


	# Load the corresponding meta_df
	meta_df = load_signature_meta_from_db(graph_doc['coll'], 
		query={'sig_id': {'$in': graph_df.index.tolist()}},
		drug_meta_df=drug_meta_df
		)

	graph_df = graph_df.merge(meta_df, how='left', left_index=True, right_index=True)
	# Check form of sig_id
	if len(graph_df.index[0].split(':')) == 3:
		graph_df['Batch'] = graph_df.index.map(lambda x:x.split('_')[0])
	# graph_df['pert_id'] = graph_df.index.map(lambda x:x.split(':')[1])

	graph_df.rename(
		index=str, 
		columns={
			'SCS_centered_by_batch': 'p-value', 'cell': 'Cell', 'pert_time': 'Time', 
			'drug_class': 'Drug class', 'dose': 'Dose',
			'pert_desc': 'Perturbation',
			'pert_id': 'Perturbation_ID',
			'most_frequent_rx': 'EHR_Coprescribed_Drugs',
			'most_frequent_dx': 'EHR_Diagnoses',
			},
		inplace=True)

	return graph_df, meta_df


def get_all_sig_ids_from_graphs():
	cur = mongo.db.graphs.find(
		{'$and': [
			{'coll': 'sigs'}, 
			{'name': {'$ne': 'Signature_Graph_17041nodes_0.56_ERSC.cyjs'}}
		]},
		{'_id':False, 'sig_ids':True})
	sig_ids = [doc['sig_ids'] for doc in cur]
	sig_ids = reduce(lambda x, y: x+y, sig_ids)
	sig_ids = sorted(set(sig_ids))	
	return sig_ids

def get_download_meta():
	files = [
	('Adjacency_matrix_LM_space_42809x42809.gctx', 'Adjacency matrix of all significant CD signatures. Cosine similarity is used to compute the signature similarity.'),
	('CD_signatures_LM_42809x978.gctx', 'CD signature matrix in the Landmark gene space.'), 
	('CD_signatures_full_42809x22268.gctx', 'CD signature matrix in the full space (22268 probes).'), 
	('CD_signature_metadata.csv', 'Metadata of the signatures indexed by the `sig_id`.'), 
	('Probes_L1000_metadata.csv', 'Metadata of the probes of the Landmark genes.'),
	('Probes_full_metadata.csv', 'Metadata of the probes used in the full space.'), 	
	]
	def sizeof_fmt(num, suffix='B'):
		for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
			if abs(num) < 1024.0:
				return "%3.1f%s%s" % (num, unit, suffix)
			num /= 1024.0
		return "%.1f%s%s" % (num, 'Yi', suffix)


	files_meta = []
	for filename, desc in files:
		size = os.path.getsize(os.path.join('data/download', filename))
		files_meta.append({
			'name': filename,
			'desc': desc,
			'size': sizeof_fmt(size)
			})

	return files_meta

### ORMs for user imput
class EnrichmentResult(object):
	"""EnrichmentResult: object for documents in the userResults collection"""
	projection = {'_id':0}
	default_score = 0.

	def __init__(self, rid):
		'''To retrieve a result using _id'''
		self.rid = ObjectId(rid)
		doc = mongo.db.userResults.find_one({'_id': self.rid}, self.projection)
		self.data = doc['data']
		self.result = doc['result']
		self.type = doc['type']
		self.graph_name = doc['graph_name']

	def bind_to_graph(self, df):
		'''Bind the enrichment results to the graph df'''
		d_sig_id_score = dict(zip(df.index, self.result['scores']))
		df['Scores'] = [d_sig_id_score[sig_id] for sig_id in df.index]
		return df

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


class UserSubset(object):
	"""Metadata (pert_ids, cells, times) from users to subset signatures.
	"""
	def __init__(self, data):
		self.data = data
	
	def save(self):
		res = mongo.db.userSubset.insert_one({
			'data': self.data
			})
		self.rid = res.inserted_id # <class 'bson.objectid.ObjectId'>
		return str(self.rid)

	def subset_graph(self, graph_df):
		'''Based on self's metadata to subset a graph_df
		'''
		mask = graph_df['Perturbation_ID'].isin(self.data['pert_ids']) &\
			graph_df['Cell'].isin(self.data['cells']) &\
			graph_df['Time'].isin(self.data['times'])
		graph_df_sub = graph_df.loc[mask]
		# Scale the x, y 
		graph_df_sub['x'] = _minmax_scaling(graph_df_sub['x'].values)
		graph_df_sub['y'] = _minmax_scaling(graph_df_sub['y'].values)
		return graph_df_sub

	@classmethod
	def get(cls, rid):
		rid = ObjectId(rid)
		data = mongo.db.userSubset.find_one({'_id': rid}, {'_id':0})
		return cls(data['data'])


class Signature(object):
	"""Signature object in the MongoDB."""

	projection = {
		'_id': False,
		'avg_center_LM': False,
		'CD_nocenter_LM': False,
		'avg_center_LM_det': False,
		'CDavg_center_LM_det': False,
		'CDavg_nocenter_LM_det': False,
		'CD_center_Full': False,
		'pvalues_Full':False,
		'sigIdx': False,
	}

	dtypes = {
		'pert_time': int,
	}

	collections = {
		# map for finding which collection to look for the signature
		3: ('sigs', 'SCS_centered_by_batch'),
		2: ('sigs_pert_cell', 'avg_pvalue'),
		1: ('sigs_pert', 'avg_pvalue'),
	}

	def __init__(self, sig_id, mongo):
		self.sig_id = sig_id
		coll_name, pvalue = self.collections[len(sig_id.split(':'))]

		doc = mongo.db[coll_name].find_one({'sig_id':sig_id}, self.projection)
		if coll_name == 'sigs_pert':
			doc['pert_id'] = sig_id

		for key, value in doc.items():
			value = self.dtypes.get(key, lambda x:x)(value)
			if key == pvalue:
				setattr(self, 'pvalue', value)
			else:
				setattr(self, key, value)

		# Get combined_genes
		self.combined_genes = list(set(doc.get('upGenes', [])) | set(doc.get('dnGenes', [])))

	def __repr__(self):
		return '<Signature %r>' % self.sig_id

	def json_data(self):
		data = self.__dict__
		data['up_genes'] = data.pop('upGenes')
		data['down_genes'] = data.pop('dnGenes')
		data.pop('CD_center_LM')
		data.pop('CD_center_LM_det')
		return json.dumps(data)
