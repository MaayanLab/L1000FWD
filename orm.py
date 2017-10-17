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

def load_graph(cyjs_filename, meta_df):
	'''Load graph node coordinates and other metadata from a cyjs file,
	then return a df indexed by sig_id'''
	json_data = json.load(open('notebooks/%s' % cyjs_filename, 'rb'))
	json_data = json_data['elements']['nodes']

	scl = MinMaxScaler((-10, 10))

	coords = np.array([
		[rec['position']['x'] for rec in json_data], 
		[rec['position']['y'] for rec in json_data]
		]).T
	coords = scl.fit_transform(coords)

	df = pd.DataFrame({
		'sig_id': [rec['data']['name'] for rec in json_data],
		'x': coords[:, 0],
		'y': coords[:, 1],
		}).set_index('sig_id')
	df['z'] = 0

	df = df.merge(meta_df, how='left', left_index=True, right_index=True)
	df = df.sort_index()
	return df


def load_graphs_meta():
	'''Load and preprocess the meta for graphs in the `graphs` collection.
	'''
	graph_names = mongo.db.graphs.distinct('name')
	graphs = {
		'cells':[
			{'name': 'Signature_Graph_CD_center_LM_sig-only_16848nodes.gml.cyjs', 'display_name': 'All cell lines'}
		],
		'agg': [
			{'name': 'graph_pert_cell_12894nodes_99.9.gml.cyjs', 'display_name': 'Aggregated by drugs and cells'},
			{'name': 'kNN_5_layout', 'display_name': 'Aggregated by drugs (kNN)'},
			{'name': 'threshold_99.5', 'display_name': 'Aggregated by drugs (thresholding)'},
		],
	}

	# process graphs for individual cells
	for graph_name in graph_names:
		if graph_name.endswith('-tSNE_layout.csv'):
			cell = graph_name.split('-')[0]
			rec = {
				'name': graph_name, 
				'display_name': '%s (tSNE)' % cell
				}
			graphs['cells'].append(rec)

		elif graph_name.endswith('kNN_5'):
			cell = graph_name.split('_')[0]
			rec = {
				'name': graph_name,
				'display_name': '%s (kNN)' % cell
			}
			graphs['cells'].append(rec)
	return graphs

def load_drug_meta_from_db():
	drug_meta_df = pd.read_sql_query('''
		SELECT drug_repurposedb.pert_id, drug_repurposedb.pert_iname AS pert_desc,
		most_frequent_dx_rx.most_frequent_rx, most_frequent_dx_rx.most_frequent_dx, 
		drug_repurposedb.Phase, drug_repurposedb.MOA
		FROM most_frequent_dx_rx
		RIGHT JOIN drug_repurposedb
		ON drug_repurposedb.pert_id=most_frequent_dx_rx.pert_id
		''', engine, 
		index_col='pert_id')
	print drug_meta_df.shape

	# Borrow the info from pert_ids with the same pert_desc
	for col in ['most_frequent_rx','most_frequent_dx','Phase', 'MOA']:
		d = drug_meta_df[['pert_desc', col]].dropna(axis=0)
		d = dict(zip(d['pert_desc'], d[col]))
		drug_meta_df[col] = [d.get(name, None) for name in drug_meta_df['pert_desc']]

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
			'drug_class': 'Drug class', 'pert_dose': 'Dose',
			'pert_desc': 'Perturbation'},
		inplace=True)

	return graph_df, meta_df




### ORMs for user imput
class EnrichmentResult(object):
	"""EnrichmentResult: object for documents in the userResults collection"""
	projection = {'_id':0}
	default_score = 0.

	def __init__(self, rid):
		'''To retrieve a result using _id'''
		self.rid = ObjectId(rid)
		doc = COLL_RES.find_one({'_id': self.rid}, self.projection)
		self.data = doc['data']
		self.result = doc['result']
		self.type = doc['type']

	def bind_to_graph(self, df):
		'''Bind the enrichment results to the graph df'''
		df['scores'] = self.result['scores']
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

	def enrich(self):
		'''POST to Rook API to get enriched LJP signatures'''
		self.config['method'] = self.type
		payload = dict(self.data.items() + self.config.items())
		response = requests.post(RURL, data=json.dumps(payload),headers=self.headers)
		result = pd.DataFrame(response.json())
		# Get the top N as list of records:
		topn = {
			'similar': result.iloc[:50].to_dict(orient='records'),
			'opposite': result.iloc[-50:][::-1].to_dict(orient='records'),
			}
		# Sort scores by sig_ids to ensure consistency with the graph
		result.sort_values(by='sig_ids', inplace=True, ascending=True)
		self.result = {
			'scores': result['scores'].tolist(), 
			'topn': topn
			}
		return self.result

	def save(self):
		'''Save the UserInput as well as the EnrichmentResult to a document'''
		res = COLL_RES.insert_one({
			'result': self.result, 
			'data': self.data, 
			'type': self.type,
			})
		self.rid = res.inserted_id # <class 'bson.objectid.ObjectId'>
		return str(self.rid)

	def bind_enrichment_to_graph(self, net):
		'''Bind the enrichment results to the graph df'''
		df['scores'] = self.result['scores']
		return df


class GeneSets(UserInput):
	"""docstring for GeneSets"""
	def __init__(self, up_genes, dn_genes):
		data = {'upGenes': up_genes, 'dnGenes': dn_genes}
		UserInput.__init__(self, data)
		self.type = 'geneSet'

	def json_data(self):
		'''Return an object to be encoded to json format'''
		return self.data



