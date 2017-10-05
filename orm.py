import os
import json, requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import create_engine
from bson.objectid import ObjectId
from pymongo import MongoClient

RURL = os.environ['RURL']
MONGOURI = os.environ['MONGOURI']
client = MongoClient(MONGOURI)
DB = client['DMOA']
COLL_RES = DB['userResults']
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
	# df.drop('pert_id', axis=1, inplace=True)
	# df['neglogp'] = -np.log10(df['pvalue']+1e-4)

	# df = encode_rare_categories(df, 'cell')
	# df = encode_rare_categories(df, 'perturbation')

	df = df.sort_index()
	return df


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



