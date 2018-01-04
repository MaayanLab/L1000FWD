'''
Insert network layouts in cyjs and csv files to MongoDB

'''
import os, json
from pymongo import MongoClient
from graph_utils import *

def csv_file_to_doc(fn):
	df = pd.read_csv(fn)
	
	doc = {
		'sig_ids': df['sig_id'].tolist(),
		'x': df['x'].tolist(),
		'y': df['y'].tolist(),
		'name': fn
	}
	return doc


def cyjs_file_to_doc(fn):
	json_data = json.load(open(fn, 'rb'))
	json_data = json_data['elements']['nodes']

	doc = {
		'sig_ids': [rec['data']['name'] for rec in json_data],
		'x': [rec['position']['x'] for rec in json_data],
		'y': [rec['position']['y'] for rec in json_data],
		'name': fn
		}
	return doc


## 0. Set up DB conn
client = MongoClient('mongodb://146.203.54.131:27017/')
coll_graphs = client['L1000FWD']['graphs']


## 1. Work on cyjs
# cyjs_files = [
# 	'Signature_Graph_17041nodes_0.56_ERSC.cyjs',
# 	'Signature_Graph_CD_center_LM_sig-only_16848nodes.gml.cyjs',
# ]

# for cyjs_file in cyjs_files:
# 	doc = cyjs_file_to_doc(cyjs_file)
# 	doc['coll'] = 'sigs'
# 	coll_graphs.insert_one(doc)

# doc = cyjs_file_to_doc('graph_pert_cell_12894nodes_99.9.gml.cyjs')
# doc['coll'] = 'sigs_pert_cell'
# coll_graphs.insert_one(doc)

## 2. work on csv files
os.chdir('graphs_for_cells')

for fn in os.listdir(os.getcwd()):
	print fn
	doc = csv_file_to_doc(fn)
	doc['coll'] = 'sigs'
	coll_graphs.insert_one(doc)





