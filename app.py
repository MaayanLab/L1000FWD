import os, sys
import json
import time
import StringIO
import numpy as np
np.random.seed(10)
import requests
import pandas as pd

import re
from werkzeug.routing import Rule, RequestRedirect

class CIRule(Rule):
	def compile(self):
		Rule.compile(self)
		self._regex = re.compile(self._regex.pattern, 
			re.UNICODE | re.IGNORECASE)


from flask import Flask, request, redirect, render_template, \
	jsonify, send_from_directory, abort, Response, send_file

class CIFlask(Flask):
    url_rule_class = CIRule


from orm import *
# from subset_signatures import *
from crossdomain import crossdomain

ENTER_POINT = os.environ['ENTER_POINT']
SCRIPT_PATH = os 
app = CIFlask(__name__, static_url_path=ENTER_POINT, static_folder=os.getcwd())
app.debug = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 6
app.config['MONGO_URI'] = MONGOURI

mongo.init_app(app)

@app.before_first_request
def load_globals():
	global meta_df, graph_df, drug_synonyms, drug_meta_df
	global graphs # meta data of the graphs for the header
	global all_sig_ids
	global d_all_graphs # preload all graphs
	global graph_name_full
	global creeds_meta_df
	global api_docs
	global download_files_meta
	# global meta_df_full
	graph_name_full = 'Signature_Graph_CD_center_LM_sig-only_16848nodes.gml.cyjs'
	graphs = load_graphs_meta()

	drug_meta_df = load_drug_meta_from_db()

	graph_df, meta_df = load_graph_from_db(graph_name_full,
		drug_meta_df=drug_meta_df)
	print meta_df.shape

	drug_synonyms = load_drug_synonyms_from_db(meta_df, drug_meta_df)

	all_sig_ids = get_all_sig_ids_from_graphs()

	# Load all the graphs
	d_all_graphs = {}
	for graph_rec in graphs['cells']:# + graphs['agg']:
		graph_name = graph_rec['name']
		graph_df_, _ = load_graph_from_db(graph_name, drug_meta_df=drug_meta_df)
		d_all_graphs[graph_name] = graph_df_

	# # The full metadata for all_sig_ids, then filter out signatures from poscons
	# meta_df_full = load_signature_meta_from_db('sigs', 
	# 	query={'sig_id': {'$in': all_sig_ids}},
	# 	drug_meta_df=drug_meta_df		
	# 	)
	# meta_df_full.rename(
	# 	index=str, 
	# 	columns={
	# 		'SCS_centered_by_batch': 'p-value', 'cell': 'Cell', 'pert_time': 'Time', 
	# 		'drug_class': 'Drug class', 'dose': 'Dose',
	# 		'pert_desc': 'Perturbation',
	# 		'pert_id': 'Perturbation_ID',
	# 		'most_frequent_rx': 'EHR_Coprescribed_Drugs',
	# 		'most_frequent_dx': 'EHR_Diagnoses',
	# 		},
	# 	inplace=True)
	# # meta_df_full = meta_df_full.groupby('Perturbation_ID')['p-value'].apply(lambda x: x.sort_values(ascending=True).head(20))
	# # meta_df_full = meta_df_full.reset_index().set_index('sig_id')
	# print meta_df_full.shape

	creeds_meta_df = pd.read_csv('data/CREEDS_meta.csv').set_index('id')
	api_docs = json.load(open('api_docs.json', 'rb'))
	download_files_meta = get_download_meta()
	return

@app.route(ENTER_POINT + '/')
def index_page():
	return render_template('index.html', 
		ENTER_POINT=ENTER_POINT)

@app.route(ENTER_POINT + '/main')
def main_page():
	# The default main page
	sdvConfig = {
		'colorKey': 'MOA',
		'shapeKey': 'Time',
		'labelKey': ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time', 'Phase', 'MOA', 'predicted_MOA'],
	}

	return render_template('scatter.html', 
		script='main',
		ENTER_POINT=ENTER_POINT,
		result_id='hello',
		graphs=graphs,
		graph_name=graph_name_full,
		sdvConfig=json.dumps(sdvConfig),
		)

@app.route(ENTER_POINT + '/graph_page/<string:graph_name>')
def graph_page(graph_name):
	url = 'graph/%s' % graph_name
	# defaults
	sdvConfig = {
		'colorKey': 'MOA',
		'shapeKey': 'Time',
		'labelKey': ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time', 'Phase', 'MOA', 'predicted_MOA'],
	}

	if graph_name == 'graph_pert_cell_12894nodes_99.9.gml.cyjs': 
		# Signatures aggregated for pert_id x cell_id
		sdvConfig['colorKey'] = 'Cell'
		sdvConfig['shapeKey'] = 'avg_time'
		sdvConfig['labelKey'] = ['Perturbation', 'Cell', 'avg_dose', 'avg_time', 
			'Phase', 'MOA', 'n_signatures_aggregated']
	elif graph_name in ('kNN_5_layout', 'threshold_99.5'):
		# Signatures aggregated for pert_id
		sdvConfig['shapeKey'] = 'avg_pvalue'
		sdvConfig['labelKey'] = ['Perturbation', 'avg_dose', 'avg_time', 'avg_pvalue', 
					'Phase', 'MOA', 'n_signatures_aggregated']
	return render_template('scatter.html', 
		script='main',
		ENTER_POINT=ENTER_POINT,
		result_id='hello',
		graphs=graphs,
		graph_name=graph_name,
		sdvConfig=json.dumps(sdvConfig),
		)


## /subset endpoints and pages
@app.route(ENTER_POINT + '/subset_page', methods=['GET'])
def send_subset_input_page():
	return render_template('subset.html',
		ENTER_POINT=ENTER_POINT,
		graphs=graphs,

		)

@app.route(ENTER_POINT + '/subset', methods=['POST'])
def create_graph_from_user_subset():
	if request.method == 'POST':

		post_data = request.get_json()
		times = post_data.get('times', None)
		pert_ids = post_data.get('pert_ids', None)
		cells = post_data.get('cells', None)
		if times is None:
			times = [6, 24, 48]
		if cells is None:
			cells = graph_df['Cell'].unique().tolist()

		user_subset = UserSubset({
			'pert_ids': pert_ids,
			'cells': cells,
			'times': times,
			})
		graph_df_sub = user_subset.subset_graph(d_all_graphs)
		# sig_ids_sub = get_subset_sig_ids(user_subset)
		if graph_df_sub.shape[0] == 1:
		# if len(sig_ids_sub) == 1:
			# Redirect to signature page 
			sig_id = graph_df_sub.index[0]
			# sig_id = sig_ids_sub[0]
			return json.dumps({
				'url': 'http://amp.pharm.mssm.edu/dmoa/sig/%s'%sig_id,
				'absolute': True
				})
		# elif len(sig_ids_sub) == 0:
		elif graph_df_sub.shape[0] == 0:
			# No results found
			return json.dumps({'error': 'No signatures can be found for the subset criteria.'})
		else:
			rid = user_subset.save()
			print rid
			return json.dumps({'url':'/subset/' + rid, 'absolute': False})

@app.route(ENTER_POINT + '/subset/<string:subset_id>', methods=['GET'])
def send_subset_result_page(subset_id):
	sdvConfig = {
		'colorKey': 'MOA',
		'shapeKey': 'Time',
		'labelKey': ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time', 'Phase', 'MOA', 'predicted_MOA'],
	}

	return render_template('scatter.html', 
		ENTER_POINT=ENTER_POINT,
		subset_id=subset_id,
		script='subset',
		sdvConfig=json.dumps(sdvConfig),
		graphs=graphs,
		graph_name=graph_name_full
		)

@app.route(ENTER_POINT + '/subset/graph/<string:subset_id>', methods=['GET'])
def retrieve_subset_id_and_subset_graph(subset_id):
	user_subset = UserSubset.get(subset_id)
	graph_df_sub = user_subset.subset_graph(d_all_graphs)
	# coords_df = subset_mat_and_do_tsne(user_subset)
	# graph_df_sub = meta_df_full.merge(coords_df, 
	# 	left_index=True,
	# 	right_index=True,
	# 	how='inner'
	# 	)

	# This somehow works
	time.sleep(1)
	return graph_df_sub.reset_index().to_json(orient='records')


@app.route('/<path:filename>')
def serve_static_file(filename):
	'''Serve static files.
	'''
	return send_from_directory(app.static_folder, filename)

@app.route(ENTER_POINT + '/search_drug')
def search_drug():
	'''Redirect to drug search page.
	'''
	return redirect('http://amp.pharm.mssm.edu/dmoa/search_drug', code=302)	


@app.route(ENTER_POINT + '/graph/<string:graph_name>', methods=['GET'])
def load_graph_layout_coords(graph_name):
	'''API for different graphs'''
	if request.method == 'GET':
		if graph_name == 'full':
			print graph_df.shape
			return graph_df.reset_index().to_json(orient='records')
		else:
			# graph_df_, meta_df_ = load_graph_from_db(graph_name, drug_meta_df)
			# print graph_df_.head()
			graph_df_ = d_all_graphs[graph_name]
			return graph_df_.reset_index().to_json(orient='records')


@app.route(ENTER_POINT + '/sig_ids', methods=['GET'])
def get_all_sig_ids():
	'''Get all sig_ids across the all available graphs. 
	This endpoint is for sigine to precompute the signature matrix for the search engine.
	'''
	if request.method == 'GET':
		return json.dumps({'sig_ids': all_sig_ids, 'n_sig_ids': len(all_sig_ids)})

@app.route(ENTER_POINT + '/sig/<string:sig_id>', methods=['GET'])
@crossdomain(origin='*')
def get_sig_by_id(sig_id):
	sig = Signature(sig_id, mongo)
	return sig.json_data()


@app.route(ENTER_POINT + '/sig_search', methods=['POST'])
@crossdomain(origin='*')
def post_to_sigine_api():
	'''Endpoint handling signature similarity search, POST the up/down genes 
	to the RURL and return the result_id.'''
	if request.method == 'POST':
		data = json.loads(request.data)
		up_genes = data['up_genes']
		down_genes = data['down_genes']
		# init GeneSets instance
		gene_sets = GeneSets(up_genes, down_genes)
		# perform similarity search
		graph_df_ = d_all_graphs[graph_name_full]
		result = gene_sets.enrich(graph_df_, graph_name_full)
		# save gene_sets and results to MongoDB
		rid = gene_sets.save()
		return json.dumps({'result_id': rid})


@app.route(ENTER_POINT + '/search/<string:graph_name>', methods=['POST'])
def post_to_sigine(graph_name):
	'''Endpoint handling signature similarity search, POST the up/down genes 
	to the RURL and redirect to the result page.'''
	if request.method == 'POST':
		# retrieve data from the form 
		up_genes = request.form.get('upGenes', '').split()
		down_genes = request.form.get('dnGenes', '').split()
		# init GeneSets instance
		gene_sets = GeneSets(up_genes, down_genes)
		# perform similarity search
		graph_df_ = d_all_graphs[graph_name]
		result = gene_sets.enrich(graph_df_, graph_name)
		# save gene_sets and results to MongoDB
		rid = gene_sets.save()
		print rid
		return redirect(ENTER_POINT + '/result/' + rid, code=302)


@app.route(ENTER_POINT + '/synonyms/<string:query_string>', methods=['GET'])
@crossdomain(origin='*')
def search_drug_by_synonyms(query_string):
	'''Endpoint handling synonym search for drugs in the graph.
	'''
	if request.method == 'GET':
		mask = drug_synonyms['Name'].str.contains(query_string, case=False)
		return drug_synonyms.loc[mask].to_json(orient='records') 

@app.route(ENTER_POINT + '/synonyms_by_graph/<string:graph_name>/<string:query_string>', methods=['GET'])
@crossdomain(origin='*')
def search_drug_by_synonyms_on_graph(graph_name, query_string):
	'''Endpoint handling synonym search for drugs in the graph.
	'''
	if request.method == 'GET':
		graph_df_ = d_all_graphs[graph_name]
		pert_ids_in_this_graph = graph_df_['Perturbation_ID'].unique()
		drug_synonyms_sub = drug_synonyms.loc[drug_synonyms['pert_id'].isin(pert_ids_in_this_graph)]
		mask = drug_synonyms_sub['Name'].str.contains(query_string, case=False)
		return drug_synonyms_sub.loc[mask].to_json(orient='records') 


@app.route(ENTER_POINT + '/cells', methods=['GET'])
def get_cell_lines():
	'''Get a list of cell line names in the default graph.
	'''
	cells = graph_df['Cell'].unique().tolist()
	cells = [{'name': cell, 'value': cell} for cell in cells]
	return json.dumps(cells)	

## /CREEDS/ endpoints

@app.route(ENTER_POINT + '/CREEDS/search/<string:query_string>', methods=['GET'])
def search_signature_from_creeds(query_string):
	'''Handles searching signatures in CREEDS.
	'''
	mask = creeds_meta_df['name'].str.contains(query_string, flags=re.IGNORECASE)
	return creeds_meta_df.loc[mask].reset_index().to_json(orient='records')

@app.route(ENTER_POINT + '/CREEDS/genes/<string:creeds_id>', methods=['GET'])
def retrieve_creeds_genes_by_id(creeds_id):
	CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
	response = requests.get(CREEDS_URL + 'api', params={'id': creeds_id})
	signature = response.json()
	gene_sets = {
		'upGenes': map(lambda x:x[0], signature['up_genes']), 
		'dnGenes': map(lambda x:x[0], signature['down_genes']),
		}
	return json.dumps(gene_sets)

## /result/ endpoints

@app.route(ENTER_POINT + '/result/graph/<string:result_id>', methods=['GET'])
def result(result_id):
	"""
	Retrieve a simiarity search result using id and combine it
	with graph layout.
	"""
	# retrieve enrichment results from db
	result_obj = EnrichmentResult(result_id)
	# bind enrichment result to the network layout
	graph_df_res = result_obj.bind_to_graph(d_all_graphs[result_obj.graph_name])

	return graph_df_res.reset_index().to_json(orient='records')


@app.route(ENTER_POINT + '/result/genes/<string:result_id>', methods=['GET'])
def result_genes(result_id):
	'''Retrieve user input genes.
	'''
	# retrieve enrichment results from db
	result_obj = EnrichmentResult(result_id)
	return jsonify(result_obj.data)


@app.route(ENTER_POINT + '/result/topn/<string:result_id>', methods=['GET'])
@crossdomain(origin='*')
def result_topn(result_id):
	'''Retrieve topn signatures to visualize in a table.
	'''
	result_obj = EnrichmentResult(result_id)
	return json.dumps(result_obj.result['topn'])

# @app.route(ENTER_POINT + '/result/graph_name/<string:result_id>', methods=['GET'])
# def get_result_graph_name(result_id):
# 	result_obj = EnrichmentResult(result_id)
# 	return json.dumps({'graph_name': result_obj.graph_name})

@app.route(ENTER_POINT + '/result/modal/<string:result_id>', methods=['GET'])
def result_modal(result_id):
	'''Template for the signature similarity search result modal.
	'''
	result_obj = EnrichmentResult(result_id)
	# add pert_id and perturbation to topn for the modal to render
	n = len(result_obj.result['topn']['similar'])
	topn = {'similar': [None]*n, 'opposite': [None]*n}
	graph_df_ = d_all_graphs[result_obj.graph_name]
	for i in range(n):
		rec = result_obj.result['topn']['similar'][i]
		sig_id = rec['sig_id']
		rec['pert_id'] = graph_df_.ix[sig_id]['Perturbation_ID']
		rec['perturbation'] = graph_df_.ix[sig_id]['Perturbation']
		topn['similar'][i] = rec
		
		rec = result_obj.result['topn']['opposite'][i]
		sig_id = rec['sig_id']
		rec['pert_id'] = graph_df_.ix[sig_id]['Perturbation_ID']
		rec['perturbation'] = graph_df_.ix[sig_id]['Perturbation']
		topn['opposite'][i] = rec

	return render_template('result-modal.html', 
		topn=topn,
		result_id=result_id)


@app.route(ENTER_POINT + '/result/download/<string:result_id>', methods=['GET'])
def result_download(result_id):
	'''To download the results to a csv file.
	'''
	result_obj = EnrichmentResult(result_id)
	# Prepare a DataFrame for the result
	scores = result_obj.result['scores']
	graph_df_ = d_all_graphs[result_obj.graph_name]
	result_df = pd.DataFrame({'similarity_scores': scores, 
		'drug': graph_df_['Perturbation'],
		'pert_id': graph_df_.index.map(lambda x:x.split(':')[1]),
		}, index=graph_df_.index)\
		.sort_values('similarity_scores', ascending=False)
	# Write into memory
	s = StringIO.StringIO()
	result_df.to_csv(s)
	# Prepare response
	resp = Response(s.getvalue(), mimetype='text/csv')
	resp.headers['Content-Disposition'] = 'attachment; filename=similarity_search_result-%s.csv' \
		% result_id
	return resp


@app.route(ENTER_POINT + '/result/<string:result_id>', methods=['GET'])
def result_page(result_id):
	'''The result page.
	'''
	sdvConfig = {
		'colorKey': 'Scores',
		'shapeKey': 'Time',
		'labelKey': ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time', 'Phase', 'MOA', 'predicted_MOA'],
	}
	result_obj = EnrichmentResult(result_id)
	return render_template('scatter.html', 
		script='result', 
		ENTER_POINT=ENTER_POINT,
		result_id=result_id,
		graphs=graphs,
		graph_name=result_obj.graph_name,
		sdvConfig=json.dumps(sdvConfig),
		)


@app.route(ENTER_POINT + '/download_page', methods=['GET'])
def download_page():
	return render_template('download.html', 
		ENTER_POINT=ENTER_POINT,
		graphs=graphs,
		download_files_meta=download_files_meta
		)


@app.route(ENTER_POINT + '/download/<path:filename>')
def download_file(filename):
	'''Serve download files.
	'''
	download_file_path = os.path.join(app.static_folder, 'data/download')
	return send_from_directory(download_file_path, filename)

@app.route(ENTER_POINT + '/download_graph/<string:name>')
def download_graph_by_name(name):
	temp_file = StringIO.StringIO()
	d_all_graphs[name]\
		[['x','y','Cell','Dose','Time','Perturbation_ID',
		'Perturbation','MOA','Phase','Batch','p-value']]\
		.to_csv(temp_file, encoding='utf-8')
	temp_file.seek(0)
	return send_file(temp_file,
		attachment_filename="%s.csv"%name.split('.')[0],
		as_attachment=True)

@app.route(ENTER_POINT + '/api_page', methods=['GET'])
def api_doc_page():
	return render_template('apis.html', 
		ENTER_POINT=ENTER_POINT,
		graphs=graphs,
		api_docs=api_docs
		)

from jinja2 import Markup
app.jinja_env.globals['include_raw'] = lambda filename : Markup(app.jinja_loader.get_source(app.jinja_env, filename)[0])


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, threaded=True)


