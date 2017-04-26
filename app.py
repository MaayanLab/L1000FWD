import os, sys
import json
import StringIO
import numpy as np
np.random.seed(10)
import requests
import pandas as pd

from flask import Flask, request, redirect, render_template, \
	jsonify, send_from_directory, abort, Response

from orm import *

ENTER_POINT = os.environ['ENTER_POINT']
app = Flask(__name__, static_url_path=ENTER_POINT, static_folder=os.getcwd())
app.debug = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 6


@app.before_first_request
def load_globals():
	global meta_df, N_SIGS, graph_df
	# meta_df = pd.read_csv('data/metadata.tsv', sep='\t')
	# meta_df = pd.read_csv('data/metadata-sig-only.tsv', sep='\t')
	meta_df = pd.read_csv('data/metadata-full-anno.tsv', sep='\t')
	meta_df = meta_df.set_index('sig_id')
	print meta_df.shape
	N_SIGS = meta_df.shape[0]

	cyjs_filename = os.environ['CYJS']
	graph_df = load_graph(cyjs_filename, meta_df)
	return


@app.route(ENTER_POINT + '/')
def index_page():
	return render_template('index.html', 
		script='main',
		ENTER_POINT=ENTER_POINT,
		result_id='hello')

@app.route('/<path:filename>')
def send_file(filename):
	'''Serve static files.
	'''
	return send_from_directory(app.static_folder, filename)

@app.route(ENTER_POINT + '/toy', methods=['GET'])
def toy_data():
	if request.method == 'GET':
		n = int(request.args.get('n', 10))
		rand_idx = np.random.choice(range(N_SIGS), n, replace=False)

		rand_coords = np.random.randn(n, 3)
		df = meta_df.iloc[rand_idx]
		df = df.assign(x=rand_coords[:,0], y=rand_coords[:,1], z=rand_coords[:,2])

		return df.to_json(orient='records')
		# return jsonify(df.to_dict(orient='list'))

'''
@app.route(ENTER_POINT + '/pca', methods=['GET'])
def load_pca_coords():
	if request.method == 'GET':
		# coords = np.load('data/pca_coords.npy')
		# coords = np.load('data/zscored_pca_coords.npy')
		coords = np.load('data/pca_coords-sig-only.npy')
		scl = MinMaxScaler((-10, 10))
		# scl.fit(coords[:, 0].reshape(-1, 1))
		# for j in range(coords.shape[1]):
		# 	coords[:, j] = scl.transform(coords[:,j].reshape(-1, 1))[:,0]
		coords = scl.fit_transform(coords)

		print 'coords shape:', coords.shape
		print 'meta_df.shape', meta_df.shape
		df = meta_df.assign(x=coords[:,0], y=coords[:,1], z=coords[:,2])
		df['neglogp'] = -np.log10(df['pvalue'])
		df['z'] = 0
		frequent_cells = set(df['cell'].value_counts()[:19].index)
		def encode_rare_cell(cell):
			if cell in frequent_cells:
				return cell
			else:
				return 'rare_cell'
		df['cell'] = df['cell'].apply(encode_rare_cell)
		print 'df.shape: ', df.shape
	return df.to_json(orient='records')
'''

@app.route(ENTER_POINT + '/graph', methods=['GET'])
def load_graph_layout_coords():
	if request.method == 'GET':
		print graph_df.shape
		return graph_df.reset_index().to_json(orient='records')

@app.route(ENTER_POINT + '/sig_ids', methods=['GET'])
def get_all_sig_ids():
	if request.method == 'GET':
		cyjs_filename = os.environ['CYJS']
		json_data = json.load(open('notebooks/%s' % cyjs_filename, 'rb'))
		json_data = json_data['elements']['nodes']
		sig_ids = [rec['data']['name'] for rec in json_data]
		return json.dumps({'sig_ids': sig_ids, 'n_sig_ids': len(sig_ids)})

@app.route(ENTER_POINT + '/search', methods=['POST'])
def post_to_sigine():
	'''Endpoint handling signature similarity search, POST the up/down genes 
	to the RURL and redirect to the result page.'''
	if request.method == 'POST':
		# retrieve data from the form 
		up_genes = request.form.get('upGenes', '').split()
		down_genes = request.form.get('dnGenes', '').split()
		# init GeneSets instance
		gene_sets = GeneSets(up_genes, down_genes)
		# perform similarity search
		result = gene_sets.enrich()
		# save gene_sets and results to MongoDB
		rid = gene_sets.save()
		print rid
		return redirect(ENTER_POINT + '/result/' + rid, code=302)


@app.route(ENTER_POINT + '/result/graph/<string:result_id>', methods=['GET'])
def result(result_id):
	"""
	Retrieve a simiarity search result using id and combine it
	with graph layout.
	"""
	# retrieve enrichment results from db
	result_obj = EnrichmentResult(result_id)
	# bind enrichment result to the network layout
	graph_df_res = result_obj.bind_to_graph(graph_df)

	return graph_df_res.reset_index().to_json(orient='records')


@app.route(ENTER_POINT + '/result/genes/<string:result_id>', methods=['GET'])
def result_genes(result_id):
	'''Retrieve user input genes.
	'''
	# retrieve enrichment results from db
	result_obj = EnrichmentResult(result_id)
	return jsonify(result_obj.data)


@app.route(ENTER_POINT + '/result/topn/<string:result_id>', methods=['GET'])
def result_topn(result_id):
	'''Retrieve topn signatures to visualize in a table.
	'''
	result_obj = EnrichmentResult(result_id)
	return json.dumps(result_obj.result['topn'])


@app.route(ENTER_POINT + '/result/modal/<string:result_id>', methods=['GET'])
def result_modal(result_id):
	'''Template for the signature similarity search result modal.
	'''
	result_obj = EnrichmentResult(result_id)
	return render_template('result-modal.html', 
		result_obj=result_obj,
		result_id=result_id)


@app.route(ENTER_POINT + '/result/download/<string:result_id>', methods=['GET'])
def result_download(result_id):
	'''To download the results to a csv file.
	'''
	result_obj = EnrichmentResult(result_id)
	# Prepare a DataFrame for the result
	scores = result_obj.result['scores']
	result_df = pd.DataFrame({'similarity_scores': scores}, index=graph_df.index)\
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
	return render_template('index.html', 
		script='result', 
		ENTER_POINT=ENTER_POINT,
		result_id=result_id)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, threaded=True)


