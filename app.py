import os, sys
import json
import numpy as np
np.random.seed(10)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, request, redirect, render_template, jsonify, send_from_directory



ENTER_POINT = '/embed'
app = Flask(__name__, static_url_path=ENTER_POINT, static_folder=os.getcwd())
app.debug = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 6


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

@app.before_first_request
def load_globals():
	global meta_df, N_SIGS
	# meta_df = pd.read_csv('data/metadata.tsv', sep='\t')
	# meta_df = pd.read_csv('data/metadata-sig-only.tsv', sep='\t')
	meta_df = pd.read_csv('data/metadata-full.tsv', sep='\t')
	meta_df = meta_df.set_index('sig_id')
	print meta_df.shape
	N_SIGS = meta_df.shape[0]
	return


@app.route(ENTER_POINT + '/')
def index_page():
	return app.send_static_file('index.html')

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
		# json_data = json.load(open('notebooks/Signature_Graph_12761nodes_99.9_SC.cyjs', 'rb'))
		# json_data = json.load(open('notebooks/Signature_Graph_12761nodes_99.9_ERSC.cyjs', 'rb'))
		json_data = json.load(open('notebooks/Signature_Graph_17041nodes_0.56_ERSC.cyjs', 'rb'))
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
		print df.shape
		df = df.merge(meta_df, how='left', left_index=True, right_index=True)
		
		df['neglogp'] = -np.log10(df['pvalue']+1e-4)
		df['z'] = 0

		# df = encode_rare_categories(df, 'cell')
		# df = encode_rare_categories(df, 'perturbation')

		print df.shape
		return df.reset_index().to_json(orient='records')


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000)


