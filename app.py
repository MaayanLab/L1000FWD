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

@app.before_first_request
def load_globals():
	global meta_df, N_SIGS
	# meta_df = pd.read_csv('data/metadata.tsv', sep='\t')
	meta_df = pd.read_csv('data/metadata-sig-only.tsv', sep='\t')
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


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000)


