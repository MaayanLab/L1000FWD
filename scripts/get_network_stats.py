import os, sys
import json
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

## Load dmoa-embed network
json_data = json.load(open('../notebooks/Signature_Graph_17041nodes_0.56_ERSC.cyjs', 'rb'))
json_data = json_data['elements']['nodes']
sig_ids_in_net = [rec['data']['name'] for rec in json_data] 

meta_df = pd.read_csv('../data/metadata-full-anno.tsv', sep='\t').set_index('sig_id')
meta_df = meta_df.ix[sig_ids_in_net]
print meta_df.shape

for col in meta_df.columns:
	print col, meta_df[col].nunique()

## Load LJP network
ljp_json_file = '/Users/zichen/Documents/Zichen_Projects/HarvardNetwork/scripts/data/harvard_net_with_pos_Cidx_enriched_terms_combined_score.json'
graph = json.load(open(ljp_json_file, 'rb'))

nodes = pd.DataFrame.from_records(graph['nodes'])
print nodes.shape
for col in nodes.columns:
	print col, nodes[col].nunique()

