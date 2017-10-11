import os
import sys
import json
import cPickle as pickle
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing, manifold, decomposition, neighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance as dist
import scipy as sp
from scipy.sparse import linalg, csr_matrix

import matplotlib.pyplot as plt

import requests
# modules to interact with CyREST, the API of Cytoscape
from py2cytoscape.data.cyrest_client import CyRestClient
import py2cytoscape.cytoscapejs as renderer

# Plot degree distribution
def plot_degree_distribution(G):
    fig, ax = plt.subplots()
    
    degrees = G.degree().values()
    degrees = dict(Counter(degrees))
    x = degrees.keys()
    y = degrees.values()

    ax.scatter(x, y, s=10, alpha=.6)
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig

def compute_adjcency_mat(X, metric='cosine'):
    adj_mat = dist.squareform(pairwise_distances(X, metric=metric))
    adj_mat = 1 - adj_mat
    adj_mat[adj_mat<0] = 0
    return adj_mat

def create_graph_by_threshold(adj_mat, percentile):
    threshold = np.percentile(adj_mat, percentile)
    print threshold
    adj_mat_ = adj_mat.copy()
    adj_mat_[adj_mat<threshold] = 0
    G = nx.from_numpy_matrix(dist.squareform(adj_mat_))
    return G

def create_graph_by_threshold2(adj_mat, cosine_cutoff):
    adj_mat_ = adj_mat.copy()
    adj_mat_[adj_mat<cosine_cutoff] = 0
    G = nx.from_numpy_matrix(dist.squareform(adj_mat))
    return G
    
def create_knn_graph(X, k=30, metric='euclidean', n_jobs=1):
    '''Create a graph from a data matrix (sample x features).
    '''
    adj_mat = neighbors.kneighbors_graph(X, k, mode='connectivity', 
                                         metric=metric,
                                         n_jobs=n_jobs
                                        )
    G = nx.from_scipy_sparse_matrix(adj_mat)
    return G


def filter_out_small_ccs(G, cutoff=10):
    ## Create a new graph only keeping the large connected components
    G_new = nx.Graph()
    for cc in nx.connected_component_subgraphs(G):
        if cc.number_of_nodes() > cutoff:
            G_new = nx.compose(G_new, cc)
    return G_new


def compute_eigenvector_centrality(adj_mat):
    M = csr_matrix(dist.squareform(adj_mat))
    eigenvalue, eigenvector = linalg.eigs(M.T, k=1, which='LR')
    largest = eigenvector.flatten().real
    norm = sp.sign(largest.sum())*sp.linalg.norm(largest)
    centrality = largest / norm
    return centrality


# Params for CyREST
IP = '127.0.0.1'
PORT = 1234
BASE_URL = 'http://%s:%s/v1' % (IP, PORT)
HEADERS = {'Content-Type': 'application/json'}

def network_layout(G):
    '''POST the network to CyREST, then run layout algorithm, 
    finally return the coordinates and the cy_network.
    '''
    # Create Py2cytoscape client
    cy = CyRestClient(ip=IP, port=PORT)
    # Reset session
    cy.session.delete()
    # POST the graph to CyREST
    G_cy = cy.network.create_from_networkx(G)

    # Change the layout params
    layout_parameters = [
        {"name": "numIterations", "value": 10}, 
        {"name":"numIterationsEdgeRepulsive", "value":10}
    ]
    resp = requests.put(BASE_URL+ '/apply/layouts/force-directed-cl/parameters', 
                 data=json.dumps(layout_parameters),
                 headers=HEADERS)
    cy.layout.apply(name='force-directed-cl', network=G_cy)


    layout_parameters = [
        {"name": "maxIterations", "value": 10000}, 
        {"name":"randomize", "value":True},
    ]
    resp = requests.put(BASE_URL+ '/apply/layouts/allegro-edge-repulsive-strong-clustering/parameters', 
                 data=json.dumps(layout_parameters),
                 headers=HEADERS)
    cy.layout.apply(name='allegro-edge-repulsive-strong-clustering', network=G_cy)
    # Get current view
    view = G_cy.get_first_view()
    nodes = view['elements']['nodes']
    # Get a coord matrix ordered by id_original
    ids_original = np.array([n['data']['id_original'] for n in nodes]).astype(np.int32)
    xs = [n['position']['x'] for n in nodes]
    ys = [n['position']['y'] for n in nodes]
    coords = np.array([xs, ys]).T
    return coords, ids_original
    

