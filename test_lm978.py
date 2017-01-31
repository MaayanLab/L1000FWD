'''
Create a embedding projector for all the CD signatures.
'''

import os
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from tensorflow.contrib.tensorboard.plugins import projector

N_DIMS = 978

LOG_DIR = 'log/CD_lm978_%sdims/' % N_DIMS
try:
	os.makedirs(LOG_DIR)
except OSError:
	pass

step = 0


f = h5py.File('data/CD_matrix_45012xlm978.h5', 'r')
mat = f['matrix']
sig_ids = f['meta']['sig_ids']
# print type(mat)
# print type(mat[:, :5])

print len(sig_ids), mat.shape
print sig_ids[:5]


# 0) Create metadata.tsv
engine = create_engine("mysql://euclid:elements@amp.pharm.mssm.edu:3306/euclid3?charset=utf8")
meta_df = pd.read_sql('optional_metadata', engine, columns=['gene_signature_fk', 'name', 'value'])
meta_df = pd.pivot_table(meta_df, values='value', 
	columns=['name'], index=['gene_signature_fk'], aggfunc=lambda x:x)
# print meta_df.head()
# print meta_df.shape

meta_df = meta_df.reset_index().set_index('sig_id').drop(['gene_signature_fk', 'disease', 'gene'], axis=1)
# print meta_df.head()

# find shared sig_ids
sig_ids_shared = list(set(list(sig_ids)) & set(meta_df.index))
print '# shared sig_ids: %d' % len(sig_ids_shared)
mask_shared = np.in1d(list(sig_ids), sig_ids_shared)
sig_ids_shared = np.array(sig_ids)[mask_shared]

# subset and reorder
meta_df = meta_df.ix[sig_ids_shared]
print meta_df.head()
print meta_df.shape

metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

print meta_df['cell'].nunique()
print meta_df['pert_id'].nunique()
print meta_df['perturbation'].nunique()
meta_df.to_csv(metadata_path, sep='\t')


# 1) Setup a 2D tensor variable(s) that holds your embedding(s).
embedding_var = tf.Variable(mat[mask_shared, :N_DIMS],
	name='CD_LM')


init_op = tf.global_variables_initializer()


# 2) Periodically save your embeddings in a LOG_DIR.
saver = tf.train.Saver()
with tf.Session() as session:
	session.run(init_op)
	saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)

	# 3) Associate metadata with your embedding.
	# Use the same LOG_DIR where you stored your checkpoint.
	summary_writer = tf.train.SummaryWriter(LOG_DIR)

	# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
	config = projector.ProjectorConfig()

	# You can add multiple embeddings. Here we add only one.
	embedding = config.embeddings.add()
	embedding.tensor_name = embedding_var.name
	# Link this tensor to its metadata file (e.g. labels).
	embedding.metadata_path = metadata_path

	# Saves a configuration file that TensorBoard will read during startup.
	projector.visualize_embeddings(summary_writer, config)


f.close() # close the h5 file
