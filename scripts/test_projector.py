import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'log/'
step = 0


# 1) Setup a 2D tensor variable(s) that holds your embedding(s).
embedding_var = tf.Variable(tf.random_normal([10, 5]),
	name='embedding_var')


init_op = tf.global_variables_initializer()

# write metadata.tsv
meta_df = pd.DataFrame({
	'Name': range(10),
	'Type': ['type-1'] * 5 + ['type-2'] * 5
	})
metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
meta_df.to_csv(metadata_path, sep='\t')

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

