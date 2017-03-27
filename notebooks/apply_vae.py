import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 110
original_dim = 978
latent_dim = 2
intermediate_dim = 300
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
# decoder_mean = Dense(original_dim, activation='relu')

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
# vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.compile(optimizer='adam', loss=vae_loss)

## Load data from h5 file
import h5py
f = h5py.File('../data/CD_matrix_105939xlm978.h5', 'r')
mat = f['matrix']
sig_ids = f['meta']['sig_ids']
print mat.shape, type(mat), len(sig_ids)
print mat.dtype

residual = mat.shape[0] % batch_size

n_train = 950 * batch_size
n_test = mat.shape[0] - residual - n_train 


## subset into train and test
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()

f2 = h5py.File('../data/CD_matrix_%sxlm978_train-test.h5' % mat.shape[0], 'w')
dset_train = f2.create_dataset('train', shape=(n_train, 978))
dset_train.write_direct(scl.fit_transform(mat)[:n_train, :])

dset_test = f2.create_dataset('test', shape=(n_test, 978))
dset_test.write_direct(scl.fit_transform(mat)[n_train:(mat.shape[0]-residual), :])
f2.close()



f2 = h5py.File('../data/CD_matrix_105939xlm978_train-test.h5','r')
mat_train = f2['train']
mat_test = f2['test']

print mat_train.shape, mat_test.shape
print mat_test[:,:].min(), mat_test[:,:].max()
print mat_train[:,:].min(), mat_train[:,:].max()


vae.fit(mat_train, mat_train,
        shuffle="batch",
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(mat_test, mat_test)
       )

       
