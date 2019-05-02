
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import tile_raster_images
import PIL.Image as Image


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def Vauto_encoder(input_shape,original_dim,loss):
    intermediate_dim=400
    latent_dim=200
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2]) #why [2]????
    vae = Model(inputs, outputs, name='vae_mlp')

    if loss=='MSE':
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae,encoder 

if __name__ == '__main__':
    x_train=np.load(r"data.npk")
    image_size = x_train.shape[1]
    original_dim = image_size 
    #x_train = x_train.astype('float32') / 255

# network parameters
    input_shape = (original_dim, )
    batch_size = 133
    epochs = 100
    inputsize=28
    Image.fromarray(tile_raster_images(X=x_train,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"X.png")
    # VAE loss = mse_loss or xent_loss + kl_loss
    vae,encoder =Vauto_encoder(input_shape=input_shape,original_dim=original_dim,loss='MSE')


        # train the autoencoder
    vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size)
        #vae.save_weights('vae_mlp_mnist.h5')
    inputsize=28
    VAES=x_train-vae.predict(x_train)
    VAEL=vae.predict(x_train)
    Image.fromarray(tile_raster_images(X=VAES,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"VAES.png")
    Image.fromarray(tile_raster_images(X=VAEL,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"VAEL.png")
