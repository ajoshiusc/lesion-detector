from keras.models import Model
from keras.layers import (Input, Flatten, Dense, Reshape, Multiply)
import numpy as np


def pca_autoencoder_old(img_shape=[64, 64, 3], code_size=32):
    pca_ae = keras.models.Sequential()
    # Input layer
    pca_ae.add(L.InputLayer(img_shape))
    # Flattening the layer
    pca_ae.add(L.Flatten())
    # Encoded space
    pca_ae.add(L.Dense(code_size))
    # Output units should be image_size * image_size * channels
    pca_ae.add(L.Dense(np.prod(img_shape)))
    # Last layer
    pca_ae.add(L.Reshape(img_shape))

    return pca_ae


def pca_autoencoder(img_shape=[64, 64, 3], code_size=32):

    img_shape_out = img_shape.copy()
    img_shape_out[-1] = 1

    input_imgs = Input(shape=img_shape)

    flattened_input = Flatten()(input_imgs)
    # Encoded space
    encoded_space = Dense(code_size)(flattened_input)
    # Decoded space
    decompressed = Dense(np.prod(img_shape_out))(encoded_space)
    # Output units should be image_size * image_size * channels
    output_imgs = Reshape(img_shape_out)(decompressed)

    return input_imgs, output_imgs


def pca_autoencoder_masked(image_size, code_size=32):

    input_imgs, output_imgs = pca_autoencoder([image_size, image_size, 3],
                                              code_size)

    msk_input = Input(shape=[image_size, image_size, 1])
    mskd_output_imgs = Multiply()([output_imgs, msk_input])

    model = Model(inputs=[input_imgs, msk_input], outputs=mskd_output_imgs)
    model.compile(optimizer='adamax', loss='mse')

    return model
