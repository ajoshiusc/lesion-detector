import keras, keras.layers as L
import numpy as np


def build_pca_autoencoder(img_shape=[64, 64, 3], code_size=32):
    """
    Here we define a simple linear autoencoder as described above.
    We also flatten and un-flatten data to be compatible with image shapes
    """

    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())  #flatten image to vector
    encoder.add(L.Dense(code_size))  #actual encoder

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size, )))
    decoder.add(L.Dense(
        np.prod(img_shape)))  #actual decoder, height*width*3 units
    decoder.add(L.Reshape(img_shape))  #un-flatten

    return encoder, decoder


def pca_autoencoder(img_shape=[64, 64, 3], code_size=32):
    encoder, decoder = build_pca_autoencoder(img_shape, code_size)
    inp = L.Input(img_shape)
    code = encoder(inp)
    reconstruction = decoder(code)

    # Merging the models
    autoencoder = keras.models.Model(inp, reconstruction)

    return autoencoder  # encoder, decoder


""" 
def pca_autoencoder(img_shape=[64, 64, 3], code_size=32):
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
    return pca_ae """