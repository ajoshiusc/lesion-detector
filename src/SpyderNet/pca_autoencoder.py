import keras, keras.layers as L
import numpy as np


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
    return pca_ae