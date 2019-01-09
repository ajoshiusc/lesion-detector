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


def my_nn(image_size, code_size=32):
    pca_ae=pca_autoencoder(image_size, code_size)
    msk = keras.models.Sequential()
    msk.add(L.InputLayer(img_shape))

    L.Multiply(pca_ae, msk)
