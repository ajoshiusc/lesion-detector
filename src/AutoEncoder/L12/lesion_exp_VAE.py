import numpy as np
from VAE import Vauto_encoder
from keras import backend as K
from keras.datasets import mnist
from utils import tile_raster_images
from myutils import lession_generator
import PIL.Image as Image

x_train=np.load('x_train.npy')
ind=np.load('lesion_ind.npy')
inputsize=28

x_in= x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train=np.load('lesion_x_train.npy')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

image_size = x_train.shape[1]
original_dim = image_size
input_shape = (original_dim, )
batch_size = 133
epochs = 100



vae,encoder =Vauto_encoder(input_shape=input_shape,original_dim=original_dim,loss='MSE')
vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size)
        #vae.save_weights('vae_mlp_mnist.h5')
inputsize=28
VAES=x_train-vae.predict(x_train)
VAEL=vae.predict(x_train)
Image.fromarray(tile_raster_images(X=VAES,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"VAES.png")
Image.fromarray(tile_raster_images(X=VAEL,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"VAEL.png")
