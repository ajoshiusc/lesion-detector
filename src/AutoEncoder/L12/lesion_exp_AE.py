import tensorflow as tf
import numpy as np
import DeepAE as DAE
import time
import os
from keras.datasets import mnist
from utils import tile_raster_images
from myutils import lession_generator
import PIL.Image as Image

(x_train, _), (x_test, _) = mnist.load_data()
x_train=x_train[0:10000,:,:]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

inputsize=28

x_in= x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
Image.fromarray(tile_raster_images(X=x_train,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"Xin_lession.png")
np.save('x_train', x_train)

x_train,ind=lession_generator(x_train, corNum=500)


np.save('lesion_x_train', x_train)
np.save('lesion_ind', ind)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

start_time = time.time()
Image.fromarray(tile_raster_images(X=x_train,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"X_lession.png")
with tf.Session() as sess:
    ae = DAE.Deep_Autoencoder(sess = sess, input_dim_list=[784,400,200])
    ae.fit(x_train ,sess = sess, learning_rate=0.01, batch_size = 133, iteration = 100, verbose=True)
    VAEL= ae.getRecon(x_train, sess = sess)
    print ("size 100 Runing time:" + str(time.time() - start_time) + " s")
    VAES=x_train-VAEL
    
    Image.fromarray(tile_raster_images(X=VAES,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"AES_lession.png")
    Image.fromarray(tile_raster_images(X=VAEL,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"AEL_lession.png")

