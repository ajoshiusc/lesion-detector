import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os
import time

import sys
#sys.path.append("../../model")
import l21RobustDeepAutoencoderOnST as l21RDA

#sys.path.append("../../data")
from utils import tile_raster_images

def l21RDAE(X, layers, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, 
            batch_size = 133,re_init=False,inputsize = (28,28)):
            if not os.path.isdir(folder):
                os.makedirs(folder)
            os.chdir(folder)
            with tf.Session() as sess:
                rael21 = l21RDA.RobustL21Autoencoder(sess = sess, lambda_= lamda, 
                                                 layers_sizes=layers)
                l21L, l21S = rael21.fit(X = X, sess = sess, inner_iteration = inner, iteration = outer, 
                                    batch_size = batch_size, learning_rate = learning_rate,  
                                    re_init=re_init,verbose = False)
                l21R = rael21.getRecon(X = X, sess = sess)
                l21H = rael21.transform(X, sess)
                Image.fromarray(tile_raster_images(X=l21S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21S.png")
                Image.fromarray(tile_raster_images(X=l21R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21R.png")
                Image.fromarray(tile_raster_images(X=l21L,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21L.png")
                l21S.dump("l21S.npk")


def compare_frame():

    
    #x_train=np.load('x_train.npy')
    ind=np.load('lesion_ind.npy')
    #inputsize=28

    #x_in= x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train=np.load('lesion_x_train.npy')
    X = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    inner = 50
    outer = 20



    lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045, 
         0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]
#     lambda_list = [0.00015,0.00018,0.0002,0.00025,0.00028,0.0003]
    print(lambda_list)
    
    layers = [784, 400, 200] ## S trans
    print("start")
    start_time = time.time()
    image_X = Image.fromarray(tile_raster_images(X = X, img_shape = (28,28), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    for lam in lambda_list:
        folder = "lam" + str(lam)
        l21RDAE(X = X, layers=layers, lamda = lam, folder = folder, learning_rate = 0.001, 
                inner = inner, outer = outer, batch_size = 133,re_init=True,inputsize = (28,28))
        print("done: lam", str(lam))
    print ("Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    compare_frame()