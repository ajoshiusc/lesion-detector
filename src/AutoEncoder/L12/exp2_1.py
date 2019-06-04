import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os
import time

import sys
#sys.path.append("../../model")
import l21RobustDeepAutoencoderOnST2 as l21RDA

#sys.path.append("../../data")
from utils import tile_raster_images
from myutils import lession_generator

def l21RDAE(X, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, 
            batch_size = 133,inputsize = 28):
            if not os.path.isdir(folder):
                os.makedirs(folder)
            os.chdir(folder)
            rael21 = l21RDA.RobustL21Autoencoder(lambda_= lamda, input_size=inputsize)
            l21L, l21S=rael21.fit_T(X = X, inner_iteration = inner, iteration = outer, 
                                    batch_size = batch_size, learning_rate = learning_rate)
            l21L=l21L.reshape(l21L.shape[0],784)
            l21S=l21S.reshape(l21S.shape[0],784)

            l21R = rael21.getRecon(X = X)
            l21R =l21R .reshape(l21R.shape[0],784)
            #l21H = rael21.transform(X = X)

            Image.fromarray(tile_raster_images(X=l21S,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21S.png")
            Image.fromarray(tile_raster_images(X=l21R,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21R.png")
            Image.fromarray(tile_raster_images(X=l21L,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21L.png")
            l21S.dump("l21S.npk")


def compare_frame():

    from keras.datasets import mnist
    import numpy as np
    x_train=np.load('x_train.npy')
    ind=np.load('lesion_ind.npy')
    #inputsize=28

    #x_in= x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train=np.load('lesion_x_train.npy')
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    inner = 50
    outer = 20



    lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045, 
         0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]
#     lambda_list = [0.00015,0.00018,0.0002,0.00025,0.00028,0.0003]
    print(lambda_list)
    
    #layers = [784, 400, 200] ## S trans
    print("start")
    start_time = time.time()
    image_X = Image.fromarray(tile_raster_images(X = x_train, img_shape = (28,28), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    for lam in lambda_list:
        folder = "lam" + str(lam)
        x=x_train.reshape(x_train.shape[0],28,28,1)
        l21RDAE(X = x, lamda = lam, folder = folder, learning_rate = 0.001, 
                inner = inner, outer = outer, batch_size = 133,inputsize = 28)
        print("done: lam", str(lam))
    print ("Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    compare_frame()