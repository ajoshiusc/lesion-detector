from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import optimizers
from tensorflow.python.keras import backend as K
import numpy as np
from sklearn.metrics import mean_squared_error
import math

#from keras.losses import _logcosh

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
k_out=math.pi

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import tensorflow as tf


from keras.models import *
from keras.layers import *
from keras.optimizers import *
delta=0.05

###SV
def square_variation(images, name=None):
    ndims=images.get_shape().ndims
    if ndims == 3:
      pixel_var1=K.zeros_like(images[1:2, :, :])
      pixel_var2=K.zeros_like(images[:, 1:2, :])
      pixel_dif1  = tf.concat(((images[1:, :, :] - images[:-1, :, :]), pixel_var1), axis=1)
      pixel_dif2  = tf.concat(((images[:, 1:, :] - images[:, :-1, :]), pixel_var2), axis=2)
    elif ndims == 4:
      pixel_var1=K.zeros_like(images[:, 1:2, :, :])
      pixel_var2=K.zeros_like(images[:, :, 1:2, :])
      pixel_dif1  = tf.concat(((images[:, 1:, :, :] - images[:, :-1, :, :]), pixel_var1), axis=1)
      pixel_dif2  = tf.concat(((images[:, :, 1:, :] - images[:, :, :-1, :]), pixel_var2), axis=2)
    #sum_axis = -1
    square_variation = K.mean((math_ops.square(pixel_dif1)) +(math_ops.square(pixel_dif2)),axis=-1)
    return square_variation
def square_coef(y_true, y_pred, alpha):
    #sum_axis = -1
    term1= K.mean(math_ops.square(y_true- y_pred),axis=-1)
    term2=square_variation((y_true-y_pred))
    return ((1-alpha)*term1+alpha*term2)

def square_loss(alpha):
    def SV(y_true, y_pred):
        return square_coef(y_true, y_pred, alpha)
    return SV

### k_delta_RAE
def k_delta(inp,outp):
    k_var=tf.math.exp(-tf.squared_difference(inp, outp)/(2*math_ops.square(delta)))
    k_out=tf.reduce_mean(k_var/(math_ops.sqrt(2*math.pi*delta)))
    return k_out

def corrent_coef(y_true, y_pred, alpha):
    term1=-k_delta(y_true ,y_pred)
    term2= square_variation((y_true - y_pred),name=None)
    return ((1-alpha)*term1+alpha*term2) 

def corrent_loss(alpha):
    def RAE(y_true, y_pred):
        return corrent_coef(y_true, y_pred, alpha)
    return RAE
def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.log(2.)
### TV of reconstructed error
def total_variation_of_reconstructed(images):
    ndims=images.get_shape().ndims
    if ndims == 3:
      pixel_var1=K.zeros_like(images[1:2, :, :])
      pixel_var2=K.zeros_like(images[:, 1:2, :])
      pixel_dif1  = tf.concat(((images[1:, :, :] - images[:-1, :, :]), pixel_var1), axis=1)
      pixel_dif2  = tf.concat(((images[:, 1:, :] - images[:, :-1, :]), pixel_var2), axis=2)
    elif ndims == 4:
      pixel_var1=K.zeros_like(images[:, 1:2, :, :])
      pixel_var2=K.zeros_like(images[:, :, 1:2, :])
      pixel_dif1  = tf.concat(((images[:, 1:, :, :] - images[:, :-1, :, :]), pixel_var1), axis=1)
      pixel_dif2  = tf.concat(((images[:, :, 1:, :] - images[:, :, :-1, :]), pixel_var2), axis=2)
    apx_total_variation = K.mean((math_ops.abs(pixel_dif1)) +(math_ops.abs(pixel_dif2)),axis=-1)
    
    return apx_total_variation

def TV_reconstructed_coef(y_true, y_pred, alpha):
    term1= K.mean(math_ops.square(y_true-y_pred),axis=-1)
    term2=total_variation_of_reconstructed(y_pred)
    return ((1-alpha)*term1+alpha*term2)

def TV_reconstructed_loss(alpha):
    def TV_R(y_true, y_pred):
        return TV_reconstructed_coef(y_true, y_pred, alpha)
    return TV_R
### mean square
def FLAIR_coef(y_true, y_pred,alpha):
        term1=K.mean(math_ops.square(y_true- y_pred),axis=-1)
        return term1

def FLAIR_loss(alpha):
    def MSE_FLAIR(y_true, y_pred):
        return FLAIR_coef(y_true, y_pred, alpha)
    return MSE_FLAIR



def matlab_style_gauss2D(shape=(64,64),sigma=3):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return np.float32(h)


#### Auto Encoder
def auto_encoder(input_size,loss1,alpha):

    input_img = Input(shape=(input_size, input_size,3))  # adapt this if using `channels_first` image data format



    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    
    

    # at this point the representation is (7, 7, 32)
    x=BatchNormalization()(encoded)
    x = Conv2D(512, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(16, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    
    if loss1=='SV':
      loss2=square_loss(alpha)
    elif loss1=='RAE':
      loss2=corrent_loss(alpha)
    elif loss1=='TV_R':
      loss2=TV_reconstructed_loss(alpha)
    elif loss1=='MSE_FLAIR':
      loss2=FLAIR_loss(alpha)
    else:
      loss2=loss1



    model = Model(input_img, decoded)
    #opt = optimizers.Adam(lr=0.0001)
    opt=optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=loss2)
     



    return model

def highpass_auto_encoder(input_size,loss1,alpha):

    
    K=matlab_style_gauss2D(shape=(input_size,input_size),sigma=3)
    input_img = Input(shape=(input_size, input_size,3))  # adapt this if using `channels_first` image data format
    filth = tf.constant(K.reshape(input_size,input_size, 1, 1))
    # get low/high pass ops
    lowpass = tf.nn.conv2d(input_img[:,:,:,2:3], filth, strides=[1, 1, 1, 1], padding='SAME')
    highpass = input_img[:,:,:,2:3] -lowpass

    x2= Conv2D(16, (3, 3), activation='relu', padding='same')(highpass)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (4, 4), activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    
    

    # at this point the representation is (7, 7, 32)
    x=BatchNormalization()(encoded)
    x = Conv2D(512, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(64, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(32, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    x = Conv2D(16, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=BatchNormalization()(x)
    merge1=concatenate([x,x2], axis = 3)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(merge1)
    
    if loss1=='SV':
      loss2=square_loss(alpha)
    elif loss1=='RAE':
      loss2=corrent_loss(alpha)
    elif loss1=='TV_R':
      loss2=TV_reconstructed_loss(alpha)
    elif loss1=='MSE_FLAIR':
      loss2=FLAIR_loss(alpha)
    else:
      loss2=loss1



    model = Model(input_img, decoded)
    #opt = optimizers.Adam(lr=0.0001)
    opt=optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=loss2)
     



    return model
  

def l21shrink(epsilon, x):
    output = x.copy()
    for i in range(x.shape[0]):
        temp=np.ravel(x[i,:,:,:])
        norm = np.linalg.norm(temp, ord=2, axis=0)
        #rint(x.shape[0])
        if (norm > epsilon):
          output[i,:,:,:] = x[i,:,:,:] - epsilon * x[i,:,:,:] / norm
        else:
          output[i,:,:,:] = 0
    
    return output

def L12Cost_Cal(X,L,S,lamb):
    LD=X-S
    term1= tf.reduce_mean(tf.squared_difference(LD, L))
    norm1=np.zeros(1,)
    for i in range(X.shape[0]):
        temp=np.ravel(X[i,:,:,:])
        norm1[i] = tf.math.sqrt(np.linalg.norm(temp, ord=2, axis=0))
    term2=lamb*tf.reduce_mean(norm1)
    cost=term1+term2
    return cost

