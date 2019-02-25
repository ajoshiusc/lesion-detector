from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers
import keras.backend as K
import numpy as np
from sklearn.metrics import mean_squared_error
import math

 
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
k_out=math.pi

import tensorflow as tf
delta=0.05
def square_variation(images, name=None):
    pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
    pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
    sum_axis = None
    square_variation = (
        math_ops.reduce_mean(math_ops.square(pixel_dif1), axis=sum_axis) +
        math_ops.reduce_mean(math_ops.square(pixel_dif2), axis=sum_axis))
    return square_variation

def k_delta(inp,outp):
    k_var=tf.math.exp(-tf.squared_difference(inp, outp)/(2*math_ops.square(delta)))
    k_out=tf.reduce_mean(k_var/(math_ops.sqrt(2*math.pi*delta)))
    return k_out

def square_coef(y_true, y_pred, alpha):
    term1= tf.reduce_mean(tf.squared_difference(y_true, y_pred))
    term2=square_variation((y_true - y_pred),name=None)
    return ((1-alpha)*term1+alpha*term2)

def square_loss(alpha):
    def SV(y_true, y_pred):
        return square_coef(y_true, y_pred, alpha)
    return SV

def corrent_coef(y_true, y_pred, alpha):
    term1=-k_delta(y_true ,y_pred)
    term2= square_variation((y_true - y_pred),name=None)
    return ((1-alpha)*term1+alpha*term2) 

def corrent_loss(alpha):
    def RAE(y_true, y_pred):
        return corrent_coef(y_true, y_pred, alpha)
    return RAE


  


def auto_encoder(input_size,loss1,alpha):

    input_img = Input(shape=(input_size, input_size,3))  # adapt this if using `channels_first` image data format



    
    Conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    max1 = MaxPooling2D((2, 2), padding='same')(Conv1)
    Conv2 = Conv2D(32, (4, 4), activation='relu', padding='same')(max1)
    Conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(Conv2)
    max2 = MaxPooling2D((2, 2), padding='same')(Conv2)
    Conv3 = Conv2D(64, (4, 4), activation='relu', padding='same')(max2)
    Conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(Conv3)
    max3 = MaxPooling2D((2, 2), padding='same')(Conv3)
    Conv4= Conv2D(128, (4, 4), activation='relu', padding='same')(max3)
    Conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(Conv4)
    max4 = MaxPooling2D((2, 2), padding='same')(Conv4)
    Conv5 = Conv2D(256, (4, 4), activation='relu', padding='same')(max4)
    Conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(Conv5)
    drop5 = Dropout(0.5)(Conv5)
    max5 = MaxPooling2D((2, 2), padding='same')(drop5)
    Conv6 = Conv2D(512, (4, 4), activation='relu', padding='same')(max5)
    Conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(Conv6)
    
    

    # at this point the representation is (7, 7, 32)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv6))
    merge7 = concatenate([drop5,up6], axis = 3)
    Conv7 = Conv2D(256, (4, 4,), activation='relu', padding='same')(merge7)
    Conv7 = Conv2D(256, (3, 3,), activation='relu', padding='same')(Conv7)
    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv7))
    merge8 = concatenate([Conv4,up7], axis = 3)
    Conv8 = Conv2D(128, (4, 4,), activation='relu', padding='same')(merge8)
    Conv8 = Conv2D(128, (3, 3,), activation='relu', padding='same')(Conv8)
    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv8))
    merge9 = concatenate([Conv3,up8], axis = 3)
    Conv9 = Conv2D(64, (4, 4,), activation='relu', padding='same')(merge9)
    Conv9 = Conv2D(64, (3, 3,), activation='relu', padding='same')(Conv9)
    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv9))
    merge10 = concatenate([Conv2,up9], axis = 3)
    Conv10 = Conv2D(32, (4, 4,), activation='relu', padding='same')(merge10)
    Conv10 = Conv2D(32, (3, 3,), activation='relu', padding='same')(Conv10)
    up10 = Conv2D(16, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv10))
    #merge11 = concatenate([Conv1,up10], axis = 3)
    Conv11 = Conv2D(16, (4, 4,), activation='relu', padding='same')(up10)
    Conv11 = Conv2D(16, (3, 3,), activation='relu', padding='same')(Conv11)

    #up11 = Conv2D(13, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(Conv11))
    Conv12 = Conv2D(3, (4, 4,), activation='relu', padding='same')(Conv11)
    decoded = Conv2D(3, (3, 3,), activation='relu', padding='same')(Conv12)

    if loss1=='SV':
      loss2=square_loss(alpha)
    elif loss1=='RAE':
      loss2=corrent_loss(alpha)
    else:
      loss2=loss1



    model = Model(input_img, decoded)
    opt = optimizers.Adam(lr=0.0001)
    #opt=optimizers.Nadam(lr=0.002)
    model.compile(optimizer=opt, loss=loss2)
     #model_dice = dice_loss(alpha=0.5)
  #model.compile(loss=model_dice)



    return model