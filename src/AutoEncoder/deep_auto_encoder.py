from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers
import keras.backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def TV_coef(y_true, y_pred, alpha):
    term1= tf.reduce_mean(tf.squared_difference(y_true, y_pred))
    term2=tf.image.total_variation((y_true - y_pred),name=None)/(64*64)
    return ((1-alpha)*term1+alpha*term2)
    
def TV_loss(alpha):
    def TV(y_true, y_pred):
        return TV_coef(y_true, y_pred, alpha)
    return TV

  


def auto_encoder(input_size,loss1,alpha):

    input_img = Input(shape=(input_size, input_size,3))  # adapt this if using `channels_first` image data format



    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (4, 4), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    
    

    # at this point the representation is (7, 7, 32)

    x = Conv2D(512, (4, 4,), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (4, 4,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    if loss1=='TV':
      loss2=TV_loss(alpha)
    else:
      loss2=loss1



    model = Model(input_img, decoded)
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss2)
     #model_dice = dice_loss(alpha=0.5)
  #model.compile(loss=model_dice)



    return model