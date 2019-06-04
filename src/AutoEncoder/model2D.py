from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import optimizers

def auto_encoder(input_size):
    input_img = Input(shape=(input_size, input_size,1))  # adapt this if using `channels_first` image data format

    x = Conv2D(8, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    encoded= MaxPooling2D((2, 2), padding='same')(x)
    

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5,), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (6, 6), activation='relu', padding='valid')(x)

    model = Model(input_img, decoded)
    opt=optimizers.adadelta(lr=0.01)
    model.compile(opt, loss='mean_squared_error')
    
    return model
