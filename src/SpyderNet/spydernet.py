# Author: Anand A Joshi (ajoshi@usc.edu)
from keras.models import Input, Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
import numpy as np


def encoder(isize, namestr):
    input_img = Input(shape=isize)  # tensorflow follows NHWC (HWC here)

    x = Conv2D(
        4, (3, 3), activation='relu', padding='same',
        name=namestr + 'enc1')(input_img)
    #    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(
        4, (3, 3), activation='relu', padding='same', name=namestr + 'enc2')(x)
    #   x = MaxPooling2D((2, 2), padding='same')(x)

    return input_img, x


def decoder(input_enc, namestr):
    x = Conv2D(
        4, (3, 3), activation='relu', padding='same',
        name=namestr + 'dec1')(input_enc)
    #    x = UpSampling2D((2, 2))(x)
    x = Conv2D(
        4, (3, 3), activation='relu', padding='same', name=namestr + 'dec2')(x)
    #    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(
        1, (3, 3), activation='relu', padding='same', name=namestr + 'dec3')(x)
    return decoded


def spyder_net(isize, n_channel):

    I1, E1 = encoder(isize, 'i1')
    I2, E2 = encoder(isize, 'i2')
    I3, E3 = encoder(isize, 'i3')

    x = concatenate([E1, E2, E3], axis=-1)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    I_1 = decoder(x, 'd1')
    I_2 = decoder(x, 'd2')
    I_3 = decoder(x, 'd3')

    print("==Defining Model  ==")
    autoencoder = Model(inputs=[I1, I2, I3], outputs=[I_1, I_2, I_3])
    opt = optimizers.adam()  #adadelta(lr=1)
    autoencoder.compile(opt, loss='mean_squared_error')

    return autoencoder


def train_model(data):
    sz = np.array(data.shape[1:4])
    sz[2] = 1
    autoenc = spyder_net(isize=sz, n_channel=data.shape[3])
    history = autoenc.fit([
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ], [
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ],
                          batch_size=256,
                          epochs=200,
                          verbose=1,
                          shuffle=True,
                          validation_split=0.2)
    #        callbacks=[model_checkpoint])

    return autoenc


def test_model(model, data):

    return predictions


def get_neural_net(self, isize=[32, 32], subc_size=31870):
    """VGG model with one FC layer added at the end for continuous output"""
    lh_input, lh_out = get_mynet(isize, 'lh_')
    rh_input, rh_out = get_mynet(isize, 'rh_')

    subco_input = Input(shape=(subc_size, 36), dtype='float32')
    fc = Flatten()(subco_input)
    subco_out = Dense(256, activation='relu')(fc)

    cc = concatenate([lh_out, rh_out, subco_out], axis=-1)
    cc = Dense(64, activation='relu')(cc)
    out_theta = Dense(3)(cc)

    print("==Defining Model  ==")
    model = Model(
        inputs=[lh_input, rh_input, subco_input], outputs=[out_theta])
    optz = adam(lr=1e-4)  #, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=optz, loss=losses.mean_squared_error, metrics=['mse'])

    return model
