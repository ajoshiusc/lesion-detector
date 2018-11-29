# Author: Anand A Joshi (ajoshi@usc.edu)
from keras.models import Input, Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
import numpy as np
from keras.callbacks import TensorBoard
from time import time
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm


def custom_loss(y_true, y_pred):
    alpha = 0.1
    loss1 = K.sum(C.binary_cross_entropy(y_pred, y_true)) / 200
    loss2 = C.binary_cross_entropy(y_pred[200, :], y_true[200, :])
    loss = alpha * loss1 + (1 - alpha) * loss2

    loss = tf.reduce_sum(tf.image.total_variation(images))

    return loss


def encoder(isize, namestr):
    input_img = Input(shape=isize)  # tensorflow follows NHWC (HWC here)

    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'enc1')(input_img)
    #    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'enc2')(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'enc3')(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'enc4')(x)

    #   x = MaxPooling2D((2, 2), padding='same')(x)

    return input_img, x


def decoder(input_enc, namestr):
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'dec1')(input_enc)
    #    x = UpSampling2D((2, 2))(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'dec2')(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'dec3')(x)
    x = Conv2D(
        16, (3, 3), activation='relu', padding='same',
        name=namestr + 'dec4')(x)
    #    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(
        1, (3, 3), activation='relu', padding='same', name=namestr + 'dec5')(x)
    return decoded


def spyder_net(isize, n_channel):

    I1, E1 = encoder(isize, 'i1')
    I2, E2 = encoder(isize, 'i2')
    I3, E3 = encoder(isize, 'i3')

    x = concatenate([E1, E2, E3], axis=-1)
    x = Conv2D(
        1, (3, 3), activation='sigmoid', padding='same', name='Trunk1')(x)

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
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    history = autoenc.fit([
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ], [
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ],
                          batch_size=256,
                          epochs=10,
                          verbose=1,
                          shuffle=True,
                          validation_split=0.2,
                          callbacks=[tensorboard])

    return autoenc


def mod_indep_rep(model, data):

    layer_name = 'Trunk1'
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict([
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ])

    pred = model.predict([
        data[:, :, :, 0, None], data[:, :, :, 1, None], data[:, :, :, 2, None]
    ])

    return intermediate_output, pred


def mod_indep_rep_vol(model, vol_data, im_size):

    layer_name = 'Trunk1'

    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)

    intermediate_output1 = intermediate_layer_model.predict([
        vol_data[:, :im_size, :im_size, 0, None],
        vol_data[:, :im_size, :im_size, 1, None],
        vol_data[:, :im_size, :im_size, 2, None]
    ])

    intermediate_output2 = intermediate_layer_model.predict([
        vol_data[:, :im_size, -im_size:, 0, None],
        vol_data[:, :im_size, -im_size:, 1, None],
        vol_data[:, :im_size, -im_size:, 2, None]
    ])

    intermediate_output3 = intermediate_layer_model.predict([
        vol_data[:, -im_size:, :im_size, 0, None],
        vol_data[:, -im_size:, :im_size, 1, None],
        vol_data[:, -im_size:, :im_size, 2, None]
    ])

    intermediate_output4 = intermediate_layer_model.predict([
        vol_data[:, -im_size:, -im_size:, 0, None],
        vol_data[:, -im_size:, -im_size:, 1, None],
        vol_data[:, -im_size:, -im_size:, 2, None]
    ])

    indf = np.zeros(vol_data.shape[:3])
    out_vol = np.zeros(vol_data.shape[:3])
    out_vol[:, :im_size, :im_size] += intermediate_output1.squeeze()
    out_vol[:, :im_size, -im_size:] += intermediate_output2.squeeze()
    out_vol[:, -im_size:, :im_size] += intermediate_output3.squeeze()
    out_vol[:, -im_size:, -im_size:] += intermediate_output4.squeeze()

    indf[:, :im_size, :im_size] += 1
    indf[:, :im_size, -im_size:] += 1
    indf[:, -im_size:, :im_size] += 1
    indf[:, -im_size:, -im_size:] += 1

    out_vol = out_vol / indf  #[...,None]

    return out_vol


def mod_indep_rep_vol_stp(model, vol_data, im_size, step_size=1):
    # This function reconstructs the 3D volume from slices

    layer_name = 'Trunk1'

    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)

    model_pred = intermediate_layer_model.predict

    out_vol = slice2vol_pred(model_pred, vol_data, im_size, step_size=1)
    
    return out_vol

def slice2vol_pred(model_pred, vol_data, im_size, step_size=1):

    out_vol = np.zeros(vol_data.shape[:3]) # output volume
    indf = np.zeros(vol_data.shape[:3]) # dividing factor
    vol_size = vol_data.shape

    print('Putting together slices to form volume')
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size +
                    k] += model_pred([
                        vol_data[:, j:im_size + j, k:im_size + k, 0, None],
                        vol_data[:, j:im_size + j, k:im_size + k, 1, None],
                        vol_data[:, j:im_size + j, k:im_size + k, 2, None]
                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf + 1e-12)  #[...,None]

    return out_vol


