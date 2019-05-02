from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import PIL.Image as Image
from utils import tile_raster_images
#import ImShow as I
def auto_encoder(input_size):
    input_img = Input(shape=(input_size, input_size, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    DAE=Model(input_img, decoded)
    DAE.compile(optimizer='adadelta', loss='binary_crossentropy')

    EDAE=Model(input_img, encoded)
    return DAE, EDAE
def test():
    from keras.datasets import mnist
    import numpy as np
    x_train=np.load(r"data.npk")

    #x_train = x_train.astype('float32') / 255

    #x_train = x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format


    from keras.callbacks import TensorBoard
    (autoencoder,hyden)=auto_encoder(28)
    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=132,
                    shuffle=True)
    inputsize=28
    AES=x_train-autoencoder.predict(x_train)
    AES=AES.reshape(AES.shape[0],784)
    AEL=autoencoder.predict(x_train)
    AEL=AEL.reshape(AEL.shape[0],784)
    Image.fromarray(tile_raster_images(X=AES,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"AES.png")
    Image.fromarray(tile_raster_images(X=AEL,img_shape=(inputsize,inputsize), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"AEL.png")

if __name__ == "__main__":
    test()


