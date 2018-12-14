from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder import auto_encoder

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data.npz')
data=d['data']
print(data.shape)
var=np.mean(data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
data=data[zplace,:,:,:]
print(data.shape)
trainsize=np.floor((data.shape[0])*0.8)
testsize=np.floor((data.shape[0])*0.9)
#trainsize.astype(int)

train_data = data[0:int(trainsize), :, :, :]
val_data = data[int(trainsize):int(testsize), :, :, :]
test_data=data[int(testsize):, :, :, :]

window_size=64
model=auto_encoder(window_size)
model.fit(train_data,train_data,

                epochs=200,

                batch_size=128,

                shuffle=True,

                validation_data=(val_data, val_data),

               callbacks=[TensorBoard(log_dir='/tmp/tb')])
model.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_4.h5')
