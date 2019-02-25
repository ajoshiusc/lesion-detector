from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder3 import auto_encoder

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz')
data=d['data']
print(data.shape)
trainsize=np.floor(0.8*30*32*182)
testsize=np.floor(0.9*30*32*182)
#trainsize.astype(int)

train_data = data[0:int(trainsize), :, :, :]
print(train_data.shape)
val_data = data[int(trainsize):int(testsize), :, :, :]
print(val_data.shape)
test_data=data[int(testsize):, :, :, :]
print(test_data.shape)
var=np.mean(train_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
train_data=train_data[zplace,:,:,:]

var=np.mean(val_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
val_data=val_data[zplace,:,:,:]

var=np.mean(test_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
test_data=test_data[zplace,:,:,:]

loss='mean_squared_error'
loss='TV'
alpha=0
window_size=64
model=auto_encoder(window_size,loss,alpha)
model.fit(train_data,train_data,

                epochs=200,

                batch_size=128,

                shuffle=True,

                validation_data=(val_data, val_data),

               callbacks=[TensorBoard(log_dir='/tmp/tb')])
model.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_SV_new_re.h5')
