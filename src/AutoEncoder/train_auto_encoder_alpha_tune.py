from time import time
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder2 import auto_encoder
from deep_auto_encoder2 import FLAIR_loss
from keras.models import load_model
from deep_auto_encoder2 import square_loss

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/ISLES2015_5__32_nf.npz')
data=d['data']
#print(data.shape)
#npatch=(182 - 64)**2
#trainsize=np.floor(0.8*6*8*30*182)
#testsize=np.floor(0.9*6*8*30*182)
trainsize=np.floor(0.8*32*5*182)
testsize=np.floor(1*32*5*182)
#trainsize.astype(int)
print(data.shape)

train_data = data[0:int(trainsize), :, :, :]
print(train_data.shape)
val_data = data[int(trainsize):int(testsize), :, :, :]
print(val_data.shape)
#test_data=data[int(testsize):, :, :, :]
#print(test_data.shape)
var=np.mean(train_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
train_data=train_data[zplace,:,:,:]

var=np.mean(val_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
val_data=val_data[zplace,:,:,:]

#var=np.mean(test_data[:,:,:,1].squeeze(),2)
#var=np.mean(var,1)
#zplace=np.where(var != 0)[0]
#test_data=test_data[zplace,:,:,:]

window_size=64
iteration=30
alpha=[]
loss='TV_R'
for  it in range(iteration):
    alpha[it]=np.random.uniform
    checkpointer = ModelCheckpoint('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_TV_R_%s bactchnorm.h5'% alpha[it], monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False)
    model=auto_encoder(window_size,loss,alpha[it])
    model.load_weights('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_bactchnorm_wights.h5')
    history_callback=model.fit(train_data,train_data[:,:,:,2:3],

                epochs=50,

                batch_size=128,

                shuffle=True,

                validation_data=(val_data, val_data[:,:,:,2:3]),

               callbacks=[checkpointer])
np.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/TR_alpha.npz', alpha)

