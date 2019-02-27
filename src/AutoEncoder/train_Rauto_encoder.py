from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder2 import auto_encoder
from deep_auto_encoder2 import l21shrink

lamb=1

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz')
data=d['data']
print(data.shape)
trainsize=np.floor(0.9*30*32*182)
#trainsize.astype(int)

train_data = data[0:int(trainsize), :, :, :]
print(train_data.shape)
test_data=data[int(trainsize):, :, :, :]
print(test_data.shape)

var=np.mean(train_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
train_data=train_data[zplace,:,:,:]


var=np.mean(test_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
test_data=test_data[zplace,:,:,:]

loss='TV_R'
#loss='SV'
alpha=0.01
window_size=64
model=auto_encoder(window_size,loss,alpha)
X=train_data
L = np.zeros(X.shape)
S = np.zeros(X.shape)
iteration=20
for it in range(iteration):
    print ("Out iteration: " , it)
    ## alternating project, first project to L
    L = X - S
    ## Using L to train the auto-encoder
    model.fit(L,L,

            epochs=50,

            batch_size=128,

            shuffle=True,

            validation_split=.1,

            callbacks=[TensorBoard(log_dir='/tmp/tb')])
            ## get optmized L
    L=model.predict(L)        
    ## alternating project, now project to S and shrink S
    S = l21shrink(lamb, (X - L))
   

model.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_L12_20iter_lamb1_alpha0.01.h5')
