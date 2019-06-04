from time import time
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder2 import auto_encoder
from deep_auto_encoder2 import l21shrink



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

loss='mean_squared_error'
#loss='SV'
alpha=0.1
window_size=64
model=auto_encoder(window_size,loss,alpha)
X=np.copy(train_data)
L = np.zeros(X.shape)
S = np.zeros(X.shape)
iteration=20
lamb=10000
model.load_weights('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_bactchnorm_wights.h5')
checkpointer = ModelCheckpoint('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_L12_50_2_MSE_10000_without5.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False)
for it in range(iteration):
    print ("Out iteration: " , it)
    epochs_num=50   
    ## alternating project, first project to L
    L = X - S
    ## Using L to train the auto-encoder
    model.fit(L,L[:,:,:,2:3],

            epochs=epochs_num,

            batch_size=128,

            shuffle=True,

            validation_split=.1,

            callbacks=[checkpointer])
            ## get optmized L
    L=model.predict(L)        
    ## alternating project, now project to S and shrink S
    S = l21shrink(lamb, (X - L))
   
#print_summary()

