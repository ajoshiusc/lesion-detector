from time import time
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder2 import auto_encoder

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz')
data=d['data']
#print(data.shape)
#npatch=(182 - 64)**2
#trainsize=np.floor(0.8*6*8*30*182)
#testsize=np.floor(0.9*6*8*30*182)
trainsize=np.floor(0.8*32*30*182)
testsize=np.floor(0.9*32*30*182)
#trainsize.astype(int)
print(data.shape)

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


checkpointer = ModelCheckpoint('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_SV_1.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False)
loss='SV'
#loss='MSE_FLAIR'
alpha=1
window_size=64
model=auto_encoder(window_size,loss,alpha)
history_callback=model.fit(train_data,train_data[:,:,:,2:3],

                epochs=200,

                batch_size=128,

                shuffle=True,

                validation_data=(val_data, val_data[:,:,:,2:3]),

               callbacks=[checkpointer])
#model.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF.h5')
#callbacks=[TensorBoard(log_dir='/tmp/tb')]

acc_history = history_callback.history["loss"]
Train_acc_history = np.array(acc_history)
acc_history = history_callback.history["val_loss"]
Val_acc_history = np.array(acc_history)

#ig, ax1 = plt.subplots()
    #ax1.set_title("Initial Learning rate =" + str(ilearning_rate) +"  Activation= relu")
#ax1.plot(Val_acc_history,label='Validation')
#ax1.set_ylabel('Loss', color='b')
#ax1.set_xlabel('epoches')
#plt.plot(Train_acc_history,label='Train')  
#plt.legend()
np.savetxt("p_model_200_512_merryland_30_MSEF2_block_Val_loss_history_SV_1.txt", Val_acc_history, delimiter=",")