from time import time
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from deep_auto_encoder3 import auto_encoder
#from deep_auto_encoder2 import FLAIR_loss
from keras.models import load_model
#from deep_auto_encoder3 import square_loss


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


checkpointer = ModelCheckpoint('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_TV_R_high_bactchnorm_0.5_abs.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False)
loss='mean_squared_error'

alpha=0
window_size=64
model=auto_encoder(window_size,loss,alpha)
#model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_SV_1.h5',custom_objects={'SV': square_loss(alpha)})
#model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_bactchnorm.h5',custom_objects={'MSE_FLAIR': FLAIR_loss(alpha)})
#model.load_weights('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSE_deep3.h5')
#model.compile(loss='SV')

history_callback=model.fit(train_data,train_data[:,:,:,2:3],

                epochs=50,

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
#np.savetxt("p_model_200_512_merryland_30_MSEF2_block_Val_loss_history_SV_bactchnorm.txt", Val_acc_history, delimiter=",")
model.summary()