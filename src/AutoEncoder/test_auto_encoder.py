from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import sklearn 



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


model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_3.h5') 

x_test_noisy=test_data.copy()
x_test_noisy[2,4:16,10:30,2] = 3
decoded_imgs = model.predict(x_test_noisy)
print(decoded_imgs.shape)
plt.figure()

for j in range(5):
  ax=plt.subplot(3, 5, j + 1)
  plt.imshow(x_test_noisy[j, :, :, 2].squeeze(), vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(3, 5, 5 + j + 1)
  plt.imshow(decoded_imgs[j, :, :, 2].squeeze(),vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(3, 5, 10 + j + 1)
  plt.imshow(x_test_noisy[j, :, :, 2].squeeze()-decoded_imgs[j, :, :, 2].squeeze(),vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  

plt.show()