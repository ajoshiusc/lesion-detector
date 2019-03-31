from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import sklearn 

from keras import losses
loss=losses.mean_squared_error

d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz')
data=d['data']
print(data.shape)
testsize=np.floor(0.9*30*32*182)
trainsize=np.floor(0.8*30*32*182)
test_data=data[1:int(trainsize), :, :, :]
print(test_data.shape)
var=np.mean(test_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
test_data=test_data[zplace,:,:,:]
np.random.shuffle(test_data)



pixel_dif1 = test_data[:,1:, :, :] - test_data[:,:-1, :, :]
pixel_dif2 = test_data[:,:, 1:, :] - test_data[:,:, :-1, :]
grad=(pixel_dif1)+(pixel_dif2)




for k in range(5):
  j=k*32*10
  ax=plt.subplot(3, 5, k + 1)
  plt.imshow(test_data[j, :, :, 2].squeeze(), vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(3, 5, 5 + k + 1)
  plt.imshow(grad[j, :, :, 2].squeeze(),vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
 

plt.show()
