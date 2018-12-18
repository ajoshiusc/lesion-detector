from time import time
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import sklearn 



d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz')
data=d['data']
print(data.shape)
testsize=np.floor(0.9*30*32*182)
test_data=data[int(testsize):, :, :, :]
print(test_data.shape)
var=np.mean(test_data[:,:,:,1].squeeze(),2)
var=np.mean(var,1)
zplace=np.where(var != 0)[0]
test_data=test_data[zplace,:,:,:]
np.random.shuffle(test_data)




model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30.h5') 

x_test_noisy=test_data.copy()
x_test_noisy[640,4:16,10:30,2] = 1.6
decoded_imgs = model.predict(x_test_noisy)
print(decoded_imgs.shape)
plt.figure()

for k in range(5):
  j=k*320
  ax=plt.subplot(3, 5, k + 1)
  plt.imshow(x_test_noisy[j, :, :, 2].squeeze(), vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(3, 5, 5 + k + 1)
  plt.imshow(decoded_imgs[j, :, :, 2].squeeze(),vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(3, 5, 10 + k + 1)
  plt.imshow(x_test_noisy[j, :, :, 2].squeeze()-decoded_imgs[j, :, :, 2].squeeze(),vmin=0., vmax=1., cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  

plt.show()