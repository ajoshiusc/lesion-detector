from keras.datasets import mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def MNIST_reshape():
   (X_t, _), (X_te,y_test) = mnist.load_data()
#np.save('x_train', x_org_train)

#X_t, _), (X_te,y_test) = mnist.load_data()

   for i in range(X_t.shape[0]):
      if i==0:
         X= cv2.resize(X_t[i,:,:], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
         X=X.reshape((1,32,32))
      else:
         temp=cv2.resize(X_t[i,:,:], dsize=(32, 32), interpolation=cv2.INTER_CUBIC) 
         temp=temp.reshape((1,32,32))
         X=np.append(X,temp ,axis=0)
       
       
       
   for i in range(X_te.shape[0]):
      if i==0:
         x_test= cv2.resize(X_te[i,:,:], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
         x_test=x_test.reshape((1,32,32))
      else:
         temp=cv2.resize(X_te[i,:,:], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
         temp=temp.reshape((1,32,32))
         x_test=np.append(x_test,temp ,axis=0)  

   np.save('X', X)
   np.save('x_test', x_test)
   return()

def TBI():
   d=np.load('data_100_AL_maryland_140.npz')
   data=d['data']
   for i in tqdm(range(data.shape[0])):
      if i==0:
         X= cv2.resize(data[i,:,:,:], dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
         X=X.reshape((1,128,128,3))
      else:
         temp=cv2.resize(data[i,:,:,:], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) 
         temp=temp.reshape((1,128,128,3))
         X=np.append(X,temp ,axis=0)
   return X
def ISEL():
   d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE/data_24_ISEL.npz')
   data=d['data']
   for i in tqdm(range(data.shape[0])):
      if i==0:
         X= cv2.resize(data[i,:,:,:], dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
         X=X.reshape((1,128,128,4))
      else:
         temp=cv2.resize(data[i,:,:,:], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) 
         temp=temp.reshape((1,128,128,4))
         X=np.append(X,temp ,axis=0)
   return X

if __name__ == "__main__":
   X_r=TBI()
   fig, ax = plt.subplots()
   im = ax.imshow(X_r[0,:,:,0])
   plt.show()
np.savez('data_100_AL_maryland_140_r.npz', data=X_r)       


