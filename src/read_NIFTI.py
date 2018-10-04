import os
import numpy as np
import nibabel as nib
from skimage.util.shape import view_as_windows
from skimage import io
from model import *
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
## data path
first_path='/data_disk/HCP_All'
second_path='T1w'
data_name='T1w_acpc_dc_restore_brain.nii.gz'
subnum=11

sublist=os.listdir(first_path)
sublist.sort()
favindex =list(range(1,subnum))
favlist= [sublist[i] for i in favindex]
## read data and divide it to 32,32,32 patches
window_size=35
window_shape = np.array([window_size, window_size])
data_all=np.array([])
check = 0
temp=np.zeros((1,window_size,window_size,1))
for filename in favlist:
    data_path=os.path.join(first_path,filename,second_path,data_name)
    img=nib.load(data_path)
    img_array=img.get_data()
    p = np.percentile(np.ravel(img_array), 95)
    img_array=np.float32(img_array)/p
    padsize=np.floor((window_size-1)/2)
    padsize=padsize.astype(int)
    img_array=np.pad(img_array,((padsize,padsize),(padsize,padsize),(0,0)), 'constant')
    plt.imshow(img_array[:,:,156].reshape(img_array.shape[0], img_array.shape[1]))
    plt.gray()
    for patchN in range(2000):
        coordinates=[np.random.randint(padsize,img_array.shape[0]-padsize),np.random.randint(padsize,img_array.shape[1]-padsize),np.random.randint(padsize,img_array.shape[2]-padsize)]
        while img_array[coordinates[0],coordinates[1],coordinates[2]] == 0:
            coordinates=[np.random.randint(padsize,img_array.shape[0]-padsize),np.random.randint(padsize,img_array.shape[1]-padsize),np.random.randint(padsize,img_array.shape[2]-padsize)]
        if check == 0 :
            temp[0,:,:,0]=img_array[coordinates[0]-padsize:coordinates[0]+padsize+1,coordinates[1]-padsize:coordinates[1]+padsize+1,coordinates[2]]
            img_newshape =temp 
            check = 1
        else:
            img_newshape=np.concatenate((img_newshape,temp))
        plt.imshow(img_newshape[0,:,:,0].reshape(img_newshape.shape[1], img_newshape.shape[2]))
        plt.gray()


    
