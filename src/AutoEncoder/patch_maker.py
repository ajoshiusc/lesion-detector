import os
import numpy as np
import nibabel as nib
from skimage import io
import matplotlib.pyplot as plt

def patch_maker(first_path,second_path,data_name,subnum,window_size,patch_num):
    sublist=os.listdir(first_path)
    sublist.sort()
    favindex =list(range(1,subnum+1))
    favlist= [sublist[i] for i in favindex]
    ## read data and divide it to 32,32,32 patches
    check = 0
    temp=np.zeros((1,window_size,window_size,1))
    ##patch the image
    for filename in favlist:
        data_path=os.path.join(first_path,filename,second_path,data_name)
        img=nib.load(data_path)
        img_array=img.get_data()
        p = np.percentile(np.ravel(img_array), 95)  #normalize to 95 percentile
        img_array=np.float32(img_array)/p
        padsize=np.floor((window_size-1)/2)
        padsize=padsize.astype(int)
        img_array=np.pad(img_array,((padsize,padsize),(padsize,padsize),(0,0)), 'constant')  # zero pad by size of the window
        #plt.imshow(img_array[:,:,156].reshape(img_array.shape[0], img_array.shape[1]))
        #plt.gray()
        for patchN in range(patch_num):
            nZ=np.nonzero(np.ravel(img_array))
            nZ=nZ[0]
            ravel_coordinates=np.random.randint(0,nZ.shape[0])
            coordinates=np.unravel_index(nZ[ravel_coordinates],(img_array.shape)) #read a nonzero coordinate
            temp[0,:,:,0]=img_array[coordinates[0]-padsize:coordinates[0]+padsize+1,coordinates[1]-padsize:coordinates[1]+padsize+1,coordinates[2]]
            if check == 0 :
                img_newshape =temp 
                check = 1
            else:
                img_newshape=np.concatenate((img_newshape,temp)) #concatenate data 
    return img_newshape



    
