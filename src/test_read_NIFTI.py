import os
import numpy as np
import nibabel as nib
from skimage.util.shape import view_as_blocks
## data path
first_path='/data_disk/HCP_All'
sublist=os.listdir(first_path)
sublist.sort()
favindex =list(range(1,20))
favlist= [sublist[i] for i in favindex]
## read data and divide it to 32,32,32 patches
window_size=32
window_shape = np.array([window_size, window_size, window_size])
data_all=np.array([])
check = 0
for filename in favlist:
    data_path=os.path.join(first_path,filename,'T1w','T1w_acpc_dc_restore_brain.nii.gz')
    img=nib.load(data_path)
    img_array=img.get_data()
    img_array=np.float32(img_array)/255
    padsize= window_shape-np.array([(img_array.shape[0] % window_shape[0]),(img_array.shape[1] % window_shape[1]),(img_array.shape[2] % window_shape[2])])
    pre_pad=(np.floor(padsize/2))
    pre_pad=pre_pad.astype(int)
    img_array=np.pad(img_array,((pre_pad[0],padsize[0]-pre_pad[0]),(pre_pad[1],padsize[1]-pre_pad[1]),(pre_pad[2],padsize[2]-pre_pad[2])), 'constant')
    img_newshape=view_as_blocks(img_array, (window_size,window_size,window_size))
    temp=img_newshape.shape[0]*img_newshape.shape[1]*img_newshape.shape[2]
    img_newshape=img_newshape.reshape(temp,32,32,32)
    if check == 0 :
        data_all = img_newshape
        check = 1
    else:
        data_all=np.concatenate((data_all,img_newshape))

    #img_newshape=np.reshape(img_newshape)





