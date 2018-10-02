import os
import numpy as np
import nibabel as nib
from skimage.util.shape import view_as_blocks
from model import *
from keras.callbacks import TensorBoard
## data path
first_path='/data_disk/HCP_All'
sublist=os.listdir(first_path)
sublist.sort()
favindex =list(range(1,11))
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
    img_newshape=img_newshape.reshape(temp,32,32,32,1)
    if check == 0 :
        data_all = img_newshape
        check = 1
    else:
        data_all=np.concatenate((data_all,img_newshape))

model=auto_encoder(window_size)
training_percent=9*temp
x_train=data_all[0:training_percent,:,:,:,:]
x_test=data_all[training_percent:,:,:,:,:]


model.fit(x_train, x_train,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
               callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])





decoded_imgs = model.predict(x_test)

""" n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
 """


