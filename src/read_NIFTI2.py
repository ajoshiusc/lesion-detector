from model2D import *
from keras.callbacks import TensorBoard
from patch_maker import *
## data path
first_path='/data_disk/HCP_All'
second_path='T1w'
data_name='T1w_acpc_dc_restore_brain.nii.gz'
subnum=5
window_size=35
patch_num=10
## patch data
img_newshape=patch_maker(first_path,second_path,data_name,subnum,window_size,patch_num)
## define test and train
x_train = img_newshape[0:patch_num+1,:,:,:]
x_test = img_newshape[patch_num+1:,:,:,:]
## fit the data
model=auto_encoder(window_size)
model.fit(x_train, x_train,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
               callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])


decoded_imgs = model.predict(x_test)

##plot the output
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(window_size, window_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(window_size, window_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

