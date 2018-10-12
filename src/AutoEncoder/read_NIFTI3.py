from model2D import *
from keras.callbacks import TensorBoard
from patch_maker import *
from skimage.util.shape import view_as_windows
## data path
first_path='/data_disk/HCP_All'
second_path='T1w'
data_name='T1w_acpc_dc_restore_brain.nii.gz'
subnum=10
window_size=35
patch_num=2000
## patch data
img_newshape=patch_maker(first_path,second_path,data_name,subnum,window_size,patch_num)
## define test and train
x_train = img_newshape
## make test data
sub=11
data_path=os.path.join(first_path,filename,second_path,data_name)
x_test = img_newshape[patch_num+1:,:,:,:]
sublist=os.listdir(first_path)
sublist.sort()
filename=sublist[sub+1]
data_path=os.path.join(first_path,filename,second_path,data_name)
img=nib.load(data_path)
img_array=img.get_data()
p = np.percentile(np.ravel(img_array), 95)  #normalize to 95 percentile
img_array=np.float32(img_array)/p
padsize=np.floor((window_size-1)/2)
padsize=padsize.astype(int)
img_array=np.pad(img_array,((padsize,padsize),(padsize,padsize),(0,0)), 'constant')  # zero pad by size of the window
img_newshape=view_as_windows(img_array[:,:,150], (window_size,window_size)
temp=img_newshape.shape[0]*img_newshape.shape[1]
x_test=img_newshape.reshape(temp,32,32,1)

## fit the data
model=auto_encoder(window_size)
model.fit(x_train, x_train,
                epochs=20,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test),
               callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])


decoded_imgs = model.predict(x_test)

        