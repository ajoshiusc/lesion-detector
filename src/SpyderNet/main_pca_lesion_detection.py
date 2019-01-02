# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
from datautils import read_data, slice2vol_pred
from pca_autoencoder import pca_autoencoder as pca_ae
import keras, keras.layers as L
import numpy as np
from keras.losses import mse
import nilearn.image as ni

data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'
tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'

with open(tbi_done_list) as f:
    tbidoneIds = f.readlines()

# Get the list of subjects that are correctly registered
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

data = read_data(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=10,
    psize=[64, 64],
    npatch_perslice=16)

model1 = pca_ae([64, 64, 3], 2*2*512)
model1.compile(optimizer='adamax', loss='mse')

model1.fit(
    x=data, y=data, shuffle=True, validation_split=.2, batch_size=32, epochs=100)

data2 = model1.predict(data)

print(np.mean((data2.flatten() - data.flatten())**2))

t1 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVBB041DZW/T1mni.nii.gz'
).get_data()
t2 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVBB041DZW/T2mni.nii.gz'
).get_data()
flair = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVBB041DZW/FLAIRmni.nii.gz'
).get_data()
t1o = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVBB041DZW/T1mni.nii.gz'
)

dat = np.stack((t1, t2, flair), axis=3)

print(dat.shape)
dat = np.float32(dat)
td = dat.copy()

#build_pca_autoencoder(model1, td, [64, 64, 3], step_size=1)
out_vol = slice2vol_pred(model1.predict, dat, 64, step_size=1)

t1=ni.new_img_like(t1o,out_vol[:,:,:,0])
t1.to_filename('rec_t1.nii.gz')

print('vol created')
print(out_vol.shape)
print('done')

# In[8]:
