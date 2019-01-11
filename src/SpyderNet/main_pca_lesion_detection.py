# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
from datautils import read_data, slice2vol_pred
from pca_autoencoder import pca_autoencoder_masked as pca_ae_msk
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

data, mask_data = read_data(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=30,
    psize=[64, 64],
    npatch_perslice=32,
    erode_sz=5)

model1 = pca_ae_msk(64, 2 * 2 * 512)

model1.fit(
    x=[data, mask_data[..., None]],
    y=data * mask_data[..., None],
    shuffle=False,
    validation_split=.2,
    batch_size=128,
    epochs=200)

model1.save('model1.h5')

data2 = model1.predict([data, mask_data[..., None]])

print(np.mean((data2.flatten() - data.flatten())**2))

t1 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/T1mni.nii.gz'
).get_data()
t1msk = np.float32(
    ni.load_img(
        '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/T1mni.mask.nii.gz'
    ).get_data() > 0)
t2 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/T2mni.nii.gz'
).get_data()
flair = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/FLAIRmni.nii.gz'
).get_data()
t1o = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/T1mni.nii.gz'
)

dat = np.stack((t1, t2, flair), axis=3)

print(dat.shape)
dat = np.float32(dat)

#build_pca_autoencoder(model1, td, [64, 64, 3], step_size=1)
out_vol = slice2vol_pred(model1.predict, dat, t1msk, 64, step_size=10)
#%%
t1 = ni.new_img_like(t1o, out_vol[:, :, :, 0])
t1.to_filename('TBI_INVJH729XF3_rec_t1.nii.gz')


t2 = ni.new_img_like(t1o, out_vol[:, :, :, 1])
t2.to_filename('TBI_INVJH729XF3_rec_t2.nii.gz')

flair = ni.new_img_like(t1o, out_vol[:, :, :, 2])
flair.to_filename('TBI_INVJH729XF3_rec_flair.nii.gz')

err = ni.new_img_like(t1o, np.mean((out_vol - dat)**2, axis=3))
err.to_filename('TBI_INVJH729XF3_rec_err.nii.gz')

np.savez('lesion_det.npz', out_vol=out_vol, dat=dat)
print('vol created')
print(out_vol.shape)
print('done')

# In[8]:
