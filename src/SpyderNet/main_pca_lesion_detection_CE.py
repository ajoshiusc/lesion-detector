# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from datautils import read_data, slice2vol_pred, binary_erosion

from pca_autoencoder import pca_autoencoder_masked as pca_ae_msk
import keras, keras.layers as L
from keras.models import load_model
import numpy as np
from keras.losses import mse
import nilearn.image as ni

ERODE_SZ = 1
DO_TRAINING = 0

data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'
tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'

with open(tbi_done_list) as f:
    tbidoneIds = f.readlines()

# Get the list of subjects that are correctly registered
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

if DO_TRAINING:
    model1 = pca_ae_msk(image_size=64, code_size=128) # was 2*2*512

    data, mask_data = read_data(
        study_dir=data_dir,
        subids=tbidoneIds,
        nsub=3,
        psize=[64, 64],
        npatch_perslice=32,
        erode_sz=ERODE_SZ)

    lesion = data[:, :, :, -1]

    l_data = data.copy()
    l_data[:, :, :, 1:3] = 0*l_data[:, :, :, 1:3] + lesion[:, :, :, None]

    model1.fit(
        x=[l_data[:, :, :, :3], mask_data[..., None]],
        y=data[:, :, :, 2:3] * mask_data[..., None],
        shuffle=False,
        validation_split=.2,
        batch_size=128,
        epochs=20)

    data2 = model1.predict([l_data[:, :, :, :3], mask_data[..., None]])
    print(np.mean((data2.flatten() - data[..., 2].flatten())**2))

    model1.save('tracktbi_pilot_pca_autoencoder.h5')

model1 = load_model('tracktbi_pilot_pca_autoencoder.h5')

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
t1msko = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/TBI_INVJH729XF3/T1mni.mask.nii.gz'
)

t1msk = binary_erosion(t1msk, iterations=ERODE_SZ)

pt1 = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
t1 = np.float32(t1) / pt1

pt2 = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
t2 = np.float32(t2) / pt2

pflair = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
flair = np.float32(flair) / pflair

dat = np.stack((t1, t2, flair), axis=3)

#build_pca_autoencoder(model1, td, [64, 64, 3], step_size=1)
out_vol = slice2vol_pred(model1.predict, dat, t1msk, 64, step_size=10)
#%%
t1 = ni.new_img_like(t1o, out_vol[:, :, :, 0] * pt1)
t1.to_filename('TBI_INVJH729XF3_rec_t1.nii.gz')

t1mskfile = ni.new_img_like(t1msko, t1msk)
t1mskfile.to_filename('TBI_INVJH729XF3_rec_t1.mask.nii.gz')

t2 = ni.new_img_like(t1o, out_vol[:, :, :, 1] * pt2)
t2.to_filename('TBI_INVJH729XF3_rec_t2.nii.gz')

flair = ni.new_img_like(t1o, out_vol[:, :, :, 2] * pflair)
flair.to_filename('TBI_INVJH729XF3_rec_flair.nii.gz')

err = ni.new_img_like(t1o, np.mean((out_vol - dat)**2, axis=3))
err.to_filename('TBI_INVJH729XF3_rec_err.nii.gz')

err = ni.new_img_like(
    t1o, np.mean((out_vol[..., 2, None] - dat[..., 2, None])**2, axis=3))
err.to_filename('TBI_INVJH729XF3_rec_flair_err.nii.gz')

np.savez('lesion_det.npz', out_vol=out_vol, dat=dat)
print('vol created')
print(out_vol.shape)
print('done')

# In[8]:
