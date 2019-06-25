# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
from datautils import read_data, slice2vol_pred, binary_erosion
#from pca_autoencoder import pca_autoencoder_masked as pca_ae_msk
import keras, keras.layers as L
from keras.models import load_model
import numpy as np
from keras.losses import mse
import nilearn.image as ni
from VAE_models import train, VAE_nf
from sklearn.model_selection import train_test_split
import torch

IM_SZ = 64
ERODE_SZ = 0
DO_TRAINING = 1
CODE_SZ = 32

data_dir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'
tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt'

with open(tbi_done_list) as f:
    tbidoneIds = f.readlines()

# Get the list of subjects that are correctly registered
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

if DO_TRAINING:
    model1 = VAE_nf(CODE_SZ)

    #    pca_ae_msk(IM_SZ, 2 * 2 * 512)

    data, mask_data = read_data(study_dir=data_dir,
                                subids=tbidoneIds,
                                nsub=32,
                                psize=[IM_SZ, IM_SZ],
                                npatch_perslice=4,
                                erode_sz=ERODE_SZ)

    #    X_train, X_valid = train_test_split(data, test_size=0.1, random_state=10002,shuffle=False)

   # data = torch.from_numpy(data).float()
    data = np.transpose(data[:, :, :, :3], (0, 3, 1, 2))

    model1.to('cuda')
#    data = (data).to('cuda')
    model1.have_cuda = True

    train(model1, data, device='cuda', epochs=200, batch_size=64)
    torch.save(model1, 'maryland_rao_v1_pca_autoencoder2rmsprop.pth')
else:
    model1 = torch.load('maryland_rao_v1_pca_autoencoder2rmsprop.pth')
"""     model1.to('cpu')
    data=(data).to('cpu')
    model1.have_cuda = False

    data2,_,_ = model1(data)
    data=data.numpy()
    data2=data2.detach().numpy()
    print(np.sum((data2.flatten() - data.flatten())**2))
 """
#    model1.save('maryland_rao_v1_pca_autoencoder.h5')

#model1 = load_model('maryland_rao_v1_pca_autoencoder.h5')

t1 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/T1mni.nii.gz'
).get_data()
t1msk = np.float32(
    ni.load_img(
        '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/T1mni.mask.nii.gz'
    ).get_data() > 0)
t2 = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/T2mni.nii.gz'
).get_data()
flair = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/FLAIRmni.nii.gz'
).get_data()
t1o = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/T1mni.nii.gz'
)
t1msko = ni.load_img(
    '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVYM889KY4/T1mni.mask.nii.gz'
)

t1msk = binary_erosion(t1msk, iterations=ERODE_SZ)

pt1 = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
t1 = np.float32(t1) / pt1

pt2 = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
t2 = np.float32(t2) / pt2

pflair = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
flair = np.float32(flair) / pflair

dat = np.stack((t1, t2, flair), axis=3)

dat = torch.from_numpy(dat).float()

#build_pca_autoencoder(model1, td, [64, 64, 3], step_size=1)
model1.have_cuda = False
model1.to('cpu')
out_vol = slice2vol_pred(model1, dat, IM_SZ, step_size=10)
#%%
t1 = ni.new_img_like(t1o, out_vol[:, :, :, 0] * pt1)
t1.to_filename('TBI_INVYM889KY4_rec_t1.nii.gz')

t1mskfile = ni.new_img_like(t1msko, t1msk)
t1mskfile.to_filename('TBI_INVYM889KY4_rec_t1.mask.nii.gz')

t2 = ni.new_img_like(t1o, out_vol[:, :, :, 1] * pt2)
t2.to_filename('TBI_INVYM889KY4_rec_t2.nii.gz')

flair = ni.new_img_like(t1o, out_vol[:, :, :, 2] * pflair)
flair.to_filename('TBI_INVYM889KY4_rec_flair.nii.gz')

err = ni.new_img_like(t1o, np.mean((out_vol - dat.numpy())**2, axis=3))
err.to_filename('TBI_INVYM889KY4_rec_err.nii.gz')

err = ni.new_img_like(
    t1o, np.mean((out_vol[..., 2, None] - dat.numpy()[..., 2, None])**2,
                 axis=3))
err.to_filename('TBI_INVYM889KY4_rec_flair_err.nii.gz')

np.savez('lesion_det.npz', out_vol=out_vol, dat=dat)
print('vol created')
print(out_vol.shape)
print('done')

# In[8]:
