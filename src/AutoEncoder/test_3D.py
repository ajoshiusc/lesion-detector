from read_test_data import read_test_data
from slice2vol_pred import slice2vol_pred
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import nilearn.image as ni
import os
from deep_auto_encoder2 import square_loss
from deep_auto_encoder2 import corrent_loss
from deep_auto_encoder2 import TV_reconstructed_loss
import math

alpha=0.5

study_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
#study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
with open('/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
subid=tbidoneIds[62]
#subid=tbidoneIds[30]
t1file = os.path.join(study_dir, subid, 'T1mni.nii.gz')
t1model=ni.load_img(t1file )
test_image=read_test_data(study_dir, subid)
im_size=64
model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_L12_20iter_lamb1_alpha0.5.h5',custom_objects={'TV_R': TV_reconstructed_loss(alpha)})
#model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_SV_new_re.h5',custom_objects={'SV': square_loss(alpha)})
#model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_RAE_0.3.h5',custom_objects={'RAE': corrent_loss(alpha)})

vol_data=np.copy(test_image)
out_vol = np.zeros(vol_data.shape[:]) # output volume
indf = np.zeros(vol_data.shape[:])
vol_size = vol_data.shape
step_size=8
for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
    for k in range(0, vol_size[2] - im_size, step_size):
        out_vol[:, j:im_size + j, k:im_size +
                k,:] += model.predict([
                    vol_data[:, j:im_size + j, k:im_size + k,:],
                ]).squeeze()
        indf[:, j:im_size + j, k:im_size + k,:] += 1
out_vol = out_vol / (indf + 1e-12)  #[...,None]

MSE_image=((vol_data[:,:,:,:]-out_vol[:,:,:,:])**2)
MSE_image=MSE_image.mean(axis=3)
img = ni.new_img_like(t1model, MSE_image)
img.to_filename('/big_disk/akrami/MSE_TBI62_TV_R_0.5.nii.gz')

img = ni.new_img_like(t1model, out_vol[:,:,:,0])
img.to_filename('/big_disk/akrami/outt1_TBI62_TV_R_0.5.nii.gz')
img = ni.new_img_like(t1model, out_vol[:,:,:,1])
img.to_filename('/big_disk/akrami/outt2_TBI62_TV_R_0.5.nii.gz')
img = ni.new_img_like(t1model, out_vol[:,:,:,2])
img.to_filename('/big_disk/akrami/outFLAIR_TBI62_TV_R_0.5.nii.gz')


img = ni.new_img_like(t1model, vol_data[:,:,:,0]-out_vol[:,:,:,0])
img.to_filename('/big_disk/akrami/dift1_TBI62_TV_R_0.5.nii.gz')
img = ni.new_img_like(t1model, vol_data[:,:,:,1]-out_vol[:,:,:,1])
img.to_filename('/big_disk/akrami/dift2_TBI62_TV_R_0.5.nii.gz')
img = ni.new_img_like(t1model, vol_data[:,:,:,2]-out_vol[:,:,:,2])
img.to_filename('/big_disk/akrami/difFLAIR_TBI62_TV_R_0.5.nii.gz')




#test_re =slice2vol_pred(model, test_image, im_size,8)
