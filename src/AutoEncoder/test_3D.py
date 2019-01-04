from read_test_data import read_test_data
from slice2vol_pred import slice2vol_pred
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import nilearn.image as ni
import os
from deep_auto_encoder2 import square_loss

alpha=1

#study_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
with open('/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
subid=tbidoneIds[30]
t1file = os.path.join(study_dir, subid, 'T1mni.nii.gz')
t1model=ni.load_img(t1file )
test_image=read_test_data(study_dir, subid)
im_size=64
model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_SV.h5',custom_objects={'SV': square_loss(alpha)})
vol_data=test_image
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

img = ni.new_img_like(t1model, out_vol[:,:,:,0])
img.to_filename('/big_disk/akrami/outt1_merry30_SV.nii.gz')
img = ni.new_img_like(t1model, out_vol[:,:,:,1])
img.to_filename('/big_disk/akrami/outt2_merry30_SV.nii.gz')
img = ni.new_img_like(t1model, out_vol[:,:,:,2])
img.to_filename('/big_disk/akrami/outFLAIR_merry30_SV.nii.gz')

img = ni.new_img_like(t1model, vol_data[:,:,:,0]-out_vol[:,:,:,0])
img.to_filename('/big_disk/akrami/dift1_merry30_SV.nii.gz')
img = ni.new_img_like(t1model, vol_data[:,:,:,1]-out_vol[:,:,:,1])
img.to_filename('/big_disk/akrami/dift2_merry30_SV.nii.gz')
img = ni.new_img_like(t1model, vol_data[:,:,:,2]-out_vol[:,:,:,2])
img.to_filename('/big_disk/akrami/difFLAIR_merry30_SV.nii.gz')




#test_re =slice2vol_pred(model, test_image, im_size,8)
