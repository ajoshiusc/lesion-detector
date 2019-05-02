from read_test_data import read_test_data
from slice2vol_pred import slice2vol_pred
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import nilearn.image as ni
import os
from deep_auto_encoder2 import square_loss
from deep_auto_encoder2 import corrent_loss

alpha=0

#study_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
study_dir ='/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
with open('/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
subids=tbidoneIds[0:100]
t1file = os.path.join(study_dir, subids[1], 'T1mni.nii.gz')
t1model=ni.load_img(t1file )
model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_SV.h5',custom_objects={'SV': square_loss(alpha)})
#model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_RAE.h5',custom_objects={'RAE': corrent_loss(alpha)})
step_size=8
im_size=64
count=0
for subj in subids:
    t1file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
    test_image=read_test_data(study_dir, subj)
    vol_data=test_image
    out_vol = np.zeros(vol_data.shape[:]) # output volume
    indf = np.zeros(vol_data.shape[:])
    vol_size = vol_data.shape
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size +
                    k,:] += model.predict([
                    vol_data[:, j:im_size + j, k:im_size + k,:],
                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k,:] += 1
    out_vol = out_vol / (indf + 1e-12)  #[...,None]
    img = ni.new_img_like(t1model, out_vol[:,:,:,0])
    img.to_filename("/big_disk/akrami/Result/SV/outt1_ %s.nii.gz" % subj)
    img = ni.new_img_like(t1model, out_vol[:,:,:,1])
    img.to_filename("/big_disk/akrami/Result/SV/outt2_ %s.nii.gz" % subj )
    img = ni.new_img_like(t1model, out_vol[:,:,:,2])
    img.to_filename("/big_disk/akrami/Result/SV/outFLAIR_ %s.nii.gz" % subj )

    img = ni.new_img_like(t1model, vol_data[:,:,:,0]-out_vol[:,:,:,0])
    img.to_filename("/big_disk/akrami/Result/SV/dift1_ %s.nii.gz" % subj)
    img = ni.new_img_like(t1model, vol_data[:,:,:,1]-out_vol[:,:,:,1])
    img.to_filename("/big_disk/akrami/Result/SV/dift2_ %s.nii.gz" % subj)
    img = ni.new_img_like(t1model, vol_data[:,:,:,2]-out_vol[:,:,:,2])
    img.to_filename("/big_disk/akrami/Result/SV/difFLAIR_ %s.nii.gz" % subj)







#test_re =slice2vol_pred(model, test_image, im_size,8)
