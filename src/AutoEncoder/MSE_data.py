from read_test_data import read_test_data
from slice2vol_pred import slice2vol_pred
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import nilearn.image as nl
import os
from deep_auto_encoder2 import square_loss
from deep_auto_encoder2 import corrent_loss

alpha=0

#study_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
study_dir ='/big_disk/akrami/Result/SV/'
with open('/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
subids=tbidoneIds[0:100]
MSE = np.zeros(100)

count= 0
for subj in subids:
    t1file = os.path.join(study_dir,"dift1_ %s.nii.gz" % subj)
    #t1file=os.path.join(study_dir,'dift1_ TBI_INVAB314ZE5.nii.gz')
    t2file = os.path.join(study_dir,"dift2_ %s.nii.gz" % subj)
    flfile = os.path.join(study_dir,"difFLAIR_ %s.nii.gz" % subj)
    t1 = nl.image.load_img(t1file).get_data()
    t2 = nl.image.load_img(t2file).get_data()
    flair = nl.image.load_img(flfile).get_data()
    imgs = np.stack((t1, t2, flair), axis=3)
    MS=imgs**2
    MS=MS.mean(axis=0)
    MS=MS.mean(axis=0)
    MS=MS.mean(axis=0)
    MS=MS.mean(axis=0)
    MSE[count]=MS
    count=count+1
order=np.argsort(MSE)
MSElist=[tbidoneIds[i] for i in order]
print(order)
with open("MSEorder.txt", "w") as output:
    output.write(str(MSElist))