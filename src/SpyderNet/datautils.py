import glob
import nilearn as nl
import os
import numpy as np


def read_data(study_dir, nsub, npatch_perslice):
    dirlist = glob.glob(study_dir + '/TBI*')
    for subj in dirlist:
        t1file = os.path.join(subj,'T1.nii.gz')
        t2file = os.path.join(subj,'T2.nii.gz')
        fl = os.path.join(subj,'FLAIR.nii.gz')

        t1 = nl.image.load_img(t1file).get_data()
        t2 = nl.image.load_img(t2file).get_data()
        flair = nl.image.load_img(fl).get_data()

        imgs = np.stack((t1,t2,flair), axis = 3)

        # Read coronal slices

#        create image

#        create random patches

        return patch_data # npatch x width x height x channels
