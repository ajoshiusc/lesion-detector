import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util.shape import view_as_windows

check=0;
def read_data(study_dir, nsub, psize, npatch_perslice):
    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    for subj in dirlist:

        t1file = os.path.join(subj, 'T1.nii.gz')
        t2file = os.path.join(subj, 'T2.nii.gz')
        fl = os.path.join(subj, 'FLAIR.nii.gz')

        if not (os.path.isfile(t1file) and os.path.isfile(t2file)
                and os.path.isfile(fl)):
            continue

        if subno < nsub:
            subno = subno + 1
            print("subject %d " % (subno))
        else:
            break
        # Read the three images
        t1 = nl.image.load_img(t1file).get_data()
        t2 = nl.image.load_img(t2file).get_data()
        flair = nl.image.load_img(fl).get_data()

        p = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
        t1 = np.float32(t1) / p

        p = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
        t2 = np.float32(t2) / p

        p = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
        flair = np.float32(flair) / p

        imgs = np.stack((t1, t2, flair), axis=3)
        
        if check==0
        
        
           

        # Read coronal slices

        #        create image

        #        create random patches

    return patch_data  # npatch x width x height x channels
