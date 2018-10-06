import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


def read_data(study_dir, nsub, psize, npatch_perslice):
    dirlist = glob.glob(study_dir + '/TBI*')
    subno=0
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
        else:
            break
        # Read the three images
        t1 = nl.image.load_img(t1file).get_data()
        t2 = nl.image.load_img(t2file).get_data()
        flair = nl.image.load_img(fl).get_data()

        imgs = np.stack((t1, t2, flair), axis=3)

        for sliceno in range(imgs.shape[2]):
            ptch = extract_patches_2d(
                image=imgs[:, :, sliceno, :],
                patch_size=psize,
                max_patches=npatch_perslice)
            if patch_data.shape[0] == 0:
                patch_data = ptch
            else:
                patch_data = np.concatenate((patch_data, ptch), axis=0)

        # Read coronal slices

        #        create image

        #        create random patches

    return patch_data  # npatch x width x height x channels
