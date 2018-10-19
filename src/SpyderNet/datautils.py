import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


def read_data(study_dir, nsub, psize, npatch_perslice):
    """
    study_dir      : is the study directory where each subjects have different
                    type of MRI stored.
    nsub           : is indicating the number of subject we have in
                    the study file
    psize          : is the patch size
    npatch_perslice: is number of patches per scan slice
    """
    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    for subj in dirlist:

        t1file = os.path.join(subj, 'T1.nii.gz')
        t2file = os.path.join(subj, 'T2.nii.gz')
        fl = os.path.join(subj, 'FLAIR.nii.gz')
        # check if three MRI contrast exist if not skip subject
        if not (os.path.isfile(t1file) and os.path.isfile(t2file)
                and os.path.isfile(fl)):
            print("one file missing (T1,T2 or FLAIR), go to the next subject")
            subno += 1
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
