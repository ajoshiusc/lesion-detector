import numpy as np
import os
import nilearn as nl
def read_test_data(study_dir, subid):
    t1file = os.path.join(study_dir, subid, 'T1mni.nii.gz')
    t2file = os.path.join(study_dir, subid, 'T1mni.nii.gz')
    fl = os.path.join(study_dir, subid, 'FLAIRmni.nii.gz')
    if not (os.path.isfile(t1file) and os.path.isfile(t2file)
            and os.path.isfile(fl)):
        return 0
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
    return imgs