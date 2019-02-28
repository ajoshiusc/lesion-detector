import numpy as np
import os
import nilearn as nl
from skimage.io import imsave
from tqdm import tqdm


def read_test_data(study_dir, subid):
    t1file = os.path.join(study_dir, subid, 'T1r.nii.gz')
    t2file = os.path.join(study_dir, subid, 'T2r.nii.gz')
    fl = os.path.join(study_dir, subid, 'FLAIRr.nii.gz')
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
    imgs[imgs < 0] = 0
    imgs[imgs > 1] = 1
    return imgs


def subdata2png(study_dir, subid, axis=2):

    imgs = read_test_data(study_dir, subid)

    for ind in tqdm(range(imgs.shape[axis])):

        im = imgs.take(indices=ind, axis=axis).transpose((1, 0, 2))
        im = np.flip(im, axis=0)

        imsave(
            os.path.join(study_dir, subid, 'pngs',
                         np.str(ind) + '.png'), im)
