import numpy as np
import os
import nilearn as nl
from scipy.misc import imsave
from tqdm import tqdm
import random
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter


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


def gen_lesion(study_dir, subid):
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

    lesion = np.zeros(t1.shape)
    mskx, msky, mskz = np.where(t1 > 0)
    ind = random.randint(0, mskx.shape[0])
    mskx = mskx[ind]
    msky = msky[ind]
    mskz = mskz[ind]
    #    xyz = np.unravel_index(ind, shape=t1.shape)
    centr = np.array([mskx, msky, mskz])[:, None]
    blob, _ = make_blobs(
        n_samples=10,
        n_features=3,
        centers=centr,
        cluster_std=random.uniform(0, 30))

    blob = np.int16(np.clip(np.round(blob), [0, 0, 0], np.array(t1.shape) - 1))
    lesion[blob[:, 0], blob[:, 1], blob[:, 2]] = 1.0

    lesion = gaussian_filter(lesion, 5)
    lesion /= lesion.max()

    #    lesion = nl.image.new_img_like()
    blb = nl.image.new_img_like(t1file, lesion)
    blb.to_filename('blob.nii.gz')

    p = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
    t1 = np.float32(t1) / p

    p = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
    t2 = np.float32(t2) / p
    t2 = t2 + lesion

    p = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
    flair = np.float32(flair) / p
    flair = flair + lesion

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
