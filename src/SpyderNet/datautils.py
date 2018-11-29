import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


def read_data(study_dir, subids, nsub, psize, npatch_perslice):
#    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    for subj in subids:

        t1file = os.path.join(study_dir, subj, 'T1.nii.gz')
        t2file = os.path.join(study_dir, subj, 'T2.nii.gz')
        fl = os.path.join(study_dir, subj, 'FLAIR.nii.gz')

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



def slice2vol_pred(model_pred, vol_data, im_size, step_size=1):
# model_pred: predictor that gives 2d images as outputs
# vol_data this is 3d images + 4th dim for different modalities
# im_size number with size of images (for 64x64 images) it is 64
# step_size : size of offset in stacked reconstruction
    
    out_vol = np.zeros(vol_data.shape[:3]) # output volume
    indf = np.zeros(vol_data.shape[:3]) # dividing factor
    vol_size = vol_data.shape

    print('Putting together slices to form volume')
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size +
                    k] += model_pred([
                        vol_data[:, j:im_size + j, k:im_size + k, 0, None],
                        vol_data[:, j:im_size + j, k:im_size + k, 1, None],
                        vol_data[:, j:im_size + j, k:im_size + k, 2, None]
                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf + 1e-12)  #[...,None]

    return out_vol
