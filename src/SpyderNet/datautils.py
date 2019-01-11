import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from scipy.ndimage.morphology import binary_erosion


def read_data(study_dir, subids, nsub, psize, npatch_perslice, erode_sz=1):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    for subj in subids:

        t1_file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        flair_file = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')

        if not (os.path.isfile(t1_file) and os.path.isfile(t1_mask_file)
                and os.path.isfile(t2_file) and os.path.isfile(flair_file)):
            continue

        if subno < nsub:
            subno = subno + 1
            print("subject %d " % (subno))
        else:
            break
        # Read the three images
        t1 = nl.image.load_img(t1_file).get_data()
        t2 = nl.image.load_img(t2_file).get_data()
        flair = nl.image.load_img(flair_file).get_data()
        t1_msk = np.float32(nl.image.load_img(t1_mask_file).get_data() > 0)

        p = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
        t1 = np.float32(t1) / p

        p = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
        t2 = np.float32(t2) / p

        p = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
        flair = np.float32(flair) / p

        t1_msk = binary_erosion(t1_msk, iterations=erode_sz)

        imgs = np.stack((t1, t2, flair, t1_msk), axis=3)

        for sliceno in range(imgs.shape[2]):
            ptch = extract_patches_2d(
                image=imgs[:, :, sliceno, :],
                patch_size=psize,
                max_patches=npatch_perslice)
            if patch_data.shape[0] == 0:
                patch_data = ptch[..., :-1]
                mask_data = ptch[..., 1]
            else:
                patch_data = np.concatenate((patch_data, ptch[..., :-1]),
                                            axis=0)
                mask_data = np.concatenate((mask_data, ptch[..., -1]), axis=0)

    return patch_data, mask_data  # npatch x width x height x channels


def slice2vol_pred(model_pred, vol_data, mask_data, im_size, step_size=1):
    # model_pred: predictor that gives 2d images as outputs
    # vol_data this is 3d images + 4th dim for different modalities
    # im_size number with size of images (for 64x64 images) it is 64
    # step_size : size of offset in stacked reconstruction

    out_vol = np.zeros(vol_data.shape)  # output volume
    indf = np.zeros(vol_data.shape[:3])  # dividing factor
    vol_size = vol_data.shape

    print('Putting together slices to form volume')
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size + k, :] += model_pred([
                vol_data[:, j:im_size + j, k:im_size + k],
                mask_data[:, j:im_size + j, k:im_size + k, None]
            ])
            #                        [
            #                        vol_data[:, j:im_size + j, k:im_size + k, 0, None],
            #                        vol_data[:, j:im_size + j, k:im_size + k, 1, None],
            #                        vol_data[:, j:im_size + j, k:im_size + k, 2, None]
            #                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf[..., None] + 1e-12)  #

    return out_vol
