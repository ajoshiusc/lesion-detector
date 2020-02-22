import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from skimage.util.shape import view_as_windows
from scipy.ndimage.morphology import binary_erosion
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter


def read_data(study_dir,
              subids,
              nsub,
              psize,
              npatch_perslice,
              slicerange,
              erode_sz=1,
              lesioned=False):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0

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
        t1 = nl.image.load_img(t1_file).get_data()[:, :, slicerange]
        t2 = nl.image.load_img(t2_file).get_data()[:, :, slicerange]
        flair = nl.image.load_img(flair_file).get_data()[:, :, slicerange]
        t1_msk = np.float32(
            nl.image.load_img(t1_mask_file).get_data()[:, :, slicerange] > 0)

        p = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
        t1 = np.float32(t1) / p

        p = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
        t2 = np.float32(t2) / p

        p = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
        flair = np.float32(flair) / p

        if erode_sz > 0:
            t1_msk = binary_erosion(t1_msk, iterations=erode_sz)

        imgs = np.stack((t1, t2, flair, t1_msk), axis=3)

        if lesioned == True:
            lesion = np.zeros(t1.shape)
            mskx, msky, mskz = np.where(t1 > 0)
            ind = np.random.randint(0, mskx.shape[0])
            mskx = mskx[ind]
            msky = msky[ind]
            mskz = mskz[ind]
            #    xyz = np.unravel_index(ind, shape=t1.shape)
            centr = np.array([mskx, msky, mskz])[None, :]
            blob, _ = make_blobs(n_samples=10000,
                                 n_features=3,
                                 centers=centr,
                                 cluster_std=np.random.uniform(20, 40))

            blob = np.int16(
                np.clip(np.round(blob), [0, 0, 0],
                        np.array(t1.shape) - 1))
            lesion[blob[:, 0], blob[:, 1], blob[:, 2]] = 1.0
            #lesion[blob.ravel]=1.0

            lesion = gaussian_filter(lesion, 5)
            lesion /= lesion.max()
            imgs = np.concatenate(
                (imgs[:, :, :, :3], lesion[:, :, :, None], imgs[:, :, :, -1:]),
                axis=3)

        # Generate random patches
        # preallocate
        if subno == 1:
            num_slices = imgs.shape[2]
            patch_data = np.zeros((nsub * npatch_perslice * num_slices,
                                   psize[0], psize[1], imgs.shape[-1]))

        for sliceno in tqdm(range(len(slicerange))):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
                                      patch_size=psize,
                                      max_patches=npatch_perslice)

            strt_ind = (
                subno -
                1) * npatch_perslice * num_slices + sliceno * npatch_perslice

            end_ind = (subno - 1) * npatch_perslice * num_slices + (
                sliceno + 1) * npatch_perslice

            patch_data[strt_ind:end_ind, :, :, :] = ptch

    mask_data = patch_data[:, :, :, -1]
    patch_data = patch_data[:, :, :, :-1]
    patch_data[:, :, :, -1] = patch_data[:, :, :, -1] * mask_data

    return patch_data, mask_data  # npatch x width x height x channels


def read_data_block(study_dir, subids, nsub, psize, stride):
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    #patch_data=[]
    for subj in subids:

        t1file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        t2file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        fl = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')

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

        vol_data = np.stack((t1, t2, flair), axis=3)

        #window_shape = (64, 64, 1, 1)
        #temp = view_as_windows(imgs, window_shape, step=8)
        #patch_data=temp.reshape(temp.shape[0]*temp.shape[1])

        out_vol = []  # output volume
        vol_size = vol_data.shape
        step_size = stride

        #im_size= psize
        for j in tqdm(range(0, vol_size[0] - psize[0], step_size)):
            for k in range(0, vol_size[1] - psize[1], step_size):

                ptch = np.transpose(
                    vol_data[j:psize[0] + j, k:psize[1] + k, :, :],
                    (2, 0, 1, 3))
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

    out_vol = np.zeros(vol_data.shape[:3])  # output volume
    indf = np.zeros(vol_data.shape[:3])  # dividing factor
    vol_size = vol_data.shape

    print('Putting together slices to form volume')
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size + k] += model_pred([
                vol_data[:, j:im_size + j, k:im_size + k, 0, None],
                vol_data[:, j:im_size + j, k:im_size + k, 1, None],
                vol_data[:, j:im_size + j, k:im_size + k, 2, None]
            ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf + 1e-12)  #[...,None]

    return out_vol


def read_data_test(study_dir, subids, nsub, psize, npatch_perslice,
                   slicerange):
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    patch_data = np.zeros((0, 0, 0, 0))
    for subj in subids:

        t1file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        t2file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        fl = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')
        seg = os.path.join(study_dir, subj, 'SEGMENTATIONmni.nii.gz')

        if not (os.path.isfile(t1file) and os.path.isfile(t2file)
                and os.path.isfile(fl) and os.path.isfile(seg)):
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
        segment = nl.image.load_img(seg).get_data()

        p = np.percentile(np.ravel(t1), 95)  #normalize to 95 percentile
        t1 = np.float32(t1) / p

        p = np.percentile(np.ravel(t2), 95)  #normalize to 95 percentile
        t2 = np.float32(t2) / p

        p = np.percentile(np.ravel(flair), 95)  #normalize to 95 percentile
        flair = np.float32(flair) / p

        imgs = np.stack((t1, t2, flair, segment), axis=3)

        for sliceno in (slicerange):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
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