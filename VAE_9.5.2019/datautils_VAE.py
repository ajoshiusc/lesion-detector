import glob
import nilearn as nl
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from scipy.ndimage.morphology import binary_erosion
import random
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
from skimage.transform import match_histograms
import torch
from torch.autograd import Variable
import fnmatch


def read_data(study_dir,
              ref_dir,
              subids,
              nsub,
              psize,
              npatch_perslice,
              slicerange,
              erode_sz=1,
              lesioned=False,
              dohisteq=False):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    ref_imgs = 0
    ref_imgs_set = False
    if ref_dir:
        ref_imgs_set = True
        t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(ref_dir,'T2mni.nii.gz')
        flair_file = os.path.join(ref_dir,'FLAIRmni.nii.gz')


        t1 = nl.image.load_img(t1_file).get_data()
        t2 = nl.image.load_img(t2_file).get_data()
        flair = nl.image.load_img(flair_file).get_data()


        p=np.max(t1)
        t1=np.float32(t1) / p

        p=np.max(t2)
        t2=np.float32(t2) / p

        p=np.max(flair)
        flair=np.float32(flair) / p

        ref_imgs = np.stack((t1, t2, flair), axis=3)

        #p = np.percentile(np.ravel(t1), 99)  #normalize to 95 percentile
        #t1 = np.float32(t1) / p

        #p = np.percentile(np.ravel(t2), 99)  #normalize to 95 percentile
        #t2 = np.float32(t2) / p

        #p = np.percentile(np.ravel(flair), 99)  #normalize to 95 percentile
        #flair = np.float32(flair) / p

        
    for subj in subids:

        t1_file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        flair_file = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')

        if not (os.path.isfile(t1_file)
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
        t1_msk = nl.image.load_img(t1_file).get_data()
        t1_msk[t1_msk!=0]=1

        

        
        #t1_msk = binary_erosion(t1_msk, iterations=erode_sz)
        

        imgs = np.stack((t1, t2, flair), axis=3)
       
        if ref_imgs_set == False:
            ref_imgs = imgs
            ref_imgs_set = True

        if dohisteq == True:
            imgs = match_histograms(image=imgs,
                                    reference=ref_imgs,
                                    multichannel=True)

        t1_msk=np.float32(t1_msk)
        t1_msk=np.reshape(t1_msk,(t1_msk.shape[0],t1_msk.shape[1],t1_msk.shape[2],1))
        imgs=imgs*t1_msk
        imgs=imgs[:, :,slicerange,:]
        print(np.max(imgs))
        if lesioned == True:
            lesion = np.zeros(t1.shape)
            mskx, msky, mskz = np.where(t1 > 0)
            ind = random.randint(0, mskx.shape[0])
            mskx = mskx[ind]
            msky = msky[ind]
            mskz = mskz[ind]
            #    xyz = np.unravel_index(ind, shape=t1.shape)
            centr = np.array([mskx, msky, mskz])[:, None]
            blob, _ = make_blobs(n_samples=10,
                                 n_features=3,
                                 centers=centr,
                                 cluster_std=random.uniform(0, 30))

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

        for sliceno in tqdm(range(num_slices)):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
                                      patch_size=psize,
                                      max_patches=npatch_perslice,
                                      random_state=1121)

            strt_ind = (
                subno -
                1) * npatch_perslice * num_slices + sliceno * npatch_perslice

            end_ind = (subno - 1) * npatch_perslice * num_slices + (
                sliceno + 1) * npatch_perslice

            patch_data[strt_ind:end_ind, :, :, :] = ptch

    #mask_data = patch_data[:, :, :, -1]
    patch_data = patch_data[:, :, :, :]
    return patch_data  # npatch x width x height x channels


def read_data_test(study_dir,
              ref_dir,
              subids,
              nsub,
              psize,
              npatch_perslice,
              slicerange,
              erode_sz=1,
              lesioned=False,
              dohisteq=False):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    ref_imgs = 0
    ref_imgs_set = False
    if ref_dir:
        ref_imgs_set = True
        t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(ref_dir,'T2mni.nii.gz')
        flair_file = os.path.join(ref_dir,'FLAIRmni.nii.gz')
        


        t1 = nl.image.load_img(t1_file).get_data()
        t2 = nl.image.load_img(t2_file).get_data()
        flair = nl.image.load_img(flair_file).get_data()
        



        p=np.max(t1)
        t1=np.float32(t1) / p

        p=np.max(t2)
        t2=np.float32(t2) / p

        p=np.max(flair)
        flair=np.float32(flair) / p

        ref_imgs = np.stack((t1, t2, flair), axis=3)

        #p = np.percentile(np.ravel(t1), 99)  #normalize to 95 percentile
        #t1 = np.float32(t1) / p

        #p = np.percentile(np.ravel(t2), 99)  #normalize to 95 percentile
        #t2 = np.float32(t2) / p

        #p = np.percentile(np.ravel(flair), 99)  #normalize to 95 percentile
        #flair = np.float32(flair) / p

        
    for subj in subids:

        t1_file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        flair_file = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')
        seg = os.path.join(study_dir, subj, 'SEGMENTATIONmni.nii.gz')

        if not (os.path.isfile(t1_file)
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
        segment = nl.image.load_img(seg).get_data()
        t1_msk = nl.image.load_img(t1_file).get_data()
        t1_msk[t1_msk!=0]=1

        
        #t1_msk = binary_erosion(t1_msk, iterations=erode_sz)

        imgs = np.stack((t1, t2, flair), axis=3)


        if ref_imgs_set == False:
            ref_imgs = imgs
            ref_imgs_set = True

        if dohisteq == True:
            imgs = match_histograms(image=imgs,
                                    reference=ref_imgs,
                                    multichannel=True)
        
        t1_msk=np.float32(t1_msk)
        t1_msk=np.reshape(t1_msk,(t1_msk.shape[0],t1_msk.shape[1],t1_msk.shape[2],1))
        imgs=imgs*t1_msk
        segment=np.reshape(segment,(segment.shape[0],segment.shape[1],segment.shape[2],1))
        imgs = np.concatenate((imgs, segment), axis=3)

        imgs=imgs[:, :,slicerange,:]


        if lesioned == True:
            lesion = np.zeros(t1.shape)
            mskx, msky, mskz = np.where(t1 > 0)
            ind = random.randint(0, mskx.shape[0])
            mskx = mskx[ind]
            msky = msky[ind]
            mskz = mskz[ind]
            #    xyz = np.unravel_index(ind, shape=t1.shape)
            centr = np.array([mskx, msky, mskz])[:, None]
            blob, _ = make_blobs(n_samples=10,
                                 n_features=3,
                                 centers=centr,
                                 cluster_std=random.uniform(0, 30))

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

        for sliceno in tqdm(range(num_slices)):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
                                      patch_size=psize,
                                      max_patches=npatch_perslice,
                                      random_state=1121)

            strt_ind = (
                subno -
                1) * npatch_perslice * num_slices + sliceno * npatch_perslice

            end_ind = (subno - 1) * npatch_perslice * num_slices + (
                sliceno + 1) * npatch_perslice

            patch_data[strt_ind:end_ind, :, :, :] = ptch

    #mask_data = patch_data[:, :, :, -1]
    patch_data = patch_data[:, :, :, :]
    return patch_data  # npatch x width x height x channels


def read_data_brats(study_dir,
              ref_dir,
              subids,
              nsub,
              psize,
              npatch_perslice,
              slicerange,
              erode_sz=1,
              lesioned=False,
              dohisteq=False):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    ref_imgs = 0
    ref_imgs_set = False
    if ref_dir:
        ref_imgs_set = True
        t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(ref_dir,'T2mni.nii.gz')
        flair_file = os.path.join(ref_dir,'FLAIRmni.nii.gz')
        


        t1 = nl.image.load_img(t1_file).get_data()
        t2 = nl.image.load_img(t2_file).get_data()
        flair = nl.image.load_img(flair_file).get_data()
        



        p=np.max(t1)
        t1=np.float32(t1) / p

        p=np.max(t2)
        t2=np.float32(t2) / p

        p=np.max(flair)
        flair=np.float32(flair) / p

        ref_imgs = np.stack((t1, t2, flair), axis=3)

        #p = np.percentile(np.ravel(t1), 99)  #normalize to 95 percentile
        #t1 = np.float32(t1) / p

        #p = np.percentile(np.ravel(t2), 99)  #normalize to 95 percentile
        #t2 = np.float32(t2) / p

        #p = np.percentile(np.ravel(flair), 99)  #normalize to 95 percentile
        #flair = np.float32(flair) / p

        
    for subj in subids:

        t1_file = os.path.join(study_dir, subj, 'T1.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(study_dir, subj, 'T2.nii.gz')
        flair_file = os.path.join(study_dir, subj, 'Flair.nii.gz')
        seg = os.path.join(study_dir, subj, 'truth.nii.gz')

        if not (os.path.isfile(t1_file)
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
        segment = nl.image.load_img(seg).get_data()
        t1_msk = nl.image.load_img(t1_file).get_data()
        t1_msk[t1_msk!=0]=1

                #p = np.percentile(np.ravel(t1), 99)  #normalize to 95 percentile
        #t1 = np.float32(t1) / p

        #p = np.percentile(np.ravel(t2), 99)  #normalize to 95 percentile
        #t2 = np.float32(t2) / p

        #p = np.percentile(np.ravel(flair), 99)  #normalize to 95 percentile
        #flair = np.float32(flair) / p
        #t1_msk = binary_erosion(t1_msk, iterations=erode_sz)

        imgs = np.stack((t1, t2, flair), axis=3)


        if ref_imgs_set == False:
            ref_imgs = imgs
            ref_imgs_set = True

        if dohisteq == True:
            imgs = match_histograms(image=imgs,
                                    reference=ref_imgs,
                                    multichannel=True)
        
        t1_msk=np.float32(t1_msk)
        t1_msk=np.reshape(t1_msk,(t1_msk.shape[0],t1_msk.shape[1],t1_msk.shape[2],1))
        imgs=imgs*t1_msk
        segment=np.reshape(segment,(segment.shape[0],segment.shape[1],segment.shape[2],1))
        imgs = np.concatenate((imgs, segment), axis=3)

        imgs=imgs[:, :,slicerange,:]


        if lesioned == True:
            lesion = np.zeros(t1.shape)
            mskx, msky, mskz = np.where(t1 > 0)
            ind = random.randint(0, mskx.shape[0])
            mskx = mskx[ind]
            msky = msky[ind]
            mskz = mskz[ind]
            #    xyz = np.unravel_index(ind, shape=t1.shape)
            centr = np.array([mskx, msky, mskz])[:, None]
            blob, _ = make_blobs(n_samples=10,
                                 n_features=3,
                                 centers=centr,
                                 cluster_std=random.uniform(0, 30))

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

        for sliceno in tqdm(range(num_slices)):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
                                      patch_size=psize,
                                      max_patches=npatch_perslice,
                                      random_state=1121)

            strt_ind = (
                subno -
                1) * npatch_perslice * num_slices + sliceno * npatch_perslice

            end_ind = (subno - 1) * npatch_perslice * num_slices + (
                sliceno + 1) * npatch_perslice

            patch_data[strt_ind:end_ind, :, :, :] = ptch

    #mask_data = patch_data[:, :, :, -1]
    patch_data = patch_data[:, :, :, :]
    return patch_data  # npatch x width x height x channels



def slice2vol_pred(model_pred, vol_data, mask_data, im_size, step_size=32):
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

            data = torch.tensor(vol_data[:, j:im_size + j, k:im_size +
                                         k].transpose((0, 3, 1, 2)))
            data = Variable(data).cuda()
            #            model_pred = model_pred.to('cpu')
            for ind in range(data.shape[0]):
                _, _, out_dat, _ = model_pred(data[None, ind, :])
                out_vol[ind, j:im_size + j, k:im_size + k, :] += out_dat.cpu().detach().numpy()[0,].transpose((1,2,0))
            #                        [
            #                        vol_data[:, j:im_size + j, k:im_size + k, 0, None],
            #                        vol_data[:, j:im_size + j, k:im_size + k, 1, None],
            #                        vol_data[:, j:im_size + j, k:im_size + k, 2, None]
            #                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf[..., None] + 1e-12)  #

    return out_vol


def  datalist(data_dir):
        #for filename in files:
    a = open("Brats2015.txt", "w")
    for name in os.listdir(data_dir):
        a.write(str(name))
        a.write("\n")