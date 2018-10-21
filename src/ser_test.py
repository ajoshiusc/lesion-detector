import nilearn as ni
from nilearn import image
from nilearn.input_data import NiftiMasker
import nibabel as nb
import os
import numpy as np
from ReNA.rena import ReNA
import matplotlib.pyplot as plt
import glob

mask_img = nb.load('../../../stochastic_regularizer/sergul_aydore/supporting_data/grey10_icbm_2mm_bin.nii.gz')
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
def get_single_subject(file_name):
    t1file = os.path.join(file_name, 'T1.nii.gz')
    t2file = os.path.join(file_name, 'T2.nii.gz')
    fl = os.path.join(file_name, 'FLAIR.nii.gz')

    # t1 = image.load_img(t1file).get_data()
    # t2 = image.load_img(t2file).get_data()
    # flair = image.load_img(fl).get_data()

    t1 = nifti_masker.transform(image.load_img(t1file))
    t2 = nifti_masker.transform(image.load_img(t2file))
    flair = nifti_masker.transform(image.load_img(fl))

    p = np.percentile(np.ravel(t1), 95)  # normalize to 95 percentile
    t1 = np.float32(t1) / p

    p = np.percentile(np.ravel(t2), 95)  # normalize to 95 percentile
    t2 = np.float32(t2) / p

    p = np.percentile(np.ravel(flair), 95)  # normalize to 95 percentile
    flair = np.float32(flair) / p

    imgs = np.concatenate((t1, t2, flair))

    return imgs

# TODO: make automated
all_imgs = None
dir_name = "../../sample_data/"
subject_list = glob.glob(dir_name + 'TBI*')

for subject_name in subject_list:
    file_name = subject_name
    imgs = get_single_subject(file_name)
    if all_imgs is None:
        all_imgs = imgs
    else:
        all_imgs = np.concatenate((all_imgs, imgs))


# print list of labels for the plot
labels_plot = []
for zz in subject_list:
    # if there are other type nii.gz not wanted included as shown with 'fse.nii.gz'
    if zz.split('/')[4][0:2] == 'fs':  # ../../sample_data/TBI*/*  >>> 0/1/2/3/4
        continue
    labels_plot.append(zz.split('/')[3] + '-' + zz.split('/')[4][0:2])

print(all_imgs.shape)

n_voxels = all_imgs.shape[1]
n_samples = all_imgs.shape[0]
n_clusters = int(20*n_voxels/100)
cluster = ReNA(scaling=True,
               n_clusters=n_clusters,
               masker=nifti_masker)

cluster.fit(all_imgs)

reduced_images = cluster.transform(all_imgs)
reconstructed_images = cluster.inverse_transform(reduced_images)

mse = np.mean(abs(all_imgs - reconstructed_images)**2, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mse, marker='o')
ax.set_xticks(range(n_samples))
ax.set_xticklabels(labels_plot, rotation=90)
ax.set_ylabel("MSE")
plt.tight_layout()
plt.show()

