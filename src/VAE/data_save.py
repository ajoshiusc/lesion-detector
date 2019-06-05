import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_utils import read_data, read_data_test


def train_save():
    data_dir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
    with open('/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt') as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

<<<<<<< HEAD
    window_H=182
    window_W=218
    slicerange=np.arange(20, 162, dtype=int)
    data = read_data(
        study_dir=data_dir,
        subids=tbidoneIds,
        nsub=179,
        psize=[window_H, window_W],
        npatch_perslice=1,
        slicerange=slicerange)
#data=data[:,:,81:101]
=======
    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)
    data, mask = read_data(study_dir=data_dir,
                        subids=tbidoneIds,
                        nsub=1,
                        psize=[window_H, window_W],
                        npatch_perslice=1,
                        slicerange=slicerange,
                        erode_sz=0)
    #data=data[:,:,81:101]
>>>>>>> 42041a1839a6a6c6f25c9445818fde54a374471b
    fig, ax = plt.subplots()
    im = ax.imshow(data[10, :, :, 3])
#    im = ax.imshow(mask[0, :, :])

    plt.show()
<<<<<<< HEAD
#np.random.shuffle(data)
    np.savez('/big_disk/akrami/git_repos/lesion-detector/src/VAE/data_179_maryland_140s.npz', data=data)
=======
    #np.random.shuffle(data)
    np.savez(
        'data_100_maryland.npz',
        data=data)


>>>>>>> 42041a1839a6a6c6f25c9445818fde54a374471b
def test_save():
    data_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training/'
    #study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
    with open('/big_disk/ajoshi/ISLES2015/ISLES2015_Training_done.txt') as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)
    data = read_data_test(study_dir=data_dir,
                          subids=tbidoneIds,
                          nsub=24,
                          psize=[window_H, window_W],
                          npatch_perslice=1,
                          slicerange=slicerange)
    #data=data[:,:,81:101]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0, :, :, 0])
    plt.show()
    np.savez(
        '/big_disk/akrami/git_repos/lesion-detector/src/VAE/data_24_ISEL.npz',
        data=data)
    return ()


#np.random.shuffle(data)
if __name__ == "__main__":
    train_save()