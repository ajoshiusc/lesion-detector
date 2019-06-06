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

    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)

    dataLesion, maskLesion = read_data(study_dir=data_dir,
                                       subids=tbidoneIds,
                                       nsub=10,
                                       psize=[window_H, window_W],
                                       npatch_perslice=1,
                                       slicerange=slicerange,
                                       erode_sz=0,
                                       lesioned=True)

    data, mask = read_data(study_dir=data_dir,
                           subids=tbidoneIds,
                           nsub=100,
                           psize=[window_H, window_W],
                           npatch_perslice=1,
                           slicerange=slicerange,
                           erode_sz=0)

    data[:dataLesion.
         shape[0], :, :, 1:3] = data[:dataLesion.shape[0], :, :, 1:
                                     3] + dataLesion[:, :, :, 3][:, :, :, None]

    mask[:dataLesion.shape[0], :, :] = maskLesion

    fig, ax = plt.subplots()
    im = ax.imshow(data[10, :, :, 2])
    #im = ax.imshow(dataLesion[10, :, :, 3])

    plt.show()

    #np.random.shuffle(data)
    np.savez('data_100_AL_maryland.npz', data=data)


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
    np.savez('data_24_ISEL.npz', data=data)
    return ()


#np.random.shuffle(data)
if __name__ == "__main__":
    train_save()