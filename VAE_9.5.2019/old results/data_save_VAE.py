import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
from datautils_VAE import read_data, read_data_test,datalist, read_data_brats



def train_save():
    data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/'
    with open('/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt') as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVNU820VND'
    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)

    data = read_data(study_dir=data_dir,
                                       ref_dir=ref_dir,
                                       subids=tbidoneIds,
                                       nsub=253,
                                       psize=[window_H, window_W],
                                       npatch_perslice=1,
                                       slicerange=slicerange,
                                       erode_sz=0,
                                       lesioned=False,
                                       dohisteq=True
                                       )

   



    fig, ax = plt.subplots()
    im = ax.imshow(data[10, :, :, 2])
    print(np.max(data))
    #im = ax.imshow(dataLesion[10, :, :, 3])

    plt.show()


    #np.random.shuffle(data)
    np.savez('data__TBI_histeq.npz', data=data)


def test_save():
    data_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training/'
    ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVNU820VND'
    #study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
    with open('/big_disk/ajoshi/ISLES2015/ISLES2015_Training_done.txt') as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)
    data = read_data_test(study_dir=data_dir,
                                       ref_dir=ref_dir,
                                       subids=tbidoneIds,
                                       nsub=24,
                                       psize=[window_H, window_W],
                                       npatch_perslice=1,
                                       slicerange=slicerange,
                                       erode_sz=0,
                                       lesioned=False,
                                       dohisteq=True
                                       )
    #data=data[:,:,81:101]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0, :, :, 0])
    plt.show()
    np.savez('data_24_ISEL_histeq.npz', data=data)
    return ()


def Brats2015_save(data_files):
    data_dir = '/ImagePTE1/ajoshi/BRATS2015_Training_preprocessed/HGG/'
    ref_dir='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVNU820VND'
    #study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
    with open(data_files) as f:
        tbidoneIds = f.readlines()
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    window_H = 182
    window_W = 218
    slicerange = np.arange(81, 101, dtype=int)
    data = read_data_brats(study_dir=data_dir,
                                       ref_dir=ref_dir,
                                       subids=tbidoneIds,
                                       nsub=220,
                                       psize=[window_H, window_W],
                                       npatch_perslice=1,
                                       slicerange=slicerange,
                                       erode_sz=0,
                                       lesioned=False,
                                       dohisteq=True
                                       )
    #data=data[:,:,81:101]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0, :, :, 0])
    plt.show()
    np.savez('Brats2015_HGG.npz', data=data)
    return ()


#np.random.shuffle(data)
if __name__ == "__main__":
    #datalist('/ImagePTE1/ajoshi/BRATS2015_Training/HGG/')
    Brats2015_save('Brats2015.txt')
    #Brats2015_save()