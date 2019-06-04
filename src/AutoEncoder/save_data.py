import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_utils import read_data

data_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training/'
with open('/big_disk/ajoshi/ISLES2015/ISLES2015_Training_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

window_size=64
data = read_data(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=100,
    psize=[window_size, window_size],
    npatch_perslice=32)
#np.random.shuffle(data)
np.savez('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/ISLES2015_5__32_nf.npz', data=data)