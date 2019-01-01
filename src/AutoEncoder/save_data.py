import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/big_disk/akrami/git_repos/lesion-detector/src')
sys.path.insert(0, '/big_disk/akrami/git_repos/lesion-detector/src/SpyderNet')
from datautils import read_data

data_dir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
with open('/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

window_size=64
data = read_data(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=30,
    psize=[window_size, window_size],
    npatch_perslice=32)
#np.random.shuffle(data)
np.savez('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__32_nf.npz', data=data)