import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
from datautils import read_data_block

data_dir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
with open('/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

window_size=64
data = read_data_block(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=30,
    psize=[window_size, window_size],
    stride=8)
#np.random.shuffle(data)
np.savez('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/data/tp_data_merryland_30__block_nf.npz', data=data)