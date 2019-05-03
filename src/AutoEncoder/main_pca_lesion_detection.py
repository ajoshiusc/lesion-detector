# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajoshiusc/lesion-detector/blob/master/main_anatomy_map.ipynb)

# In[1]:
from datautils import read_data
""" Main script that calls the functions objects"""
data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'
tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'

with open(tbi_done_list) as f:
    tbidoneIds = f.readlines()

# Get the list of subjects that are correctly registered
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

data = read_data(
    study_dir=data_dir,
    subids=tbidoneIds,
    nsub=2,
    psize=[64, 64],
    npatch_perslice=16)

model = pca_ae([64, 64, 3], 32)

print(data.shape)

data1 = np.reshape(
    data, (data.shape[0], data.shape[1] * data.shape[2], data.shape[3]))

data1.shape
data2 = np.reshape(
    data1, (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))

print(np.linalg.norm(data - data2))

# In[8]:
