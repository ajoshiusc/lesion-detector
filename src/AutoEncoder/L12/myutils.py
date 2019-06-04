from sklearn.datasets import make_blobs
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
def lession_generator(data, corNum=10):
    
    corruption_index = np.floor(np.random.random_sample(size = corNum)*data.shape[0])
    for j in corruption_index:
        lesion = np.zeros((1,data.shape[1],data.shape[2]))
        mskx=data.shape[1]
        msky=data.shape[2]
        ind = random.randint(5, mskx-5)
        mskx = np.copy(ind)
        ind = random.randint(5, msky-5)
        msky = np.copy(ind)
        centr = np.array([mskx,msky])[None,:]
        blob, _ = make_blobs(
        n_samples=1000,
        n_features=3,
        centers=centr,
        cluster_std=random.uniform(0.5, 1))

        blob = np.int16(
            np.clip(np.round(blob), [0, 0],
                    np.array((data.shape[1],data.shape[2])) - 1))
        lesion[0,blob[:, 0], blob[:, 1]] = 1.0

        #lesion = gaussian_filter(lesion, 5)
        
        
        #lesion /= lesion.max()
        #plt.imshow(data[int(j),:,:])
        #plt.show()
        data[int(j),:,:]=data[int(j),:,:]+lesion
        #plt.imshow(data[int(j),:,:])
        #plt.show()
    return data, corruption_index