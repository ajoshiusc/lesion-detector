from __future__ import print_function
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.signal
from sklearn.model_selection import train_test_split


#####read data######################
d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/data_24_ISEL.npz')
X = d['data']
####################################



if __name__ == "__main__":

    y_true = X[0:(15*20), :, :, 3]
    y_true = np.reshape(y_true, (-1, 1))
    
    y_probas = X[0:(15*20), :, :, 2]
    y_probas =y_probas /np.max(y_probas)
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
 
    
    y_probas = np.reshape(y_probas, (-1, 1,128,128))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[tpr>=0.5]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()
    
    np.savez('/big_disk/akrami/git_repos/lesion-detector/src/VAE/y_probas_VAE_notcons.npz', data=y_probas)
    #np.savez('/big_disk/akrami/git_repos/lesion-detector/src/VAE/y_true_VAE.npz', data=y_true)
    y_probas[y_probas >=0.7]=1
    y_probas[y_probas <0.7]=0

    y_probas = np.reshape(y_probas, (-1,128*128*20))
    y_true = np.reshape(y_true, (-1,128*128*20))
    
    dice=0
    for i in range(y_probas.shape[0]):
        seg=y_probas[i,:]
        gth=y_true[i,:]
        dice += np.sum(seg[gth==1])*2.0 / (np.sum(gth) + np.sum(seg))
        #print((dice))
    print((dice)/y_probas.shape[0])
    
    