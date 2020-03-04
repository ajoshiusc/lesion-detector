from __future__ import print_function
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
from torchvision.utils import make_grid, save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
from VAE_model import Encoder, Decoder, VAE_Generator
from tqdm import tqdm

pret = 0
random.seed(8)

input_size = 64


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc + '/VAE_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc + '/VAE_encoder_%d.pth' % epoch))
    encoder.cuda()


# Load all the data
d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__maryland_histeq.npz'
)
X = d['data']
X = np.transpose(X[:, ::2, ::2, :], (0, 3, 1, 2))

d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz'
)
X2 = np.transpose(d['data'][:, ::2, ::2, :], (0, 3, 1, 2))

X = np.concatenate((X, X2))

in_data = torch.from_numpy(X).float()

batch_size = 8

torch.manual_seed(7)
data_loader = torch.utils.data.DataLoader(in_data,
                                          batch_size=batch_size,
                                          shuffle=True)
###### define constant########
input_channels = 3
hidden_size = 8
max_epochs = 100
lr = 3e-4
beta = 0

#######network################
epoch = 99
LM = '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats_results'

##########load low res net##########
G = VAE_Generator(input_channels, hidden_size).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
load_model(24,
           G.encoder,
           G.decoder,
           loc='/home/ajoshi/coding_ground/lesion-detector/prob_vae/results')

out_data = np.zeros(in_data.shape)

G.eval()
with torch.no_grad():

    for i, data in enumerate(tqdm(in_data)):
        data = Variable(data[None, ]).cuda()
        mean, logvar, rec = G(data)
        out_data[i, ] = rec.cpu()

np.savez('rec_data.npz', out_data=out_data, in_data=in_data)
