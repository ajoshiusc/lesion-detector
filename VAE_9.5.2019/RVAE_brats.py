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
from VAE_model_pixel import Encoder, Decoder, VAE_Generator
pret = 0
random.seed(8)


def show_and_save(file_name, img):
    f = "/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats_results_RVAE/%s.png" % file_name
    save_image(img[2:3, :, :], f, range=[0, 1.5])

    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)


def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(), './VAE_GAN_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
  


d = np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats2015_HGG.npz')
X = d['data']
X = X[:, :, :, 0:3]
X = X.astype('float64')
max_val = np.max(X)
#X = X/ max_val
X_train, X_valid = train_test_split(X,
                                    test_size=0.1,
                                    random_state=10002,
                                    shuffle=False)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_valid = np.transpose(X_valid, (0, 3, 1, 2))

input = torch.from_numpy(X_train).float()
validation_data = torch.from_numpy(X_valid).float()

batch_size = 8

torch.manual_seed(7)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                                batch_size=batch_size,
                                                shuffle=True)
###### define constant########
input_channels = 3
hidden_size = 64
max_epochs = 100
lr = 3e-4
beta = 0.00000001

#######network################
epoch=31
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats_results'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)
opt_enc = optim.Adam(G.parameters(), lr=lr)

fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data = next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()

#######losss#################


def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret, 1)
    return ret


def BMSE_loss(Y, X, beta, sigma, Dim):
    term1 = -((1 + beta) / beta)
    K1 = 1 / pow((2 * math.pi * (sigma**2)), (beta * Dim / 2))
    term2 = MSE_loss(Y, X)
    term3 = torch.exp(-(beta / (2 * (sigma**2))) * term2)
    loss1 = torch.sum(term1 * (K1 * term3 - 1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma = 1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128 * 128 * 3),
                         x.view(-1, 128 * 128 * 3), beta, sigma, 128 * 128 * 3)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(
            MSE_loss(recon_x.view(-1, 128 * 128 * 3),
                     x.view(-1, 128 * 128 * 3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    x_temp = x.repeat(10, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()

    std = var_x.mul(0.5).exp_()
    #std_all=torch.prod(std,dim=1)
    const = (-torch.sum(var_x*msk, (1, 2, 3))) / 2
    #const=const.repeat(10,1,1,1) ##check if it is correct


    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2, 3))
    const2 = -((128 * 128 * 3) / 2) * math.log((2 * math.pi))

    #term2=torch.log(const+0.0000000000001)
    prob_term = const + (-(0.5) * term1) + const2

    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE + KLD


def beta_prob_loss_function(recon_x, logvar_x, x, mu, logvar, beta):
    x_temp = x.repeat(10, 1, 1, 1)
    std = logvar_x.mul(0.5).exp_()
    # dim=(128*128*3)
    std_all = torch.prod(std, dim=1)
    std_all = torch.prod(std_all, dim=1)
    std_all = torch.prod(std_all, dim=1)+1e-6 
    
    term1 = -(beta + 1) / (beta * torch.pow(((std_all**2) * (2 * math.pi)),
                                            (beta / 2)))
    term2 = torch.sum((((recon_x - x_temp) / std)**2), (1, 2, 3))
    term2 = torch.exp(-(0.5 * beta * term2))
    term3 = -1 / (torch.pow(std_all, beta) * (((beta + 1) *
                                               ((2 * math.pi)**beta))**(0.5)))

    prob_term = term1 * term2 + term3

    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE + KLD


################################

if pret == 1:
    load_model(499, G.encoder, G.decoder)

##############train#####################
train_loss = 0
valid_loss = 0
valid_loss_list, train_loss_list = [], []
for epoch in range(max_epochs):
    train_loss = 0
    valid_loss = 0
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0

        mean, logvar, rec_enc, var_enc = G(datav)
        if beta == 0:
            prob_err = prob_loss_function(rec_enc, var_enc, datav, mean,
                                          logvar)
        else:
            prob_err = beta_prob_loss_function(rec_enc, var_enc, datav, mean,
                                               logvar, beta)
        err_enc = prob_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss += prob_err.item()
    train_loss /= len(train_loader.dataset)

    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            mean, logvar, valid_enc, valid_var_enc = G(data)
            if beta == 0:
                prob_err = prob_loss_function(valid_enc, valid_var_enc, data,
                                              mean, logvar)
            else:
                prob_err = beta_prob_loss_function(valid_enc, valid_var_enc,
                                                   data, mean, logvar, beta)
            valid_loss += prob_err.item()
        valid_loss /= len(Validation_loader.dataset)

    if epoch == 0:
        best_val = valid_loss
    elif (valid_loss < best_val):
        save_model(epoch, G.encoder, G.decoder)
        best_val = valid_loss

    print(valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    _, _, rec_imgs, var_img = G(fixed_batch)

    show_and_save(
        'Input_epoch_%d.png' % epoch,
        make_grid((fixed_batch.data[0:8, 2:3, :, :]).cpu(), 8, range=[0, 1.5]))
    show_and_save('rec_epoch_%d.png' % epoch,
                  make_grid((rec_imgs.data[0:8, 2:3, :, :]).cpu(), 8))
    #samples = G.decoder(fixed_noise)
    #show_and_save('samples_epoch_%d.png' % epoch ,make_grid((samples.data[0:8,2:3,:,:]).cpu(),8))
    show_and_save(
        'Error_epoch_%d.png' % epoch,
        make_grid((fixed_batch.data[0:8, 2:3, :, :] -
                   rec_imgs.data[0:8, 2:3, :, :]).cpu(), 8))

    #localtime = time.asctime( time.localtime(time.time()) )
    #D_real_list_np=(D_real_list).to('cpu')
######################################

save_model(epoch, G.encoder, G.decoder)
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()
