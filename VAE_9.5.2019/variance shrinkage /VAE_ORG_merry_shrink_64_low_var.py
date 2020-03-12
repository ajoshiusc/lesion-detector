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
from VAE_model_pixel_vanilla_shrink import Encoder, Decoder, VAE_Generator
pret = 0
random.seed(8)
input_size=64


def show_and_save(file_name, img):
    f = "results/%s.png" % file_name
    save_image(img[2:3, :, :], f, range=[0, 1.5])

    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)


def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), 'results/VAE_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(), 'results/VAE_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_encoder_%d.pth' % epoch))
    encoder.cuda()
  




d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__maryland_histeq.npz')
X=d['data']
X=X[0:2380,:,:,:]
X_train=X[0:-20*20,:,:,:]
X_valid=X[-20*20:,:,:,:]


d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz')
X_train=np.concatenate((X_train,d['data'][0:-20*20,:,:,:]))
X_valid=np.concatenate((X_valid,d['data'][-20*20:,:,:,:]))







X_train = np.transpose(X_train[:,::2,::2,:], (0, 3, 1,2))
X_valid = np.transpose(X_valid[:,::2,::2,:] , (0, 3, 1,2))


input = torch.from_numpy(X_train).float()
validation_data = torch.from_numpy(X_valid).float()

batch_size=8


torch.manual_seed(7)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)
###### define constant########
input_channels = 3
hidden_size = 128
max_epochs = 100
lr = 3e-4
beta =0
logvar_const=-1.54
var_enc=torch.from_numpy(np.zeros((1))).cuda()+logvar_const

#######network################
epoch=20
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats_results'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
opt_enc = optim.Adam(G.parameters(), lr=lr)

fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data = next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()

#######losss#################






def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    dim1=1
    x_temp = x.repeat(dim1, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()
    


    msk2 = torch.tensor(x_temp > -1).float()
    NDim = torch.sum(msk2,(1,2,3))
    std = var_x.mul(0.5).exp_()
    const = (-torch.sum(var_x*msk, (1, 2, 3))) / 2



    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2, 3))
    #const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) #+const2
    BBCE = torch.sum(prob_term / dim1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   

    return -BBCE + KLD



def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 


def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar):
   
    BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*3),x.view(-1, 64*64*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    
    return BBCE +KLD


################################

#if pret == 1:
   #oad_model(499, G.encoder, G.decoder)

##############train#####################
train_loss = 0
valid_loss = 0
pay=0
valid_loss_list, train_loss_list = [], []
for epoch in range(max_epochs):
    train_loss = 0
    valid_loss = 0
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0
        
        mean, logvar, rec_enc = G(datav)
        if beta == 0:
            prob_err = prob_loss_function(rec_enc, var_enc, datav, mean,
                                          logvar)
            prob_err2=beta_loss_function(rec_enc, datav, mean, logvar)
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
            
            mean, logvar, valid_enc  = G(data)
            if beta == 0:
                prob_err = prob_loss_function(valid_enc,var_enc, data,
                                              mean, logvar)
            
            valid_loss += prob_err.item()
        valid_loss /= len(Validation_loader.dataset)

    if epoch == 0:
        best_val = valid_loss
    elif (valid_loss < best_val):
        save_model(epoch, G.encoder, G.decoder)
        pay=0
        best_val = valid_loss
    pay=pay+1
    if(pay==100):
        break


    print(valid_loss)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    _, _, rec_imgs = G(fixed_batch)

    

    show_and_save(
        'Input_epoch_%d.png' % epoch,
        make_grid((fixed_batch.data[0:8, 2:3, :, :]).cpu(), 8, range=[0, 1.5]))
    show_and_save('rec_epoch_%d.png' % epoch,
                  make_grid((rec_imgs.data[0:8, 2:3, :, :]).cpu(), 8))


    show_and_save(
        'Error_epoch_%d.png' % epoch,
        make_grid((fixed_batch.data[0:8, 2:3, :, :] -
                   rec_imgs.data[0:8, 2:3, :, :]).cpu(), 8))

 
######################################

save_model(epoch, G.encoder, G.decoder)
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()
