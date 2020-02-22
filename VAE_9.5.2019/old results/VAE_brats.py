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
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
from VAE_model_pixel_vanilla import Encoder,Decoder,VAE_Generator


pret=0
random.seed(8)

            
def show_and_save(file_name,img):
    f = "/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/result_VAE_reg/%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()

    
def load_model(epoch, encoder, decoder, D):
    #  restore models
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    D.load_state_dict(torch.load('VAE_GAN_D_%d.pth' % epoch))
    D.cuda()


d=np.load('Brats2015_HGG.npz')
X=d['data']
X=X[:,:,:,0:3]
X = X.astype('float64')
max_val=np.max(X)
#X = X/ max_val
X_train, X_valid = train_test_split(X, test_size=0.2, random_state=10002,shuffle=False)
X_valid,X_test=train_test_split(X_valid, test_size=0.5, random_state=10001,shuffle=False)
X_train = np.transpose(X_train, (0, 3, 1,2))
X_valid = np.transpose(X_valid , (0, 3, 1,2))


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
hidden_size = 64
max_epochs =40
lr = 3e-4
beta = 0





#####

G=VAE_Generator(input_channels, hidden_size).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
opt_enc = optim.Adam(G.parameters(), lr=lr)
fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data= next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()


########loss##################

def MSE_loss(Y, X):
    ret = (X- Y) ** 2
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
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,128*128*3)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD
##########load pre train model#########
if pret==1:
    load_model(499, G.encoder, G.decoder, D)

#############train##########################
train_loss=0
valid_loss=0
valid_loss_list, train_loss_list= [], []
for epoch in range(max_epochs):
    train_loss=0
    valid_loss=0
    for data in train_loader:
        batch_size = data.size()[0]
        datav = Variable(data).cuda()
        

        mean, logvar, rec_enc = G(datav)
        beta_err=beta_loss_function(rec_enc, datav, mean, logvar,beta) 
        err_enc = beta_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss+=beta_err.item()
    train_loss /= len(train_loader.dataset)



    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            mean,logvar, valid_rec = G(data)
            beta_err=beta_loss_function(valid_rec, data, mean, logvar,beta) 
            valid_loss+=beta_err.item()
        valid_loss /= len(Validation_loader.dataset)

    
    print(valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    _, _, rec_imgs = G(fixed_batch)
    show_and_save('Input_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]).cpu(),8))
    show_and_save('rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data[:,2:3,:,:]).cpu(),8))
    samples = G.decoder(fixed_noise)
    show_and_save('samples_epoch_%d.png' % epoch ,make_grid((samples.data[:,2:3,:,:]).cpu(),8))
    show_and_save('Error_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]-rec_imgs.data[:,2:3,:,:]).cpu(),8))

    #localtime = time.asctime( time.localtime(time.time()) )
    #D_real_list_np=(D_real_list).to('cpu')

##########save and plot###############
save_model(epoch, G.encoder, G.decoder)    
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()  