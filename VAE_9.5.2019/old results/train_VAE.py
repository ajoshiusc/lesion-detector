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
from torchvision.datasets import ImageFolder
import VAE_models
import math
from torchsummary import summary

seed = 10009
epochs =500
batch_size = 32
log_interval = 10
beta=0
sigma=1
z=4
model = VAE_models.VAE_nf(z)
model.have_cuda = True
model.cuda()
device = torch.device("cuda")


print(model)
summary(model, (3, 128, 128))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def show_and_save(file_name,img):
    f = "./%s.png" % file_name
    save_image(img[:,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder, D):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(D.cpu().state_dict(), 'VAE_GAN_D_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
    D.cuda()
    
def load_model(epoch, encoder, decoder, D):
    #  restore models
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    D.load_state_dict(torch.load('VAE_GAN_D_%d.pth' % epoch))
    D.cuda()

batch_size =16
###
path = '/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/'
kwargs = {'num_workers': 0, 'pin_memory': True} 

simple_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize([0.48829153, 0.45526633, 0.41688013],
    #                     [0.25974154, 0.25308523, 0.25552085])
])
train = ImageFolder(path + 'train/', simple_transform)
valid = ImageFolder(path + 'valid/', simple_transform)
train_data_gen = torch.utils.data.DataLoader(train,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             num_workers=kwargs['num_workers'])
valid_data_gen = torch.utils.data.DataLoader(valid,
                                             batch_size=batch_size,
                                             num_workers=kwargs['num_workers'])

dataset_sizes = {
    'train': len(train_data_gen.dataset),
    'valid': len(valid_data_gen.dataset)
}
dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}

# define constant





data= next(iter(dataloaders ['valid']))
fixed_batch = Variable(data[0]).cuda()
#pret=0
#if pret==1:
    #load_model(499, G.encoder, G.decoder, D)





def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,D):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*D/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):
    D=x.shape[2]*x.shape[3]*x.shape[1]
    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,D)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD


def train_(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloaders ['train']):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data=data[0]
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = beta_loss_function(recon_batch, data, mu, logvar,beta)
        loss.backward()
        if torch.isnan(loss):
            print(loss)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloaders ['train'].dataset),
                100. * batch_idx / len(dataloaders ['train']),
                loss.item() / len(data)))
    rec_imgs,_,_ = model(fixed_batch)
    show_and_save('Input_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data).cpu(),8))
    show_and_save('rec_epoch_cats_%d.png' % epoch ,make_grid((rec_imgs.data).cpu(),8))
    show_and_save('Error_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data-rec_imgs.data).cpu(),8))
         
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len((dataloaders ['train'].dataset))))
    return (train_loss / len((dataloaders['train'].dataset)))   
    
    

  
    



train_loss_list = []
valid_loss_list = []
best_loss = np.inf
for epoch in range(1, epochs + 1):
        train_loss =train_(epoch)

#save_model('cats_%d' % epoch, G.encoder, G.decoder, D)    
    #print (localtime)
    