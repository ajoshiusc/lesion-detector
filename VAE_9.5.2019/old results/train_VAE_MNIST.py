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
import VAE_models_MNIST
import math
from torchsummary import summary
import torchvision

seed = 10009
epochs =500
batch_size = 112
bs=112
log_interval = 10
beta=0
sigma=1
z=32
torch.manual_seed(10001)
model = VAE_models_MNIST.VAE(z)
model.have_cuda = True
model.cuda()
device = torch.device("cuda")


print(model)
#summary(model, (3, 128, 128))
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

#batch_size =16
###
transform=transforms.Compose([
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    
])

transform_anom = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

FRAC_ANOM=0
train_data_gen = datasets.MNIST(root='./data',
                                train=True,
                                transform=transform_anom,
                                download=True)
anom_dataset = datasets.FashionMNIST(root='./mnist_data/',
                               train=True,
                               transform=transform_anom,
                               download=True)

anom_dataset = torch.utils.data.Subset(
    anom_dataset, range(int(FRAC_ANOM * len(train_data_gen))))


train_data_gen = train_data_gen + anom_dataset


valid_data_gen = datasets.MNIST(root='./data',
                                train=False,
                                transform=transform_anom,
                                download=True)
anom_dataset2 = datasets.FashionMNIST(root='./mnist_data/',
                               train=False,
                               transform=transform_anom,
                               download=True)
valid_data_gen = torch.utils.data.Subset(
   valid_data_gen, range(int(112)))
anom_dataset2 = torch.utils.data.Subset(
    anom_dataset2, range(int(FRAC_ANOM * len(valid_data_gen ))))



valid_data_gen=valid_data_gen+anom_dataset2
train_loader = torch.utils.data.DataLoader(dataset=train_data_gen,
                                           batch_size=bs,
                                           shuffle=True,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data_gen,
                                           batch_size=bs,
                                           shuffle=True,
                                           drop_last=True)

# define constant





data= next(iter(valid_loader))
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

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,D)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD


def train_(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    rec_imgs,_,_ = model(fixed_batch)
    show_and_save('Input_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data).cpu(),8))
    show_and_save('rec_epoch_cats_%d.png' % epoch ,make_grid((rec_imgs.data).cpu(),8))
    show_and_save('Error_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data-rec_imgs.data).cpu(),8))
         
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len((train_loader.dataset))))
    return (train_loss / len((train_loader.dataset)))   
    
    

  
    



train_loss_list = []
valid_loss_list = []
best_loss = np.inf
for epoch in range(1, epochs + 1):
        train_loss =train_(epoch)

#save_model('cats_%d' % epoch, G.encoder, G.decoder, D)    
    #print (localtime)
    