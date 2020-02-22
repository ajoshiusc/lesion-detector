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
pret=0
random.seed(8)

            
def show_and_save(file_name,img):
    f = "/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/VAE_hiseq/%s.png" % file_name
    save_image(img[2:3,:,:],f,range=[0,1.5])
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
   
    
def load_model(epoch, encoder, decoder):
    #  restore models
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    



d=np.load('data__maryland_histeq.npz')
X=d['data']
#X = np.transpose(X, (0, 2, 3,1))

#max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))
#X = X/ max_val
X = X.astype('float64')
#X=np.clip(X,0,1)
D=X.shape[1]*X.shape[2]
print(np.min(X))
X=X[0:2380,:,:,:]
batch_size =16


d=np.load('data__TBI_histeq.npz')
X=np.concatenate((X,d['data']),axis=0)
X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10002,shuffle=False)

#fig, ax = plt.subplots()
#im = ax.imshow(X_train[0,:,:,0])
print(np.mean(X_valid[:,:,:,0]))
#plt.show()
X_train = np.transpose(X_train, (0, 3, 1,2))
X_valid = np.transpose(X_valid , (0, 3, 1,2))


input = torch.from_numpy(X_train).float()
#input = input.to('cuda') 

validation_data = torch.from_numpy(X_valid).float()
#validation_data = validation_data.to('cuda') 

torch.manual_seed(7)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)
#####

class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, representation_size = 64):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.features = nn.Sequential(
            # nc x 128x 128
            nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(),
            # hidden_size x 64 x 64
            nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 2),
            nn.ReLU(),
            # hidden_size*2 x 32 x 32
            nn.Conv2d(representation_size*2, representation_size*4, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 4),
            nn.ReLU())
            # hidden_size*4 x 16x 16
            
        self.mean = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 16 x 16
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2)
        self.act1 = nn.Sequential(nn.BatchNorm2d(256),
                                  nn.ReLU())
            # 256 x 32 x 32
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU())
            # 128 x 64 x 64
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.ReLU())
            # 32 x 128 x 128
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
        self.deconv5 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
            # 3 x 128 x 128
        self.activation = nn.Tanh()
        self.relu=nn.ReLU()
            
    
    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 32, 32))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 64, 64))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 128, 128))
        output = self.act3(output)
        output=self.activation(output)

        output_mu = self.deconv4(output, output_size=(bs, 3, 128, 128))
        #output_mu= self.activation(output_mu)

        output_logvar = self.deconv5(output, output_size=(bs, 3, 128, 128))
        #output_sig= self.activation(output_sig)
        return output_mu, output_logvar



class VAE_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(256, 16, 16)):
        super(VAE_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        for i in range (10):
            reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

            reparametrized_noise = mean + std * reparametrized_noise
            if i==0:
                rec_images,var_image = self.decoder(reparametrized_noise)
            else:
                rec_images_tmp,var_image_tmp=self.decoder(reparametrized_noise)
                rec_images=torch.cat([rec_images,rec_images_tmp],0)
                var_image=torch.cat([var_image,var_image_tmp],0)
        return mean, logvar, rec_images,var_image

# define constant
input_channels = 3
hidden_size = 64
max_epochs =100
lr = 3e-4
beta = 0


G = VAE_Generator(input_channels, hidden_size).cuda()
opt_enc = optim.Adam(G.parameters(), lr=lr, weight_decay=0.01)


fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data= next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()




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

def prob_loss_function(recon_x,var_x, x, mu, logvar):
    
    var_x=var_x
    std = var_x.mul(0.5).exp_()
    std=std
    #std_all=torch.prod(std,dim=1)
    const=(-torch.sum(var_x,(1,2,3)))/2
    #const=const.repeat(10,1,1,1) ##check if it is correct
    x_temp=x.repeat(10,1,1,1)
    term1=torch.sum((((recon_x-x_temp)/std)**2),(1, 2,3))
    const2=-((128*128*3)/2)*math.log((2*math.pi))
    
 
    #term2=torch.log(const+0.0000000000001)
    prob_term=const+(-(0.5)*term1)+const2
    
    BBCE=torch.sum(prob_term/10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE +KLD



if pret==1:
    load_model(499, G.encoder, G.decoder)


train_loss=0
valid_loss=0
valid_loss_list, train_loss_list= [], []
for epoch in range(max_epochs):
    train_loss=0
    valid_loss=0
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0

        mean, logvar, rec_enc, var_enc = G(datav)
        prob_err=prob_loss_function(rec_enc,var_enc, datav, mean, logvar) 
        err_enc = prob_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss+=prob_err.item()
    train_loss /= len(train_loader.dataset)
        

    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            mean, logvar, valid_enc, valid_var_enc = G(data)
            beta_err=prob_loss_function(valid_enc,valid_var_enc, data, mean, logvar)
            valid_loss+=beta_err.item()
        valid_loss /= len(Validation_loader.dataset)

    if epoch==0:
        best_val=valid_loss
    elif (valid_loss < best_val):
        save_model(epoch, G.encoder, G.decoder)
        best_val=valid_loss


    print(valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    _, _, rec_imgs,var_img = G(fixed_batch)
  
    show_and_save('Input_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[0:8,2:3,:,:]).cpu(),8,range=[0,1.5]))
    show_and_save('rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data[0:8,2:3,:,:]).cpu(),8))
    #samples = G.decoder(fixed_noise)
    #show_and_save('samples_epoch_%d.png' % epoch ,make_grid((samples.data[0:8,2:3,:,:]).cpu(),8))
    show_and_save('Error_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[0:8,2:3,:,:]-rec_imgs.data[0:8,2:3,:,:]).cpu(),8))

    #localtime = time.asctime( time.localtime(time.time()) )
    #D_real_list_np=(D_real_list).to('cpu')
#save_model(epoch, G.encoder, G.decoder) 
# 
save_model(epoch, G.encoder, G.decoder)   
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()  