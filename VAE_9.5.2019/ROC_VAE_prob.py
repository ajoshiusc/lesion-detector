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

pret=0


    
def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
  

#####read data######################
d=np.load('/big_disk/akrami/Projects/lesion_detector_data/VAE_GAN/data_24_ISEL.npz')
X = d['data']

X_data = X[0:15*20, :, :, 0:3]
max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))
X_data = X_data/ max_val
X_data = X_data.astype('float64')
X_valid=X_data[:,:,:,:]
D=X_data.shape[1]*X_data.shape[2]
####################################



##########train validation split##########
batch_size=8


X_valid = np.transpose(X_valid, (0, 3, 1,2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference= validation_data_inference.to('cuda') 


Validation_loader_inference = torch.utils.data.DataLoader(validation_data_inference,
                                          batch_size=batch_size,
                                          shuffle=False)
                                         
############################################


##########define network##########
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



#################################

########## intilaize parameters##########        
# define constant
input_channels = 3
hidden_size = 64
max_epochs = 200
lr = 3e-4
beta = 0
device='cuda'
#########################################
epoch=49
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/result_VAE_prob'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)




##########define prob loss##########
def prob_loss_function(recon_x,var_x, x, mu, logvar):
    
    var_x=var_x+0.0000000000001
    std = var_x.mul(0.5).exp_()
    std=std+0.0000000000001
    #std_all=torch.prod(std,dim=1)
    const=(-torch.sum(var_x,(1,2,3)))/2
    #const=const.repeat(10,1,1,1) ##check if it is correct
    x_temp=x.repeat(10,1,1,1)
    term1=torch.sum((((recon_x-x_temp)/std)**2),(1, 2,3))
    
 
    #term2=torch.log(const+0.0000000000001)
    prob_term=const+(-(0.5)*term1)
    
    BBCE=torch.sum(prob_term/10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE +KLD

####################################

##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            seg = X[ind:ind + batch_size, :, :, 3]
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            mean, logvar, rec_enc, var_enc = G(data)



            tem_rec_enc=rec_enc.view(8,10,3,128,128)
            tem_var_enc=var_enc.view(8,10,3,128,128)
            std2=tem_var_enc.exp_()
            mu_all=torch.mean(tem_rec_enc,(1))
            mu2_all=torch.mean((tem_rec_enc**2),(1))
            std2=torch.mean(std2,(1))

            std_all=std2+mu2_all-((mu_all)**2)
            
            thesh_all_upp=mu_all+3*(std_all**(0.5))
            #print(torch.max(thesh_all_upp))
            err=data-thesh_all_upp
            err[err<=0]=0
            
            f_recon_batch=mu_all[:,2,:,:]
            f_data=data[:,2,:,:]
            err = 4*err[:, 2, :, :]
            sig_plot=3*((std_all**(0.5))[:,2,:,:])
            print(torch.max(sig_plot))
            print(torch.min(sig_plot))
            
            if i<20:
                n = min(f_data.size(0), 100)
                err_rec=(err.view(batch_size,1, 128, 128)[:n])
                
                median=(err_rec).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
                scale_error=np.max(median,axis=2)
                scale_error=np.max(scale_error,axis=2)
                scale_error=np.reshape(scale_error,(-1,1,1,1))
                err_rec=median*4
                #err_rec=median/scale_error
                err_rec=torch.from_numpy(err_rec)
                err_rec=(err_rec).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],
                    err.view(batch_size, 1, 128, 128)[:n],
                    sig_plot.view(batch_size, 1, 128, 128)[:n],

                    seg.view(batch_size, 1, 128, 128)[:n]
                ])
                save_image(comparison.cpu(),
                           'result_VAE_prob_valid/reconstruction_b' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = err
            else:
                rec_error_all = torch.cat([rec_error_all, err])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    