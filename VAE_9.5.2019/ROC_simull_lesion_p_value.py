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
import matplotlib.pyplot as plt
import scipy.signal
from VAE_model_pixel import Encoder,Decoder,VAE_Generator
import scipy.stats as st
from sklearn.model_selection import train_test_split
import random
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
random.seed(8)

pret=0

#p_values = scipy.stats.norm.sf(abs(z_scores))*2


def lesion_generator():
    lesion = np.zeros((1,3,128,128))
    indx = random.randint(32, 96)
    indy = random.randint(32, 96)
    centr = ((np.array([indx, indy]))[:, None]).T
    #nsamples = (np.array([10, 10]))[:, None

    blob, _ = make_blobs(
            n_samples=10,
            n_features=2,
            centers=centr,
            cluster_std=random.uniform(0, 5))

    #blob = np.int16(
            #np.clip(np.round(blob), [0, 0],
                #np.array(t1.shape) - 1))
    blob=np.round(np.int16(blob))          
    lesion[:,:,blob[:, 0], blob[:, 1]]= 1.0
            #lesion[blob.ravel]=1.0

    lesion = gaussian_filter(lesion, 5)
    lesion /= lesion.max()
    return lesion

def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
  

#####read data######################
d=np.load('data__maryland_histeq.npz')
X=d['data']

#X_data = X[:, :, :, 0:3]
#max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))
#X = X/ max_val
X= X.astype('float64')
X=X[0:2380,:,:,:]
X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10002,shuffle=False)
X=X_valid
D=X.shape[1]*X.shape[2]
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



########## intilaize parameters##########        
# define constant
input_channels = 3
hidden_size = 64
max_epochs = 200
lr = 3e-4
beta = 0
device='cuda'
#########################################
epoch=21
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/VAE_hiseq'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)




##########define prob loss##########
def prob_loss_function(recon_x,var_x, x, mu, logvar):
    
    var_x=var_x+0.0000000000001
    std = var_x.mul(0.5).exp_()
    std=std+0.0000000000001
    const=(-torch.sum(var_x,(1,2,3)))/2
    x_temp=x.repeat(10,1,1,1)
    term1=torch.sum((((recon_x-x_temp)/std)**2),(1, 2,3))
    prob_term=const+(-(0.5)*term1)
    
    BBCE=torch.sum(prob_term/10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE +KLD

####################################

##########TEST##########
def Validation(X):
    G.eval()

    test_loss = 0
    ind_list=[]
    ind=0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            l1=random.randint(1,5)-1
            blb=lesion_generator()
            temp_img=(data[l1:l1+1,:,:,:]).cpu().numpy()
            temp_img[blb>0.1]=blb[blb>0.1]
            data [l1,:,:,:]=torch.from_numpy(temp_img)
            seg=np.copy(temp_img)
            seg_all=np.zeros((8,1,128,128))
            seg[blb<0.1]=0
            seg[blb>0.1]=1
            seg_all[:,:,:,:]=seg[:,0,:,:]
            seg_all=seg_all.astype('float32')
            seg_all= torch.from_numpy(seg_all)
            seg_all = (seg_all).to(device)
           
            mean, logvar, rec_enc, var_enc = G(data)



            tem_rec_enc=rec_enc.view(10,-1,3,128,128)
            tem_var_enc=var_enc.view(10,-1,3,128,128)
            std2=tem_var_enc.exp_()
            mu_all=torch.mean(tem_rec_enc,(0))
            mu2_all=torch.mean((tem_rec_enc**2),(0))
            std2=torch.mean(std2,(0))

            std_all=std2+mu2_all-((mu_all)**2)
            
            
            
            f_recon_batch=mu_all[:,2,:,:]
            f_data=data[:,2,:,:]
            sig_plot=((std_all**(0.5))[:,2,:,:])
            z_value=(f_data-f_recon_batch)/sig_plot
            
            
            if i<200:
                n = min(f_data.size(0), 100)
                err_rec=(z_value.view(batch_size,1, 128, 128)[:n])
                
                ##########median filtering#############
                median=(err_rec).to('cpu')
                median=median.numpy()
                median=1-st.norm.sf(abs(median))*2
                
                median=scipy.signal.medfilt(median,(1,1,7,7))
                scale=0.05/(128*128)
                median[median<1-scale]=0
                median=median.astype('float32')
                err_rec=torch.from_numpy(median)
                err_rec=(err_rec).to(device)
                ############save_images##############
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],

                    (f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    sig_plot.view(batch_size, 1, 128, 128)[:n],
                    err_rec.view(batch_size, 1, 128, 128)[:n],
                    seg_all.view(batch_size, 1, 128, 128)[:n]
                ])
                save_image(comparison.cpu(),
                           'VAE_hiseq/reconstruction_bst' +str(i)+ '.png',
                           nrow=n)
           #############save z values###############
            if i==0:
                rec_error_all = z_value+0
            else:
                rec_error_all = torch.cat([rec_error_all, z_value])

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all,ind_list,seg_all_all


if __name__ == "__main__":
    rec_error_all,ind_list,seg_all_all = Validation(X)
    y_true = X[0:15*20 ,:, :, 3]
    y_true = np.reshape(y_true, (-1, 1))
    
    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    
    y_probas =np.reshape(y_probas, (-1, 1,128,128))
    median=1-st.norm.sf(abs(y_probas))*2
    median=median/(128*128)
    #scale=0.05/(128*128)
    median=scipy.signal.medfilt(y_probas,(1,1,7,7))
    median=median.astype('float32')
               
            
    y_probas=median+0
    y_probas[y_probas >0]=1
    

    y_probas = np.reshape(y_probas, (-1,128*128*20))
    y_true = np.reshape(y_true, (-1,128*128*20))
    
    dice=0
    for i in range(y_probas.shape[0]):
        seg=y_probas[i,:]
        gth=y_true[i,:]
        dice += np.sum(seg[gth==1])*2.0 / (np.sum(gth) + np.sum(seg))
        #print((dice))
    print((dice)/y_probas.shape[0])
    